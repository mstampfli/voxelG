# Plan: reach 10 cm voxels (2.5× per axis ≈ 15.6× voxels)

Goal: shrink the voxel from 25 cm to 10 cm — 2.5× finer per axis, **15.6× more
voxels** (2.5³) for the same loaded world extent. Target budget: ≤ 8 GB RAM,
≤ 5 GB VRAM. This is reachable today; it's mostly a coordinated dims bump + a few
tuning knobs, not new architecture. Steps are ordered and each is independently
validatable (headless render test + `raymarch_timing`).

## Target dimensions

Current (`src/world_dims.rs`) → 10 cm target (keep multiples of 8 so the storage
chunk / tile / chunk / L4 hierarchy stays exact):

| const | now | 10 cm (×2.5) |
|---|---|---|
| `WORLD_BRICKS_X/Y/Z` | 128 / 64 / 128 | **320 / 160 / 320** |
| `WORLD_VOXELS` (=×4) | 512 / 256 / 512 | 1280 / 640 / 1280 |
| `WORLD_BRICKS_TOTAL` | 1.05 M | 16.4 M |
| total voxels | 67 M | 1.05 B |
| `WORLD_STORE_CHUNKS` | 2048 | 32 000 |

320/160/320 are all ÷8 (storage chunk) and ÷4 (tile), so `WORLD_TILES`
(80/40/80), `WORLD_CHUNKS` (20/10/20) and `WORLD_L4` stay clean. Everything else
in `world_dims.rs` is derived, so this is the **only** dims edit — it propagates
to Rust storage, the CPU raycaster, the temporal projector and all shaders via
`build.rs`.

## Memory budget (fits)

| buffer | now | 10 cm | notes |
|---|---|---|---|
| bricks (`Brick`=72 B) | 75 MB | **1.18 GB** | the big one |
| bricks_b (GPU physics) | 75 MB | 1.18 GB | **only if `VOXELG_GPU_PHYSICS` on** |
| brick_uniform (1 B) | 1 MB | 16 MB | |
| tile_mask (8 B) | 128 KB | 2 MB | |
| chunk_mask / L4 | tiny | tiny | |
| light G-buffers, cloud, output | ~35 MB | ~35 MB | per-pixel, **density-independent** |

- VRAM: **~1.25 GB** physics-off, **~2.45 GB** physics-on → both under 5 GB.
- RAM (CPU `world.bricks` + `movable_mask` + scratch): **~1.4 GB** → under 8 GB.

**Action item:** make `bricks_buf_b` allocation conditional on `gpu_physics`
(currently always allocated in `Renderer::new`) so physics-off doesn't waste
1.18 GB of VRAM.

## Step-by-step

1. **Raise GPU limits** (`src/renderer.rs`, both device requests + the headless
   test device): a single 1.18 GB storage buffer needs
   `max_storage_buffer_binding_size ≥ ~1.3 GB` and `max_buffer_size ≥ ~1.3 GB`.
   Query `adapter.limits()` and assert the GPU supports it (Blackwell/Vulkan
   `maxStorageBufferRange` is multi-GB, so fine on the RTX 5060). Fail loudly if
   an adapter can't.
2. **Bump `world_dims.rs`** to 320/160/320. Rebuild — `build.rs` regenerates the
   WGSL consts. The headless render test + far-origin test should still pass
   (they scale automatically); this validates the whole pipeline at the new size.
3. **Worldgen detail vs. extent decision.** Terrain noise is sampled in
   voxel-space. To render the *same world* at 10 cm detail (not a 2.5×-bigger
   world), divide the terrain-noise input coordinate by 2.5 in `gen_slot_bricks`
   / `sample_terrain` (features then span 2.5× more voxels = same physical size,
   finer steps). If instead you want a bigger world at the same feel, leave it.
4. **Ray budget + LOD in voxel units** (`shaders/raymarch.wgsl`): `MAX_RAY_DIST`,
   `SHADOW_MAX_DIST`, `LOD_BRICK_T`, `TILE_LOD_T`, `FOLIAGE_NEAR_T`, `AO_DIST`,
   `GOD_RAY_OCCL_DIST`, and the cloud slab heights are all in voxels. To keep the
   same *physical* view distance, ×2.5 them; or leave them for a 2.5×-shorter
   view (cheaper). Tune LOD aggressiveness up — far terrain at brick/tile
   granularity is what keeps traversal sub-linear (≈2× hit, not 2.5×).
5. **Startup generation** (`World::fill_demo_terrain`): it synchronously gens the
   whole window (2048 → 32 000 storage chunks ≈ 16 M bricks). That's a multi-
   second blocking startup at 10 cm. Either (a) gen only the chunks near spawn
   synchronously and let the async worker pool stream the rest, or (b) show a
   loading state. The first frame's `all_dirty` upload is also 1.18 GB once.
6. **Streaming budget** (`src/app.rs`): each storage chunk now carries 15.6× the
   brick data. Lower `CHUNK_INSTALL_BUDGET` and/or spread the brick upload across
   more frames so a chunk cross doesn't spike; raise the worker count for gen
   throughput. The async install + worker-computed masks already done in the
   chunk-load-spike fix carry over — just re-tune the budget.
7. **Physics** (`#25`): 15.6× more active cells makes the CPU CA the bottleneck.
   Enable the GPU physics path (`VOXELG_GPU_PHYSICS`) and finish its next stages:
   the CPU↔GPU readback so raycast picking is correct, then water/smoke. This is
   the subsystem that *requires* the GPU at 10 cm.
8. **Re-measure on AC**: `raymarch_timing` for traversal/shading, a real session
   for streaming + chunk-load smoothness, and confirm the static-only
   reprojection caches still behave (they're per-pixel/res, so density-neutral).

## What scales how (so expectations are right)

- **Shading** (shadows-blend, AO composite, clouds, TAA, material): per-pixel →
  **density-independent**. Win B/C/D already paid this tax. No change at 10 cm.
- **Traversal + shadow/AO/god-ray walks**: ~2.5× (ray length in voxels), offset
  by LOD + Win A's incremental slot → realistically **~2× worst case**, less with
  aggressive LOD. Survivable on AC.
- **Memory**: 15.6× on the brick array — the one true 3D-multiplied cost, but it
  fits the budget.
- **Worldgen + upload**: 15.6× data per chunk → the startup + streaming costs are
  where the real engineering is (steps 5–6), not the steady-state frame.
- **Physics**: 15.6× cells → GPU physics (step 7).

## Risks / watch-list

- **Startup time** (step 5) is the most likely regression — 16 M bricks gen +
  1.18 GB upload on frame 1.
- **Chunk-load hitch** could return under 15.6× per-chunk data; re-tune budget.
- **f32 precision**: already handled — origin-rebasing is window-relative and is
  tested at 3.2 M voxels from origin; coords are 2.5× larger in voxel units but
  still rebased, so no new precision work.
- **Binding limit**: if a future GPU reports `maxStorageBufferRange` < 1.3 GB,
  the brick window must be split across multiple bindings or the window shrunk —
  not a concern on the RTX 5060.

## Phasing (validate each before the next)

- **Phase A** — limits + dims to **1.5×** (192/96/192, ~270 MB): smallest real
  test that the pipeline + tests + a live run all hold at a bigger size.
- **Phase B** — full **2.5×** (320/160/320): tune ray budget + LOD + streaming
  budget; fix startup gen; measure on AC.
- **Phase C** — GPU physics live (readback + water) so the CA keeps up.

A future memory optimisation (not required for 10 cm at this window, but the path
to *larger* worlds) is **sparse brick storage** — only allocate non-empty bricks
via a brickmap indirection, so empty sky/underground costs no VRAM. With the
current dense array, 10 cm fits; going much past 2.5× or to a bigger window is
when sparse storage becomes necessary.
