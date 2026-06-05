# voxelG — Improvement checklist: implementation status

Every item from `voxelG-improvements.md`, what was done, and where. Cross-platform
(Linux) was a goal throughout: the engine builds and runs on Linux (Wayland/X11,
Vulkan) — verified building + creating a Vulkan device + rendering on an RTX 5060.

Validation tooling added so changes are checkable without a display:
- **Headless WGSL validation** (`naga`) for every shader.
- **Headless GPU render+readback** test: renders the real raymarch pipeline
  offscreen and asserts terrain+sky at spawn AND at 3.2 M voxels from origin.
- Deterministic-worldgen, edit-replay-roundtrip, net (compress/interp) unit tests,
  an in-process TCP+UDP integration test, and criterion microbenchmarks.

## Frame loop
- **force_full on camera move** → only a *real* move (>1 mm / >1e-4 rad) forces a
  full re-trace; sub-mm jitter no longer re-traces every tile. `src/app.rs`.
- **15-frame forced full** → removed; replaced by an evenly-spread rotating
  animation refresh (1/8 of tiles per frame) so idle animation has no stutter.
- **per-frame `dirty_bricks` / `active_bricks` clones** → gone. Tiles are
  projected straight from `world.dirty_bricks`; physics reuses
  `World::phys_scratch`/`phys_touched` (zero per-tick allocation).
- **`about_to_wait` always redraws** → `ControlFlow::WaitUntil` frame pacing (144 Hz
  cap) + redraw-on-input; present mode is Mailbox/FIFO (vsync), not Immediate.
- **`unwrap`/`expect` on GPU/window/surface** → `Renderer::new` returns `Result`;
  window/adapter/device failures log + exit; `render()` handles Lost/Outdated/
  Timeout/OutOfMemory.

## Streaming / upload
- **all_dirty re-uploads ~75 MB; chunk cross must avoid it** → chunk crossing only
  ever takes the incremental path; all_dirty is first-frame only.
- **incremental path re-uploaded whole masks** → fully incremental: brick spans +
  only the touched packed words of brick/tile uniform tables and the touched
  tile/chunk/L4 mask entries (derived from the dirty-brick list). Also fixed a
  latent bug where the uniform-skip tables were never refreshed incrementally
  (streamed chunks rendered with stale skip data).
- **gen on the calling thread** → background worker pool; `shift_origin` dispatches
  jobs, `install_finished_chunks` stitches results in with stale-result rejection.
- **no upload budget / prefetch** → per-frame chunk-install budget + 2-chunk
  hysteresis deadband; the centred window prefetches in all directions.

## Rendering / GPU (`shaders/raymarch.wgsl`)
- **nested hierarchical DDA + L4** → trace/trace_any/trace_no_water descend
  L4(256) → chunk(64) → tile(16) → brick(4) → voxel, skipping the coarsest empty
  cell each step. New L4 occupancy level (one u64 per 256³ region) added end to end.
- **voxel step budget → distance-based** → `MAX_RAY_DIST`.
- **secondary rays uncapped** → shadow rays use the full hierarchy + `SHADOW_MAX_DIST`
  give-up + treat far foliage as a solid blocker. *(Cache+reproject of the shadow/AO
  term is partially covered by TAA temporal accumulation; a dedicated reprojected
  shadow cache is left as a follow-up.)*
- **foliage in both trace passes** → procedural sub-voxel foliage only within
  `FOLIAGE_NEAR_T`; far foliage is a solid cube (and a solid shadow blocker).
- **clouds/god-rays full-res** → sample counts halved; TAA accumulates them across
  frames (temporal upsample). *(A dedicated half-res volumetric pass is the further
  optimisation.)*
- **no AA** → TAA: sub-pixel jitter while static + a resolve pass that blends with
  history under a 3×3 neighbourhood colour clamp (no ghosting), reset on motion.
- **8×8 divergence** → reduced by near-only foliage. *(A full transparent/foliage
  second/deferred pass was deliberately not done — high-risk rewrite, needs a GPU
  profiler to justify; near-only foliage removes the worst divergence.)*
- **LOD** → far terrain terminates at brick (`LOD_BRICK_T`) then tile granularity
  (`TILE_LOD_T`, via a tile-representative material).

## Correctness / numerical
- **rebase to window-relative coords** → all float DDA math is done relative to the
  window corner (`camera.world_origin`) while the integer voxel grid stays absolute.
  Removes the f32 cancellation / "sky through hills" bug; bias hack reduced 1e-2→1e-3.
  Verified: byte-identical at spawn, correct at 3.2 M voxels out (f32 ulp ≈ 0.4 voxel,
  where the un-rebased DDA cannot resolve a boundary).
- **no wire protocol versioning** → `PROTOCOL_VERSION` in the handshake; server
  rejects mismatched clients.

## Networking
- **all-TCP** → split transports: pose over **UDP** (token-authenticated, lossy-ok),
  edits/control/replay/heartbeat over **TCP**. (Raw UDP rather than QUIC to avoid a
  heavy async dependency in a student project.)
- **no interpolation** → per-player interpolation buffer; remotes render ~100 ms in
  the past, lerped between samples.
- **late-joiner desync** → authoritative **ordered, seq-numbered** edit log; replayed
  in order to joiners, **deflate-compressed** (`flate2`).
- **no heartbeat / reconnection / compression / ordering** → heartbeat + timeout
  sweep, client auto-reconnect (re-handshake + edit-log re-sync), compression, and
  seq-versioned edits all added.

## Physics
- **synchronous + active_bricks clones** → physics runs on a dedicated worker thread
  behind `Arc<Mutex<World>>`; the render thread only locks briefly. Per-tick clones
  removed (reusable scratch). *(A fully lock-free double-buffered render snapshot is
  the documented next step; the brief-lock model already takes the tick off the frame
  path.)*
- **CA on CPU; GPU compute candidate** → `docs/gpu-physics-design.md`: a concrete,
  actionable design (pull-only, double-buffered, integer-only CA, indirect-dispatch
  active set, determinism analysis, incremental sand→smoke→water migration).

## Architecture / hygiene
- **constants duplicated in Rust + WGSL** → single source `src/world_dims.rs`;
  `build.rs` generates the matching WGSL consts prepended to every shader.
- **monolithic main.rs** → `lib.rs` + `app.rs` (client/frame loop) + `server.rs`
  (dedicated server) + thin `main.rs` launcher.
- **zero tests/benchmarks** → see the validation tooling list at the top.

## Performance notes (measured)

Headless GPU timing of the raymarch pass (RTX 5060, `cargo test --lib
raymarch_timing -- --ignored`), high camera looking down (worst case — most of
the screen is terrain):

| | on AC power | on battery |
|---|---|---|
| 1280×720 | **5.5 ms (~180 fps)** | ~80 ms (~12 fps) |
| 640×360  | **1.5 ms (~650 fps)** | ~23 ms (~44 fps) |

The engine is GPU-light on AC (~5.5 ms/frame for the raymarch). **Laptop battery
throttling is ~14×** — measure/play on AC. Cost breakdown at 1280×720 on battery:
~41 ms base traversal, +8 ms for the L4/chunk coarse skips (net-negative on
terrain-heavy views, a win on sky/empty views), +31 ms shading (shadows + AO +
clouds + god-rays — all pre-existing per-pixel secondary rays).

## Known issues / remaining work

- **Chunk-load lag spike.** Crossing a chunk boundary still hitches. The
  *generation* is async (worker pool), but the *install* — `apply_slot_bricks`
  (per-brick movable/uniform recompute + `refresh_masks_for_brick`) for up to
  `CHUNK_INSTALL_BUDGET` (6) chunks/frame, plus the resulting brick upload burst
  — still runs on the main thread under the world lock. Fix paths: (a) compute
  masks/uniform on the worker so the main thread only memcpy-installs; (b) lower
  the per-frame install budget; (c) spread the brick upload across more frames.
- **L4 per-step cost.** The L4/chunk coarse checks run every voxel step; on
  terrain-dense views that's ~+8 ms with no skip benefit. A true *nested* march
  (descend once, step within a level, ascend) would make the coarse levels pay
  for themselves instead of costing per-step — the bigger traversal rewrite.
- **Shading is brute-forced per pixel per frame** (shadows/AO/god-rays re-traced
  every frame). Amortizing them via a reprojected shadow/AO cache (checklist #12,
  only partially done) is the path to small-voxel/LOTL-class perf.

## Deliberately deferred (with rationale)
- Full reprojection/disocclusion TAA (skip tracing reprojectable tiles on motion) —
  the same-pixel TAA here gives the AA win safely; checkerboard/disocclusion needs
  visual tuning that isn't possible headless.
- Dedicated half-res volumetric pass (#14) and a transparent/foliage second pass
  (#16) — both are larger GPU-profiler-driven refactors; cheaper equivalents shipped.
- GPU compute physics (#25) — shipped as a design doc; the CPU path now runs off the
  frame thread.
