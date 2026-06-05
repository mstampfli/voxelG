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

## Deliberately deferred (with rationale)
- Full reprojection/disocclusion TAA (skip tracing reprojectable tiles on motion) —
  the same-pixel TAA here gives the AA win safely; checkerboard/disocclusion needs
  visual tuning that isn't possible headless.
- Dedicated half-res volumetric pass (#14) and a transparent/foliage second pass
  (#16) — both are larger GPU-profiler-driven refactors; cheaper equivalents shipped.
- GPU compute physics (#25) — shipped as a design doc; the CPU path now runs off the
  frame thread.
