# voxelG

A real-time voxel engine in Rust where the entire world is rendered by a WGSL compute shader. There is no
triangle geometry for terrain: each frame, rays are marched through a three-level occupancy-bit pyramid
(chunk → tile → brick) directly on the GPU, and a fullscreen-triangle blit presents the result. The same
binary runs solo, as a headless server, or as a multiplayer client. Built on wgpu 23 and winit 0.30.

## Rendering

World rendering lives in `shaders/raymarch.wgsl`, with a coarse pre-pass in `shaders/beam.wgsl`.

Traversal and performance:

- **Hierarchical DDA** through a 3-level bit pyramid: a bit per 16³ tile in each 64³ chunk, a bit per 4³
  brick in each tile, and a 64-bit occupancy mask per brick. Empty space is skipped at 16- or 4-voxel
  granularity with exact cell-boundary snapping (`skip_to_cell`).
- **Uniform-brick / uniform-tile compaction**: CPU passes detect 4³ bricks and 16³ tiles filled with a
  single opaque material and publish packed byte LUTs; the DDA then terminates at the cell's entry face
  without descending the hierarchy.
- **Beam pre-pass**: at 1/8 resolution, one ray per 8×8 pixel block walks the world at tile granularity
  and writes the first-hit distance to an `r32float` texture; the main pass fast-forwards each ray to
  that depth minus a 16-voxel safety margin.
- **Temporal differential rendering**: dirty bricks (physics, edits) are projected into 8×8 screen tiles
  on the CPU (`src/temporal.rs`); the compute shader early-outs on clean tiles, and both compute passes
  are skipped entirely when nothing changed.
- **Distance LOD**: beyond 400 voxels the DDA terminates at brick granularity, shading with the brick's
  topmost solid material so distant terrain keeps its surface colour.
- **Scalable internal resolution** (`RENDER_SCALE` in `src/renderer.rs`) with a bilinear upscale in the
  blit; ships at native 1:1.

Shading and effects:

- **Water**: four-wave Gerstner-style normals from closed-form height-field derivatives; Schlick Fresnel
  mixes a traced reflection with a Snell-refracted trace beneath the surface (η = 1/1.33); Beer, Lambert
  per-channel absorption, shoreline foam from underwater hit distance gated by wave crests, a caustic
  approximation, specular sun glints, and a separate absorption post-effect when the camera is submerged.
- **Glass**: Fresnel reflection plus per-channel refraction for chromatic dispersion (n = 1.48/1.50/1.52),
  total-internal-reflection fallback to the reflected ray, distance-compounding tint; the 3-trace
  dispersion path is gated to grazing angles.
- **Foliage**, resolved sub-voxel inside the DDA: alpha-cutout cube leaves (procedural hashed 16×16 masks
  tested on entry and exit faces), tall grass as bundles of up to 22 tapered, bowed blades per voxel,
  flower silhouettes on crossed planes, all driven by one global wind field with per-face edge sway
  (only faces not pressed against neighbouring foliage move).
- **Lighting**: day/night sun cycle with sunset scattering, sun disc, halo and stars; 2-sample
  golden-angle PCF soft shadows jittered with interleaved gradient noise; bit-test ambient occlusion
  bilinearly interpolated across the hit face.
- **Volumetrics**: slab-raymarched cumulus clouds (fbm body under a low-frequency coverage mask, 3 cone
  samples toward the sun for self-shadowing, Henyey, Greenstein forward scattering) and god rays
  accumulated as jittered sun-visibility samples along the primary ray.
- **Tri-planar procedural materials**: world-projected luminance textures, running-bond brick on stone,
  ring and longitudinal wood grain, snow sparkle, lava cracks, continuous across voxel boundaries.

## World, simulation, multiplayer

- The world is a 512×256×512-voxel sliding window stored toroidally; crossing a chunk boundary
  regenerates only edge chunks (rayon-parallel) while the rest of GPU storage stays in place.
- Terrain from layered fbm value noise: temperature/humidity biomes (plains, forest, jungle, savanna,
  desert, tundra, beach, mountain), 3D-noise caves, ore veins, trees, rivers and sea.
- Cellular-automaton physics at 30 Hz: 8-level water (DwarfCorp-style level propagation), bitmask sand
  gravity, and rising smoke, iterating only "active" bricks that contain movable voxels.
- Multiplayer (`src/net.rs`): TCP with length-prefixed bincode messages. Each connection gets a reader
  and a writer thread bridged to the single-owner game thread through crossbeam's lock-free MPMC
  channels, no shared mutexes. World state syncs by shared seed plus a persistent edit log replayed to
  joiners; pose updates are capped at 20 Hz and fan out with distance-based interest management
  (600-voxel radius). Sphere destruction travels as a single `Explode` message expanded locally on each
  client. Remote players render as ray-traced colour-hashed boxes.

## Architecture

```
src/main.rs            event loop, input, click-to-raycast pipeline, server main loop
src/renderer.rs        wgpu setup; beam -> raymarch -> blit passes, palette, buffers
src/voxel.rs           brick/tile/chunk storage, noise terrain, biomes, streaming,
                       uniform-brick/tile compaction, edit log
src/physics.rs         sand / 8-level water / smoke cellular automata
src/temporal.rs        dirty-brick -> screen-tile projection for partial re-render
src/raycast.rs         CPU DDA for block picking (destroy/place)
src/net.rs             TCP client + server over lock-free channels
src/camera.rs          fly camera
shaders/beam.wgsl      1/8-resolution first-hit depth pre-pass
shaders/raymarch.wgsl  primary tracer and all shading (~2000 lines)
shaders/blit.wgsl      fullscreen-triangle present + crosshair
```

## Building and running

Requires a GPU and driver supported by wgpu (Vulkan, Metal or DX12).

```sh
cargo run --release                          # solo
cargo run --release -- --server 7878        # headless server
cargo run --release -- --connect host:7878  # join a server
```

Controls: WASD + Space/Shift to fly, Alt to sprint, mouse to look. Left click destroys a sphere, right
click places the selected material; keys 1, 0 select stone, sand, water, wood, leaves, glass, lava, ice,
snow or smoke. Esc releases the cursor.

## Status

Experimental graphics playground, not a game. Terrain is seed-deterministic; the server keeps its edit
log in memory only (no on-disk persistence), and remote players are placeholder markers. Constants are
tuned by eye on a single machine, expect to adjust them for yours.
