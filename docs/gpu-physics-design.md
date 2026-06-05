# GPU Compute Port of the Voxel Cellular-Automaton Physics

Design document for moving the CPU cellular-automaton physics in `src/physics.rs`
(`physics::tick`) onto a wgpu 23 compute pass. Grounded in the current code:
`World` storage in `src/voxel.rs`, the raymarch GPU upload path in
`src/renderer.rs`, and the dimensions in `src/world_dims.rs`.

Status: design only. No source changes are implied by this document.

---

## 0. What we are porting and why it is hard

`physics::tick(&mut World)` currently runs every fixed step (30 Hz, capped at 4
steps/frame in `App::render_frame`, `src/app.rs:358-365`). Each tick does, over
the `world.active_bricks` list, filtered by `world.movable_mask`:

1. **Sand gravity** — `sand_gravity_pass` / `step_brick_sand_fall`. Binary
   (occupied / not). Pure `u64` bitmask ops: intra-brick fall is
   `falling = sand & (empty << 16)`; cross-brick fall moves the bottom layer
   (`BOTTOM_LAYER = 0xFFFF`) into the top layer of the brick below.
2. **Water** — `step_brick_water`, three passes per brick on a *snapshot*:
   - Pass 1 gravity (top-down): a water cell donates its whole level downward,
     capped at `MAX_WATER_LEVEL = 8`. Level is encoded in the material id
     (`MAT_WATER_L1..MAT_WATER_L8`, see `water_level_of` / `water_mat_for_level`).
   - Pass 2 refill-from-above (bottom-up): only cells that were water in the
     snapshot pull from the cell above, so a draining column stays a connected
     beam instead of leaving mid-column gaps.
   - Pass 3 lateral spread: donate one level to a strictly-lower horizontal
     neighbour; a **per-cell integer hash** of the *global* voxel coord
     (`gx*0x9E3779B1 ^ gy*0x85EBCA77 ^ gz*0xC2B2AE3D`) picks the starting
     direction so adjacent cells don't all donate the same way.
3. **Smoke** — `step_brick_smoke`. Rises (intra- then cross-brick), spreads
   laterally if blocked, and dissipates 1/40 per tick via a per-(cell, frame)
   hash that also folds in `FRAME_COUNTER`.

Bricks are 4×4×4 voxels = one `u64` occupancy + `[u8; 64]` materials (`Brick`,
`src/voxel.rs:154`). Voxel layout inside a brick is `i = lx + lz*4 + ly*16`
(`brick_voxel_idx`), so a horizontal layer is 16 contiguous bits and `+16` is
"one cell up" — this is what makes the sand bitmask trick work and it carries
over to the GPU verbatim.

The hard parts, in order:

- **Read-after-write hazards.** On the CPU each brick is processed alone and
  reads neighbours' *current* memory; on the GPU thousands of invocations run
  concurrently and a cell must never read a neighbour another invocation is
  mid-writing.
- **Cross-brick transfers.** The CPU code freely writes into the neighbouring
  brick (`cross_apply_water`, the cross-brick branches of sand/smoke) and mutates
  shared bookkeeping (`active_bricks`, `movable_mask`, `refresh_masks_for_brick`).
  That is a data race on the GPU.
- **Determinism.** Physics is run **locally on every client**; only player edits
  (`net::Message::VoxelEdit`) and poses are networked (`src/net.rs:45`,
  `src/server.rs`). Terrain is regenerated from the shared seed and the replayed
  edit log. So *every client must compute byte-identical physics from the same
  starting state* or worlds diverge. The CPU version is integer-only and
  deterministic; the GPU version must preserve that.

The good news: **the bricks are already GPU-resident.** `renderer.rs` uploads
`world.bricks` into `bricks_buf` (`src/renderer.rs:262-267`) and keeps it in sync
incrementally via `upload_world` driven by `world.dirty_bricks`. The compute
physics pass can read and write that same data, eliminating the per-tick
72 MB round trip that a naive CPU-physics + full-reupload would cost.

---

## 1. Data layout on the GPU

### 1.1 Bricks: keep the exact `Brick` layout, double-buffer it

`Brick` is `#[repr(C)]` with `occupancy: u64` (8 B, align 8) followed by
`materials: [u8; 64]` (64 B) → **72 B per brick, 8-byte aligned**. With
`WORLD_BRICKS_TOTAL = 128*64*128 = 1,048,576` that is **72 MB** per copy — well
within the configured `max_storage_buffer_binding_size: 256 << 20`
(`src/renderer.rs:218`).

WGSL cannot declare a `u64` field. Mirror each brick as two `u32` halves plus the
already-used packed-byte pattern for materials (the renderer already packs `u8`
tables 4-per-`u32`, see `pack_u8_to_u32` and the `(packed[w] >> ((b&3)*8)) & 0xFF`
comment, `src/renderer.rs:11-24`). Brick struct in WGSL:

```wgsl
struct Brick {
    occ_lo : u32,        // low 32 bits of occupancy
    occ_hi : u32,        // high 32 bits of occupancy
    mats   : array<u32, 16>,  // 64 materials, 4 per word -> mats[i>>2] >> ((i&3)*8)
};
@group(0) @binding(0) var<storage, read>       bricks_in  : array<Brick>;
@group(0) @binding(1) var<storage, read_write> bricks_out : array<Brick>;
```

`72` is a multiple of 4 and the struct contains only `u32`, so this matches the
Rust `repr(C)` bytes exactly — `bytemuck::cast_slice(&world.bricks)` produces the
same bytes the shader reads. Helper accessors in WGSL:

```wgsl
fn brick_occ(b: Brick) -> vec2<u32> { return vec2(b.occ_lo, b.occ_hi); }
fn mat_at(b: Brick, i: u32) -> u32 { return (b.mats[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu; }
```

Note: WGSL has no 64-bit integer, so the `u64` occupancy bitmask ops in
`physics.rs` (`falling = sand & (empty << 16)`, `BOTTOM_LAYER`, `<< 48`) become
`vec2<u32>` ops. A small library of helpers (`bit_get(vec2,i)`, `bit_set`,
`shl16`, `bottom_layer()`, `top_layer()`) reproduces the exact bit semantics.
This is mechanical but must be done carefully — it is the single largest source
of "looks right but drifts" bugs. (Optional: enable `SHADER_INT64` if the target
GPU supports it; see §6. RTX 5060 / Vulkan does, but keeping a `vec2<u32>` path
preserves determinism and portability and is recommended.)

### 1.2 movable_mask: a parallel GPU buffer

`world.movable_mask: Vec<u64>` (one `u64` per brick, **8 MB**) drives the active
set on the CPU. Mirror it as `array<vec2<u32>>` in its own ping-pong pair
(`movable_in` / `movable_out`). It is recomputed cheaply from materials in the
shader, but carrying it explicitly avoids re-deriving it and matches the CPU's
invariant (`new_mask = movable_materials & occupancy`).

### 1.3 Double-buffering (ping-pong) — the core of hazard-freedom

Allocate **two** brick buffers `bricks_a`, `bricks_b` (and two movable buffers).
Each physics tick reads from `in` and writes to `out`; the next tick swaps the
bind groups. This is the safe-by-construction answer to read-after-write: an
invocation only ever **reads** `bricks_in` and **writes** `bricks_out`, so no
invocation can observe a partial write of another.

Cost: an extra 72 MB (+8 MB movable) of VRAM. On a 16 GB-class card (RTX 5060)
this is negligible. The alternative — in-place update with barriers — cannot work
across workgroups in WGSL (there is no global barrier), so ping-pong is mandatory
for cross-brick interaction.

### 1.4 Keeping the renderer's `bricks_buf` in sync

The cleanest design: **render directly from the physics buffer.** After each tick
the "current" buffer (`bricks_out` of the last tick) *is* the authoritative state.
Make the raymarch bind group point its `bricks` binding (`@binding(1)`,
`create_compute_bgl` / `make_compute_bg`, `src/renderer.rs:889-1019`) at whichever
of `bricks_a`/`bricks_b` is current. Because the bind group must reference a fixed
buffer, either:

- **(A) Two raymarch bind groups**, one per physics buffer, selected each frame by
  parity (recommended — zero copies). `make_compute_bg` already takes `bricks_buf`
  as a parameter, so build it twice.
- **(B) One stable render buffer + a copy.** Keep `bricks_buf` as today and, after
  physics, `encoder.copy_buffer_to_buffer` only the dirty spans into it. This
  reuses the existing `upload_spans` span logic but needs a GPU-built dirty list
  (§3). More moving parts; prefer (A).

The hierarchy masks (`tile_mask`, `chunk_mask`, `l4_mask`, `brick_uniform`,
`tile_uniform`) still feed the DDA. They must be refreshed wherever a brick went
empty↔non-empty or changed material uniformity. See §5.2 for how to regenerate
just the touched mask bits on the GPU instead of round-tripping to the CPU
`refresh_masks_for_brick` / `recompute_uniform_for_brick`.

---

## 2. The CA update as a compute pass: hazards and cross-brick transfers

### 2.1 The hazard, restated for the GPU

The CPU relies on three things the GPU does not give for free:
- per-brick *snapshots* (`snap_occ`, `snap_mats`, `p1_occ`, `p1_mats`) so a fall
  doesn't cascade through a column in one tick;
- strict brick processing **order** (water bottom-up, smoke top-down — see the
  `active.sort_by_key(...)` calls in `tick`) so a brick sees the *finished* state
  of its neighbour;
- exclusive mutation of the neighbour brick during a cross-brick transfer.

None of order-dependence survives parallelism. The design must remove the order
dependence, not emulate it.

### 2.2 Recommended approach: **pull-only, double-buffered, per-voxel invocation**

Dispatch **one invocation per voxel of each active brick** (workgroup = one brick
= 64 invocations, see §3/§6). Each invocation:

- computes the new state of **its own voxel only**, and
- writes **only its own slot** in `bricks_out`,
- by **reading** `bricks_in` for itself *and any neighbour it needs* (including
  voxels in adjacent bricks — a neighbour read is just another global index into
  `bricks_in`).

This is the "a cell only WRITES its own slot and PULLS from neighbours" model the
prompt names, and it is the safest. There are **no atomics and no scatter**: every
write target is unique to one invocation, so there is no contention, and every
read is from the immutable `in` buffer, so there is no read-after-write. Cross-brick
transfers stop being "writes into the neighbour" and become "the neighbour's voxel
pulls from me / I pull from the neighbour" — perfectly symmetric and race-free.

Because reads come from the snapshot-equivalent `in` buffer, the "1 cell per tick"
fall behaviour the CPU gets from its snapshot is preserved automatically: a falling
column advances exactly one cell per dispatch.

#### Re-expressing each rule as a pull

For a cell to decide its next value it must reproduce the CPU's transfer decision
from the *reader's* side. This requires that the rule be **conservative and
symmetric**: whatever cell A donates, cell B receives, and both compute the same
amount from the same inputs. The CPU rules are already mass-conserving; we just
move the decision to be computed twice (once by donor, once by receiver) instead of
once-and-scattered.

- **Sand (binary, easiest).** A cell becomes sand iff:
  *(it is sand and the cell below is occupied in `in`)* OR *(it is empty in `in`
  and the cell above is sand and the cell two-above situation permits)*. Concretely,
  next-occupancy for cell `i`:
  - falls away if `in[i]` is sand and `in[i-16]` is empty → becomes air;
  - fills if `in[i]` is empty and `in[i+16]` is sand and (`in[i+16]` would fall,
    i.e. `i` below it is empty) → becomes sand.
  These two conditions are exact mirror images, so mass is conserved with no
  atomics. Cross-brick: the same logic but `i-16`/`i+16` index into the brick
  below/above via the global brick grid. The CPU's multi-pass intra-fall (collapse
  a 4-deep stack in one tick) is **not** reproduced per dispatch — a stack falls
  one layer per tick. That is acceptable (and arguably more physically uniform);
  if single-tick collapse is required, run N inner ping-pong sub-steps per physics
  tick (cheap, just more dispatches).

- **Water (hardest).** The CPU does donate-then-pull in two passes precisely to
  keep a column connected. In the pull model, fold both into a single per-cell
  computation from `in`:
  - `inflow_from_above` = how much the cell directly above (in `in`) would donate
    downward into me this tick (gravity), computed by *me* using the same
    `level.min(space)` formula the donor uses.
  - `outflow_below` = how much *I* would donate to the cell below (same formula,
    from my side).
  - `lateral_net` = sum over the 4 horizontal neighbours of (received from that
    neighbour) − (given to it), where each pairwise amount is computed identically
    by both cells from the per-cell hash direction order (see §4). Because both
    endpoints evaluate the same `DIRS[(h+k)&3]` and the same `remaining > t_level+1`
    test against `in`, donor and receiver agree on the integer transferred.
  - `new_level = clamp(in_level − outflow_below − lateral_given + inflow_above +
    lateral_received, 0, 8)`, then `new_mat = water_mat_for_level(new_level)`.

  The subtlety the CPU solves with pass ordering (don't refill the cell that just
  received the falling drop) is handled here by computing all flows against the
  *same immutable* `in` snapshot: there is no intermediate state to be confused by.
  The behaviour differs slightly from the CPU's exact pass-ordered result, but it
  is still deterministic, mass-conserving, and visually equivalent (a column drains
  at the surface, fills at the bottom). **Bit-for-bit parity with the CPU is not a
  goal** (see §4): determinism *across GPU machines* is.

- **Smoke.** Rises if the cell above (in `in`) is empty; a cell becomes smoke if
  the cell below it is smoke and *it* is empty (mirror). Dissipation uses the same
  `(hash % 40 == 0)` test the CPU uses; fold the tick counter in via a uniform
  (see §4). Lateral-when-blocked uses the same hash-ordered `DIRS` scan; both the
  source and the target cell must agree on whether the move happens, which again
  follows from evaluating the identical predicate against `in`.

### 2.3 Why not atomic scatter?

An atomic-scatter design (each cell `atomicAdd`s donations into neighbour slots)
would let us port the CPU rules more literally, but:
- it reintroduces **order non-determinism** (atomic add order across cells is
  undefined → different machines can land on different intermediate clamps);
- it needs `atomic<u32>` per voxel level and careful clamping to `MAX_WATER_LEVEL`
  under contention (two donors can overshoot 8 before either clamps);
- it forces a two-phase dispatch anyway (scatter, then resolve) to avoid lost
  updates.

The pull model gives determinism for free and is strictly simpler. **Recommended:
pull-only, double-buffered. Do not use atomics.**

### 2.4 Per-tick sequencing

Within one physics tick, run the rules as **separate dispatches that each
ping-pong**, mirroring the CPU's ordering (`tick` runs sand, then water, then
smoke, then settle):

```
tick:
  sand   : dispatch(read A, write B)   ; swap -> current = B
  water  : dispatch(read B, write A)   ; swap -> current = A
  smoke  : dispatch(read A, write B)   ; swap -> current = B
```

Each sub-pass is hazard-free in isolation (separate in/out). Sand-settle (the
CPU's `settle_sand` re-running falls on touched bricks for up to 8 iterations) is
naturally handled by the next tick's sand pass, or by running the sand dispatch K
times back-to-back with ping-pong if same-tick settling is desired.

---

## 3. Active-set management on the GPU

Dispatching over all 1,048,576 bricks every tick is wasteful when only a few
thousand are active (the whole point of `world.active_bricks` + `movable_mask`).
Two layers:

### 3.1 Indirect dispatch driven by a GPU-built active list

Maintain a GPU **active-brick list** + count and dispatch indirectly so the GPU,
not the CPU, decides how many workgroups to launch.

Buffers:
- `active_list : array<u32>` — brick indices to process this tick.
- `active_count : atomic<u32>` (in a small "indirect args" buffer).
- An indirect-args buffer `DispatchIndirect { x:u32, y:u32, z:u32 }` with
  `USAGE = STORAGE | INDIRECT`.

Per tick, three stages:

1. **Prepass (compaction).** One invocation per brick (or per active candidate):
   if `movable_in[bi] != 0`, append `bi` to `active_list` via
   `atomicAdd(&active_count, 1u)` and set `active_list[slot] = bi`. This is the GPU
   equivalent of `rebuild_active_bricks`. To avoid scanning all 1 M bricks every
   tick, seed the candidate set from the previous tick's `active_list` plus the
   one-ring of bricks any active brick can spill into (the cross-brick neighbours
   in ±x, ±y, ±z that sand/water/smoke can reach). Append those neighbours
   speculatively; the prepass's `movable != 0` test drops the ones that turned out
   empty. This keeps the prepass O(active), matching the CPU's `active.clone()` +
   the `binary_search insert` of spill targets in `cross_apply_water` /
   `step_brick_sand_fall`.

2. **Build indirect args.** A trivial 1-thread dispatch writes
   `args.x = active_count` (workgroups = one per active brick), `args.y = args.z =
   1`. Reset `active_count` for next tick at the end.

3. **Physics dispatch (indirect).** `cp.dispatch_workgroups_indirect(&args_buf, 0)`
   with workgroup size 64 (one brick). Each workgroup reads `active_list[wg_id]` to
   find its brick index, then the 64 invocations cover the 64 voxels.

### 3.2 Coarse fallback

If the indirect plumbing is deferred, a cheaper interim is to **dispatch over all
bricks but early-out** when `movable_in[bi] == 0` (a single `vec2<u32>` compare at
the top of the workgroup). This wastes launch overhead on 1 M tiny workgroups but
costs almost no ALU; it is a fine first milestone and lets the rules be validated
before the active-list machinery exists. Move to §3.1 once correctness is proven.

### 3.3 Mapping back to CPU bookkeeping

The CPU keeps `active_bricks` sorted and edits it incrementally. On the GPU the
list is rebuilt each tick from `movable`, so sortedness/dedup is unnecessary — the
list is consumed once and regenerated. The CPU `active_bricks` becomes irrelevant
*while physics is GPU-resident*; it only needs reconstructing if/when state is read
back (see §5.3).

---

## 4. Determinism across machines and drivers

Physics runs on every client and must agree (worlds are reproduced from
seed + edit log, not synced voxel-by-voxel — `src/server.rs`, `src/net.rs:45`).
Determinism requirements:

### 4.1 Integer-only updates — mandatory

The CPU physics is **already integer-only**. Keep it that way on the GPU:
- Water level is an integer 1..8; all transfers are `min`/`+`/`−` of integers.
- The lateral-direction hash is integer: `gx*0x9E3779B1 ^ gy*0x85EBCA77 ^
  gz*0xC2B2AE3D`. WGSL `u32` wrapping multiply/xor are **bit-exact and identical on
  every conformant device** (WGSL/SPIR-V define `u32` arithmetic as modular 2³²).
  Port the hash verbatim using `gx`, `gy`, `gz` = the *global* voxel coords, which
  the shader already reconstructs from brick index + local index (the renderer's
  toroidal mapping is irrelevant here — physics works in storage/local space, the
  same space the CPU uses).
- **No floats anywhere in the physics shader.** Float results are *not* guaranteed
  bit-identical across GPUs/drivers (fused multiply-add, rounding, transcendental
  precision all vary), so any float would break cross-machine determinism. There is
  no need for floats in these rules — keep the whole pass `u32`/`i32`.

### 4.2 The frame counter

Smoke folds `FRAME_COUNTER` into its dissipation hash. On the CPU this is a process
atomic. On the GPU pass it as a **`u32` in a small uniform buffer** (a
`PhysicsParams { tick: u32, ... }`) incremented by the CPU once per physics tick
and written with `queue.write_buffer`. Every client increments it the same way per
tick, so the hash stays in lockstep. (Determinism also requires every client run
the *same number of ticks from the same start* — already true given the fixed
30 Hz step in `App::render_frame`; the `min(4)` step clamp must be applied
identically, which it is.)

### 4.3 Parallel-order independence

The pull model guarantees the result is independent of invocation execution order
(every output is a pure function of the immutable `in` buffer). This is the
property that makes GPU determinism achievable at all: there is no atomic-add
order, no first-writer-wins, no race. Two different GPUs running the same shader on
the same `in` buffer produce the same `out` buffer, bit for bit, because only
integer ops and a fixed dataflow are involved.

### 4.4 Caveat to document for the team

The GPU result will **not** be bit-identical to the *current CPU* result, because
the pull reformulation collapses the CPU's ordered multi-pass water into a single
simultaneous update (§2.2). That is fine **as long as all clients run the GPU
path**. If a mixed deployment is ever possible (some clients CPU, some GPU), they
will diverge — so the migration must be all-or-nothing per server, or the CPU path
must be rewritten to the same pull formulation. Recommendation: make the pull
formulation the single source of truth and, if a CPU fallback is needed (no GPU
adapter — see the `render_luma_stats_at` headless guard pattern,
`src/renderer.rs:1112`), port the *same* pull rules to a CPU function so both
agree.

---

## 5. Sync with the rest of the engine

### 5.1 Dirty tracking for the renderer

Today physics calls `mark_brick_dirty` (pushing to `world.dirty_bricks`) and the
renderer drains that list in `upload_world` to refresh brick data + the
`tile_mask` / `chunk_mask` / `l4_mask` / `brick_uniform` / `tile_uniform`
hierarchy (`src/renderer.rs:545-635`), and `App` projects dirty bricks to screen
tiles for the temporal-differential re-trace (`temporal::project_brick_to_tiles`,
`src/app.rs:396-405`).

With GPU physics, brick data never leaves the GPU, so there is nothing to *upload*
for changed bricks. But the **derived masks still drive the DDA** and **the
temporal system still needs to know which bricks changed** (otherwise the
raymarcher skips clean tiles and physics looks frozen). Plan:

- **Masks on GPU.** Add a small "mask fixup" compute pass after physics that, for
  each brick the physics pass changed, recomputes its `tile_mask` bit
  (occupied↔empty) and `brick_uniform`/`tile_uniform` entry, then bubbles
  `chunk_mask` / `l4_mask`. This is the GPU port of `refresh_masks_for_brick` +
  `recompute_uniform_for_brick` + `recompute_uniform_for_tile`. The tricky bit is
  the bubble-up writes can race (multiple bricks in one tile / multiple tiles in
  one chunk). Use the GPU active-list as input and resolve hierarchy levels in
  **separate dispatches** (bricks→tiles, tiles→chunks, chunks→L4), each writing one
  level from the level below with one invocation per parent cell — no races within
  a level. Mask buffers must become `read_write` storage and have a render-side
  bind group pointing at them (they are `storage_ro` in `create_compute_bgl`
  today; physics needs a separate writable view, but the raymarch can keep its RO
  view of the same buffers since reads and the physics write happen in different
  passes within the same submit and are ordered by the compute-pass boundary).

- **A GPU "changed bricks" list for temporal.** Have the physics pass append, to a
  `changed_list` buffer (same atomic-compaction trick as §3.1), every brick whose
  output differs from its input. Read that list back to the CPU **only the count +
  the list** (a few KB) so `App` can keep feeding `project_brick_to_tiles`. If the
  readback latency is unwelcome, an alternative is to move the brick→screen-tile
  projection itself onto the GPU and write directly into the `tile_dirty_buf` the
  raymarch already consumes (`@binding(7)` / `upload_tile_dirty`,
  `src/renderer.rs:637`). That fully closes the loop with zero per-frame readback.
  Recommended end state; the changed-list readback is the simpler first step.

### 5.2 Coexistence with streaming and edit-replay

Streaming (`shift_origin` / `install_finished_chunks` / `clear_slot` /
`apply_slot_bricks`) and player edits (`apply_edit` → `set_voxel`) are **CPU
writes into `world.bricks`** that then upload via the existing `dirty_bricks`
path. These must land in the **same buffer the GPU physics reads next tick.**
Ordering per frame (extending `App::render_frame`):

1. CPU streaming + edits mutate `world.bricks`, push to `dirty_bricks`.
2. Upload those dirty spans into the **current** physics buffer (the one physics
   will read this tick) — reuse `upload_world`'s span logic, but target the live
   ping-pong buffer instead of a separate `bricks_buf`.
3. Run the physics dispatches (which ping-pong; the streamed/edited voxels are now
   visible to the pull rules).
4. Render from the resulting current buffer.

Hazard to call out: an edit and physics could touch the same brick the same frame.
Because the CPU upload in step 2 writes into the *input* buffer before the physics
dispatch reads it, the edit wins as the initial condition and physics evolves from
it — which is the desired semantics (same as CPU today, where `apply_edit` runs in
`poll_net`/`consume_clicks` before `physics::tick`). The only new rule: **never
upload into the buffer physics is currently writing** (the `out` buffer); always
upload into the `in` buffer for the upcoming tick. With per-buffer bind groups
this is just "upload into `current`, then dispatch."

Edit-replay on chunk reload (`replay_edits_for_chunk`) and `clear_slot` similarly
just need to write the active buffer and mark the affected bricks for the §5.1 mask
fixup. Movable recompute (the `is_movable_mat` loop in `apply_slot_bricks`) can run
on the CPU as today (it already does) and upload the `movable` buffer span, or be
folded into the GPU mask-fixup pass.

### 5.3 Reading results back to the CPU — only when needed

Three CPU consumers of voxel state remain:

- **CPU raycaster (`raycast::raycast`)** for click-picking (`src/app.rs:294`,
  `src/raycast.rs`). It reads `world.bricks[bi].occupancy`. If physics is
  GPU-resident, picking against fluid surfaces would be stale. Options, in order of
  preference:
  1. **Pick against terrain-only CPU state** and accept that you can't break a
     mid-air water/smoke voxel — usually fine, since picks target solid blocks.
     Keep `world.bricks` as the authoritative *solid/terrain* copy (edits +
     streaming write it); fluids live only on the GPU. The raycaster already only
     cares about occupancy of solids in practice.
  2. **On-demand readback** of the brick(s) along the pick ray (a handful of
     bricks, KB-scale) via `copy_buffer_to_buffer` into a `MAP_READ` staging buffer
     — only on click, not per frame. The headless test already shows the
     map_async + `device.poll(Wait)` pattern (`src/renderer.rs:1261-1264`).
  3. Move picking to the GPU (write the hit into a tiny buffer, read it back). Most
     work; defer.

- **Networking / edit log.** This is **already physics-independent.** Only
  *player* edits are networked (`broadcast_edit_world` → `VoxelEdit`); physics is
  never serialized — every client recomputes it. So GPU physics needs **no
  readback for networking** at all, *provided* determinism (§4) holds. This is the
  big payoff of keeping it deterministic: the edit log stays tiny and physics stays
  free of the wire.

- **`material_at_world`** (`src/voxel.rs:1027`) and any gameplay query — same as the
  raycaster: serve from the CPU terrain copy, or on-demand readback for fluids.

Net: **keep state GPU-resident; read back only the small changed-brick list (for
temporal, §5.1) and, on click, the few bricks under the pick ray.** No per-frame
bulk readback.

---

## 6. wgpu 23 specifics

### 6.1 Buffer usages

- `bricks_a`, `bricks_b`: `STORAGE | COPY_DST` (COPY_DST for the CPU edit/stream
  span uploads; add `COPY_SRC` to whichever may be read back for picking). They are
  bound `read` in one bind group and `read_write` in another within a pass.
- `movable_a`, `movable_b`: `STORAGE | COPY_DST`.
- `tile_mask` / `chunk_mask` / `l4_mask` / `brick_uniform` / `tile_uniform`: today
  `STORAGE | COPY_DST` (`src/renderer.rs:269-299`). For the GPU mask-fixup they
  also need to be bound `read_write`; the binding *usage* is still `STORAGE`, only
  the BGL entry's `read_only` flag changes per bind group.
- `active_list`, `changed_list`: `STORAGE | COPY_DST` (+ `COPY_SRC` for changed
  list readback).
- `indirect_args`: `STORAGE | INDIRECT`.
- `physics_params` uniform (`tick: u32`, dims if needed): `UNIFORM | COPY_DST`.

### 6.2 Bind group layout sketch (physics pass)

Following the style of `create_compute_bgl` (`src/renderer.rs:889`):

```
@group(0):
  binding 0  uniform                         physics_params   (tick counter)
  binding 1  storage, read                    bricks_in
  binding 2  storage, read_write              bricks_out
  binding 3  storage, read                    movable_in
  binding 4  storage, read_write              movable_out
  binding 5  storage, read                    active_list       (which bricks)
  binding 6  storage, read_write              changed_list      (+ its atomic count)
```

A separate, smaller BGL for the prepass (build `active_list` from `movable_in`) and
for the indirect-args writer. The mask-fixup pass gets its own BGL binding the five
mask buffers `read_write` plus the `changed_list` as input.

Two physics bind groups are created (`bg_ab` reads A writes B, `bg_ba` reads B
writes A); the frame loop selects by tick parity, exactly as the renderer already
builds bind groups twice across `Renderer::new` / `Renderer::resize`.

### 6.3 Dispatch sizing

- **Workgroup size `@workgroup_size(64)`** = one brick (64 voxels), one invocation
  per voxel. `local_invocation_index` (0..63) is the voxel index `i`;
  `workgroup_id.x` indexes into `active_list` to get the brick. 64 is a good
  occupancy multiple (2 warps on NVIDIA, 1 on AMD wave64).
- **Indirect dispatch:** `cp.dispatch_workgroups_indirect(&indirect_args, 0)` after
  the prepass fills `args.x = active_count`. (wgpu 23 exposes
  `ComputePass::dispatch_workgroups_indirect`.)
- **Interim direct dispatch (§3.2):** `dispatch_workgroups(WORLD_BRICKS_TOTAL, 1, 1)`
  with an early-out — 1,048,576 workgroups is within the 65,535-per-dim limit only
  if split; prefer dispatching `ceil(total/N)` workgroups of N bricks each, or just
  use the indirect path from the start.
- Prepass over candidate bricks: `@workgroup_size(64)` over the candidate count.
- Mask-fixup levels: one invocation per parent cell, sized to the changed set.

These passes slot into the existing `encoder.begin_compute_pass` structure in
`Renderer::render` (`src/renderer.rs:683-736`), *before* the beam/raymarch passes,
within the same encoder/submit so the compute-pass ordering provides the needed
read-after-write barrier between physics-out and raymarch-in.

### 6.4 Features

- **None required** for the recommended `vec2<u32>` integer path. The current
  device requests `Features::empty()` (`src/renderer.rs:216`) and that suffices.
- Indirect dispatch is core wgpu, no feature flag.
- *Optional:* `Features::SHADER_INT64` would let occupancy stay a real `u64` and
  simplify the bit ops, but it is **not portable** (and unnecessary). Avoid it to
  keep the determinism/portability story clean and the headless test
  (`Features::empty()`, `src/renderer.rs:1126`) representative.
- `max_storage_buffer_binding_size` is already raised to 256 MB
  (`src/renderer.rs:218`); two 72 MB brick buffers + masks fit comfortably.

---

## 7. Incremental migration plan and risks

Port in order of difficulty. Each milestone is independently shippable and
testable against the CPU implementation.

**M0 — Plumbing (no behaviour change).**
- Add `bricks_a`/`bricks_b` (+movable) ping-pong buffers, `physics_params`
  uniform, two raymarch bind groups (§1.4-A). Add the WGSL `Brick`/`vec2<u32>`
  helpers and a no-op physics shader that copies `in→out`.
- Wire `App::render_frame` to run the (no-op) GPU pass instead of/alongside
  `physics::tick`, render from the current buffer.
- Validate: image identical to today (the no-op pass changes nothing). Reuse the
  headless render test (`render_luma_stats_at`) as the regression net.

**M1 — Sand (simplest, binary).**
- Implement the pull-based sand rule (§2.2), direct dispatch with `movable`
  early-out (§3.2). One layer fall per tick.
- Validate: drop a sand column, compare settle behaviour to CPU. Add a headless
  test that places sand and asserts it reaches the floor in the expected tick
  count.
- Risk: the `vec2<u32>` reimplementation of `falling = sand & (empty << 16)` and
  cross-brick `<<48`/`>>48` is the bug-prone part — unit-test the bit helpers in
  isolation first (a tiny WGSL or even a Rust `vec2<u32>` mirror with property
  tests against the `u64` originals).

**M2 — Active list + indirect dispatch (§3.1).**
- Replace the brute-force dispatch with the prepass-built `active_list` +
  `dispatch_workgroups_indirect`.
- Validate: identical sand behaviour, far fewer launched workgroups (profile).
- Risk: speculative neighbour inclusion must cover every direction sand/water/smoke
  can spill (±y for gravity/rise, ±x/±z for lateral). Missing a direction = a brick
  that should activate stays frozen. Mirror the exact spill targets from the CPU
  cross-brick branches.

**M3 — Smoke.**
- Pull-based rise + dissipation (`hash % 40`, tick from `physics_params`) +
  blocked-lateral.
- Validate: smoke rises one cell/tick, fades at the right rate, deterministic
  across two runs with the same tick sequence (run twice, compare buffers).

**M4 — Water (hardest).**
- Implement the combined gravity+refill+lateral pull formula (§2.2). This is where
  behaviour intentionally diverges from the CPU's ordered passes; lock in the new
  behaviour with golden tests (drain a lake into a cave, assert column stays
  connected and total water volume is conserved tick-over-tick).
- Risk areas:
  - **Mass conservation under simultaneous lateral.** Donor and receiver must
    compute the identical pairwise transfer from `in`; an asymmetry leaks or
    duplicates water. Prove conservation with a test summing total level before/
    after across many ticks (must be invariant except at world boundaries /
    sinks).
  - **The hash direction order** must be evaluated identically by both endpoints of
    every lateral pair, or A thinks it gave and B thinks it didn't receive. Use the
    *donor's* coords to decide direction and have the receiver reproduce the donor's
    decision (the receiver knows the donor's coords).
  - **Cross-brick at brick faces** — the `vec2<u32>` indexing into the neighbour
    brick's bottom/top layer must match the CPU's `below_i_in = lx + lz*4 + 3*16`
    etc. exactly.

**M5 — Masks + temporal closure (§5.1).**
- GPU mask-fixup passes (brick→tile→chunk→L4, uniform recompute) so the DDA sees
  fluid changes; GPU `changed_list` (or direct `tile_dirty` writes) so the temporal
  re-trace doesn't skip animating fluids.
- Validate: water visibly flows in the rendered image with the temporal-diff path
  enabled (not just full re-trace). This is the milestone that makes the feature
  actually visible without forcing full-frame re-traces.

**M6 — Decommission the CPU hot path; keep a deterministic CPU fallback.**
- Remove the per-tick `physics::tick` call; keep `world.bricks` as the CPU
  terrain/edit copy (§5.2) and, if a no-GPU fallback is needed, a CPU
  implementation of the *same pull rules* (§4.4) for `Features::empty()`/no-adapter
  environments and tests.
- Picking: switch to terrain-only CPU picks or on-demand readback (§5.3).

### Cross-cutting risks

- **Determinism regressions are silent.** A single float, or an order-dependent
  atomic, won't crash — worlds just drift over minutes of play. Mitigate with a
  "run physics N ticks on two adapters / two runs, hash the brick buffer, assert
  equal" test (use the headless device helper, `src/renderer.rs:1112`).
- **`vec2<u32>` vs `u64` parity.** The whole port hinges on faithfully reproducing
  the `u64` bit tricks. Build the helper library + property tests *first* (M1).
- **Mixed CPU/GPU clients diverge** (§4.4). Treat GPU physics as a server-wide /
  build-wide switch, not a per-client toggle, until the CPU path is rewritten to
  the pull formulation.
- **Edit/physics buffer ordering** (§5.2): always upload edits into the buffer
  physics will *read*, never the one it is *writing*. A wrong-buffer upload yields
  flicker or lost edits.
- **VRAM:** ~160 MB for the doubled brick+movable buffers plus the existing 72 MB
  render copy if path (B) is used; negligible on the target RTX 5060 but worth
  noting for lower-end adapters.

---

## Appendix: file/symbol cross-reference

| Concept | CPU location |
| --- | --- |
| `physics::tick` entry, pass ordering, frame counter | `src/physics.rs:24-54` |
| Sand bitmask fall / cross-brick | `src/physics.rs:101-169` |
| Water 3-pass + snapshot rationale | `src/physics.rs:173-451` |
| Lateral-direction per-cell hash | `src/physics.rs:370-372` |
| `cross_apply_water` (the scatter we replace with pulls) | `src/physics.rs:586-625` |
| Smoke rise/dissipate/lateral, frame hash | `src/physics.rs:455-581` |
| `Brick` layout (`u64` + `[u8;64]`, 72 B) | `src/voxel.rs:154-184` |
| `movable_mask`, `active_bricks`, `dirty_bricks` | `src/voxel.rs:186-204` |
| `refresh_masks_for_brick`, uniform recompute | `src/voxel.rs:701-808` |
| `material_at_world` (toroidal CPU query) | `src/voxel.rs:1027` |
| `bricks_buf` upload + incremental `upload_world` | `src/renderer.rs:262-635` |
| `pack_u8_to_u32` / packed-byte read pattern | `src/renderer.rs:11-24` |
| `create_compute_bgl` / `make_compute_bg` (bind group style) | `src/renderer.rs:889-1019` |
| Compute-pass structure in `render` | `src/renderer.rs:683-736` |
| Device features/limits (`empty()`, 256 MB) | `src/renderer.rs:213-225` |
| Headless device + readback pattern | `src/renderer.rs:1112-1264` |
| Fixed-step physics call, 30 Hz, `min(4)` | `src/app.rs:357-365` |
| Edit application order vs physics | `src/app.rs:181-230, 294-313` |
| Networking carries only edits, not physics | `src/net.rs:45`, `src/server.rs` |
| World dimensions | `src/world_dims.rs` |
