// Bit-packed voxel world with a 3-level u64 hierarchy.
//
// Hierarchy (every level fits in a single u64):
//
//   level   cell-size voxels   one cell stores                   axes
//   ─────   ───────────────    ──────────────────────────────    ─────
//   L1      4³  = 64           u64 = 1 bit per voxel             64×16×64 cells
//   L2      16³ = 4³ bricks    u64 = 1 bit per child brick       16× 4×16 cells
//   L3      64³ = 4³ tiles     u64 = 1 bit per child tile         4× 1× 4 cells
//
// At every level the cell is a 4³ subgrid → exactly one u64. So a single
// bit-test "is this 64³ region of the world empty?" reads 8 bytes; if so we
// skip that whole region during ray traversal.
//
// Within a brick, voxels are ordered (x, z, y) — y is the slowest axis. That
// makes a 4×4 horizontal layer 16 contiguous bits, so falling-sand physics on
// a u64 is `intra = sand & (~occupancy << 16)` — see physics.rs. Tile-in-chunk
// and brick-in-tile linearisations follow the same convention.

use glam::UVec3;

pub const BRICK_DIM: u32 = 4;
pub const BRICK_VOXELS: u32 = BRICK_DIM * BRICK_DIM * BRICK_DIM;

// 1.5× scale-up — chunk count drops 60% vs 2× (2304 vs 8192 storage chunks),
// so initial gen + chunk-streaming is ~3× faster while still keeping more
// detail than the original 1× scale.
pub const WORLD_BRICKS_X: u32 = 192;
pub const WORLD_BRICKS_Y: u32 = 96;
pub const WORLD_BRICKS_Z: u32 = 192;
pub const WORLD_BRICKS_TOTAL: u32 = WORLD_BRICKS_X * WORLD_BRICKS_Y * WORLD_BRICKS_Z;

pub const WORLD_VOXELS_X: u32 = WORLD_BRICKS_X * BRICK_DIM;
pub const WORLD_VOXELS_Y: u32 = WORLD_BRICKS_Y * BRICK_DIM;
pub const WORLD_VOXELS_Z: u32 = WORLD_BRICKS_Z * BRICK_DIM;

pub const WORLD_TILES_X: u32 = WORLD_BRICKS_X / 4;
pub const WORLD_TILES_Y: u32 = WORLD_BRICKS_Y / 4;
pub const WORLD_TILES_Z: u32 = WORLD_BRICKS_Z / 4;
pub const WORLD_TILES_TOTAL: u32 = WORLD_TILES_X * WORLD_TILES_Y * WORLD_TILES_Z;

pub const WORLD_CHUNKS_X: u32 = (WORLD_TILES_X + 3) / 4;
pub const WORLD_CHUNKS_Y: u32 = (WORLD_TILES_Y + 3) / 4;
pub const WORLD_CHUNKS_Z: u32 = (WORLD_TILES_Z + 3) / 4;
pub const WORLD_CHUNKS_TOTAL: u32 = WORLD_CHUNKS_X * WORLD_CHUNKS_Y * WORLD_CHUNKS_Z;

// ---- storage chunks (the "chunked world") ----
// A storage chunk holds 8x8x8 bricks = 32x32x32 voxels. Generation, dirty
// tracking, and (eventually) GPU streaming all operate at this granularity.
pub const STORAGE_CHUNK_BRICKS: u32 = 8;
pub const STORAGE_CHUNK_VOXELS: u32 = STORAGE_CHUNK_BRICKS * BRICK_DIM;
pub const WORLD_STORE_CX: u32 = WORLD_BRICKS_X / STORAGE_CHUNK_BRICKS;
pub const WORLD_STORE_CY: u32 = WORLD_BRICKS_Y / STORAGE_CHUNK_BRICKS;
pub const WORLD_STORE_CZ: u32 = WORLD_BRICKS_Z / STORAGE_CHUNK_BRICKS;
pub const WORLD_STORE_CHUNKS: u32 = WORLD_STORE_CX * WORLD_STORE_CY * WORLD_STORE_CZ;

#[inline(always)]
pub const fn storage_chunk_idx(cx: u32, cy: u32, cz: u32) -> u32 {
    cx + cy * WORLD_STORE_CX + cz * WORLD_STORE_CX * WORLD_STORE_CY
}

#[derive(Clone, Copy)]
pub struct ChunkMeta {
    pub generated: bool,
}

pub const MAT_AIR: u8 = 0;
pub const MAT_SAND: u8 = 1;
pub const MAT_GRASS: u8 = 2;
pub const MAT_DIRT: u8 = 3;
pub const MAT_STONE: u8 = 4;
// 8 water-level variants encode mass per voxel (DwarfCorp-style cellular
// fluid). L8 = a full cell of water (also what set_voxel places); the
// physics step bleeds level into neighbours each tick.
pub const MAT_WATER_L1: u8 = 5;
pub const MAT_WATER_L2: u8 = 6;
pub const MAT_WATER_L3: u8 = 7;
pub const MAT_WATER_L4: u8 = 8;
pub const MAT_WATER_L5: u8 = 9;
pub const MAT_WATER_L6: u8 = 10;
pub const MAT_WATER_L7: u8 = 11;
pub const MAT_WATER_L8: u8 = 12;
pub const MAT_WATER: u8 = MAT_WATER_L8; // alias for callers that just want "full water"
pub const MAT_WOOD: u8 = 13;
pub const MAT_LEAVES: u8 = 14;
pub const MAT_SNOW: u8 = 15;
pub const MAT_LAVA: u8 = 16;
pub const MAT_ICE: u8 = 17;
pub const MAT_GLASS: u8 = 18;
pub const MAT_COAL: u8 = 19;
pub const MAT_IRON: u8 = 20;
pub const MAT_GOLD: u8 = 21;
pub const MAT_DIAMOND: u8 = 22;
pub const MAT_WOOD_BIRCH: u8 = 23;
pub const MAT_WOOD_PINE: u8 = 24;
pub const MAT_LEAVES_BIRCH: u8 = 25;
pub const MAT_LEAVES_PINE: u8 = 26;
pub const MAT_LEAVES_AUTUMN: u8 = 27;
pub const MAT_SMOKE: u8 = 28;
pub const MAT_FIRE: u8 = 29;
pub const MAT_FLOWER: u8 = 30;
pub const MAT_TALL_GRASS: u8 = 31;

#[inline(always)]
pub fn is_leaf_mat(m: u8) -> bool {
    m == MAT_LEAVES || m == MAT_LEAVES_BIRCH || m == MAT_LEAVES_PINE || m == MAT_LEAVES_AUTUMN
}
#[inline(always)]
pub fn is_wood_mat(m: u8) -> bool {
    m == MAT_WOOD || m == MAT_WOOD_BIRCH || m == MAT_WOOD_PINE
}
pub const MAX_WATER_LEVEL: u8 = 8;

#[inline(always)]
pub fn is_water_mat(m: u8) -> bool {
    m >= MAT_WATER_L1 && m <= MAT_WATER_L8
}

#[inline(always)]
pub fn is_movable_mat(m: u8) -> bool {
    m == MAT_SAND || is_water_mat(m) || m == MAT_SMOKE
}

#[inline(always)]
pub fn water_level_of(m: u8) -> u8 {
    if is_water_mat(m) { m - MAT_WATER_L1 + 1 } else { 0 }
}

#[inline(always)]
pub fn water_mat_for_level(level: u8) -> u8 {
    if level == 0 { MAT_AIR } else { MAT_WATER_L1 + (level.min(MAX_WATER_LEVEL) - 1) }
}

#[inline(always)]
pub const fn brick_voxel_idx(x: u32, y: u32, z: u32) -> u32 {
    x + z * BRICK_DIM + y * BRICK_DIM * BRICK_DIM
}

#[inline(always)]
pub const fn brick_idx(bx: u32, by: u32, bz: u32) -> u32 {
    bx + by * WORLD_BRICKS_X + bz * WORLD_BRICKS_X * WORLD_BRICKS_Y
}

#[inline(always)]
pub const fn tile_idx(tx: u32, ty: u32, tz: u32) -> u32 {
    tx + ty * WORLD_TILES_X + tz * WORLD_TILES_X * WORLD_TILES_Y
}

#[inline(always)]
pub const fn chunk_idx(cx: u32, cy: u32, cz: u32) -> u32 {
    cx + cy * WORLD_CHUNKS_X + cz * WORLD_CHUNKS_X * WORLD_CHUNKS_Y
}

#[inline(always)]
pub const fn brick_bit_in_tile(lx: u32, ly: u32, lz: u32) -> u32 {
    lx + lz * 4 + ly * 16
}

#[inline(always)]
pub const fn tile_bit_in_chunk(lx: u32, ly: u32, lz: u32) -> u32 {
    lx + lz * 4 + ly * 16
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Brick {
    pub occupancy: u64,
    pub materials: [u8; BRICK_VOXELS as usize],
}

/// Compute the brick_uniform[bi] hint from a brick's current contents.
/// Returns 0 if the brick is non-uniform (mixed materials or partial fill);
/// otherwise the material id that fills every voxel.
#[inline]
pub fn compute_uniform_hint(b: &Brick) -> u8 {
    if b.occupancy != !0u64 { return 0; }
    let m0 = b.materials[0];
    if m0 == 0 { return 0; }
    for i in 1..(BRICK_VOXELS as usize) {
        if b.materials[i] != m0 { return 0; }
    }
    m0
}

impl Brick {
    pub const EMPTY: Self = Self {
        occupancy: 0,
        materials: [0; BRICK_VOXELS as usize],
    };

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.occupancy == 0
    }

    #[inline(always)]
    pub fn set(&mut self, x: u32, y: u32, z: u32, mat: u8) {
        let i = brick_voxel_idx(x, y, z);
        let bit = 1u64 << i;
        if mat == MAT_AIR {
            self.occupancy &= !bit;
            self.materials[i as usize] = 0;
        } else {
            self.occupancy |= bit;
            self.materials[i as usize] = mat;
        }
    }
}

pub struct World {
    /// DENSE brick storage — indexed directly by brick_idx(bx, by, bz).
    /// 2× scale = 8.4M bricks × 72B = 605 MB. Tried sparse-pool earlier:
    /// the extra brick_slot[bi] fetch per DDA step in the shader cost 5-10x
    /// the per-pixel cost. Speed wins over memory until we hit hard limits.
    pub bricks: Vec<Brick>,
    pub tile_mask: Vec<u64>,
    pub chunk_mask: Vec<u64>,
    pub movable_mask: Vec<u64>,
    /// Per-brick "this whole brick is one material" hint (0 = non-uniform,
    /// else mat id). The dense brick STILL stores the data; this just
    /// lets the DDA fast-skip uniform 4-voxel cubes in one test, avoiding
    /// the 72B Brick fetch entirely for those.
    pub brick_uniform: Vec<u8>,
    /// Per-tile uniform hint at 16-voxel scale.
    pub tile_uniform: Vec<u8>,
    /// Per-brick baked AO (0..255). One byte per brick, computed at gen
    /// + edit time by counting solid 6-axis brick-neighbours. Replaces
    /// compute_ao's 12 hierarchical descents per primary hit with a
    /// single byte fetch. 8 MB total at 2× scale.
    pub brick_ao: Vec<u8>,
    pub active_bricks: Vec<u32>,
    pub dirty_bricks: Vec<u32>,
    pub all_dirty: bool,
    pub chunk_meta: Vec<ChunkMeta>,
    pub seed: u64,
    /// Sliding-window origin in chunk coords (xz only — y axis is fixed).
    /// Voxels stored locally at index `(x, y, z)` correspond to world voxel
    /// `(world_origin.x * 32 + x, y, world_origin.z * 32 + z)`. As the camera
    /// moves the origin shifts and edge chunks regenerate to give "infinite"
    /// terrain. y stays in [0, WORLD_VOXELS_Y).
    pub world_origin_chunk: glam::IVec2,
    /// For each slot, the world chunk coord it currently holds. None = stale.
    pub slot_world_chunk: Vec<Option<glam::IVec3>>,
    pub regen_queue: std::collections::VecDeque<(u32, glam::IVec3)>,
    /// Channel back from the BACKGROUND chunk-gen worker pool. Each
    /// (slot, want, scratch) is a finished chunk ready to apply. The
    /// worker uses rayon::spawn so generation runs across the thread
    /// pool — render thread NEVER blocks on it. Eliminates the
    /// "chunk-load frame drop" players hate.
    pub gen_result_tx: std::sync::mpsc::Sender<(u32, glam::IVec3, Vec<Brick>)>,
    pub gen_result_rx: std::sync::mpsc::Receiver<(u32, glam::IVec3, Vec<Brick>)>,
    /// Number of gen tasks currently in flight on the worker pool.
    /// Bounded so we don't spawn millions of pending tasks during fast
    /// flight (each holds a Vec<Brick> of 32 KB).
    pub gen_in_flight: usize,
    /// Persistent voxel edits keyed by *world* voxel coord. Survives chunk
    /// unload/regen — applied on top of fresh noise when a chunk reloads,
    /// and synced over the network so all clients agree on player builds.
    pub edits: std::collections::HashMap<(i32, i32, i32), u8>,
}

impl World {
    pub fn new() -> Self {
        Self::with_seed(0xC0FFEE_F00D_BEEFu64)
    }

    pub fn with_seed(seed: u64) -> Self {
        let (gen_tx, gen_rx) = std::sync::mpsc::channel();
        Self {
            // Dense alloc — every logical brick gets a slot at its bi.
            // Cheap initial alloc: 8.4M × 72B ≈ 600 MB at 2× scale.
            bricks: vec![Brick::EMPTY; WORLD_BRICKS_TOTAL as usize],
            tile_mask: vec![0u64; WORLD_TILES_TOTAL as usize],
            chunk_mask: vec![0u64; WORLD_CHUNKS_TOTAL as usize],
            movable_mask: vec![0u64; WORLD_BRICKS_TOTAL as usize],
            brick_uniform: vec![0u8; WORLD_BRICKS_TOTAL as usize],
            tile_uniform: vec![0u8; WORLD_TILES_TOTAL as usize],
            brick_ao: vec![255u8; WORLD_BRICKS_TOTAL as usize],
            active_bricks: Vec::with_capacity(4096),
            dirty_bricks: Vec::with_capacity(4096),
            all_dirty: true,
            chunk_meta: vec![ChunkMeta { generated: false }; WORLD_STORE_CHUNKS as usize],
            seed,
            world_origin_chunk: glam::IVec2::ZERO,
            slot_world_chunk: vec![None; WORLD_STORE_CHUNKS as usize],
            regen_queue: std::collections::VecDeque::with_capacity(256),
            gen_result_tx: gen_tx,
            gen_result_rx: gen_rx,
            gen_in_flight: 0,
            edits: std::collections::HashMap::new(),
        }
    }

    // ---- dense brick access helpers ----
    // Storage is direct: bricks[bi as usize] for every operation. The
    // brick_uniform[bi] is kept purely as a GPU fast-skip hint that lets
    // shader rays skip whole 4-voxel cubes (and tile_uniform skips 16).

    /// Return a copy of the brick at logical index `bi`.
    #[inline]
    pub fn brick_read(&self, bi: u32) -> Brick {
        self.bricks[bi as usize]
    }

    /// Mutable borrow — caller is responsible for updating masks, uniform
    /// hints, and dirty bits afterward (set_voxel does this for the
    /// public single-voxel path).
    #[inline]
    pub fn brick_mut(&mut self, bi: u32) -> &mut Brick {
        &mut self.bricks[bi as usize]
    }

    /// Cheap "any solid voxel?" check that doesn't copy the 72 B Brick.
    #[inline]
    pub fn brick_is_empty(&self, bi: u32) -> bool {
        self.bricks[bi as usize].occupancy == 0
    }

    /// Occupancy mask — direct fetch.
    #[inline]
    pub fn brick_occupancy(&self, bi: u32) -> u64 {
        self.bricks[bi as usize].occupancy
    }

    /// Material at a specific voxel inside a brick.
    #[inline]
    pub fn brick_voxel_mat(&self, bi: u32, vi: u32) -> u8 {
        self.bricks[bi as usize].materials[vi as usize]
    }

    /// Reset to fully empty air. Used by chunk stream-out.
    pub fn clear_brick(&mut self, bi: u32) {
        self.bricks[bi as usize] = Brick::EMPTY;
        self.brick_uniform[bi as usize] = 0;
    }

    /// Drop a dense Brick into this brick index. Updates the uniform hint
    /// so the GPU fast-skip path stays accurate.
    pub fn assign_brick(&mut self, bi: u32, src: Brick) {
        self.bricks[bi as usize] = src;
        self.brick_uniform[bi as usize] = compute_uniform_hint(&src);
    }

    /// Recompute the uniform hint from the current brick contents.
    pub fn try_compact_brick(&mut self, bi: u32) {
        self.brick_uniform[bi as usize] = compute_uniform_hint(&self.bricks[bi as usize]);
    }

    /// World-voxel offset of the loaded window's lower corner.
    pub fn world_origin_voxel(&self) -> glam::IVec3 {
        glam::IVec3::new(
            self.world_origin_chunk.x * STORAGE_CHUNK_VOXELS as i32,
            0,
            self.world_origin_chunk.y * STORAGE_CHUNK_VOXELS as i32,
        )
    }

    /// Record a persistent edit at WORLD-voxel coords and (if it's currently
    /// inside the loaded window) apply it locally. The edit map drives
    /// replay-on-regen so builds survive crossing the chunk-streaming edge.
    pub fn apply_edit(&mut self, wx: i32, wy: i32, wz: i32, mat: u8) {
        self.edits.insert((wx, wy, wz), mat);
        let origin = self.world_origin_voxel();
        // Bounds: only apply locally if the world voxel is inside the loaded
        // window (relative to origin).
        let rel_x = wx - origin.x;
        let rel_y = wy - origin.y;
        let rel_z = wz - origin.z;
        if rel_x < 0 || rel_y < 0 || rel_z < 0
            || (rel_x as u32) >= WORLD_VOXELS_X
            || (rel_y as u32) >= WORLD_VOXELS_Y
            || (rel_z as u32) >= WORLD_VOXELS_Z
        {
            return;
        }
        // Storage is TOROIDAL — the GPU shader maps world voxels to slots via
        // `pos_mod(wx, WORLD_VOXELS_X)`. Naive `wx - origin.x` only matches
        // that when origin == 0; for any other origin (player walked away
        // from spawn) we'd write to the wrong brick. Use the same mapping
        // the shader uses.
        let lx = wx.rem_euclid(WORLD_VOXELS_X as i32) as u32;
        let ly = rel_y as u32;
        let lz = wz.rem_euclid(WORLD_VOXELS_Z as i32) as u32;
        self.set_voxel(lx, ly, lz, mat);
    }

    /// Target origin chunk-coord for a camera at the given world position.
    /// Centres the loaded window on the camera.
    pub fn target_origin_chunk(camera_world: glam::Vec3) -> glam::IVec2 {
        let cam_cx = (camera_world.x / STORAGE_CHUNK_VOXELS as f32).floor() as i32;
        let cam_cz = (camera_world.z / STORAGE_CHUNK_VOXELS as f32).floor() as i32;
        let half_x = (WORLD_STORE_CX as i32) / 2;
        let half_z = (WORLD_STORE_CZ as i32) / 2;
        glam::IVec2::new(cam_cx - half_x, cam_cz - half_z)
    }

    /// Local voxel coords for `world_voxel`, given the current origin.
    /// Returns None if `world_voxel` is outside the loaded window.
    pub fn world_to_local(&self, world_voxel: glam::IVec3) -> Option<glam::UVec3> {
        let origin_vox = glam::IVec3::new(
            self.world_origin_chunk.x * STORAGE_CHUNK_VOXELS as i32,
            0,
            self.world_origin_chunk.y * STORAGE_CHUNK_VOXELS as i32,
        );
        let local = world_voxel - origin_vox;
        if local.x < 0 || local.y < 0 || local.z < 0
            || local.x >= WORLD_VOXELS_X as i32
            || local.y >= WORLD_VOXELS_Y as i32
            || local.z >= WORLD_VOXELS_Z as i32 { return None; }
        Some(glam::UVec3::new(local.x as u32, local.y as u32, local.z as u32))
    }

    /// Shift the sliding window using TOROIDAL slot indexing. A slot at
    /// store-coord `(sx, _, sz)` represents the world chunk in the loaded
    /// window whose `mod WORLD_STORE_*` equals `(sx, sz)` — so a +1 origin
    /// shift only invalidates the single column of slots that just dropped
    /// out of the window.
    pub fn shift_origin(&mut self, new_origin: glam::IVec2) {
        if new_origin == self.world_origin_chunk { return; }
        self.world_origin_chunk = new_origin;
        // DIAGNOSTIC: force bulk upload of entire bricks buffer.
        // If this fixes "wrong chunks" bug, delta upload was missing writes.
        self.all_dirty = true;
        let store_x = WORLD_STORE_CX as i32;
        let store_z = WORLD_STORE_CZ as i32;
        for cz in 0..WORLD_STORE_CZ {
            for cy in 0..WORLD_STORE_CY {
                for cx in 0..WORLD_STORE_CX {
                    // For slot (cx, cy, cz), the world chunk currently in
                    // the window with `wc mod store == slot` is computed via
                    // the offset from origin's mod.
                    let want_x = new_origin.x + (cx as i32 - new_origin.x).rem_euclid(store_x);
                    let want_z = new_origin.y + (cz as i32 - new_origin.y).rem_euclid(store_z);
                    let want = glam::IVec3::new(want_x, cy as i32, want_z);
                    let slot = storage_chunk_idx(cx, cy, cz) as usize;
                    if self.slot_world_chunk[slot] != Some(want) {
                        self.clear_slot(cx, cy, cz);
                        self.slot_world_chunk[slot] = Some(want);
                        self.regen_queue.push_back((slot as u32, want));
                    }
                }
            }
        }
    }

    /// Regenerate up to `budget` slots from the queue. Each regenerated slot
    /// gets the persistent-edit map replayed on top so player builds survive
    /// the chunk-streaming round-trip.
    ///
    /// Noise generation is **parallelised across rayon's thread pool** — each
    /// worker computes a scratch `Vec<Brick>` for its slot independently
    /// (pure function of world-chunk coord + seed), then the main thread
    /// merges results into the flat world array serially. No shared mutable
    /// state, no locks.
    pub fn process_regen_queue(&mut self, max_inflight: u32) {
        // Apply UNCAPPED — drain everything pending each frame. Better
        // to take a one-frame hitch than show wrong chunks for any
        // duration.
        const APPLY_CAP_PER_FRAME: usize = usize::MAX;
        let mut applied = 0usize;
        // -------- NON-BLOCKING WORKER PUMP --------
        // 1. Drain anything the worker pool has finished.
        // 2. Top up the in-flight queue with new requests.
        // The main thread NEVER blocks on chunk gen. Even if 1000 chunks
        // are queued, we just keep `max_inflight` workers running and
        // poll results next frame. This is what eliminates the
        // "chunk-load frame drop" players notice in unoptimised engines.

        let seed = self.seed;

        // ---- Step 1: drain completed gen results (cheap, non-blocking) ----
        let mut regenerated_chunks: Vec<glam::IVec3> = Vec::new();
        while applied < APPLY_CAP_PER_FRAME {
            let Ok((slot, want, scratch)) = self.gen_result_rx.try_recv() else { break; };
            self.gen_in_flight = self.gen_in_flight.saturating_sub(1);
            // Discard stale results — the slot may have been reassigned to
            // a different world chunk while gen was running. Without this
            // we'd see chunk-overlap corruption ("multiple worlds mixed").
            if self.slot_world_chunk[slot as usize] != Some(want) { continue; }
            let cx = slot % WORLD_STORE_CX;
            let cy = (slot / WORLD_STORE_CX) % WORLD_STORE_CY;
            let cz = slot / (WORLD_STORE_CX * WORLD_STORE_CY);
            self.apply_slot_bricks(cx, cy, cz, &scratch);
            // DIAGNOSTIC: also force bulk upload after any apply.
            self.all_dirty = true;
            regenerated_chunks.push(want);
            applied += 1;
        }

        // ---- Step 2: top up the worker pool ----
        // Submit up to `max_inflight - gen_in_flight` new gen tasks via
        // rayon::spawn. Each runs on a worker thread in parallel.
        while self.gen_in_flight < max_inflight as usize {
            let Some((slot, want)) = self.regen_queue.pop_front() else { break; };
            // Skip stale queue entries before even spawning the work.
            if self.slot_world_chunk[slot as usize] != Some(want) { continue; }
            let tx = self.gen_result_tx.clone();
            rayon::spawn(move || {
                let scratch = gen_slot_bricks(want, seed);
                let _ = tx.send((slot, want, scratch));
            });
            self.gen_in_flight += 1;
        }
        if !regenerated_chunks.is_empty() && !self.edits.is_empty() {
            let cv = STORAGE_CHUNK_VOXELS as i32;
            let origin = self.world_origin_voxel();
            // Pre-filter into a vec so we don't hold an immutable borrow on
            // self.edits while calling self.set_voxel.
            let mut to_apply: Vec<(i32, i32, i32, u8)> = Vec::new();
            for (&(wx, wy, wz), &mat) in &self.edits {
                let edit_chunk = glam::IVec3::new(
                    wx.div_euclid(cv),
                    wy.div_euclid(cv),
                    wz.div_euclid(cv),
                );
                if regenerated_chunks.contains(&edit_chunk) {
                    to_apply.push((wx, wy, wz, mat));
                }
            }
            for (wx, wy, wz, mat) in to_apply {
                let rel_x = wx - origin.x;
                let rel_y = wy - origin.y;
                let rel_z = wz - origin.z;
                if rel_x < 0 || rel_y < 0 || rel_z < 0
                    || (rel_x as u32) >= WORLD_VOXELS_X
                    || (rel_y as u32) >= WORLD_VOXELS_Y
                    || (rel_z as u32) >= WORLD_VOXELS_Z
                {
                    continue;
                }
                // Toroidal local mapping — must match the shader.
                let lx = wx.rem_euclid(WORLD_VOXELS_X as i32) as u32;
                let ly = rel_y as u32;
                let lz = wz.rem_euclid(WORLD_VOXELS_Z as i32) as u32;
                self.set_voxel(lx, ly, lz, mat);
            }
        }
    }

    /// Wipe one slot's voxels (and the corresponding mask bits) so the
    /// renderer immediately shows sky/air there. The slot stays empty until
    /// `process_regen_queue` gets to it.
    fn clear_slot(&mut self, slot_cx: u32, slot_cy: u32, slot_cz: u32) {
        let x0 = slot_cx * STORAGE_CHUNK_BRICKS;
        let y0 = slot_cy * STORAGE_CHUNK_BRICKS;
        let z0 = slot_cz * STORAGE_CHUNK_BRICKS;
        for dz in 0..STORAGE_CHUNK_BRICKS {
            for dy in 0..STORAGE_CHUNK_BRICKS {
                for dx in 0..STORAGE_CHUNK_BRICKS {
                    let bi = brick_idx(x0 + dx, y0 + dy, z0 + dz);
                    let was_nonempty = !self.brick_is_empty(bi);
                    self.clear_brick(bi);
                    self.movable_mask[bi as usize] = 0;
                    if was_nonempty {
                        self.refresh_masks_for_brick(x0 + dx, y0 + dy, z0 + dz);
                    }
                    self.mark_brick_dirty(bi);
                }
            }
        }
        // CRITICAL: recompute tile_uniform for the 8 tiles touched by this
        // slot. Without this the GPU shader's tile_uniform fast-skip keeps
        // returning the stale OLD chunk's uniform material — producing the
        // "floor turning to stone everywhere" visual corruption that fires
        // when streaming chunks in/out.
        let base_tx = x0 / 4;
        let base_ty = y0 / 4;
        let base_tz = z0 / 4;
        for dtz in 0..2u32 {
            for dty in 0..2u32 {
                for dtx in 0..2u32 {
                    let tx = base_tx + dtx;
                    let ty = base_ty + dty;
                    let tz = base_tz + dtz;
                    if tx < WORLD_TILES_X && ty < WORLD_TILES_Y && tz < WORLD_TILES_Z {
                        self.recompute_uniform_for_tile(tile_idx(tx, ty, tz));
                    }
                }
            }
        }
    }

    /// Apply a precomputed slot's bricks (from `gen_slot_bricks`) into the
    /// flat world array, refresh masks, mark bricks dirty.
    fn apply_slot_bricks(&mut self, slot_cx: u32, slot_cy: u32, slot_cz: u32, scratch: &[Brick]) {
        let base_bx = slot_cx * STORAGE_CHUNK_BRICKS;
        let base_by = slot_cy * STORAGE_CHUNK_BRICKS;
        let base_bz = slot_cz * STORAGE_CHUNK_BRICKS;
        for dz in 0..STORAGE_CHUNK_BRICKS {
            for dy in 0..STORAGE_CHUNK_BRICKS {
                for dx in 0..STORAGE_CHUNK_BRICKS {
                    let scratch_idx =
                        (dx + dy * STORAGE_CHUNK_BRICKS + dz * STORAGE_CHUNK_BRICKS * STORAGE_CHUNK_BRICKS)
                            as usize;
                    let bx = base_bx + dx;
                    let by = base_by + dy;
                    let bz = base_bz + dz;
                    let bi = brick_idx(bx, by, bz);
                    let src = scratch[scratch_idx];
                    // Recompute movable_mask BEFORE compacting so we still
                    // see the materials.
                    let mut m = 0u64;
                    for i in 0..64usize {
                        m |= (is_movable_mat(src.materials[i]) as u64) << i;
                    }
                    let new_movable = m & src.occupancy;
                    let was_movable = self.movable_mask[bi as usize] != 0;
                    self.movable_mask[bi as usize] = new_movable;
                    let is_movable = new_movable != 0;
                    if was_movable != is_movable {
                        if is_movable {
                            if let Err(pos) = self.active_bricks.binary_search(&bi) {
                                self.active_bricks.insert(pos, bi);
                            }
                        } else if let Ok(pos) = self.active_bricks.binary_search(&bi) {
                            self.active_bricks.remove(pos);
                        }
                    }
                    // assign_brick detects uniform fills and stores them
                    // sparsely (no slot allocated) — saves memory on the
                    // ~95% of bricks that are uniform stone / dirt / air.
                    self.assign_brick(bi, src);
                    self.refresh_masks_for_brick(bx, by, bz);
                    self.mark_brick_dirty(bi);
                }
            }
        }
        // tile_uniform flags get recomputed across all touched tiles. A
        // storage chunk = 8×8×8 bricks = 2×2×2 tiles → 8 tiles touched.
        let base_tx = (base_bx) / 4;
        let base_ty = (base_by) / 4;
        let base_tz = (base_bz) / 4;
        for dtz in 0..2u32 {
            for dty in 0..2u32 {
                for dtx in 0..2u32 {
                    let tx = base_tx + dtx;
                    let ty = base_ty + dty;
                    let tz = base_tz + dtz;
                    if tx < WORLD_TILES_X && ty < WORLD_TILES_Y && tz < WORLD_TILES_Z {
                        self.recompute_uniform_for_tile(tile_idx(tx, ty, tz));
                    }
                }
            }
        }
    }

    // ---- old recompute_uniform_for_brick is now unused in apply_slot_bricks
    // because assign_brick handles uniform-detection inline. Keep the
    // single-voxel set_voxel path using try_compact_brick. ----

    pub fn recompute_movable_for_brick(&mut self, bi: u32) {
        let b = self.brick_read(bi);
        let mut m = 0u64;
        for i in 0..64usize {
            let movable = is_movable_mat(b.materials[i]) as u64;
            m |= movable << i;
        }
        let was_movable = self.movable_mask[bi as usize] != 0;
        let new_mask = m & b.occupancy;
        self.movable_mask[bi as usize] = new_mask;
        let is_movable = new_mask != 0;
        if was_movable != is_movable {
            if is_movable {
                self.active_bricks.push(bi);
            } else if let Ok(pos) = self.active_bricks.binary_search(&bi) {
                self.active_bricks.remove(pos);
            }
        }
    }

    pub fn rebuild_active_bricks(&mut self) {
        self.active_bricks.clear();
        for (i, m) in self.movable_mask.iter().enumerate() {
            if *m != 0 {
                self.active_bricks.push(i as u32);
            }
        }
        // Already in ascending order because we walk indices in order.
    }

    pub fn mark_brick_dirty(&mut self, bi: u32) {
        if !self.all_dirty {
            self.dirty_bricks.push(bi);
        }
    }

    /// Refresh tile/chunk bits for a brick after the brick's occupancy may
    /// have changed. Called by physics and by set_voxel().
    pub fn refresh_masks_for_brick(&mut self, bx: u32, by: u32, bz: u32) {
        let bi = brick_idx(bx, by, bz);
        let solid = !self.brick_is_empty(bi);
        let (tx, ty, tz) = (bx / 4, by / 4, bz / 4);
        let ti = tile_idx(tx, ty, tz);
        let bit = brick_bit_in_tile(bx & 3, by & 3, bz & 3);
        let prev = self.tile_mask[ti as usize];
        if solid {
            self.tile_mask[ti as usize] |= 1u64 << bit;
        } else {
            self.tile_mask[ti as usize] &= !(1u64 << bit);
        }
        let now = self.tile_mask[ti as usize];
        if (prev == 0) != (now == 0) {
            let (cx, cy, cz) = (tx / 4, ty / 4, tz / 4);
            let ci = chunk_idx(cx, cy, cz);
            let cbit = tile_bit_in_chunk(tx & 3, ty & 3, tz & 3);
            if now == 0 {
                self.chunk_mask[ci as usize] &= !(1u64 << cbit);
            } else {
                self.chunk_mask[ci as usize] |= 1u64 << cbit;
            }
        }
    }

    pub fn set_voxel(&mut self, x: u32, y: u32, z: u32, mat: u8) {
        if x >= WORLD_VOXELS_X || y >= WORLD_VOXELS_Y || z >= WORLD_VOXELS_Z {
            return;
        }
        let (bx, by, bz) = (x / BRICK_DIM, y / BRICK_DIM, z / BRICK_DIM);
        let (lx, ly, lz) = (x % BRICK_DIM, y % BRICK_DIM, z % BRICK_DIM);
        let bi = brick_idx(bx, by, bz);
        let was_empty = self.brick_is_empty(bi);
        // Materialize (allocates a slot if the brick was uniform/empty,
        // pre-filling with the prior contents so we can edit one voxel
        // without losing the rest).
        self.brick_mut(bi).set(lx, ly, lz, mat);
        let is_empty = self.brick_is_empty(bi);
        if was_empty != is_empty {
            self.refresh_masks_for_brick(bx, by, bz);
        }
        self.recompute_movable_for_brick(bi);
        // try_compact_brick re-collapses the brick back to a uniform hint
        // (freeing the slot) if the edit returned us to all-air or all-X.
        // This is the "split on edit, recombine when uniform again" loop.
        self.try_compact_brick(bi);
        // Refresh AO for this brick + 6 neighbours (their AO depends on
        // this brick's solidity).
        self.refresh_ao_for_brick(bi);
        // The tile this brick lives in may have lost its uniform status.
        let ti = tile_idx(bx / 4, by / 4, bz / 4);
        self.recompute_uniform_for_tile(ti);
        self.mark_brick_dirty(bi);
    }

    /// Recompute brick_uniform[bi] from the brick's current contents.
    pub fn recompute_uniform_for_brick(&mut self, bi: u32) {
        self.brick_uniform[bi as usize] = compute_uniform_hint(&self.bricks[bi as usize]);
    }

    /// Recompute tile_uniform[ti] from its 64 child bricks. Tile is uniform
    /// iff every child brick is uniform with the same material.
    pub fn recompute_uniform_for_tile(&mut self, ti: u32) {
        let tx = ti % WORLD_TILES_X;
        let ty = (ti / WORLD_TILES_X) % WORLD_TILES_Y;
        let tz = ti / (WORLD_TILES_X * WORLD_TILES_Y);
        let bx0 = tx * 4;
        let by0 = ty * 4;
        let bz0 = tz * 4;
        let first_bi = brick_idx(bx0, by0, bz0);
        let m0 = self.brick_uniform[first_bi as usize];
        if m0 == 0 {
            self.tile_uniform[ti as usize] = 0;
            return;
        }
        for dz in 0..4 {
            for dy in 0..4 {
                for dx in 0..4 {
                    let bi = brick_idx(bx0 + dx, by0 + dy, bz0 + dz);
                    if self.brick_uniform[bi as usize] != m0 {
                        self.tile_uniform[ti as usize] = 0;
                        return;
                    }
                }
            }
        }
        self.tile_uniform[ti as usize] = m0;
    }

    /// Bake AO for a single brick: how shadowed is the brick? Counts solid
    /// 6-axis brick neighbours (0..6) and converts to an AO byte.
    /// Lower = darker (more occluded). 0 = surrounded, 255 = wide open.
    pub fn compute_brick_ao(&self, bi: u32) -> u8 {
        let bx = bi % WORLD_BRICKS_X;
        let by = (bi / WORLD_BRICKS_X) % WORLD_BRICKS_Y;
        let bz = bi / (WORLD_BRICKS_X * WORLD_BRICKS_Y);
        let mut solid_n = 0u32;
        // 6 face neighbours.
        let offs: [(i32, i32, i32); 6] = [
            ( 1, 0, 0), (-1, 0, 0),
            ( 0, 1, 0), ( 0,-1, 0),
            ( 0, 0, 1), ( 0, 0,-1),
        ];
        for (dx, dy, dz) in offs {
            let nbx = bx as i32 + dx;
            let nby = by as i32 + dy;
            let nbz = bz as i32 + dz;
            if nbx < 0 || nbx >= WORLD_BRICKS_X as i32
            || nby < 0 || nby >= WORLD_BRICKS_Y as i32
            || nbz < 0 || nbz >= WORLD_BRICKS_Z as i32 { continue; }
            let nbi = brick_idx(nbx as u32, nby as u32, nbz as u32);
            if !self.brick_is_empty(nbi) { solid_n += 1; }
        }
        // Map 0..6 → 255..(255-6*30)=75. 6 neighbours = darkest, 0 = brightest.
        255u8.saturating_sub((solid_n * 30) as u8)
    }

    /// Recompute brick_ao for a brick and its 6 neighbours (their AO changes
    /// when this brick's occupancy changes).
    pub fn refresh_ao_for_brick(&mut self, bi: u32) {
        let bx = bi % WORLD_BRICKS_X;
        let by = (bi / WORLD_BRICKS_X) % WORLD_BRICKS_Y;
        let bz = bi / (WORLD_BRICKS_X * WORLD_BRICKS_Y);
        self.brick_ao[bi as usize] = self.compute_brick_ao(bi);
        let offs: [(i32, i32, i32); 6] = [
            ( 1, 0, 0), (-1, 0, 0),
            ( 0, 1, 0), ( 0,-1, 0),
            ( 0, 0, 1), ( 0, 0,-1),
        ];
        for (dx, dy, dz) in offs {
            let nbx = bx as i32 + dx;
            let nby = by as i32 + dy;
            let nbz = bz as i32 + dz;
            if nbx < 0 || nbx >= WORLD_BRICKS_X as i32
            || nby < 0 || nby >= WORLD_BRICKS_Y as i32
            || nbz < 0 || nbz >= WORLD_BRICKS_Z as i32 { continue; }
            let nbi = brick_idx(nbx as u32, nby as u32, nbz as u32);
            self.brick_ao[nbi as usize] = self.compute_brick_ao(nbi);
        }
    }

    /// Recompute ALL uniform flags from current brick contents. Use after
    /// bulk gen.
    pub fn rebuild_all_uniform(&mut self) {
        use rayon::prelude::*;
        // Parallel recompute of per-brick uniform hints.
        let bricks = &self.bricks;
        self.brick_uniform = bricks.par_iter().map(compute_uniform_hint).collect();
        // Tiles depend on the brick_uniform array we just refreshed.
        let bu = &self.brick_uniform;
        self.tile_uniform = (0..WORLD_TILES_TOTAL as usize).into_par_iter().map(|ti| {
            let tx = (ti as u32) % WORLD_TILES_X;
            let ty = ((ti as u32) / WORLD_TILES_X) % WORLD_TILES_Y;
            let tz = (ti as u32) / (WORLD_TILES_X * WORLD_TILES_Y);
            let bx0 = tx * 4;
            let by0 = ty * 4;
            let bz0 = tz * 4;
            let first_bi = brick_idx(bx0, by0, bz0);
            let m0 = bu[first_bi as usize];
            if m0 == 0 { return 0u8; }
            for dz in 0..4 {
                for dy in 0..4 {
                    for dx in 0..4 {
                        let bi = brick_idx(bx0 + dx, by0 + dy, bz0 + dz);
                        if bu[bi as usize] != m0 { return 0u8; }
                    }
                }
            }
            m0
        }).collect();
    }

    pub fn rebuild_all_masks(&mut self) {
        self.tile_mask.iter_mut().for_each(|m| *m = 0);
        self.chunk_mask.iter_mut().for_each(|m| *m = 0);
        for bz in 0..WORLD_BRICKS_Z {
            for by in 0..WORLD_BRICKS_Y {
                for bx in 0..WORLD_BRICKS_X {
                    if !self.brick_is_empty(brick_idx(bx, by, bz)) {
                        let (tx, ty, tz) = (bx / 4, by / 4, bz / 4);
                        let ti = tile_idx(tx, ty, tz) as usize;
                        self.tile_mask[ti] |= 1u64 << brick_bit_in_tile(bx & 3, by & 3, bz & 3);
                    }
                }
            }
        }
        for tz in 0..WORLD_TILES_Z {
            for ty in 0..WORLD_TILES_Y {
                for tx in 0..WORLD_TILES_X {
                    let ti = tile_idx(tx, ty, tz) as usize;
                    if self.tile_mask[ti] != 0 {
                        let (cx, cy, cz) = (tx / 4, ty / 4, tz / 4);
                        let ci = chunk_idx(cx, cy, cz) as usize;
                        self.chunk_mask[ci] |= 1u64 << tile_bit_in_chunk(tx & 3, ty & 3, tz & 3);
                    }
                }
            }
        }
    }

    /// Bake brick_ao for every brick in parallel. Run after bulk gen.
    pub fn rebuild_all_ao(&mut self) {
        use rayon::prelude::*;
        let bu = &self.brick_uniform;
        // Empty bricks AND uniform bricks are both treated as "non-blocking
        // for AO purposes" — only solid bricks contribute to neighbour
        // occlusion. We can use brick_uniform == 0 + brick.is_empty as the
        // "is solid?" test.
        let is_solid_ref = |bi: u32| -> bool {
            let u = bu[bi as usize];
            if u != 0 { return true; }  // uniform-solid (with mat)
            !self.bricks[bi as usize].is_empty()  // detailed brick with any solid
        };
        self.brick_ao = (0..WORLD_BRICKS_TOTAL)
            .into_par_iter()
            .map(|bi| {
                let bx = bi % WORLD_BRICKS_X;
                let by = (bi / WORLD_BRICKS_X) % WORLD_BRICKS_Y;
                let bz = bi / (WORLD_BRICKS_X * WORLD_BRICKS_Y);
                let mut solid_n = 0u32;
                let offs: [(i32, i32, i32); 6] = [
                    ( 1, 0, 0), (-1, 0, 0),
                    ( 0, 1, 0), ( 0,-1, 0),
                    ( 0, 0, 1), ( 0, 0,-1),
                ];
                for (dx, dy, dz) in offs {
                    let nbx = bx as i32 + dx;
                    let nby = by as i32 + dy;
                    let nbz = bz as i32 + dz;
                    if nbx < 0 || nbx >= WORLD_BRICKS_X as i32
                    || nby < 0 || nby >= WORLD_BRICKS_Y as i32
                    || nbz < 0 || nbz >= WORLD_BRICKS_Z as i32 { continue; }
                    let nbi = brick_idx(nbx as u32, nby as u32, nbz as u32);
                    if is_solid_ref(nbi) { solid_n += 1; }
                }
                255u8.saturating_sub((solid_n * 30) as u8)
            })
            .collect();
    }

    /// Top-level demo generation: walks every storage chunk and generates it
    /// (terrain + ores + sea + trees). Trees place into neighbour chunks so
    /// the tree pass runs after the terrain pass for the whole world.
    pub fn fill_demo_terrain(&mut self) {
        use rayon::prelude::*;
        let seed = self.seed;
        // ---- 4× scale: chunked-batch generation ----
        // 131K storage chunks × 40 KB scratch each = 5+ GB intermediate if
        // we collected everything up-front. Process in batches small enough
        // to keep peak memory bounded but big enough to amortise rayon's
        // dispatch overhead. Each batch generates in parallel, applies to
        // the sparse pool sequentially, then drops its scratches before
        // moving on.
        let total_slots = (WORLD_STORE_CX * WORLD_STORE_CY * WORLD_STORE_CZ) as usize;
        let batch_size: usize = 1024;
        let mut start = 0usize;
        while start < total_slots {
            let end = (start + batch_size).min(total_slots);
            let scratches: Vec<(u32, glam::IVec3, Vec<Brick>)> = (start..end)
                .into_par_iter()
                .map(|i| {
                    let i = i as u32;
                    let slot_cx = i % WORLD_STORE_CX;
                    let slot_cy = (i / WORLD_STORE_CX) % WORLD_STORE_CY;
                    let slot_cz = i / (WORLD_STORE_CX * WORLD_STORE_CY);
                    let world_chunk = glam::IVec3::new(slot_cx as i32, slot_cy as i32, slot_cz as i32);
                    let scratch = gen_slot_bricks(world_chunk, seed);
                    (i, world_chunk, scratch)
                })
                .collect();
            for (i, world_chunk, scratch) in scratches {
                let slot_cx = i % WORLD_STORE_CX;
                let slot_cy = (i / WORLD_STORE_CX) % WORLD_STORE_CY;
                let slot_cz = i / (WORLD_STORE_CX * WORLD_STORE_CY);
                let base_bx = slot_cx * STORAGE_CHUNK_BRICKS;
                let base_by = slot_cy * STORAGE_CHUNK_BRICKS;
                let base_bz = slot_cz * STORAGE_CHUNK_BRICKS;
                for db_z in 0..STORAGE_CHUNK_BRICKS {
                    for db_y in 0..STORAGE_CHUNK_BRICKS {
                        for db_x in 0..STORAGE_CHUNK_BRICKS {
                            let scratch_idx = (db_x
                                + db_y * STORAGE_CHUNK_BRICKS
                                + db_z * STORAGE_CHUNK_BRICKS * STORAGE_CHUNK_BRICKS)
                                as usize;
                            let bi = brick_idx(base_bx + db_x, base_by + db_y, base_bz + db_z);
                            // assign_brick auto-compacts uniform fills into
                            // brick_uniform without allocating a slot —
                            // ~95% of terrain bricks (interior stone, deep
                            // dirt, air, ocean) take this path → real 4×
                            // resolution fits in a ~1 GB CPU budget instead
                            // of the 5+ GB a dense layout would need.
                            self.assign_brick(bi, scratch[scratch_idx]);
                        }
                    }
                }
                let slot_idx = storage_chunk_idx(slot_cx, slot_cy, slot_cz) as usize;
                self.slot_world_chunk[slot_idx] = Some(world_chunk);
            }
            start = end;
        }

        self.rebuild_all_masks();
        self.rebuild_all_uniform();
        self.rebuild_all_ao();
        // movable_mask is a 67M-entry × 8B = 536 MB array. We walk it once
        // and bake in the per-brick movable bitmask.
        for bi in 0..WORLD_BRICKS_TOTAL {
            let b = self.brick_read(bi);
            let mut m = 0u64;
            for i in 0..64usize {
                m |= (is_movable_mat(b.materials[i]) as u64) << i;
            }
            self.movable_mask[bi as usize] = m & b.occupancy;
        }
        self.rebuild_active_bricks();
        self.all_dirty = true;
        for cm in self.chunk_meta.iter_mut() { cm.generated = true; }
    }

    pub fn dims_voxels(&self) -> UVec3 {
        UVec3::new(WORLD_VOXELS_X, WORLD_VOXELS_Y, WORLD_VOXELS_Z)
    }

    /// Build a per-column terrain-top heightmap of the loaded window. Used
    /// by the shadow-map pass: 95% of shadow rays just hit terrain, and
    /// terrain is fully determined by `sample_terrain` — so we can answer
    /// "is this point in terrain shadow?" with one heightmap lookup,
    /// completely bypassing the per-pixel hierarchical DDA.
    ///
    /// Heights are stored as f32 (y-coord of the topmost solid voxel + 1).
    /// Trees, edits, and other non-terrain occluders are NOT in here —
    /// callers fall back to the regular shadow_tex for those.
    pub fn build_terrain_heightmap(&self) -> Vec<f32> {
        use rayon::prelude::*;
        let seed = self.seed;
        let origin = self.world_origin_voxel();
        let w = WORLD_VOXELS_X as i32;
        let d = WORLD_VOXELS_Z as i32;
        (0..(w * d) as usize)
            .into_par_iter()
            .map(|i| {
                let lx = (i as i32) % w;
                let lz = (i as i32) / w;
                let wx = origin.x + lx;
                let wz = origin.z + lz;
                let ts = sample_terrain(wx as f32, wz as f32, seed);
                // Top-of-terrain voxel y (heightmap value). Water surface
                // is at sea_level; we want the SOLID top, so use ts.h.
                (ts.h + 1) as f32
            })
            .collect()
    }
}

/// Decompose a 64-bit world seed into a pair of (x, z) float offsets used to
/// shift noise queries. Different seeds → different terrain by sampling a
/// different region of the same infinite noise field.
#[inline(always)]
pub fn seed_offset_xz(seed: u64) -> (f32, f32) {
    let hi = ((seed >> 32) as u32) as i32 as f32;
    let lo = ((seed & 0xFFFF_FFFF) as u32) as i32 as f32;
    (hi * 0.01734, lo * 0.02153)
}

// ---------- value noise ----------

#[inline(always)]
fn hash2(x: i32, z: i32) -> f32 {
    let h = (x as u32)
        .wrapping_mul(0x9E3779B1)
        .wrapping_add((z as u32).wrapping_mul(0x85EBCA77));
    let h = h.wrapping_mul(0xC2B2AE3D);
    let h = h ^ (h >> 16);
    let h = h.wrapping_mul(0x85EBCA6B);
    ((h & 0xFFFFFF) as f32) / (0xFFFFFF as f32) * 2.0 - 1.0
}

#[inline(always)]
fn hash3(x: i32, y: i32, z: i32) -> f32 {
    let h = (x as u32)
        .wrapping_mul(0x9E3779B1)
        .wrapping_add((y as u32).wrapping_mul(0x85EBCA77))
        .wrapping_add((z as u32).wrapping_mul(0xC2B2AE3D));
    let h = h.wrapping_mul(0xD2B74407);
    let h = h ^ (h >> 16);
    let h = h.wrapping_mul(0x85EBCA6B);
    ((h & 0xFFFFFF) as f32) / (0xFFFFFF as f32) * 2.0 - 1.0
}

#[inline(always)]
fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

pub fn value_noise_2d(x: f32, z: f32) -> f32 {
    let xi = x.floor() as i32;
    let zi = z.floor() as i32;
    let xf = smoothstep(x - xi as f32);
    let zf = smoothstep(z - zi as f32);
    let v00 = hash2(xi, zi);
    let v10 = hash2(xi + 1, zi);
    let v01 = hash2(xi, zi + 1);
    let v11 = hash2(xi + 1, zi + 1);
    let a = v00 * (1.0 - xf) + v10 * xf;
    let b = v01 * (1.0 - xf) + v11 * xf;
    a * (1.0 - zf) + b * zf
}

/// Ridge noise: 1 - |fbm|. Output in [0, 1] with thin "ridge" lines along
/// the fbm = 0 contours. Used for rivers + ravines.
pub fn ridge_noise_2d(x: f32, z: f32) -> f32 {
    let n = fbm_2d(x, z, 4);
    (1.0 - n.abs()).clamp(0.0, 1.0)
}

pub fn fbm_2d(x: f32, z: f32, octaves: u32) -> f32 {
    let mut total = 0.0;
    let mut amp = 1.0;
    let mut freq = 1.0;
    let mut max_amp = 0.0;
    for _ in 0..octaves {
        total += value_noise_2d(x * freq, z * freq) * amp;
        max_amp += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    total / max_amp
}

/// Per-column terrain sample. `water_top` is the y level the topmost water
/// voxel reaches (0 = no water). A river fills its carved channel up to
/// 1 voxel above the surrounding terrain so the player sees a brimming
/// stream, not a sliver at the bottom of a ditch.
pub struct TerrainSample {
    pub h: i32,
    pub water_top: i32,
    pub is_river: bool,
}

pub fn sample_terrain(wx: f32, wz: f32, seed: u64) -> TerrainSample {
    let (s_x, s_z) = seed_offset_xz(seed);
    let px = wx + s_x;
    let pz = wz + s_z;

    // 1.5× scale. Noise freqs × 0.67 → features 1.5× wider in voxels.
    // Amps × 1.5 → 1.5× taller.
    let warp_x = fbm_2d(px * 0.00333, pz * 0.00333, 2) * 12.0;
    let warp_z = fbm_2d(px * 0.00333 + 50.0, pz * 0.00333 + 50.0, 2) * 12.0;
    let wpx = px + warp_x;
    let wpz = pz + warp_z;

    let base = fbm_2d(wpx * 0.008, wpz * 0.008, 4) * 33.0;

    let mountain_mask = (fbm_2d(wpx * 0.00187, wpz * 0.00187, 2) + 0.2).max(0.0);
    let mountain_amp = mountain_mask.min(1.0);
    let mountain_h = fbm_2d(wpx * 0.006, wpz * 0.006, 5).max(0.0).powf(1.15)
        * mountain_amp * 165.0;

    let ravine_n = ridge_noise_2d(wpx * 0.008, wpz * 0.008);
    let ravine_cut = ((ravine_n - 0.97).max(0.0) * 20.0).min(1.0) * 9.0;

    let sea_level: f32 = 64.0;
    let base_h = sea_level + 12.0 + base + mountain_h - ravine_cut;

    let river_n = ridge_noise_2d(px * 0.00333 + 1000.0, pz * 0.00333 + 1000.0);
    let river_strength_raw = ((river_n - 0.85) / 0.15).clamp(0.0, 1.0);
    let elevation_above_sea = (base_h - sea_level).max(0.0);
    let elevation_fade = if elevation_above_sea < 6.0 {
        1.0
    } else if elevation_above_sea < 12.0 {
        1.0 - (elevation_above_sea - 6.0) / 6.0
    } else {
        0.0
    };
    let actual_strength = river_strength_raw * elevation_fade;

    let bed_target = sea_level - 4.0;
    let h_blended = base_h * (1.0 - actual_strength) + bed_target * actual_strength;
    let h = h_blended.clamp(2.0, (WORLD_VOXELS_Y - 1) as f32);
    let h_i = h as i32;
    let is_river = actual_strength > 0.0 && h_i < sea_level as i32;

    // Single GLOBAL water level. Anywhere terrain dips below sea_level (ocean
    // or river) fills with water to sea_level. Cannot overflow because every
    // water cell shares the same surface.
    let water_top = if h_i < sea_level as i32 { sea_level as i32 } else { 0 };
    TerrainSample { h: h_i, water_top, is_river }
}

/// Pure function: produce one storage chunk's worth of bricks from a world
/// chunk coord + seed. No shared state — safe to call from rayon workers.
/// Returns 512 bricks in (x, y, z) order with x innermost.
pub fn gen_slot_bricks(world_chunk: glam::IVec3, seed: u64) -> Vec<Brick> {
    let total = (STORAGE_CHUNK_BRICKS * STORAGE_CHUNK_BRICKS * STORAGE_CHUNK_BRICKS) as usize;
    let mut bricks: Vec<Brick> = vec![Brick::EMPTY; total];
    let sea_level: u32 = 64;
    let (s_x, s_z) = seed_offset_xz(seed);
    let world_x0 = world_chunk.x * STORAGE_CHUNK_VOXELS as i32;
    let world_y0 = world_chunk.y * STORAGE_CHUNK_VOXELS as i32;
    let world_z0 = world_chunk.z * STORAGE_CHUNK_VOXELS as i32;

    for dz in 0..STORAGE_CHUNK_VOXELS {
        for dx in 0..STORAGE_CHUNK_VOXELS {
            let wx_int = world_x0 + dx as i32;
            let wz_int = world_z0 + dz as i32;
            let wx = wx_int as f32;
            let wz = wz_int as f32;
            let ts = sample_terrain(wx, wz, seed);
            let h_signed = ts.h;
            let h_u32 = h_signed as u32;

            // Skip the whole column if it's entirely below or above this
            // chunk's Y range AND has no water that reaches into our range.
            let col_top = h_signed.max(ts.water_top);
            let col_bottom = 0;
            if col_top < world_y0 || col_bottom >= world_y0 + STORAGE_CHUNK_VOXELS as i32 { continue; }

            let temperature = fbm_2d((wx + s_x) * 0.0006, (wz + s_z) * 0.0006, 3);
            let humidity = fbm_2d((wx + s_x) * 0.0008 + 100.0, (wz + s_z) * 0.0008 + 100.0, 3);
            let biome = pick_biome(temperature, humidity, h_u32, sea_level);

            // Compute the Y range that actually overlaps this chunk to skip
            // iterating Y values above terrain (was iterating empty air).
            let y_start = world_y0.max(0);
            let y_end = (world_y0 + STORAGE_CHUNK_VOXELS as i32).min(WORLD_VOXELS_Y as i32);
            // Seal the top 5 voxels below a water column so caves don't
            // perforate the river/lake bed and let the water drain into
            // them. Caves are still allowed deeper underground.
            let has_water_above = ts.water_top > h_signed;
            // 1.5× scale: seal 5 → 8, cave noise freqs × 0.67, subsoil 4 → 6.
            let cave_seal_y = if has_water_above { h_signed - 8 } else { i32::MIN };
            for world_y in y_start..y_end {
                if world_y > h_signed { break; }
                let in_water_seal = world_y >= cave_seal_y;
                if !in_water_seal {
                    let cn = value_noise_3d(wx * 0.030, world_y as f32 * 0.057, wz * 0.030);
                    let cn2 = value_noise_3d(wx * 0.073, world_y as f32 * 0.040, wz * 0.073);
                    if world_y > 6 && world_y + 5 < h_signed && (cn + cn2 * 0.6) > 0.30 { continue; }
                }
                let mat = if ts.is_river && world_y as u32 >= h_u32 {
                    MAT_SAND
                } else if world_y as u32 >= h_u32 {
                    biome.top_block(h_u32, sea_level)
                } else if (world_y as u32) + 6 >= h_u32 {
                    biome.subsoil()
                } else {
                    stone_or_ore(wx, world_y as f32, wz, h_u32)
                };
                let dy = (world_y - world_y0) as u32;
                write_into_scratch(&mut bricks, dx, dy, dz, mat);
            }

            // Water fill — rivers brim above their carved banks, ocean fills
            // any column with terrain below sea level.
            if ts.water_top > h_signed {
                let fill_top = ts.water_top;
                let fill_bottom = (h_signed + 1).max(world_y0);
                let fill_end = fill_top.min(world_y0 + STORAGE_CHUNK_VOXELS as i32 - 1);
                if fill_bottom <= fill_end {
                    for wy in fill_bottom..=fill_end {
                        let dy = (wy - world_y0) as u32;
                        write_into_scratch(&mut bricks, dx, dy, dz, MAT_WATER);
                    }
                }
            }

            // Surface decoration: tall grass + flowers on grass tops. No
            // decoration over water/sand/etc — only on actual grass blocks.
            if ts.water_top == 0 && (h_signed as u32) < WORLD_VOXELS_Y - 1 {
                let surface_top = biome.top_block(h_u32, sea_level);
                if surface_top == MAT_GRASS {
                    let dec_y = h_signed + 1;
                    if dec_y >= world_y0 && dec_y < world_y0 + STORAGE_CHUNK_VOXELS as i32 {
                        let h = hash3(wx_int, dec_y, wz_int);
                        let v = h * 0.5 + 0.5;  // 0..1
                        let dec_mat = if v > 0.985 { MAT_FLOWER }
                                      else if v > 0.92 { MAT_TALL_GRASS }
                                      else { 0u8 };
                        if dec_mat != 0 {
                            let dy = (dec_y - world_y0) as u32;
                            write_into_scratch(&mut bricks, dx, dy, dz, dec_mat);
                        }
                    }
                }
            }
        }
    }

    // ---------- TREE PASS ----------
    // Trees with their *base* in this chunk or any of the 8 xz neighbours.
    // Small trees (canopy radius ≤ 5 vox), so a 1-chunk scan covers them.
    // We write straight into the brick scratch — no per-voxel allocation.
    let chunk_min = (world_x0, world_y0, world_z0);
    let chunk_max = (
        world_x0 + STORAGE_CHUNK_VOXELS as i32,
        world_y0 + STORAGE_CHUNK_VOXELS as i32,
        world_z0 + STORAGE_CHUNK_VOXELS as i32,
    );
    for ncz in -1..=1i32 {
        for ncx in -1..=1i32 {
            let src_chunk = glam::IVec2::new(world_chunk.x + ncx, world_chunk.z + ncz);
            let trees = trees_for_chunk(src_chunk, seed, sea_level);
            for tree in trees {
                let tree_top = tree.base_y + 30;
                if tree.base_y > chunk_max.1 || tree_top < chunk_min.1 { continue; }
                paint_tree(&tree, &mut bricks, chunk_min, chunk_max);
            }
        }
    }

    bricks
}

#[inline(always)]
fn try_write_tree_voxel(
    bricks: &mut [Brick],
    wx: i32, wy: i32, wz: i32, mat: u8,
    cmin: (i32, i32, i32), cmax: (i32, i32, i32),
) {
    if wx < cmin.0 || wx >= cmax.0 { return; }
    if wy < cmin.1 || wy >= cmax.1 { return; }
    if wz < cmin.2 || wz >= cmax.2 { return; }
    let dx = (wx - cmin.0) as u32;
    let dy = (wy - cmin.1) as u32;
    let dz = (wz - cmin.2) as u32;
    let bb_x = dx / BRICK_DIM;
    let bb_y = dy / BRICK_DIM;
    let bb_z = dz / BRICK_DIM;
    let bb_idx = (bb_x + bb_y * STORAGE_CHUNK_BRICKS
        + bb_z * STORAGE_CHUNK_BRICKS * STORAGE_CHUNK_BRICKS) as usize;
    let vi = brick_voxel_idx(dx % BRICK_DIM, dy % BRICK_DIM, dz % BRICK_DIM);
    // Trees never overwrite existing solid voxels (terrain wins).
    if (bricks[bb_idx].occupancy & (1u64 << vi)) == 0 {
        bricks[bb_idx].set(dx % BRICK_DIM, dy % BRICK_DIM, dz % BRICK_DIM, mat);
    }
}

#[derive(Clone, Copy)]
struct TreeSpec {
    base_x: i32,
    base_y: i32,
    base_z: i32,
    ttype: u32,
    hash: u32,
}

/// Deterministic tree positions for a given (xz) chunk.
fn trees_for_chunk(chunk_xz: glam::IVec2, seed: u64, sea_level: u32) -> Vec<TreeSpec> {
    let (s_x, s_z) = seed_offset_xz(seed);
    // Climate at chunk centre — coarse enough that whole forests stay in
    // the same biome.
    let cx_center = (chunk_xz.x as f32 + 0.5) * STORAGE_CHUNK_VOXELS as f32;
    let cz_center = (chunk_xz.y as f32 + 0.5) * STORAGE_CHUNK_VOXELS as f32;
    let temperature = fbm_2d((cx_center + s_x) * 0.0006, (cz_center + s_z) * 0.0006, 3);
    let humidity = fbm_2d((cx_center + s_x) * 0.0008 + 100.0,
                           (cz_center + s_z) * 0.0008 + 100.0, 3);
    let biome = pick_biome(temperature, humidity, sea_level + 10, sea_level);
    let density = biome.tree_density();
    let n_candidates = (density * 5.0).round() as u32;
    if n_candidates == 0 { return Vec::new(); }

    // Patch noise — clearings AND dense thickets within the same biome.
    // Maps fbm ∈ [-1, 1] → [0, 2]: bottom 40% is a clearing (0 trees),
    // the rest scales linearly up to 2x the biome's base density. Forest
    // biome ends up with grove + glade patterns instead of uniform cover.
    let patch_raw = fbm_2d(cx_center * 0.0035, cz_center * 0.0035, 2);
    let patch_mul = ((patch_raw + 0.4).max(0.0) * 1.6).min(2.0);
    let n = ((n_candidates as f32) * patch_mul) as u32;
    if n == 0 { return Vec::new(); }

    let chunk_hash = hash_chunk(chunk_xz.x, chunk_xz.y, seed);
    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n {
        let h = chunk_hash.wrapping_mul(2654435761).wrapping_add(i.wrapping_mul(7919));
        let dx = (h & 31) as i32;
        let dz = ((h >> 5) & 31) as i32;
        let wx = chunk_xz.x * STORAGE_CHUNK_VOXELS as i32 + dx;
        let wz = chunk_xz.y * STORAGE_CHUNK_VOXELS as i32 + dz;
        let ts = sample_terrain(wx as f32, wz as f32, seed);
        if ts.is_river || (ts.h as u32) <= sea_level + 2 { continue; }
        if ts.h + 30 >= WORLD_VOXELS_Y as i32 { continue; }
        let h_terrain = ts.h;
        let local_t = fbm_2d((wx as f32 + s_x) * 0.0006, (wz as f32 + s_z) * 0.0006, 3);
        let local_h = fbm_2d((wx as f32 + s_x) * 0.0008 + 100.0,
                             (wz as f32 + s_z) * 0.0008 + 100.0, 3);
        let local_biome = pick_biome(local_t, local_h, h_terrain as u32, sea_level);
        if matches!(local_biome, Biome::Desert | Biome::Beach) { continue; }
        let ttype = local_biome.tree_type(h);
        out.push(TreeSpec { base_x: wx, base_y: h_terrain + 1, base_z: wz, ttype, hash: h });
    }
    out
}

fn hash_chunk(cx: i32, cz: i32, seed: u64) -> u32 {
    let s = (seed as u32) ^ ((seed >> 32) as u32);
    let mut h = (cx as u32).wrapping_mul(0x9E3779B1);
    h = h.wrapping_add((cz as u32).wrapping_mul(0x85EBCA77));
    h = h.wrapping_add(s.wrapping_mul(0xC2B2AE3D));
    h = h ^ (h >> 16);
    h = h.wrapping_mul(0xD2B74407);
    h ^ (h >> 13)
}

// ---------- branched-tree generator ----------
// Writes the tree's voxels DIRECTLY into the brick scratch — no per-voxel
// allocation, no sort. The bounds check is the only overhead per voxel.

fn paint_tree(
    t: &TreeSpec,
    bricks: &mut [Brick],
    cmin: (i32, i32, i32),
    cmax: (i32, i32, i32),
) {
    let base = glam::IVec3::new(t.base_x, t.base_y, t.base_z);
    let h = t.hash;
    match t.ttype {
        // Pine — 1.5× scale. Slim trunk: 2-vox-wide circular cross section.
        2 => {
            let trunk_h = 15 + (h % 9) as i32;
            let trunk_top = base + glam::IVec3::new(0, trunk_h, 0);
            // 1×1 slim pine trunk.
            paint_trunk(bricks, cmin, cmax, base, trunk_h, 0, MAT_WOOD_PINE);
            let layers: i32 = 6;
            for i in 0..layers {
                let t_f = i as f32 / layers as f32;
                let y = base.y + (trunk_h as f32 * (0.35 + t_f * 0.78)) as i32;
                let r = ((1.0 - t_f).powf(0.85) * 5.0 + 1.5) as i32;
                paint_sphere(bricks, cmin, cmax, glam::IVec3::new(base.x, y, base.z), r, MAT_LEAVES_PINE);
            }
        }
        // Birch — 1.5× scale. Slim trunk (2 vox cross section).
        1 => {
            let trunk_h = 12 + (h % 8) as i32;
            let trunk_top = base + glam::IVec3::new(0, trunk_h, 0);
            // 1×1 slim birch trunk.
            paint_trunk(bricks, cmin, cmax, base, trunk_h, 0, MAT_WOOD_BIRCH);
            let n = 2 + (h % 2) as i32;
            for b in 0..n {
                let angle = (b as f32 / n as f32) * std::f32::consts::TAU
                    + branch_jitter(h, b as u32, 0) * 0.5;
                let len = 3 + ((h.wrapping_mul(b as u32 + 1)) % 4) as i32;
                let sy = base.y + (trunk_h as f32 * 0.7) as i32;
                let end = glam::IVec3::new(
                    base.x + (angle.cos() * len as f32) as i32,
                    sy + 2,
                    base.z + (angle.sin() * len as f32) as i32,
                );
                paint_line(bricks, cmin, cmax, glam::IVec3::new(base.x, sy, base.z), end, 0, MAT_WOOD_BIRCH);
                paint_sphere(bricks, cmin, cmax, end, 3, MAT_LEAVES_BIRCH);
            }
            paint_sphere(bricks, cmin, cmax, trunk_top, 5, MAT_LEAVES_BIRCH);
        }
        // Oak / autumn — 1.5× scale. Thick trunk (3-vox cross section).
        _ => {
            let leaf_mat = if t.ttype == 3 { MAT_LEAVES_AUTUMN } else { MAT_LEAVES };
            let trunk_h = 12 + (h % 8) as i32;
            let trunk_top = base + glam::IVec3::new(0, trunk_h, 0);
            // 3×3 oak trunk.
            paint_trunk(bricks, cmin, cmax, base, trunk_h, 1, MAT_WOOD);
            let n = 3 + (h % 2) as i32;
            for b in 0..n {
                let angle = (b as f32 / n as f32) * std::f32::consts::TAU
                    + branch_jitter(h, b as u32, 0) * 0.6;
                let len = 5 + ((h.wrapping_mul(b as u32 + 7)) % 4) as i32;
                let sy = base.y + (trunk_h as f32 * 0.65) as i32;
                let end = glam::IVec3::new(
                    base.x + (angle.cos() * len as f32) as i32,
                    sy + (len as f32 * 0.5) as i32,
                    base.z + (angle.sin() * len as f32) as i32,
                );
                paint_line(bricks, cmin, cmax, glam::IVec3::new(base.x, sy, base.z), end, 0, MAT_WOOD);
                paint_sphere(bricks, cmin, cmax, end, 5, leaf_mat);
            }
            paint_sphere(bricks, cmin, cmax, trunk_top, 6, leaf_mat);
        }
    }
}

// thickness=0 → 1-voxel-wide line (no spheres along the line). Otherwise a
// small radius is splatted at each step. Keep small to avoid voxel blowup.
fn paint_line(
    bricks: &mut [Brick], cmin: (i32, i32, i32), cmax: (i32, i32, i32),
    a: glam::IVec3, b: glam::IVec3, thickness: i32, mat: u8,
) {
    let d = b - a;
    let len = ((d.x * d.x + d.y * d.y + d.z * d.z) as f32).sqrt();
    let steps = (len * 1.5).ceil() as i32;
    if steps <= 0 {
        try_write_tree_voxel(bricks, a.x, a.y, a.z, mat, cmin, cmax);
        return;
    }
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        let cx = (a.x as f32 + d.x as f32 * t).round() as i32;
        let cy = (a.y as f32 + d.y as f32 * t).round() as i32;
        let cz = (a.z as f32 + d.z as f32 * t).round() as i32;
        if thickness == 0 {
            try_write_tree_voxel(bricks, cx, cy, cz, mat, cmin, cmax);
        } else {
            let r2 = thickness * thickness;
            for dy in -thickness..=thickness {
                for dx in -thickness..=thickness {
                    for dz in -thickness..=thickness {
                        if dx * dx + dy * dy + dz * dz > r2 { continue; }
                        try_write_tree_voxel(bricks, cx + dx, cy + dy, cz + dz, mat, cmin, cmax);
                    }
                }
            }
        }
    }
}

/// Vertical SQUARE trunk — paints an axis-aligned square cross section of
/// `(2*radius+1)²` voxels in xz, stacked for `height` voxels in y.
/// radius=0 → 1×1, radius=1 → 3×3, radius=2 → 5×5.
fn paint_trunk(
    bricks: &mut [Brick], cmin: (i32, i32, i32), cmax: (i32, i32, i32),
    base: glam::IVec3, height: i32, radius: i32, mat: u8,
) {
    for dy in 0..height {
        for dz in -radius..=radius {
            for dx in -radius..=radius {
                try_write_tree_voxel(bricks, base.x + dx, base.y + dy, base.z + dz, mat, cmin, cmax);
            }
        }
    }
}

fn paint_sphere(
    bricks: &mut [Brick], cmin: (i32, i32, i32), cmax: (i32, i32, i32),
    center: glam::IVec3, r: i32, mat: u8,
) {
    let r2 = r * r;
    for dy in -r..=r {
        for dx in -r..=r {
            for dz in -r..=r {
                if dx * dx + dy * dy + dz * dz > r2 { continue; }
                try_write_tree_voxel(bricks, center.x + dx, center.y + dy, center.z + dz, mat, cmin, cmax);
            }
        }
    }
}

fn branch_jitter(hash: u32, b: u32, salt: u32) -> f32 {
    let h = hash
        .wrapping_mul(0x9E3779B1)
        .wrapping_add(b.wrapping_mul(2654435761))
        .wrapping_add(salt.wrapping_mul(40503));
    ((h & 0xFFFF) as f32 / 65535.0) * 2.0 - 1.0
}

#[inline]
fn write_into_scratch(bricks: &mut [Brick], dx: u32, dy: u32, dz: u32, mat: u8) {
    let bb_x = dx / BRICK_DIM;
    let bb_y = dy / BRICK_DIM;
    let bb_z = dz / BRICK_DIM;
    let bb_idx = (bb_x + bb_y * STORAGE_CHUNK_BRICKS + bb_z * STORAGE_CHUNK_BRICKS * STORAGE_CHUNK_BRICKS)
        as usize;
    bricks[bb_idx].set(dx % BRICK_DIM, dy % BRICK_DIM, dz % BRICK_DIM, mat);
}

// ---------------- biome + world-gen helpers ----------------

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Biome {
    Tundra,
    Plains,
    Forest,
    Jungle,
    Savanna,
    Desert,
    Beach,
    Mountain,
}

pub fn pick_biome(temp: f32, humid: f32, h: u32, sea_level: u32) -> Biome {
    if h > sea_level + 135 { return Biome::Mountain; }
    if h <= sea_level + 2 { return Biome::Beach; }
    if temp < -0.20 { return Biome::Tundra; }
    if temp > 0.25 && humid < -0.05 { return Biome::Desert; }
    if temp > 0.15 && humid > 0.25 { return Biome::Jungle; }
    if temp > 0.10 && humid < 0.10 { return Biome::Savanna; }
    if humid > 0.20 { return Biome::Forest; }
    Biome::Plains
}

impl Biome {
    pub fn top_block(self, h: u32, sea_level: u32) -> u8 {
        match self {
            Biome::Tundra => MAT_SNOW,
            Biome::Desert | Biome::Beach | Biome::Savanna => MAT_SAND,
            Biome::Mountain => if h > sea_level + 150 { MAT_SNOW }
                               else { MAT_GRASS },
            _ => MAT_GRASS,
        }
    }
    pub fn subsoil(self) -> u8 {
        match self {
            Biome::Desert | Biome::Beach => MAT_SAND,
            // Mountain subsoil = DIRT (was STONE) so exposed mountain
            // slope faces look natural instead of a wall of stone.
            Biome::Mountain => MAT_DIRT,
            Biome::Savanna => MAT_DIRT,
            _ => MAT_DIRT,
        }
    }
    pub fn tree_type(self, hash: u32) -> u32 {
        match self {
            Biome::Tundra | Biome::Mountain => 2, // pine
            Biome::Plains => if hash % 3 == 0 { 1 } else { 0 }, // birch/oak
            Biome::Forest => match hash % 4 { 0 => 1, 1 => 3, _ => 0 }, // birch/autumn/oak
            Biome::Jungle => match hash % 4 { 0 => 3, _ => 0 },  // oak/autumn — dense
            Biome::Savanna => 0,
            _ => 0,
        }
    }
    /// Trees per chunk multiplier — Jungle is dense, Savanna sparse.
    /// Trees per chunk in a "dense patch" of this biome. Clearings (low
    /// patch noise) bring it down to zero, dense patches scale by ~2x.
    pub fn tree_density(self) -> f32 {
        // Halved again — trees still too dense at 2× scale.
        match self {
            Biome::Jungle => 0.28,
            Biome::Forest => 0.13,
            Biome::Tundra => 0.04,
            Biome::Plains => 0.02,
            Biome::Mountain => 0.015,
            Biome::Savanna => 0.01,
            _ => 0.0,
        }
    }
    /// Elevation contribution — mountains are noticeably taller, jungles are
    /// rolling, plains are nearly flat.
    pub fn height_mult(self) -> f32 {
        match self {
            Biome::Mountain => 2.5,
            Biome::Jungle => 1.4,
            Biome::Forest => 1.0,
            Biome::Plains => 0.4,
            Biome::Savanna => 0.6,
            Biome::Tundra => 1.2,
            Biome::Desert => 0.5,
            Biome::Beach => 0.2,
        }
    }
}

/// Replace some stone voxels with ore. Rarer / more valuable ores cluster
/// deeper. Three noise scales give chunkier veins instead of single specks.
pub fn stone_or_ore(x: f32, y: f32, z: f32, h: u32) -> u8 {
    let depth = (h as f32 - y).max(0.0);
    let n1 = value_noise_3d(x * 0.27, y * 0.27, z * 0.27);
    let n2 = value_noise_3d(x * 0.55, y * 0.55, z * 0.55);
    let combined = n1 + n2 * 0.30;
    if depth > 30.0 && combined > 0.50 { return MAT_DIAMOND; }
    if depth > 20.0 && combined > 0.36 { return MAT_GOLD; }
    if depth > 10.0 && combined > 0.24 { return MAT_IRON; }
    if combined > 0.32 { return MAT_COAL; }
    MAT_STONE
}

pub fn value_noise_3d(x: f32, y: f32, z: f32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let zi = z.floor() as i32;
    let xf = smoothstep(x - xi as f32);
    let yf = smoothstep(y - yi as f32);
    let zf = smoothstep(z - zi as f32);
    let v000 = hash3(xi, yi, zi);
    let v100 = hash3(xi + 1, yi, zi);
    let v010 = hash3(xi, yi + 1, zi);
    let v110 = hash3(xi + 1, yi + 1, zi);
    let v001 = hash3(xi, yi, zi + 1);
    let v101 = hash3(xi + 1, yi, zi + 1);
    let v011 = hash3(xi, yi + 1, zi + 1);
    let v111 = hash3(xi + 1, yi + 1, zi + 1);
    let a = v000 * (1.0 - xf) + v100 * xf;
    let b = v010 * (1.0 - xf) + v110 * xf;
    let c = v001 * (1.0 - xf) + v101 * xf;
    let d = v011 * (1.0 - xf) + v111 * xf;
    let ab = a * (1.0 - yf) + b * yf;
    let cd = c * (1.0 - yf) + d * yf;
    ab * (1.0 - zf) + cd * zf
}
