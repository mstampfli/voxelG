// GPU terrain generation. Replaces the CPU `gen_slot_bricks` for the
// terrain + caves + decoration passes. Trees are still painted CPU-side
// after readback because they cross chunk boundaries (each chunk reads
// its 8 neighbours' tree placements).
//
// Layout:
//   - One dispatch processes a BATCH of N chunks.
//   - Workgroup size (4, 4, 4) = 64 threads = 1 brick.
//   - Dispatch (8, 8, 8 * N) workgroups → 8×8 bricks in xy, brick_z * N in z.
//   - Each thread = 1 voxel. Threads cooperate via workgroup-shared memory
//     and atomics to pack the brick.
//   - Output: array<Brick, 512 * N> — linear, indexed by
//     chunk_in_batch * 512 + (bx + by*8 + bz*64).

struct Brick {
    occ_lo: u32,
    occ_hi: u32,
    materials: array<u32, 16>,
};

struct GenRequest {
    world_cx: i32,
    world_cy: i32,
    world_cz: i32,
    lod: i32, // 0 = full octaves, 1 = halved (distant chunks, faster gen)
};

struct GenParams {
    seed_x: f32,
    seed_z: f32,
    sea_level: f32,
    _pad: f32,
};

@group(0) @binding(0) var<storage, read> requests: array<GenRequest>;
@group(0) @binding(1) var<storage, read_write> out_bricks: array<Brick>;
@group(0) @binding(2) var<uniform> params: GenParams;

const MAT_AIR:        u32 = 0u;
const MAT_SAND:       u32 = 1u;
const MAT_GRASS:      u32 = 2u;
const MAT_DIRT:       u32 = 3u;
const MAT_STONE:      u32 = 4u;
const MAT_WATER:      u32 = 12u; // MAT_WATER_L8
const MAT_SNOW:       u32 = 15u;
const MAT_COAL:       u32 = 19u;
const MAT_IRON:       u32 = 20u;
const MAT_GOLD:       u32 = 21u;
const MAT_DIAMOND:    u32 = 22u;
const MAT_FLOWER:     u32 = 30u;
const MAT_TALL_GRASS: u32 = 31u;

const BRICK_DIM:      u32 = 4u;
const STORAGE_CHUNK_VOXELS: i32 = 32;

// ---------------- hashes / value noise ----------------

fn hash2(x: i32, z: i32) -> f32 {
    var h = bitcast<u32>(x) * 0x9E3779B1u + bitcast<u32>(z) * 0x85EBCA77u;
    h = h * 0xC2B2AE3Du;
    h = h ^ (h >> 16u);
    h = h * 0x85EBCA6Bu;
    return f32(h & 0xFFFFFFu) / f32(0xFFFFFFu) * 2.0 - 1.0;
}

fn hash3(x: i32, y: i32, z: i32) -> f32 {
    var h = bitcast<u32>(x) * 0x9E3779B1u
          + bitcast<u32>(y) * 0x85EBCA77u
          + bitcast<u32>(z) * 0xC2B2AE3Du;
    h = h * 0xD2B74407u;
    h = h ^ (h >> 16u);
    h = h * 0x85EBCA6Bu;
    return f32(h & 0xFFFFFFu) / f32(0xFFFFFFu) * 2.0 - 1.0;
}

fn smoothstep_(t: f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
}

fn value_noise_2d(x: f32, z: f32) -> f32 {
    let xi = i32(floor(x));
    let zi = i32(floor(z));
    let xf = smoothstep_(x - f32(xi));
    let zf = smoothstep_(z - f32(zi));
    let v00 = hash2(xi, zi);
    let v10 = hash2(xi + 1, zi);
    let v01 = hash2(xi, zi + 1);
    let v11 = hash2(xi + 1, zi + 1);
    let a = v00 * (1.0 - xf) + v10 * xf;
    let b = v01 * (1.0 - xf) + v11 * xf;
    return a * (1.0 - zf) + b * zf;
}

fn value_noise_3d(x: f32, y: f32, z: f32) -> f32 {
    let xi = i32(floor(x));
    let yi = i32(floor(y));
    let zi = i32(floor(z));
    let xf = smoothstep_(x - f32(xi));
    let yf = smoothstep_(y - f32(yi));
    let zf = smoothstep_(z - f32(zi));
    let v000 = hash3(xi,     yi,     zi);
    let v100 = hash3(xi + 1, yi,     zi);
    let v010 = hash3(xi,     yi + 1, zi);
    let v110 = hash3(xi + 1, yi + 1, zi);
    let v001 = hash3(xi,     yi,     zi + 1);
    let v101 = hash3(xi + 1, yi,     zi + 1);
    let v011 = hash3(xi,     yi + 1, zi + 1);
    let v111 = hash3(xi + 1, yi + 1, zi + 1);
    let a00 = v000 * (1.0 - xf) + v100 * xf;
    let a10 = v010 * (1.0 - xf) + v110 * xf;
    let a01 = v001 * (1.0 - xf) + v101 * xf;
    let a11 = v011 * (1.0 - xf) + v111 * xf;
    let b0 = a00 * (1.0 - yf) + a10 * yf;
    let b1 = a01 * (1.0 - yf) + a11 * yf;
    return b0 * (1.0 - zf) + b1 * zf;
}

fn fbm_2d(x: f32, z: f32, octaves: u32) -> f32 {
    var total: f32 = 0.0;
    var amp: f32 = 1.0;
    var freq: f32 = 1.0;
    var max_amp: f32 = 0.0;
    for (var i: u32 = 0u; i < octaves; i = i + 1u) {
        total = total + value_noise_2d(x * freq, z * freq) * amp;
        max_amp = max_amp + amp;
        amp = amp * 0.5;
        freq = freq * 2.0;
    }
    return total / max_amp;
}

fn ridge_noise_2d(x: f32, z: f32) -> f32 {
    let n = fbm_2d(x, z, 4u);
    return clamp(1.0 - abs(n), 0.0, 1.0);
}

// ---------------- terrain sample (port of sample_terrain) ----------------

struct TerrainSample {
    h:         i32,
    water_top: i32,
    is_river:  u32,
};

// Halve octave count when LOD > 0, floor at 1.
fn oct_lod(base: u32, lod: i32) -> u32 {
    return max(1u, base >> u32(lod));
}

fn ridge_noise_2d_lod(x: f32, z: f32, lod: i32) -> f32 {
    let n = fbm_2d(x, z, oct_lod(4u, lod));
    return clamp(1.0 - abs(n), 0.0, 1.0);
}

fn sample_terrain(wx: f32, wz: f32, lod: i32) -> TerrainSample {
    let px = wx + params.seed_x;
    let pz = wz + params.seed_z;

    let warp_x = fbm_2d(px * 0.0025, pz * 0.0025, oct_lod(2u, lod)) * 16.0;
    let warp_z = fbm_2d(px * 0.0025 + 50.0, pz * 0.0025 + 50.0, oct_lod(2u, lod)) * 16.0;
    let wpx = px + warp_x;
    let wpz = pz + warp_z;

    let base_n = fbm_2d(wpx * 0.006, wpz * 0.006, oct_lod(4u, lod)) * 44.0;

    let mountain_mask = max(fbm_2d(wpx * 0.0014, wpz * 0.0014, oct_lod(2u, lod)) + 0.2, 0.0);
    let mountain_amp = min(mountain_mask, 1.0);
    let mountain_h = pow(max(fbm_2d(wpx * 0.0045, wpz * 0.0045, oct_lod(5u, lod)), 0.0), 1.15)
                   * mountain_amp * 220.0;

    let ravine_n = ridge_noise_2d_lod(wpx * 0.006, wpz * 0.006, lod);
    let ravine_cut = min(max(ravine_n - 0.97, 0.0) * 20.0, 1.0) * 12.0;

    let sea = params.sea_level;
    let base_h = sea + 16.0 + base_n + mountain_h - ravine_cut;

    let river_n = ridge_noise_2d_lod(px * 0.0025 + 1000.0, pz * 0.0025 + 1000.0, lod);
    let river_strength_raw = clamp((river_n - 0.85) / 0.15, 0.0, 1.0);
    let elevation_above_sea = max(base_h - sea, 0.0);
    var elevation_fade: f32;
    if (elevation_above_sea < 8.0) {
        elevation_fade = 1.0;
    } else if (elevation_above_sea < 16.0) {
        elevation_fade = 1.0 - (elevation_above_sea - 8.0) * 0.125;
    } else {
        elevation_fade = 0.0;
    }
    let actual_strength = river_strength_raw * elevation_fade;

    let bed_target = sea - 6.0;
    let h_blended = base_h * (1.0 - actual_strength) + bed_target * actual_strength;
    let h = clamp(h_blended, 4.0, 511.0);
    let h_i = i32(h);
    let sea_i = i32(sea);
    let is_river = u32(actual_strength > 0.0 && h_i < sea_i);
    let water_top = select(0, sea_i, h_i < sea_i);

    return TerrainSample(h_i, water_top, is_river);
}

// ---------------- biome ----------------
// 0 Tundra 1 Plains 2 Forest 3 Jungle 4 Savanna 5 Desert 6 Beach 7 Mountain
fn pick_biome(temp: f32, humid: f32, h: i32, sea: i32) -> u32 {
    if (h > sea + 72) { return 7u; }
    if (h <= sea + 2) { return 6u; }
    if (temp < -0.20) { return 0u; }
    if (temp > 0.25 && humid < -0.05) { return 5u; }
    if (temp > 0.15 && humid > 0.25) { return 3u; }
    if (temp > 0.10 && humid < 0.10) { return 4u; }
    if (humid > 0.20) { return 2u; }
    return 1u;
}

fn biome_top_block(biome: u32, h: i32, sea: i32) -> u32 {
    switch (biome) {
        case 0u: { return MAT_SNOW; }
        case 5u, 6u, 4u: { return MAT_SAND; }
        case 7u: { return select(MAT_STONE, MAT_SNOW, h > sea + 110); }
        default: { return MAT_GRASS; }
    }
}

fn biome_subsoil(biome: u32) -> u32 {
    switch (biome) {
        case 5u, 6u: { return MAT_SAND; }
        case 7u: { return MAT_STONE; }
        default: { return MAT_DIRT; }
    }
}

fn stone_or_ore(x: f32, y: f32, z: f32, h: i32) -> u32 {
    let depth = max(f32(h) - y, 0.0);
    let n1 = value_noise_3d(x * 0.27, y * 0.27, z * 0.27);
    let n2 = value_noise_3d(x * 0.55, y * 0.55, z * 0.55);
    let combined = n1 + n2 * 0.30;
    if (depth > 30.0 && combined > 0.50) { return MAT_DIAMOND; }
    if (depth > 20.0 && combined > 0.36) { return MAT_GOLD; }
    if (depth > 10.0 && combined > 0.24) { return MAT_IRON; }
    if (combined > 0.32) { return MAT_COAL; }
    return MAT_STONE;
}

// ---------------- voxel material at (wx, wy, wz) ----------------

fn voxel_material(wx: i32, wy: i32, wz: i32, ts: TerrainSample, biome: u32, sea: i32) -> u32 {
    // Top decoration (tall grass / flowers) — only on grass-topped columns
    // above sea, exactly 1 voxel above terrain top.
    if (wy == ts.h + 1 && ts.water_top == 0) {
        if (biome_top_block(biome, ts.h, sea) == MAT_GRASS) {
            let hh = hash3(wx, wy, wz);
            let v = hh * 0.5 + 0.5;
            if (v > 0.985) { return MAT_FLOWER; }
            if (v > 0.92)  { return MAT_TALL_GRASS; }
        }
    }

    // Water fill — above terrain top, up to water_top.
    if (wy > ts.h) {
        if (ts.water_top > 0 && wy >= ts.h + 1 && wy <= ts.water_top) {
            return MAT_WATER;
        }
        return MAT_AIR;
    }

    // Cave check — same as CPU.
    let has_water_above = ts.water_top > ts.h;
    let cave_seal_y = select(-2147483648, ts.h - 10, has_water_above);
    let in_water_seal = wy >= cave_seal_y;
    if (!in_water_seal) {
        let fx = f32(wx); let fy = f32(wy); let fz = f32(wz);
        let cn  = value_noise_3d(fx * 0.0225, fy * 0.0425, fz * 0.0225);
        let cn2 = value_noise_3d(fx * 0.055,  fy * 0.030,  fz * 0.055);
        if (wy > 8 && wy + 6 < ts.h && (cn + cn2 * 0.6) > 0.30) {
            return MAT_AIR;
        }
    }

    // Terrain materials.
    if (ts.is_river != 0u && wy >= ts.h) { return MAT_SAND; }
    if (wy == ts.h) { return biome_top_block(biome, ts.h, sea); }
    if (wy + 8 >= ts.h) { return biome_subsoil(biome); }
    return stone_or_ore(f32(wx), f32(wy), f32(wz), ts.h);
}

// ---------------- compute entry point ----------------

var<workgroup> brick_mats: array<atomic<u32>, 64>;
var<workgroup> brick_occ_lo: atomic<u32>;
var<workgroup> brick_occ_hi: atomic<u32>;

// Per-column cache. Each brick has 4×4 = 16 distinct (x, z) columns.
// The heightmap (sample_terrain, ~21 fbm calls) + biome (6 fbm calls)
// depend only on (x, z), so we compute them ONCE per column (16 threads
// at lid.y == 0) and have all 64 threads read from the cache.
// Cuts terrain-noise work ~4× per brick.
struct ColCache {
    h:         i32,
    water_top: i32,
    is_river:  u32,
    biome:     u32,
};
var<workgroup> col_cache: array<ColCache, 16>;

fn brick_voxel_idx(lx: u32, ly: u32, lz: u32) -> u32 {
    return lx + lz * 4u + ly * 16u;
}

@compute @workgroup_size(4, 4, 4)
fn cs_gen(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) lin_lid: u32,
) {
    // Decode which chunk + brick this workgroup handles.
    let brick_x = wid.x;       // 0..8
    let brick_y = wid.y;       // 0..8
    let chunk_in_batch = wid.z / 8u;
    let brick_z = wid.z % 8u;

    // Init shared state (once per workgroup).
    if (lin_lid == 0u) {
        atomicStore(&brick_occ_lo, 0u);
        atomicStore(&brick_occ_hi, 0u);
    }
    let vi = brick_voxel_idx(lid.x, lid.y, lid.z);
    atomicStore(&brick_mats[vi], 0u);
    workgroupBarrier();

    let req = requests[chunk_in_batch];
    let world_x0 = req.world_cx * STORAGE_CHUNK_VOXELS;
    let world_y0 = req.world_cy * STORAGE_CHUNK_VOXELS;
    let world_z0 = req.world_cz * STORAGE_CHUNK_VOXELS;

    let wx = world_x0 + i32(brick_x * BRICK_DIM + lid.x);
    let wy = world_y0 + i32(brick_y * BRICK_DIM + lid.y);
    let wz = world_z0 + i32(brick_z * BRICK_DIM + lid.z);
    let sea_i = i32(params.sea_level);
    let lod = req.lod;

    // ---- per-column cache fill (only y == 0 threads do the noise work) ----
    // 16 threads do the heightmap + biome; remaining 48 threads in the
    // workgroup skip the expensive ~27 fbm calls and read the cache.
    let col_id = lid.x + lid.z * 4u;
    if (lid.y == 0u) {
        let ts = sample_terrain(f32(wx), f32(wz), lod);
        let temperature = fbm_2d((f32(wx) + params.seed_x) * 0.0006,
                                 (f32(wz) + params.seed_z) * 0.0006, oct_lod(3u, lod));
        let humidity    = fbm_2d((f32(wx) + params.seed_x) * 0.0008 + 100.0,
                                 (f32(wz) + params.seed_z) * 0.0008 + 100.0, oct_lod(3u, lod));
        let biome = pick_biome(temperature, humidity, ts.h, sea_i);
        col_cache[col_id] = ColCache(ts.h, ts.water_top, ts.is_river, biome);
    }
    workgroupBarrier();

    let cached = col_cache[col_id];
    let ts = TerrainSample(cached.h, cached.water_top, cached.is_river);
    let biome = cached.biome;

    let mat = voxel_material(wx, wy, wz, ts, biome, sea_i);

    atomicStore(&brick_mats[vi], mat);
    if (mat != 0u) {
        if (vi < 32u) { atomicOr(&brick_occ_lo, 1u << vi); }
        else          { atomicOr(&brick_occ_hi, 1u << (vi - 32u)); }
    }
    workgroupBarrier();

    // Thread 0 packs and writes the Brick.
    if (lin_lid == 0u) {
        let brick_lin_in_chunk = brick_x + brick_y * 8u + brick_z * 64u;
        let out_idx = chunk_in_batch * 512u + brick_lin_in_chunk;
        var b: Brick;
        b.occ_lo = atomicLoad(&brick_occ_lo);
        b.occ_hi = atomicLoad(&brick_occ_hi);
        for (var w: u32 = 0u; w < 16u; w = w + 1u) {
            let m0 = atomicLoad(&brick_mats[w * 4u + 0u]);
            let m1 = atomicLoad(&brick_mats[w * 4u + 1u]);
            let m2 = atomicLoad(&brick_mats[w * 4u + 2u]);
            let m3 = atomicLoad(&brick_mats[w * 4u + 3u]);
            b.materials[w] = m0 | (m1 << 8u) | (m2 << 16u) | (m3 << 24u);
        }
        out_bricks[out_idx] = b;
    }
}
