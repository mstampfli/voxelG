// Hierarchical DDA raymarch through a 3-level bit pyramid.
//
//   L3 chunk_mask  — bit per child tile  (one u64 covers a 64³ voxel region)
//   L2 tile_mask   — bit per child brick (one u64 covers a 16³ voxel region)
//   L1 brick.occ   — bit per voxel       (one u64 covers a  4³ voxel region)
//
// On a hit we compute per-corner ambient occlusion by reading 12 neighbour
// occupancy bits across the hit face (4 corners × 3 samples each). With the
// hierarchy, each "is this neighbour solid?" lookup is ~3 u32 fetches max
// (chunk → tile → brick) and most return false at L3/L2 immediately.

struct Camera {
    origin: vec3<f32>,
    _pad0: f32,
    forward: vec3<f32>,
    _pad1: f32,
    right: vec3<f32>,
    _pad2: f32,
    up: vec3<f32>,
    tan_half_fov: f32,
    resolution: vec2<f32>,
    time: f32,
    _pad3: f32,
    world_origin: vec3<i32>,
    _pad4: i32,
};

// Toroidal storage: world voxel coords get folded into [0, WORLD_VOXELS_*)
// for the lookup. As the camera shifts the origin, only the small edge
// region's slots get reused — the rest of storage stays put.
fn pos_mod(a: i32, b: i32) -> i32 {
    let r = a % b;
    return select(r, r + b, r < 0);
}
fn world_to_slot_voxel(wv: vec3<i32>) -> vec3<i32> {
    return vec3<i32>(
        pos_mod(wv.x, WORLD_VOXELS_X),
        wv.y,
        pos_mod(wv.z, WORLD_VOXELS_Z),
    );
}

struct Brick {
    occ_lo: u32,
    occ_hi: u32,
    materials: array<u32, 16>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> bricks: array<Brick>;
@group(0) @binding(2) var<storage, read> tile_mask: array<u32>;
@group(0) @binding(3) var<storage, read> chunk_mask: array<u32>;
@group(0) @binding(4) var<uniform> palette: array<vec4<f32>, 256>;
@group(0) @binding(5) var output_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(6) var beam_depth: texture_2d<f32>;
@group(0) @binding(7) var<storage, read> tile_dirty: array<u32>;

struct PlayersBuf {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    positions: array<vec4<f32>, 16>,
};
@group(0) @binding(8) var<storage, read> players: PlayersBuf;

fn ray_aabb_t(origin: vec3<f32>, inv_dir: vec3<f32>, mn: vec3<f32>, mx: vec3<f32>) -> f32 {
    let t0 = (mn - origin) * inv_dir;
    let t1 = (mx - origin) * inv_dir;
    let tmin3 = min(t0, t1);
    let tmax3 = max(t0, t1);
    let t_enter = max(max(tmin3.x, tmin3.y), max(tmin3.z, 0.0));
    let t_exit = min(min(tmax3.x, tmax3.y), tmax3.z);
    if (t_enter >= t_exit || t_exit < 0.0) { return 1e30; }
    return t_enter;
}

fn player_color_for(id: u32) -> vec3<f32> {
    let h = (id * 2654435761u) & 0xFFu;
    let r = f32((h * 73u) & 0xFFu) / 255.0;
    let g = f32((h * 41u + 91u) & 0xFFu) / 255.0;
    let b = f32((h * 113u + 53u) & 0xFFu) / 255.0;
    return vec3<f32>(0.5 + r * 0.5, 0.5 + g * 0.5, 0.5 + b * 0.5);
}

const BRICK_DIM: i32 = 4;

const WORLD_BRICKS_X: i32 = 128;
const WORLD_BRICKS_Y: i32 = 64;
const WORLD_BRICKS_Z: i32 = 128;

const WORLD_VOXELS_X: i32 = 512;
const WORLD_VOXELS_Y: i32 = 256;
const WORLD_VOXELS_Z: i32 = 512;

const WORLD_TILES_X: i32 = 32;
const WORLD_TILES_Y: i32 = 16;
const WORLD_TILES_Z: i32 = 32;

const WORLD_CHUNKS_X: i32 = 8;
const WORLD_CHUNKS_Y: i32 = 4;
const WORLD_CHUNKS_Z: i32 = 8;

fn brick_voxel_idx(lx: i32, ly: i32, lz: i32) -> i32 {
    return lx + lz * 4 + ly * 16;
}

fn world_brick_idx(bx: i32, by: i32, bz: i32) -> i32 {
    return bx + by * WORLD_BRICKS_X + bz * WORLD_BRICKS_X * WORLD_BRICKS_Y;
}

fn world_tile_idx(tx: i32, ty: i32, tz: i32) -> i32 {
    return tx + ty * WORLD_TILES_X + tz * WORLD_TILES_X * WORLD_TILES_Y;
}

fn world_chunk_idx(cx: i32, cy: i32, cz: i32) -> i32 {
    return cx + cy * WORLD_CHUNKS_X + cz * WORLD_CHUNKS_X * WORLD_CHUNKS_Y;
}

fn tile_has_child(ti: i32, child_lin: i32) -> bool {
    let base = ti * 2;
    if (child_lin < 32) {
        return (tile_mask[base] & (1u << u32(child_lin))) != 0u;
    }
    return (tile_mask[base + 1] & (1u << u32(child_lin - 32))) != 0u;
}

fn chunk_has_child(ci: i32, child_lin: i32) -> bool {
    let base = ci * 2;
    if (child_lin < 32) {
        return (chunk_mask[base] & (1u << u32(child_lin))) != 0u;
    }
    return (chunk_mask[base + 1] & (1u << u32(child_lin - 32))) != 0u;
}

fn brick_voxel_solid(bi: i32, vi: i32) -> bool {
    let b = bricks[bi];
    if (vi < 32) {
        return (b.occ_lo & (1u << u32(vi))) != 0u;
    }
    return (b.occ_hi & (1u << u32(vi - 32))) != 0u;
}

fn brick_voxel_material(bi: i32, vi: i32) -> u32 {
    let word = vi / 4;
    let byte = vi - word * 4;
    return (bricks[bi].materials[word] >> u32(byte * 8)) & 0xFFu;
}

fn is_voxel_solid(world_v: vec3<i32>) -> bool {
    // Bounds: voxel must be inside the loaded window.
    let rel = world_v - camera.world_origin;
    if (rel.x < 0 || rel.x >= WORLD_VOXELS_X
     || rel.y < 0 || rel.y >= WORLD_VOXELS_Y
     || rel.z < 0 || rel.z >= WORLD_VOXELS_Z) {
        return false;
    }
    // Fold into slot voxel for storage lookup.
    let v = world_to_slot_voxel(world_v);
    let bp = v >> vec3<u32>(2u);
    let tp = v >> vec3<u32>(4u);
    let cp = v >> vec3<u32>(6u);
    let ci = world_chunk_idx(cp.x, cp.y, cp.z);
    let tile_lin = (tp.x & 3) + (tp.z & 3) * 4 + (tp.y & 3) * 16;
    if (!chunk_has_child(ci, tile_lin)) { return false; }
    let ti = world_tile_idx(tp.x, tp.y, tp.z);
    let brick_lin = (bp.x & 3) + (bp.z & 3) * 4 + (bp.y & 3) * 16;
    if (!tile_has_child(ti, brick_lin)) { return false; }
    let bi = world_brick_idx(bp.x, bp.y, bp.z);
    let local = v - bp * BRICK_DIM;
    let vi = brick_voxel_idx(local.x, local.y, local.z);
    return brick_voxel_solid(bi, vi);
}

fn sky(dir: vec3<f32>) -> vec3<f32> {
    return sky_color(dir);
}

/// Sky-like colour with **no stars, sun disc, or cloud emitters** — used for
/// the fog blend on distant terrain so far-away blocks don't visibly show
/// pinpoint stars through them at night.
fn fog_atmospheric(dir: vec3<f32>) -> vec3<f32> {
    let s = sun_dir();
    let day_t = sun_intensity(s);
    let up = clamp(dir.y, -0.2, 1.0);
    let zen_f = smoothstep(0.0, 0.55, up);
    let day_zenith  = vec3<f32>(0.22, 0.50, 0.95);
    let day_horizon = vec3<f32>(0.78, 0.86, 0.95);
    let day_sky = mix(day_horizon, day_zenith, zen_f);
    let night_zenith  = vec3<f32>(0.025, 0.035, 0.065);
    let night_horizon = vec3<f32>(0.055, 0.065, 0.110);
    let night_sky = mix(night_horizon, night_zenith, zen_f);
    return mix(night_sky, day_sky, day_t);
}

struct Hit {
    hit: bool,
    mat: u32,
    normal: vec3<f32>,
    voxel: vec3<i32>,
    last_axis: i32,
    t_hit: f32,
};

fn safe_inv(x: f32) -> f32 {
    if (abs(x) < 1e-8) { return 1e30; }
    return 1.0 / x;
}

// IGN (interleaved gradient noise) — high-quality low-discrepancy per-pixel
// hash. Used for shadow PCF jitter so adjacent pixels get well-distributed
// offsets without forming visible patterns.
fn ign(x: f32, y: f32, frame: f32) -> f32 {
    return fract(52.9829189 * fract(0.06711056 * (x + frame * 5.588238) + 0.00583715 * (y + frame * 4.182857)));
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = vec2<u32>(camera.resolution);
    if (gid.x >= res.x || gid.y >= res.y) { return; }

    // Temporal-differential gate.
    let tiles_w = (i32(camera.resolution.x) + 7) / 8;
    let tile_x = i32(gid.x) / 8;
    let tile_y = i32(gid.y) / 8;
    let tile_idx = tile_x + tile_y * tiles_w;
    let word = tile_idx >> 5;
    let bit = tile_idx & 31;
    if ((tile_dirty[word] & (1u << u32(bit))) == 0u) { return; }

    // Per-pixel + per-frame jitter, threaded through shading.
    let pix_jitter = ign(f32(gid.x), f32(gid.y), camera.time * 60.0);

    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) / camera.resolution;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let aspect = camera.resolution.x / camera.resolution.y;
    let dir = normalize(
        camera.forward
        + camera.right * (ndc.x * camera.tan_half_fov * aspect)
        + camera.up    * (ndc.y * camera.tan_half_fov)
    );

    // Beam pre-pass: read the coarse first-tile-hit t for this 8×8 block
    // and start the per-pixel ray there. A 16-voxel safety margin (one
    // tile) keeps grazing-angle pixels from over-skipping.
    let beam_xy = vec2<i32>(i32(gid.x / 8u), i32(gid.y / 8u));
    let beam_t_raw = textureLoad(beam_depth, beam_xy, 0).r;
    let beam_skip = max(0.0, beam_t_raw - 16.0);
    let ray_origin = camera.origin + dir * beam_skip;

    var hit = trace(ray_origin, dir);
    if (hit.hit) {
        hit.t_hit = hit.t_hit + beam_skip;
    }
    var col: vec3<f32>;
    if (hit.hit) {
        if (is_water_mat(hit.mat) && hit.normal.y > 0.5) {
            col = shade_water_top(hit, camera.origin, dir);
        } else if (hit.mat == MAT_GLASS) {
            col = shade_glass(hit, camera.origin, dir);
        } else {
            col = shade(hit, camera.origin, dir, pix_jitter);
        }
    } else {
        col = sky(dir);
    }

    // Remote-player markers: each player is a 1.6×2×1.6 box in world coords.
    // Pick the closest hit (player vs terrain) and override colour if a
    // player marker wins.
    let inv_dir_p = vec3<f32>(safe_inv(dir.x), safe_inv(dir.y), safe_inv(dir.z));
    var closest_t = select(1e30, hit.t_hit, hit.hit);
    for (var pi: u32 = 0u; pi < players.count; pi = pi + 1u) {
        let pp = players.positions[pi];
        let center = pp.xyz;
        let bmin = center - vec3<f32>(0.8, 0.5, 0.8);
        let bmax = center + vec3<f32>(0.8, 1.5, 0.8);
        let t = ray_aabb_t(camera.origin, inv_dir_p, bmin, bmax);
        if (t < closest_t) {
            closest_t = t;
            let pid = u32(pp.w);
            col = player_color_for(pid);
        }
    }

    // Volumetric god rays — accumulate sun visibility along the primary ray.
    let t_far = select(200.0, hit.t_hit, hit.hit);
    col += god_rays(camera.origin, dir, t_far, vec2<f32>(f32(gid.x), f32(gid.y)));
    textureStore(output_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(col, 1.0));
}

const MAT_WATER_L1: u32 = 5u;
const MAT_WATER_L8: u32 = 12u;
const MAT_LEAVES:   u32 = 14u;
const MAT_GLASS:    u32 = 18u;

fn is_water_mat(m: u32) -> bool {
    return m >= MAT_WATER_L1 && m <= MAT_WATER_L8;
}
fn is_transparent_mat(m: u32) -> bool {
    return is_water_mat(m) || m == MAT_GLASS;
}

// Sun rotates east→up→west→under. Start near midday so the very first frame
// isn't dim/orange; cycle slows to ~5 minutes for a less twitchy feel.
fn sun_dir() -> vec3<f32> {
    let a = camera.time * 0.025 + 1.20;
    return normalize(vec3<f32>(cos(a), sin(a), 0.30));
}

fn sun_intensity(s: vec3<f32>) -> f32 {
    // Smoothstep into night below the horizon.
    return smoothstep(-0.05, 0.10, s.y);
}

fn sun_color(s: vec3<f32>) -> vec3<f32> {
    let h = clamp(s.y, 0.0, 1.0);
    // Sunset/sunrise = warm orange. Midday = neutral. Lerp on solar elevation.
    let warm = vec3<f32>(1.40, 0.60, 0.25);
    let mid = vec3<f32>(1.10, 1.02, 0.92);
    return mix(warm, mid, smoothstep(0.05, 0.40, h)) * sun_intensity(s);
}

// IQ-style fract hash. The previous sin-based hash had visible periodic
// patterns at integer-aligned coords (that's where the "chess grid" came
// from). This one is uniform across all positions we sample.
fn hash3f(pin: vec3<f32>) -> f32 {
    var q = fract(pin * vec3<f32>(0.1031, 0.1030, 0.0973));
    q = q + dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}
fn vnoise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let n000 = hash3f(i + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = hash3f(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash3f(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash3f(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash3f(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash3f(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash3f(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash3f(i + vec3<f32>(1.0, 1.0, 1.0));
    let a = mix(n000, n100, u.x);
    let b = mix(n010, n110, u.x);
    let c = mix(n001, n101, u.x);
    let d = mix(n011, n111, u.x);
    return mix(mix(a, b, u.y), mix(c, d, u.y), u.z);
}

// Real cumulus-style cloud density: a sparse coverage mask (so most of the
// sky has zero clouds) gates a multi-octave body. Result: discrete fluffy
// clumps, not a uniform haze.
fn cloud_density(p: vec3<f32>, t: f32) -> f32 {
    let pa = p * 0.0060 + vec3<f32>(t * 0.07, 0.0, t * 0.04);
    // Coverage: low-frequency mask. Subtracting a constant gives wide gaps
    // between cloud regions.
    let coverage = vnoise3(pa * 0.35) - 0.45;
    if (coverage < 0.0) { return 0.0; }
    // Inside cloud regions, multi-octave detail builds fluffy texture.
    let n1 = vnoise3(pa);
    let n2 = vnoise3(pa * 2.7);
    let n3 = vnoise3(pa * 7.3);
    let body = n1 * 0.55 + n2 * 0.30 + n3 * 0.15;
    let d = (body - 0.40) * (coverage * 4.0);
    return clamp(d, 0.0, 1.0);
}

fn sky_color(dir: vec3<f32>) -> vec3<f32> {
    let s = sun_dir();
    let day_t = sun_intensity(s);

    // Atmosphere: smoothstep elevation, hazier near the horizon (Rayleigh-ish
    // tint). Down-facing rays inherit the horizon color (camera looking
    // *under* shouldn't see deep-blue zenith showing through ground gaps).
    let up = clamp(dir.y, -0.2, 1.0);
    let zen_f = smoothstep(0.0, 0.55, up);
    let day_zenith  = vec3<f32>(0.22, 0.50, 0.95);
    let day_horizon = vec3<f32>(0.78, 0.86, 0.95);
    let day_sky = mix(day_horizon, day_zenith, zen_f);
    let night_zenith  = vec3<f32>(0.005, 0.010, 0.035);
    let night_horizon = vec3<f32>(0.030, 0.040, 0.090);
    let night_sky = mix(night_horizon, night_zenith, zen_f);

    var col = mix(night_sky, day_sky, day_t);

    // Sunset / sunrise warmth: warm wash near horizon weighted by alignment
    // to the sun's azimuth and the sun's altitude (peak around horizon).
    let dusk_w = smoothstep(-0.08, 0.18, s.y) * (1.0 - smoothstep(0.05, 0.40, s.y));
    let azi = clamp(dot(normalize(vec2<f32>(dir.x, dir.z)), normalize(vec2<f32>(s.x, s.z))), 0.0, 1.0);
    let warm_horizon = vec3<f32>(1.50, 0.55, 0.18) * dusk_w * pow(azi, 3.0) * (1.0 - zen_f);
    col = col + warm_horizon;

    // General halo / forward-scatter near the sun.
    let sun_align = max(0.0, dot(dir, s));
    let halo = vec3<f32>(1.0, 0.78, 0.52) * pow(sun_align, 6.0) * day_t * 0.45;
    col = col + halo;

    // Sun disc.
    if (sun_align > 0.9985) {
        col = mix(col, vec3<f32>(2.4, 2.0, 1.5), 0.9);
    }

    // Stars when sun is below the horizon. Simple high-frequency noise
    // threshold; faint and scattered.
    if (day_t < 0.30) {
        let stn = vnoise3(dir * 95.0);
        if (stn > 0.93) {
            let amp = (stn - 0.93) * 14.0 * (1.0 - day_t);
            col = col + vec3<f32>(0.9, 0.92, 1.0) * amp;
        }
    }

    // Clouds disabled for now.
    return col;
}

fn ambient_color() -> vec3<f32> {
    let s = sun_dir();
    let day_t = sun_intensity(s);
    let day = vec3<f32>(0.30, 0.42, 0.58);
    let night = vec3<f32>(0.04, 0.05, 0.10);
    return mix(night, day, day_t);
}

fn water_normal(p: vec3<f32>, t: f32) -> vec3<f32> {
    // Three octaves of moving waves. Amplitudes and frequencies tuned so
    // the perturbed normal tilts ~5-15° from vertical — visibly rippling
    // but not so steep the reflection looks shattered.
    let a1 = 0.18; let f1 = 0.45; let s1 = 1.4;
    let a2 = 0.12; let f2 = 0.35; let s2 = 1.1;
    let a3 = 0.06; let f3 = 0.90; let s3 = 2.3;
    let dx = a1 * f1 *  cos(p.x * f1 + t * s1)
           + a2 * f2 *  cos((p.x + p.z) * f2 + t * s2)
           + a3 * f3 *  cos(p.x * f3 - p.z * 0.3 + t * s3);
    let dz = a1 * f1 * -sin(p.z * f1 + t * s1 * 0.9)
           + a2 * f2 *  cos((p.x + p.z) * f2 + t * s2)
           + a3 * f3 * -sin(p.z * f3 + p.x * 0.3 + t * s3);
    return normalize(vec3<f32>(-dx, 1.0, -dz));
}

// Variant of `trace` that treats every water voxel as empty. Used to find
// what lies *beneath* a water surface for refraction-style transparency.
fn trace_no_water(origin: vec3<f32>, dir: vec3<f32>) -> Hit {
    var out: Hit;
    out.hit = false;
    out.mat = 0u;
    out.normal = vec3<f32>(0.0);
    out.voxel = vec3<i32>(0);
    out.last_axis = -1;
    out.t_hit = 0.0;

    let win_min = vec3<f32>(camera.world_origin);
    let win_max = win_min + vec3<f32>(f32(WORLD_VOXELS_X), f32(WORLD_VOXELS_Y), f32(WORLD_VOXELS_Z));
    let inv_dir = vec3<f32>(safe_inv(dir.x), safe_inv(dir.y), safe_inv(dir.z));
    let t0 = (win_min - origin) * inv_dir;
    let t1 = (win_max - origin) * inv_dir;
    let tmin3 = min(t0, t1);
    let tmax3 = max(t0, t1);
    let t_enter = max(max(tmin3.x, tmin3.y), max(tmin3.z, 0.0));
    let t_exit = min(min(tmax3.x, tmax3.y), tmax3.z);
    if (t_enter >= t_exit || t_exit < 0.0) { return out; }

    let bias = 1e-3;
    var p = origin + dir * (t_enter + bias);
    p = clamp(p, win_min + vec3<f32>(0.001), win_max - vec3<f32>(0.001));
    let step = vec3<i32>(sign(dir));
    let t_delta = abs(inv_dir);

    var voxel = vec3<i32>(floor(p));
    var t_max: vec3<f32>;
    if (step.x > 0) { t_max.x = (f32(voxel.x + 1) - origin.x) * inv_dir.x; } else { t_max.x = (f32(voxel.x) - origin.x) * inv_dir.x; }
    if (step.y > 0) { t_max.y = (f32(voxel.y + 1) - origin.y) * inv_dir.y; } else { t_max.y = (f32(voxel.y) - origin.y) * inv_dir.y; }
    if (step.z > 0) { t_max.z = (f32(voxel.z + 1) - origin.z) * inv_dir.z; } else { t_max.z = (f32(voxel.z) - origin.z) * inv_dir.z; }

    var last_axis: i32 = -1;
    for (var s: i32 = 0; s < 1024; s = s + 1) {
        let rel = voxel - camera.world_origin;
        if (rel.x < 0 || rel.x >= WORLD_VOXELS_X
         || rel.y < 0 || rel.y >= WORLD_VOXELS_Y
         || rel.z < 0 || rel.z >= WORLD_VOXELS_Z) { return out; }

        let slot_v = world_to_slot_voxel(voxel);
        let bp = slot_v >> vec3<u32>(2u);
        let tp = slot_v >> vec3<u32>(4u);
        let cp = slot_v >> vec3<u32>(6u);
        let ci = world_chunk_idx(cp.x, cp.y, cp.z);
        let tile_lin = (tp.x & 3) + (tp.z & 3) * 4 + (tp.y & 3) * 16;
        if (!chunk_has_child(ci, tile_lin)) {
            skip_to_cell(16, &voxel, &t_max, origin, dir, inv_dir, step, &last_axis);
            continue;
        }
        let ti = world_tile_idx(tp.x, tp.y, tp.z);
        let brick_lin = (bp.x & 3) + (bp.z & 3) * 4 + (bp.y & 3) * 16;
        if (!tile_has_child(ti, brick_lin)) {
            skip_to_cell(4, &voxel, &t_max, origin, dir, inv_dir, step, &last_axis);
            continue;
        }
        let bi = world_brick_idx(bp.x, bp.y, bp.z);
        let local = slot_v - bp * BRICK_DIM;
        let vi = brick_voxel_idx(local.x, local.y, local.z);
        if (brick_voxel_solid(bi, vi)) {
            let m = brick_voxel_material(bi, vi);
            if (!is_transparent_mat(m)) {
                var n = vec3<f32>(0.0);
                var t_hit: f32;
                if (last_axis == 0) { n.x = -f32(step.x); t_hit = t_max.x - t_delta.x; }
                else if (last_axis == 1) { n.y = -f32(step.y); t_hit = t_max.y - t_delta.y; }
                else if (last_axis == 2) { n.z = -f32(step.z); t_hit = t_max.z - t_delta.z; }
                else { t_hit = t_enter; if (tmin3.x >= tmin3.y && tmin3.x >= tmin3.z) { n.x = -f32(step.x); } else if (tmin3.y >= tmin3.z) { n.y = -f32(step.y); } else { n.z = -f32(step.z); } }
                out.hit = true;
                out.mat = m;
                out.normal = n;
                out.voxel = voxel;
                out.last_axis = last_axis_after_entry(last_axis, tmin3);
                out.t_hit = t_hit;
                return out;
            }
            // Water cell — fall through to the regular step so we keep going.
        }

        if (t_max.x < t_max.y && t_max.x < t_max.z) {
            voxel.x = voxel.x + step.x;
            t_max.x = t_max.x + t_delta.x;
            last_axis = 0;
        } else if (t_max.y < t_max.z) {
            voxel.y = voxel.y + step.y;
            t_max.y = t_max.y + t_delta.y;
            last_axis = 1;
        } else {
            voxel.z = voxel.z + step.z;
            t_max.z = t_max.z + t_delta.z;
            last_axis = 2;
        }
    }
    return out;
}

fn shade_water_top(hit: Hit, origin: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let p_hit = origin + dir * hit.t_hit;
    let n = water_normal(p_hit, camera.time);
    let s = sun_dir();
    let sc = sun_color(s);
    // Per-pixel jitter substitute for shading-of-reflections — derive from
    // hit position since we're not in cs_main scope.
    let jit = fract(p_hit.x * 17.0 + p_hit.z * 23.0 + camera.time * 13.0);

    // ---- reflection: primary ray reflected off the perturbed normal ----
    let refl_dir = reflect(dir, n);
    let refl_origin = p_hit + n * 0.01;
    let refl_hit = trace(refl_origin, refl_dir);
    var refl_col: vec3<f32>;
    if (refl_hit.hit) {
        refl_col = shade(refl_hit, refl_origin, refl_dir, jit);
    } else {
        refl_col = sky(refl_dir);
    }

    // ---- refraction: primary ray bent into the water, trace through it ----
    // Snell's law via WGSL `refract`. eta = n_air / n_water ≈ 1/1.33.
    let eta = 1.0 / 1.33;
    var refr_dir = refract(dir, n, eta);
    // Total internal reflection would return zero; fall back to dir.
    if (length(refr_dir) < 0.01) { refr_dir = dir; }
    let refr_origin = p_hit + dir * 0.001; // step inside the water column
    let under = trace_no_water(refr_origin, refr_dir);
    var under_col: vec3<f32>;
    if (under.hit) {
        under_col = shade(under, refr_origin, refr_dir, jit);
    } else {
        under_col = sky(refr_dir) * 0.6;
    }
    // Beer-Lambert absorption — red and green are eaten faster than blue.
    let depth = max(0.0, under.t_hit);
    let absorb = vec3<f32>(0.55, 0.25, 0.10); // per-unit-distance attenuation
    let transmittance = exp(-absorb * depth);
    let water_tint = vec3<f32>(0.10, 0.32, 0.42);
    let refr_col = under_col * transmittance + water_tint * (1.0 - transmittance.x);

    // ---- Fresnel mix of reflection and refraction ----
    let cos_theta = clamp(dot(-dir, n), 0.0, 1.0);
    let f0 = 0.02;
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);

    // ---- specular sun glint on the surface ----
    let h = normalize(s - dir);
    let spec = pow(max(0.0, dot(n, h)), 128.0);
    var shadow = 0.0;
    if (sun_intensity(s) > 0.0 && dot(n, s) > 0.0) {
        shadow = select(1.0, 0.0, trace_any(refl_origin, s));
    }

    var col = mix(refr_col, refl_col, fresnel) + sc * spec * shadow * 1.2;
    let fog_t = clamp(hit.t_hit / 280.0, 0.0, 0.85);
    return mix(col, fog_atmospheric(dir), fog_t);
}

fn shade(hit: Hit, origin: vec3<f32>, dir: vec3<f32>, pix_jit: f32) -> vec3<f32> {
    var base = palette[hit.mat].rgb;
    let ao = compute_ao(hit, origin, dir);
    let p_hit = origin + dir * hit.t_hit;

    // ---- swaying foliage ----
    // Leaves and grass-tops perturb their shading normal with a wind field
    // so they look animated even though the underlying voxel is rigid.
    var n = hit.normal;
    if (hit.mat == MAT_LEAVES) {
        let t = camera.time;
        let sway = sin(p_hit.x * 0.40 + t * 1.8) * cos(p_hit.z * 0.40 + t * 1.2)
                 + 0.4 * sin((p_hit.x + p_hit.z) * 0.25 + t * 2.4);
        n.x += sway * 0.30;
        n.z += sway * 0.20;
        n = normalize(n);
        base *= 1.0 + sway * 0.10;
    } else if (hit.mat == 2u /* MAT_GRASS */ && hit.normal.y > 0.5) {
        let t = camera.time;
        let sway = sin(p_hit.x * 0.55 + t * 2.1) * cos(p_hit.z * 0.55 + t * 1.7);
        n.x += sway * 0.20;
        n.z += sway * 0.18;
        n = normalize(n);
        base *= 1.0 + sway * 0.07;
    }

    let s = sun_dir();
    let s_int = sun_intensity(s);
    let p_off = p_hit + n * 0.001;
    let n_dot_l = max(0.0, dot(n, s));
    var shadow_term = 0.0;
    if (n_dot_l > 0.0 && s_int > 0.0) {
        // PCF using per-PIXEL jitter (not position-based — that's what made
        // the chess pattern). 4 golden-spiral samples.
        let tau = 6.28318530;
        let golden = 2.39996323; // 137.5° in radians
        let cone = 0.07;
        var sum = 0.0;
        for (var i: i32 = 0; i < 4; i = i + 1) {
            let theta = (f32(i) + pix_jit) * golden;
            let radius = cone * sqrt((f32(i) + pix_jit) * 0.25);
            // Build a tangent frame around the sun direction (s) so the offset
            // is in the *plane perpendicular to s* — not just in xz. That way
            // shadows are uniformly soft regardless of sun azimuth.
            var tangent = normalize(cross(s, vec3<f32>(0.0, 1.0, 0.0)));
            if (length(cross(s, vec3<f32>(0.0, 1.0, 0.0))) < 0.01) {
                tangent = vec3<f32>(1.0, 0.0, 0.0);
            }
            let bitangent = cross(s, tangent);
            let off = (tangent * cos(theta) + bitangent * sin(theta)) * radius;
            let ss = normalize(s + off);
            if (!trace_any(p_off, ss)) { sum = sum + 1.0; }
        }
        shadow_term = sum * 0.25;
    }

    let direct = sun_color(s) * (n_dot_l * shadow_term);
    let ambient = ambient_color() * ao;
    let lit = base * (direct + ambient);

    let fog_t = clamp(hit.t_hit / 280.0, 0.0, 0.85);
    return mix(lit, fog_atmospheric(dir), fog_t);
}

// Glass — high-fresnel reflection + low-tint refraction. Reuses the same
// trace_no_water that skips water (now also skips glass) to look through.
fn shade_glass(hit: Hit, origin: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let p_hit = origin + dir * hit.t_hit;
    let n = hit.normal;
    let s = sun_dir();
    let sc = sun_color(s);
    let jit = fract(p_hit.x * 17.0 + p_hit.z * 23.0 + camera.time * 13.0);

    let refl_dir = reflect(dir, n);
    let refl_origin = p_hit + n * 0.01;
    let refl_hit = trace(refl_origin, refl_dir);
    var refl_col: vec3<f32>;
    if (refl_hit.hit) {
        refl_col = shade(refl_hit, refl_origin, refl_dir, jit);
    } else {
        refl_col = sky(refl_dir);
    }

    let eta = 1.0 / 1.50;
    var refr_dir = refract(dir, n, eta);
    if (length(refr_dir) < 0.01) { refr_dir = dir; }
    let refr_origin = p_hit + dir * 0.001;
    let under = trace_no_water(refr_origin, refr_dir);
    var under_col: vec3<f32>;
    if (under.hit) {
        under_col = shade(under, refr_origin, refr_dir, jit);
    } else {
        under_col = sky(refr_dir);
    }
    // Very faint cyan tint per unit travelled — glass is mostly clear.
    let depth = max(0.0, under.t_hit);
    let tint = vec3<f32>(0.04, 0.02, 0.02) * depth;
    let glass_col = under_col * exp(-tint);

    let cos_theta = clamp(dot(-dir, n), 0.0, 1.0);
    let f0 = 0.04;
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);

    let fog_t = clamp(hit.t_hit / 280.0, 0.0, 0.85);
    let combined = mix(glass_col, refl_col, fresnel);
    return mix(combined, fog_atmospheric(dir), fog_t);
}

// Volumetric god rays: accumulate sun visibility along the primary ray.
// Stochastic sampling — combined with the temporal-differential gate that
// reuses prior-frame pixels, this denoises well in steady-state views.
fn god_rays(origin: vec3<f32>, dir: vec3<f32>, t_far: f32, pix: vec2<f32>) -> vec3<f32> {
    let s = sun_dir();
    let s_int = sun_intensity(s);
    if (s_int <= 0.0) { return vec3<f32>(0.0); }
    // EARLY-OUT: god rays are a Mie-style forward-scatter effect, so the
    // contribution drops sharply as the camera looks away from the sun.
    // Compute the phase weight first and skip the entire shadow-ray loop
    // when it's negligible — this saves the cost on ~75% of the screen.
    let cos_sun = max(0.0, dot(dir, s));
    let phase = 0.4 + 0.6 * pow(cos_sun, 6.0);
    if (phase < 0.45) { return vec3<f32>(0.0); }

    let t_max = min(t_far, 100.0);
    if (t_max <= 1.0) { return vec3<f32>(0.0); }
    let N: i32 = 4;
    let step_t = t_max / f32(N);
    let h = fract(sin(pix.x * 12.9898 + pix.y * 78.233 + camera.time * 0.13) * 43758.5453);
    var sum = 0.0;
    for (var i: i32 = 0; i < N; i = i + 1) {
        let t = (f32(i) + h) * step_t;
        let p = origin + dir * t;
        if (!trace_any(p + s * 0.5, s)) {
            sum = sum + 1.0;
        }
    }
    let frac = sum / f32(N);
    return sun_color(s) * frac * phase * 0.18;
}

// Stripped-down DDA — same hierarchy as `trace()` but returns the moment we
// know the ray is occluded. No normal / material work.
fn trace_any(origin: vec3<f32>, dir: vec3<f32>) -> bool {
    let win_min = vec3<f32>(camera.world_origin);
    let win_max = win_min + vec3<f32>(f32(WORLD_VOXELS_X), f32(WORLD_VOXELS_Y), f32(WORLD_VOXELS_Z));
    let inv_dir = vec3<f32>(safe_inv(dir.x), safe_inv(dir.y), safe_inv(dir.z));
    let t0 = (win_min - origin) * inv_dir;
    let t1 = (win_max - origin) * inv_dir;
    let tmin3 = min(t0, t1);
    let tmax3 = max(t0, t1);
    let t_enter = max(max(tmin3.x, tmin3.y), max(tmin3.z, 0.0));
    let t_exit = min(min(tmax3.x, tmax3.y), tmax3.z);
    if (t_enter >= t_exit || t_exit < 0.0) { return false; }

    let bias = 1e-3;
    var p = origin + dir * (t_enter + bias);
    p = clamp(p, win_min + vec3<f32>(0.001), win_max - vec3<f32>(0.001));
    let step = vec3<i32>(sign(dir));
    let t_delta = abs(inv_dir);

    var voxel = vec3<i32>(floor(p));
    var t_max: vec3<f32>;
    if (step.x > 0) { t_max.x = (f32(voxel.x + 1) - origin.x) * inv_dir.x; } else { t_max.x = (f32(voxel.x) - origin.x) * inv_dir.x; }
    if (step.y > 0) { t_max.y = (f32(voxel.y + 1) - origin.y) * inv_dir.y; } else { t_max.y = (f32(voxel.y) - origin.y) * inv_dir.y; }
    if (step.z > 0) { t_max.z = (f32(voxel.z + 1) - origin.z) * inv_dir.z; } else { t_max.z = (f32(voxel.z) - origin.z) * inv_dir.z; }

    var last_axis: i32 = -1;
    for (var s: i32 = 0; s < 768; s = s + 1) {
        let rel = voxel - camera.world_origin;
        if (rel.x < 0 || rel.x >= WORLD_VOXELS_X
         || rel.y < 0 || rel.y >= WORLD_VOXELS_Y
         || rel.z < 0 || rel.z >= WORLD_VOXELS_Z) { return false; }

        let slot_v = world_to_slot_voxel(voxel);
        let bp = slot_v >> vec3<u32>(2u);
        let tp = slot_v >> vec3<u32>(4u);
        let cp = slot_v >> vec3<u32>(6u);
        let ci = world_chunk_idx(cp.x, cp.y, cp.z);
        let tile_lin = (tp.x & 3) + (tp.z & 3) * 4 + (tp.y & 3) * 16;
        if (!chunk_has_child(ci, tile_lin)) {
            skip_to_cell(16, &voxel, &t_max, origin, dir, inv_dir, step, &last_axis);
            continue;
        }
        let ti = world_tile_idx(tp.x, tp.y, tp.z);
        let brick_lin = (bp.x & 3) + (bp.z & 3) * 4 + (bp.y & 3) * 16;
        if (!tile_has_child(ti, brick_lin)) {
            skip_to_cell(4, &voxel, &t_max, origin, dir, inv_dir, step, &last_axis);
            continue;
        }
        let bi = world_brick_idx(bp.x, bp.y, bp.z);
        let local = slot_v - bp * BRICK_DIM;
        let vi = brick_voxel_idx(local.x, local.y, local.z);
        if (brick_voxel_solid(bi, vi)) { return true; }

        if (t_max.x < t_max.y && t_max.x < t_max.z) {
            voxel.x = voxel.x + step.x;
            t_max.x = t_max.x + t_delta.x;
            last_axis = 0;
        } else if (t_max.y < t_max.z) {
            voxel.y = voxel.y + step.y;
            t_max.y = t_max.y + t_delta.y;
            last_axis = 1;
        } else {
            voxel.z = voxel.z + step.z;
            t_max.z = t_max.z + t_delta.z;
            last_axis = 2;
        }
    }
    return false;
}

// Bit-packed AO. We project the hit point onto the face we entered through,
// compute fractional (fa, fb) coords on that face, sample 4 corner AOs, and
// bilinear-interpolate. Each corner samples 3 neighbours (two side voxels
// and the diagonal) — classic "Minecraft" AO formula, but every lookup is a
// hierarchical bit test rather than a struct fetch.
fn compute_ao(hit: Hit, origin: vec3<f32>, dir: vec3<f32>) -> f32 {
    let p_hit = origin + dir * hit.t_hit;
    let v = hit.voxel;
    let na = hit.last_axis;
    if (na < 0) { return 1.0; }
    // n_dir is +1 or -1 — the outward-facing component of the normal axis.
    let n_dir = i32(hit.normal[na]);
    var n_off = vec3<i32>(0);
    if (na == 0) { n_off.x = n_dir; }
    else if (na == 1) { n_off.y = n_dir; }
    else { n_off.z = n_dir; }

    var da_pos: vec3<i32>;
    var db_pos: vec3<i32>;
    var fa: f32;
    var fb: f32;
    let local_frac = p_hit - vec3<f32>(f32(v.x), f32(v.y), f32(v.z));
    if (na == 0) {
        da_pos = vec3<i32>(0, 1, 0); db_pos = vec3<i32>(0, 0, 1);
        fa = local_frac.y; fb = local_frac.z;
    } else if (na == 1) {
        da_pos = vec3<i32>(1, 0, 0); db_pos = vec3<i32>(0, 0, 1);
        fa = local_frac.x; fb = local_frac.z;
    } else {
        da_pos = vec3<i32>(1, 0, 0); db_pos = vec3<i32>(0, 1, 0);
        fa = local_frac.x; fb = local_frac.y;
    }
    let da_neg = -da_pos;
    let db_neg = -db_pos;
    let base = v + n_off;

    let ao00 = ao_corner(base, da_neg, db_neg);
    let ao10 = ao_corner(base, da_pos, db_neg);
    let ao01 = ao_corner(base, da_neg, db_pos);
    let ao11 = ao_corner(base, da_pos, db_pos);

    let fa_c = clamp(fa, 0.0, 1.0);
    let fb_c = clamp(fb, 0.0, 1.0);
    let ao_x0 = mix(ao00, ao10, fa_c);
    let ao_x1 = mix(ao01, ao11, fa_c);
    return mix(ao_x0, ao_x1, fb_c);
}

fn ao_corner(face_base: vec3<i32>, da: vec3<i32>, db: vec3<i32>) -> f32 {
    let s1 = is_voxel_solid(face_base + da);
    let s2 = is_voxel_solid(face_base + db);
    let cd = is_voxel_solid(face_base + da + db);
    // Full occlusion if both side voxels are solid (corner case).
    if (s1 && s2) { return 0.35; }
    let cnt = i32(s1) + i32(s2) + i32(cd);
    return 1.0 - f32(cnt) * 0.22;
}

fn trace(origin: vec3<f32>, dir: vec3<f32>) -> Hit {
    var out: Hit;
    out.hit = false;
    out.mat = 0u;
    out.normal = vec3<f32>(0.0);
    out.voxel = vec3<i32>(0);
    out.last_axis = -1;
    out.t_hit = 0.0;

    // Loaded window in world voxel coords.
    let win_min = vec3<f32>(camera.world_origin);
    let win_max = win_min + vec3<f32>(f32(WORLD_VOXELS_X), f32(WORLD_VOXELS_Y), f32(WORLD_VOXELS_Z));
    let inv_dir = vec3<f32>(safe_inv(dir.x), safe_inv(dir.y), safe_inv(dir.z));

    let t0 = (win_min - origin) * inv_dir;
    let t1 = (win_max - origin) * inv_dir;
    let tmin3 = min(t0, t1);
    let tmax3 = max(t0, t1);
    let t_enter = max(max(tmin3.x, tmin3.y), max(tmin3.z, 0.0));
    let t_exit = min(min(tmax3.x, tmax3.y), tmax3.z);
    if (t_enter >= t_exit || t_exit < 0.0) { return out; }

    // Larger bias — at large world coords f32 precision around the voxel
    // boundary made the ray skip cells and show sky/sun through them.
    let bias = 1e-2;
    var p = origin + dir * (t_enter + bias);
    p = clamp(p, win_min + vec3<f32>(0.01), win_max - vec3<f32>(0.01));

    let step = vec3<i32>(sign(dir));
    let t_delta = abs(inv_dir);

    // DDA in WORLD voxel coords. The slot lookup folds via mod.
    var voxel = vec3<i32>(floor(p));
    var t_max: vec3<f32>;
    if (step.x > 0) { t_max.x = (f32(voxel.x + 1) - origin.x) * inv_dir.x; } else { t_max.x = (f32(voxel.x) - origin.x) * inv_dir.x; }
    if (step.y > 0) { t_max.y = (f32(voxel.y + 1) - origin.y) * inv_dir.y; } else { t_max.y = (f32(voxel.y) - origin.y) * inv_dir.y; }
    if (step.z > 0) { t_max.z = (f32(voxel.z + 1) - origin.z) * inv_dir.z; } else { t_max.z = (f32(voxel.z) - origin.z) * inv_dir.z; }

    var last_axis: i32 = -1;
    let max_steps: i32 = 1024;
    for (var s: i32 = 0; s < max_steps; s = s + 1) {
        // Bounds check on the LOADED WINDOW (world coords).
        let rel = voxel - camera.world_origin;
        if (rel.x < 0 || rel.x >= WORLD_VOXELS_X
         || rel.y < 0 || rel.y >= WORLD_VOXELS_Y
         || rel.z < 0 || rel.z >= WORLD_VOXELS_Z) {
            return out;
        }

        let slot_v = world_to_slot_voxel(voxel);
        let bp = slot_v >> vec3<u32>(2u);
        let tp = slot_v >> vec3<u32>(4u);
        let cp = slot_v >> vec3<u32>(6u);

        let ci = world_chunk_idx(cp.x, cp.y, cp.z);
        let tile_in_chunk_lin = (tp.x & 3) + (tp.z & 3) * 4 + (tp.y & 3) * 16;
        if (!chunk_has_child(ci, tile_in_chunk_lin)) {
            skip_to_cell(16, &voxel, &t_max, origin, dir, inv_dir, step, &last_axis);
            continue;
        }

        let ti = world_tile_idx(tp.x, tp.y, tp.z);
        let brick_in_tile_lin = (bp.x & 3) + (bp.z & 3) * 4 + (bp.y & 3) * 16;
        if (!tile_has_child(ti, brick_in_tile_lin)) {
            skip_to_cell(4, &voxel, &t_max, origin, dir, inv_dir, step, &last_axis);
            continue;
        }

        let bi = world_brick_idx(bp.x, bp.y, bp.z);
        let local = slot_v - bp * BRICK_DIM;
        let vi = brick_voxel_idx(local.x, local.y, local.z);
        if (brick_voxel_solid(bi, vi)) {
            let m = brick_voxel_material(bi, vi);
            var n = vec3<f32>(0.0);
            var t_hit: f32;
            if (last_axis == 0) {
                n.x = -f32(step.x);
                t_hit = t_max.x - t_delta.x;
            } else if (last_axis == 1) {
                n.y = -f32(step.y);
                t_hit = t_max.y - t_delta.y;
            } else if (last_axis == 2) {
                n.z = -f32(step.z);
                t_hit = t_max.z - t_delta.z;
            } else {
                // First voxel — we entered through the world AABB face.
                t_hit = t_enter;
                if      (tmin3.x >= tmin3.y && tmin3.x >= tmin3.z) { n.x = -f32(step.x); }
                else if (tmin3.y >= tmin3.z)                      { n.y = -f32(step.y); }
                else                                              { n.z = -f32(step.z); }
            }
            out.hit = true;
            out.mat = m;
            out.normal = n;
            out.voxel = voxel;
            out.last_axis = last_axis_after_entry(last_axis, tmin3);
            out.t_hit = t_hit;
            return out;
        }

        if (t_max.x < t_max.y && t_max.x < t_max.z) {
            voxel.x = voxel.x + step.x;
            t_max.x = t_max.x + t_delta.x;
            last_axis = 0;
        } else if (t_max.y < t_max.z) {
            voxel.y = voxel.y + step.y;
            t_max.y = t_max.y + t_delta.y;
            last_axis = 1;
        } else {
            voxel.z = voxel.z + step.z;
            t_max.z = t_max.z + t_delta.z;
            last_axis = 2;
        }
    }
    return out;
}

fn last_axis_after_entry(la: i32, tmin3: vec3<f32>) -> i32 {
    if (la >= 0) { return la; }
    if (tmin3.x >= tmin3.y && tmin3.x >= tmin3.z) { return 0; }
    if (tmin3.y >= tmin3.z) { return 1; }
    return 2;
}

fn skip_to_cell(
    cell_size: i32,
    voxel: ptr<function, vec3<i32>>,
    t_max: ptr<function, vec3<f32>>,
    origin: vec3<f32>,
    dir: vec3<f32>,
    inv_dir: vec3<f32>,
    step: vec3<i32>,
    last_axis: ptr<function, i32>,
) {
    // voxel can be negative (negative world-origin window), so use a
    // Euclidean modulo to find the in-cell offset.
    let cell_origin = vec3<i32>(
        (*voxel).x - pos_mod((*voxel).x, cell_size),
        (*voxel).y - pos_mod((*voxel).y, cell_size),
        (*voxel).z - pos_mod((*voxel).z, cell_size),
    );
    var bnd: vec3<f32>;
    bnd.x = select(f32(cell_origin.x), f32(cell_origin.x + cell_size), step.x > 0);
    bnd.y = select(f32(cell_origin.y), f32(cell_origin.y + cell_size), step.y > 0);
    bnd.z = select(f32(cell_origin.z), f32(cell_origin.z + cell_size), step.z > 0);
    let t_face = (bnd - origin) * inv_dir;
    let eps = 1e-6;
    var t_min: f32 = 1e30;
    var ax: i32 = 0;
    if (step.x != 0 && t_face.x > eps && t_face.x < t_min) { t_min = t_face.x; ax = 0; }
    if (step.y != 0 && t_face.y > eps && t_face.y < t_min) { t_min = t_face.y; ax = 1; }
    if (step.z != 0 && t_face.z > eps && t_face.z < t_min) { t_min = t_face.z; ax = 2; }
    let bias = 1e-3;
    let p_new = origin + dir * (t_min + bias);
    (*voxel) = vec3<i32>(floor(p_new));
    if (step.x > 0) { (*t_max).x = (f32((*voxel).x + 1) - origin.x) * inv_dir.x; } else { (*t_max).x = (f32((*voxel).x) - origin.x) * inv_dir.x; }
    if (step.y > 0) { (*t_max).y = (f32((*voxel).y + 1) - origin.y) * inv_dir.y; } else { (*t_max).y = (f32((*voxel).y) - origin.y) * inv_dir.y; }
    if (step.z > 0) { (*t_max).z = (f32((*voxel).z + 1) - origin.z) * inv_dir.z; } else { (*t_max).z = (f32((*voxel).z) - origin.z) * inv_dir.z; }
    *last_axis = ax;
}
