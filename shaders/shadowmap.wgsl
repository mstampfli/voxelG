// Shadow-map pre-pass.
//
// For each of N×N pixels on the sun's orthographic projection plane, fire a
// ray from the BACK plane (far side of the world from the sun) along the
// -sun direction. Write the distance from the back plane to the first
// occluder into an R32Float depth texture. The raymarch shader replaces every
// `trace_any(p_world, sun_dir)` call with a single textureLoad on this map.
//
// Storage layout for `sun`:
//   sun_dir         — unit vector pointing TOWARD the sun
//   sun_basis_x/y   — orthonormal basis perpendicular to sun_dir (u,v on map)
//   sun_center      — world-space centre of the loaded AABB
//   sun_half_size   — half-extent of the square ortho frustum in u and v
//   sun_far         — depth range along -sun_dir (back plane → front plane)
//
// Coverage: the whole loaded window. World is 1024³ at 2× scale; with a
// 2048² map that's 0.75 vox per texel — close to per-voxel granularity.

struct Sun {
    sun_dir: vec3<f32>,
    sun_intensity: f32,
    sun_basis_x: vec3<f32>,
    sun_far: f32,
    sun_basis_y: vec3<f32>,
    sun_half_size: f32,
    sun_center: vec3<f32>,
    _pad0: f32,
    sun_color: vec3<f32>,
    _pad1: f32,
    ambient_color: vec3<f32>,
    _pad2: f32,
};

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

struct Brick {
    occ_lo: u32,
    occ_hi: u32,
    materials: array<u32, 16>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> bricks: array<Brick>;
@group(0) @binding(2) var<storage, read> tile_mask: array<u32>;
@group(0) @binding(3) var<storage, read> chunk_mask: array<u32>;
@group(0) @binding(4) var<uniform> sun: Sun;
@group(0) @binding(5) var shadow_out: texture_storage_2d<r32float, write>;
@group(0) @binding(6) var<storage, read> brick_uniform_packed: array<u32>;
@group(0) @binding(7) var<storage, read> tile_uniform_packed: array<u32>;

const WORLD_BRICKS_X: i32 = 192;
const WORLD_BRICKS_Y: i32 = 96;
const WORLD_BRICKS_Z: i32 = 192;
const WORLD_VOXELS_X: i32 = 768;
const WORLD_VOXELS_Y: i32 = 384;
const WORLD_VOXELS_Z: i32 = 768;
const WORLD_TILES_X: i32 = 48;
const WORLD_TILES_Y: i32 = 24;
const WORLD_TILES_Z: i32 = 48;
const WORLD_CHUNKS_X: i32 = 12;
const WORLD_CHUNKS_Y: i32 = 6;
const WORLD_CHUNKS_Z: i32 = 12;

const SHADOW_RES: u32 = 512u;

// Materials that act as opaque occluders for sun shadows. Foliage IS treated
// as opaque here (the dappling is then approximated in the lookup pass via a
// PCF kernel — cheaper than per-leaf sub-voxel rays).
const MAT_WATER_L1: u32 = 5u;
const MAT_WATER_L8: u32 = 12u;
const MAT_GLASS:    u32 = 18u;

fn pos_mod(a: i32, b: i32) -> i32 {
    let r = a % b;
    return select(r, r + b, r < 0);
}

fn world_to_slot_voxel(wv: vec3<i32>) -> vec3<u32> {
    return vec3<u32>(
        u32(pos_mod(wv.x, WORLD_VOXELS_X)),
        u32(wv.y),
        u32(pos_mod(wv.z, WORLD_VOXELS_Z)),
    );
}

fn world_brick_idx(bx: u32, by: u32, bz: u32) -> i32 {
    return i32(bx) + i32(by) * WORLD_BRICKS_X + i32(bz) * WORLD_BRICKS_X * WORLD_BRICKS_Y;
}
fn world_tile_idx(tx: u32, ty: u32, tz: u32) -> i32 {
    return i32(tx) + i32(ty) * WORLD_TILES_X + i32(tz) * WORLD_TILES_X * WORLD_TILES_Y;
}
fn world_chunk_idx(cx: u32, cy: u32, cz: u32) -> i32 {
    return i32(cx) + i32(cy) * WORLD_CHUNKS_X + i32(cz) * WORLD_CHUNKS_X * WORLD_CHUNKS_Y;
}
fn brick_voxel_idx(x: u32, y: u32, z: u32) -> i32 {
    return i32(x) + i32(z) * 4 + i32(y) * 16;
}
fn chunk_has_child(ci: i32, child_lin: i32) -> bool {
    let base = ci * 2;
    if (child_lin < 32) {
        return (chunk_mask[base] & (1u << u32(child_lin))) != 0u;
    }
    return (chunk_mask[base + 1] & (1u << u32(child_lin - 32))) != 0u;
}
fn tile_has_child(ti: i32, child_lin: i32) -> bool {
    let base = ti * 2;
    if (child_lin < 32) {
        return (tile_mask[base] & (1u << u32(child_lin))) != 0u;
    }
    return (tile_mask[base + 1] & (1u << u32(child_lin - 32))) != 0u;
}
fn brick_uniform_mat(bi: i32) -> u32 {
    let w = brick_uniform_packed[bi >> 2];
    let shift = u32(bi & 3) * 8u;
    return (w >> shift) & 0xFFu;
}
fn tile_uniform_mat(ti: i32) -> u32 {
    let w = tile_uniform_packed[ti >> 2];
    let shift = u32(ti & 3) * 8u;
    return (w >> shift) & 0xFFu;
}
fn is_water_mat(m: u32) -> bool {
    return m >= MAT_WATER_L1 && m <= MAT_WATER_L8;
}
fn is_transparent_mat(m: u32) -> bool {
    return is_water_mat(m) || m == MAT_GLASS;
}
fn brick_voxel_solid(bi: i32, vi: i32) -> bool {
    let b = bricks[bi];
    if (vi < 32) { return (b.occ_lo & (1u << u32(vi))) != 0u; }
    return (b.occ_hi & (1u << u32(vi - 32))) != 0u;
}

fn safe_inv(x: f32) -> f32 {
    if (abs(x) < 1e-8) { return 1e30; }
    return 1.0 / x;
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
    let bias = 1e-2;
    let p_new = origin + dir * (t_min + bias);
    var nv = vec3<i32>(floor(p_new));
    if (ax == 0) {
        if (step.x > 0) { nv.x = cell_origin.x + cell_size; }
        else            { nv.x = cell_origin.x - 1; }
    } else if (ax == 1) {
        if (step.y > 0) { nv.y = cell_origin.y + cell_size; }
        else            { nv.y = cell_origin.y - 1; }
    } else {
        if (step.z > 0) { nv.z = cell_origin.z + cell_size; }
        else            { nv.z = cell_origin.z - 1; }
    }
    *voxel = nv;
    let new_t_face = (vec3<f32>(nv) + vec3<f32>(select(0.0, 1.0, step.x > 0), select(0.0, 1.0, step.y > 0), select(0.0, 1.0, step.z > 0))) - origin;
    *t_max = new_t_face * inv_dir;
    *last_axis = ax;
}

// Trace a sun ray into the world. Returns t (distance from origin along
// dir) of the first opaque hit, or SUN_FAR if no hit. Identical hierarchy
// to raymarch.wgsl::trace_any but writes a depth instead of returning bool.
fn trace_shadow(origin: vec3<f32>, dir: vec3<f32>, max_t: f32) -> f32 {
    let inv_dir = vec3<f32>(safe_inv(dir.x), safe_inv(dir.y), safe_inv(dir.z));
    let win_min = vec3<f32>(camera.world_origin);
    let win_max = win_min + vec3<f32>(f32(WORLD_VOXELS_X), f32(WORLD_VOXELS_Y), f32(WORLD_VOXELS_Z));
    let t0 = (win_min - origin) * inv_dir;
    let t1 = (win_max - origin) * inv_dir;
    let tmin3 = min(t0, t1);
    let tmax3 = max(t0, t1);
    let t_enter = max(max(tmin3.x, tmin3.y), max(tmin3.z, 0.0));
    let t_exit = min(min(min(tmax3.x, tmax3.y), tmax3.z), max_t);
    if (t_enter >= t_exit) { return max_t; }

    let bias = 1e-2;
    var p = origin + dir * (t_enter + bias);
    p = clamp(p, win_min + vec3<f32>(0.01), win_max - vec3<f32>(0.01));
    let step = vec3<i32>(sign(dir));
    let t_delta = abs(inv_dir);

    var voxel = vec3<i32>(floor(p));
    var t_max: vec3<f32>;
    if (step.x > 0) { t_max.x = (f32(voxel.x + 1) - origin.x) * inv_dir.x; } else { t_max.x = (f32(voxel.x) - origin.x) * inv_dir.x; }
    if (step.y > 0) { t_max.y = (f32(voxel.y + 1) - origin.y) * inv_dir.y; } else { t_max.y = (f32(voxel.y) - origin.y) * inv_dir.y; }
    if (step.z > 0) { t_max.z = (f32(voxel.z + 1) - origin.z) * inv_dir.z; } else { t_max.z = (f32(voxel.z) - origin.z) * inv_dir.z; }

    var last_axis: i32 = -1;
    var t_cur: f32 = t_enter;
    // Need enough steps to walk the full sun_far depth — ~1500 voxels at
    // 2× scale, at brick (4-vox) granularity that's ~375 steps. 512
    // accommodates worst-case grazing rays. Was 160 → distant rays
    // timed out, falsely flagging "no occluder" so distant terrain
    // reported as LIT regardless of actual sun visibility.
    for (var s: i32 = 0; s < 512; s = s + 1) {
        let rel = voxel - camera.world_origin;
        if (rel.x < 0 || rel.x >= WORLD_VOXELS_X
         || rel.y < 0 || rel.y >= WORLD_VOXELS_Y
         || rel.z < 0 || rel.z >= WORLD_VOXELS_Z) { return max_t; }
        if (t_cur > max_t) { return max_t; }

        let slot_v = world_to_slot_voxel(voxel);
        let bp = slot_v >> vec3<u32>(2u);
        let tp = slot_v >> vec3<u32>(4u);
        let cp = slot_v >> vec3<u32>(6u);
        let ci = world_chunk_idx(cp.x, cp.y, cp.z);
        let tile_lin = i32((tp.x & 3u) + (tp.z & 3u) * 4u + (tp.y & 3u) * 16u);
        if (!chunk_has_child(ci, tile_lin)) {
            skip_to_cell(16, &voxel, &t_max, origin, dir, inv_dir, step, &last_axis);
            t_cur = max(max(t_max.x, t_max.y), t_max.z) - t_delta[max(last_axis, 0)];
            continue;
        }
        let ti = world_tile_idx(tp.x, tp.y, tp.z);
        // Uniform tile = opaque (unless transparent) → return t at entry.
        let tum = tile_uniform_mat(ti);
        if (tum != 0u && !is_transparent_mat(tum)) {
            return t_cur;
        }
        let brick_lin = i32((bp.x & 3u) + (bp.z & 3u) * 4u + (bp.y & 3u) * 16u);
        if (!tile_has_child(ti, brick_lin)) {
            skip_to_cell(4, &voxel, &t_max, origin, dir, inv_dir, step, &last_axis);
            t_cur = max(max(t_max.x, t_max.y), t_max.z) - t_delta[max(last_axis, 0)];
            continue;
        }
        let bi = world_brick_idx(bp.x, bp.y, bp.z);
        let bum = brick_uniform_mat(bi);
        if (bum != 0u && !is_transparent_mat(bum)) {
            return t_cur;
        }
        let local = slot_v - bp * vec3<u32>(4u);
        let vi = brick_voxel_idx(local.x, local.y, local.z);
        if (brick_voxel_solid(bi, vi)) {
            // Look up material — water/glass don't cast shadow here.
            let b = bricks[bi];
            let word = vi / 4;
            let byte = vi - word * 4;
            let m = (b.materials[word] >> u32(byte * 8)) & 0xFFu;
            if (!is_transparent_mat(m)) {
                return t_cur;
            }
        }

        if (t_max.x < t_max.y && t_max.x < t_max.z) {
            t_cur = t_max.x; voxel.x = voxel.x + step.x; t_max.x = t_max.x + t_delta.x; last_axis = 0;
        } else if (t_max.y < t_max.z) {
            t_cur = t_max.y; voxel.y = voxel.y + step.y; t_max.y = t_max.y + t_delta.y; last_axis = 1;
        } else {
            t_cur = t_max.z; voxel.z = voxel.z + step.z; t_max.z = t_max.z + t_delta.z; last_axis = 2;
        }
    }
    return max_t;
}

@compute @workgroup_size(8, 8, 1)
fn cs_shadow(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= SHADOW_RES || gid.y >= SHADOW_RES) { return; }

    // (u, v) ∈ [-1, 1] on the sun's orthographic projection plane.
    let inv = 1.0 / f32(SHADOW_RES);
    let u = (f32(gid.x) + 0.5) * inv * 2.0 - 1.0;
    let v = (f32(gid.y) + 0.5) * inv * 2.0 - 1.0;

    // Back-plane origin: far from the sun along -sun_dir, offset by (u,v).
    let origin = sun.sun_center
        + sun.sun_basis_x * (u * sun.sun_half_size)
        + sun.sun_basis_y * (v * sun.sun_half_size)
        + sun.sun_dir * sun.sun_far;
    let dir = -sun.sun_dir;

    // Trace and store the first-hit depth. The depth is measured along
    // -sun_dir from the BACK plane. A point's "depth" in sample_shadow
    // is computed identically so the comparison works out.
    let t_hit = trace_shadow(origin, dir, sun.sun_far);
    textureStore(shadow_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(t_hit, 0.0, 0.0, 0.0));
}
