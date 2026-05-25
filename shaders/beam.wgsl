// Beam pre-pass.
//
// At 1/8 resolution, fire a single ray per pixel-tile and traverse the world
// at TILE (16 voxels) granularity until we hit the first non-empty tile.
// Write that t value to a coarse depth texture. The main raymarch reads it
// per-pixel and skips the ray forward to that t — eliminating the cost of
// crawling through empty sky/underground from the camera.

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

fn pos_mod(a: i32, b: i32) -> i32 {
    let r = a % b;
    return select(r, r + b, r < 0);
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> chunk_mask: array<u32>;
@group(0) @binding(2) var beam_out: texture_storage_2d<r32float, write>;

const WORLD_VOXELS_X: i32 = 512;
const WORLD_VOXELS_Y: i32 = 256;
const WORLD_VOXELS_Z: i32 = 512;
const WORLD_TILES_X: i32 = 32;
const WORLD_TILES_Y: i32 = 16;
const WORLD_TILES_Z: i32 = 32;
const WORLD_CHUNKS_X: i32 = 8;
const WORLD_CHUNKS_Y: i32 = 4;
const WORLD_CHUNKS_Z: i32 = 8;

fn world_chunk_idx(cx: i32, cy: i32, cz: i32) -> i32 {
    return cx + cy * WORLD_CHUNKS_X + cz * WORLD_CHUNKS_X * WORLD_CHUNKS_Y;
}

fn chunk_has_child(ci: i32, child_lin: i32) -> bool {
    let base = ci * 2;
    if (child_lin < 32) {
        return (chunk_mask[base] & (1u << u32(child_lin))) != 0u;
    }
    return (chunk_mask[base + 1] & (1u << u32(child_lin - 32))) != 0u;
}

fn safe_inv(x: f32) -> f32 {
    if (abs(x) < 1e-8) { return 1e30; }
    return 1.0 / x;
}

@compute @workgroup_size(8, 8, 1)
fn cs_beam(@builtin(global_invocation_id) gid: vec3<u32>) {
    let beam_w = (u32(camera.resolution.x) + 7u) / 8u;
    let beam_h = (u32(camera.resolution.y) + 7u) / 8u;
    if (gid.x >= beam_w || gid.y >= beam_h) { return; }

    // Fire a ray through the centre of the 8×8 pixel block.
    let px = vec2<f32>(f32(gid.x) * 8.0 + 4.0, f32(gid.y) * 8.0 + 4.0);
    let uv = px / camera.resolution;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let aspect = camera.resolution.x / camera.resolution.y;
    let dir = normalize(
        camera.forward
        + camera.right * (ndc.x * camera.tan_half_fov * aspect)
        + camera.up    * (ndc.y * camera.tan_half_fov)
    );

    let t = trace_first_tile(camera.origin, dir);
    textureStore(beam_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(t, 0.0, 0.0, 0.0));
}

fn trace_first_tile(origin: vec3<f32>, dir: vec3<f32>) -> f32 {
    let win_min = vec3<f32>(camera.world_origin);
    let win_max = win_min + vec3<f32>(f32(WORLD_VOXELS_X), f32(WORLD_VOXELS_Y), f32(WORLD_VOXELS_Z));
    let inv_dir = vec3<f32>(safe_inv(dir.x), safe_inv(dir.y), safe_inv(dir.z));
    let t0 = (win_min - origin) * inv_dir;
    let t1 = (win_max - origin) * inv_dir;
    let tmin3 = min(t0, t1);
    let tmax3 = max(t0, t1);
    let t_enter = max(max(tmin3.x, tmin3.y), max(tmin3.z, 0.0));
    let t_exit = min(min(tmax3.x, tmax3.y), tmax3.z);
    if (t_enter >= t_exit || t_exit < 0.0) { return 1e9; }

    // Step at tile granularity (16 voxels) in WORLD coords.
    let bias = 0.05;
    var p = origin + dir * (t_enter + bias);
    p = clamp(p, win_min + vec3<f32>(0.01), win_max - vec3<f32>(0.01));

    var tile = vec3<i32>(floor(p / 16.0));
    let step = vec3<i32>(sign(dir));
    let t_delta = abs(inv_dir) * 16.0;

    var t_max: vec3<f32>;
    if (step.x > 0) { t_max.x = (f32((tile.x + 1) * 16) - origin.x) * inv_dir.x; }
    else            { t_max.x = (f32(tile.x * 16)       - origin.x) * inv_dir.x; }
    if (step.y > 0) { t_max.y = (f32((tile.y + 1) * 16) - origin.y) * inv_dir.y; }
    else            { t_max.y = (f32(tile.y * 16)       - origin.y) * inv_dir.y; }
    if (step.z > 0) { t_max.z = (f32((tile.z + 1) * 16) - origin.z) * inv_dir.z; }
    else            { t_max.z = (f32(tile.z * 16)       - origin.z) * inv_dir.z; }

    let origin_tile = camera.world_origin / 16;
    var t_cur = t_enter;
    for (var s = 0; s < 80; s = s + 1) {
        // Bounds in WORLD tile coords vs loaded window.
        let rel = tile - origin_tile;
        if (rel.x < 0 || rel.x >= WORLD_TILES_X
         || rel.y < 0 || rel.y >= WORLD_TILES_Y
         || rel.z < 0 || rel.z >= WORLD_TILES_Z) {
            return 1e9;
        }
        // Fold into slot tile coords.
        let slot_tile = vec3<i32>(pos_mod(tile.x, WORLD_TILES_X), tile.y, pos_mod(tile.z, WORLD_TILES_Z));
        let cp = slot_tile / 4;
        let ci = world_chunk_idx(cp.x, cp.y, cp.z);
        let lin = (slot_tile.x & 3) + (slot_tile.z & 3) * 4 + (slot_tile.y & 3) * 16;
        if (chunk_has_child(ci, lin)) {
            return t_cur;
        }
        if (t_max.x < t_max.y && t_max.x < t_max.z) {
            tile.x = tile.x + step.x;
            t_cur = t_max.x;
            t_max.x = t_max.x + t_delta.x;
        } else if (t_max.y < t_max.z) {
            tile.y = tile.y + step.y;
            t_cur = t_max.y;
            t_max.y = t_max.y + t_delta.y;
        } else {
            tile.z = tile.z + step.z;
            t_cur = t_max.z;
            t_max.z = t_max.z + t_delta.z;
        }
    }
    return 1e9;
}
