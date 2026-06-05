// Temporal anti-aliasing resolve pass.
//
// Blends the freshly-raymarched frame with the accumulated history, gated by a
// 3x3 neighbourhood colour clamp so disocclusions / motion don't ghost. While
// the camera is static the main pass jitters its rays sub-pixel each frame and
// this accumulation converges to an anti-aliased image (small voxels stop
// shimmering). On motion the host sets taa_blend = 0, so the current frame
// passes through unchanged (sharp, no smearing).

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
    jitter: vec2<f32>,
    taa_blend: f32,
    _pad5: f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var current_tex: texture_2d<f32>;
@group(0) @binding(2) var history_tex: texture_2d<f32>;
@group(0) @binding(3) var resolve_out: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_taa(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = vec2<i32>(camera.resolution);
    let p = vec2<i32>(i32(gid.x), i32(gid.y));
    if (p.x >= res.x || p.y >= res.y) { return; }

    let cur = textureLoad(current_tex, p, 0).rgb;

    // taa_blend <= 0 → reset / motion: pass the current frame straight through.
    if (camera.taa_blend <= 0.0) {
        textureStore(resolve_out, p, vec4<f32>(cur, 1.0));
        return;
    }

    // 3x3 neighbourhood colour bounds of the current frame.
    var mn = cur;
    var mx = cur;
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let q = clamp(p + vec2<i32>(dx, dy), vec2<i32>(0), res - vec2<i32>(1));
            let n = textureLoad(current_tex, q, 0).rgb;
            mn = min(mn, n);
            mx = max(mx, n);
        }
    }

    // Clamp history into the current neighbourhood (rejects ghosting) and blend.
    let hist = textureLoad(history_tex, p, 0).rgb;
    let hist_clamped = clamp(hist, mn, mx);
    let resolved = mix(cur, hist_clamped, camera.taa_blend);
    textureStore(resolve_out, p, vec4<f32>(resolved, 1.0));
}
