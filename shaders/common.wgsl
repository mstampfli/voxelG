// Shared shader prelude. renderer.rs prepends this (after world_consts.wgsl)
// to raymarch.wgsl, beam.wgsl and taa.wgsl so the Camera uniform layout and
// these tiny helpers have ONE definition instead of a per-shader copy that
// silently drifts (beam previously carried a TRUNCATED Camera). Do not add
// anything here that depends on a per-shader binding (e.g. chunk_mask).

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
    reproject_lighting: f32,
    prev_origin: vec3<f32>,
    _pad6: f32,
    prev_forward: vec3<f32>,
    _pad7: f32,
    prev_right: vec3<f32>,
    _pad8: f32,
    prev_up: vec3<f32>,
    _pad9: f32,
};

// Floored modulo for toroidal world-coord folding (result always in [0, b)).
fn pos_mod(a: i32, b: i32) -> i32 {
    let r = a % b;
    return select(r, r + b, r < 0);
}

// Reciprocal that won't blow up on a near-zero ray component.
fn safe_inv(x: f32) -> f32 {
    if (abs(x) < 1e-8) { return 1e30; }
    return 1.0 / x;
}
