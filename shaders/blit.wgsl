// Fullscreen blit pass: samples the compute-shader output into the swapchain.

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    // Single oversized triangle.
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var o: VsOut;
    o.pos = vec4<f32>(positions[vid], 0.0, 1.0);
    o.uv = uvs[vid];
    return o;
}

// Bilinear upscale + light contrast-aware sharpening to restore detail lost
// by the half-resolution primary raymarch. Sharpening is a 5-tap unsharp
// mask (cross neighbours + centre) — cheap and removes the "blurry" look
// that plain bilinear leaves behind.
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(src_tex, 0));
    let px = 1.0 / dims;
    let e  = textureSample(src_tex, src_sampler, in.uv);
    let n  = textureSample(src_tex, src_sampler, in.uv + vec2<f32>(0.0, -px.y));
    let s  = textureSample(src_tex, src_sampler, in.uv + vec2<f32>(0.0,  px.y));
    let w  = textureSample(src_tex, src_sampler, in.uv + vec2<f32>(-px.x, 0.0));
    let ee = textureSample(src_tex, src_sampler, in.uv + vec2<f32>( px.x, 0.0));
    let avg = (n + s + w + ee) * 0.25;
    let amount = 0.35;
    var col = e + (e - avg) * amount;
    col = max(col, vec4<f32>(0.0));
    return vec4<f32>(col.rgb, 1.0);
}
