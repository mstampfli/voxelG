// Microbenchmarks for the measurable CPU hot paths, so optimisation claims are
// backed by numbers (checklist: tests/benchmarks). The GPU `trace` shader can't
// be timed from Rust directly; the closest CPU analogues are the picking DDA
// (`raycast`, same hierarchical traversal the shader walks) and worldgen
// (`gen_slot_bricks`, the chunk-load hitch the streaming work targets).

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use glam::{IVec3, Vec3};
use voxelg::voxel::{self, World};

fn bench_worldgen(c: &mut Criterion) {
    let seed = 0xC0FFEE_F00D_BEEFu64;
    c.bench_function("gen_slot_bricks", |b| {
        let mut i = 0i32;
        b.iter(|| {
            // Vary the chunk each iter so we don't measure a single cache-hot column.
            i = i.wrapping_add(1);
            let chunk = IVec3::new(i & 7, 0, (i >> 3) & 7);
            black_box(voxel::gen_slot_bricks(black_box(chunk), black_box(seed)))
        })
    });
}

fn bench_raycast(c: &mut Criterion) {
    let mut world = World::new();
    world.fill_demo_terrain();
    let origin = IVec3::ZERO;
    c.bench_function("raycast_pick", |b| {
        let mut a = 0.0f32;
        b.iter(|| {
            a += 0.013;
            let dir = Vec3::new(a.sin(), -0.5, a.cos()).normalize();
            black_box(voxelg::raycast::raycast(
                black_box(Vec3::new(256.0, 90.0, 256.0)),
                black_box(dir),
                black_box(&world),
                black_box(origin),
            ))
        })
    });
}

criterion_group!(benches, bench_worldgen, bench_raycast);
criterion_main!(benches);
