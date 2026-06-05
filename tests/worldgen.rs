// Determinism + edit-replay tests (checklist: tests). These protect the two
// invariants multiplayer relies on:
//   1. Worldgen is a pure function of (chunk, seed) — every client regenerates
//      identical terrain from the shared seed.
//   2. Player edits survive a chunk streaming round-trip (unload + regenerate),
//      because the edit log is replayed on top of fresh noise. A regression
//      here is exactly the "late joiner / re-entered chunk desync" bug.

use glam::IVec3;
use voxelg::voxel::{self, World, MAT_GLASS};

fn bricks_eq(a: &[voxel::Brick], b: &[voxel::Brick]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.occupancy == y.occupancy && x.materials == y.materials)
}

#[test]
fn worldgen_is_deterministic() {
    let seed = 0xC0FFEE_F00D_BEEFu64;
    let chunk = IVec3::new(3, 0, 5);
    let a = voxel::gen_slot_bricks(chunk, seed);
    let b = voxel::gen_slot_bricks(chunk, seed);
    assert!(bricks_eq(&a, &b), "same (chunk, seed) must produce identical bricks");
}

#[test]
fn worldgen_depends_on_seed() {
    // cy=2 spans the surface (terrain sits ~y72); the surface layer + biome +
    // tree placement are what the seed actually shifts (deep-underground stone
    // is seed-independent by design).
    let chunk = IVec3::new(3, 2, 5);
    let a = voxel::gen_slot_bricks(chunk, 1);
    let b = voxel::gen_slot_bricks(chunk, 2);
    assert!(!bricks_eq(&a, &b), "different seeds must produce different terrain");
}

#[test]
fn worldgen_depends_on_chunk() {
    let seed = 42;
    let a = voxel::gen_slot_bricks(IVec3::new(0, 2, 0), seed);
    let b = voxel::gen_slot_bricks(IVec3::new(1, 2, 0), seed);
    assert!(!bricks_eq(&a, &b), "neighbouring surface chunks must differ");
}

#[test]
fn edits_survive_chunk_streaming_roundtrip() {
    let mut world = World::with_seed(0xABCD_1234_5678_9999);
    world.fill_demo_terrain();

    // A handful of edits well inside the initial window.
    let edits = [
        (10, 100, 10),
        (40, 70, 80),
        (200, 64, 130),
    ];
    for &(x, y, z) in &edits {
        world.apply_edit(x, y, z, MAT_GLASS);
        assert_eq!(world.material_at_world(x, y, z), MAT_GLASS, "edit must apply locally");
    }

    // Stream the whole window away, then back — forces every slot (including
    // the edited columns) to unload and regenerate from noise + replayed edits.
    let far = glam::IVec2::new(
        voxel::WORLD_STORE_CX as i32,
        voxel::WORLD_STORE_CZ as i32,
    );
    world.shift_origin(far);
    world.process_regen_queue(voxel::WORLD_STORE_CHUNKS);
    world.shift_origin(glam::IVec2::ZERO);
    world.process_regen_queue(voxel::WORLD_STORE_CHUNKS);

    for &(x, y, z) in &edits {
        assert_eq!(
            world.material_at_world(x, y, z),
            MAT_GLASS,
            "edit at ({x},{y},{z}) must survive the streaming round-trip"
        );
    }
}
