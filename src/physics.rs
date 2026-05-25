// Cellular-automaton water physics following the DwarfCorp approach
// (https://www.gamedeveloper.com/programming/how-water-works-in-dwarfcorp):
// each water voxel carries an integer "level" 1..8 stored in its material
// (MAT_WATER_L1..MAT_WATER_L8). Per tick:
//   1. Gravity:  push level from a cell into the cell below, capped at L8.
//   2. Lateral:  give 1 level to a horizontal neighbour that has strictly
//      less water, so columns equalise over time. A per-cell hash chooses
//      which side gets the donation so adjacent cells don't all donate the
//      same way.
//   3. Cleanup:  any cell whose level falls to 0 becomes MAT_AIR.
//
// Optimizations layered on:
//   * Active bricks list — physics only iterates bricks that contain a
//     movable voxel (sand or water). Stone-only bricks are completely
//     skipped.
//   * Bitmask gravity for sand (still binary).
//   * Multi-iteration intra-brick fall so a stack of floating water cells
//     collapses in one tick instead of one cell per tick.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::voxel::*;

static FRAME_COUNTER: AtomicU64 = AtomicU64::new(0);

const BOTTOM_LAYER: u64 = 0xFFFF;
const TOP_LAYER: u64 = 0xFFFF << 48;

pub fn tick(world: &mut World) {
    let frame = FRAME_COUNTER.fetch_add(1, Ordering::Relaxed);
    // 1. Sand gravity (bitmask, multi-pass).
    sand_gravity_pass(world);
    // 2. Water — process every active brick.
    let active = world.active_bricks.clone();
    let mut touched: Vec<u32> = Vec::with_capacity(active.len() * 2);
    for bi in &active {
        if world.movable_mask[*bi as usize] == 0 { continue; }
        step_brick_water(world, *bi, &mut touched);
    }
    // 3. Smoke — rises top-down so it doesn't teleport up a column in one tick.
    let mut smoke_active = active;
    smoke_active.sort_by_key(|&bi| std::cmp::Reverse((bi / WORLD_BRICKS_X) % WORLD_BRICKS_Y));
    for bi in smoke_active {
        if world.movable_mask[bi as usize] == 0 { continue; }
        step_brick_smoke(world, bi, frame, &mut touched);
    }
    // Settle sand on bricks the fluids touched.
    settle_sand(world, &mut touched);
}

// ---------------- sand gravity ----------------

fn sand_gravity_pass(world: &mut World) {
    let mut active: Vec<u32> = world.active_bricks.clone();
    active.sort_by_key(|&bi| (bi / WORLD_BRICKS_X) % WORLD_BRICKS_Y);
    for bi in active {
        if world.movable_mask[bi as usize] == 0 { continue; }
        let bx = bi % WORLD_BRICKS_X;
        let by = (bi / WORLD_BRICKS_X) % WORLD_BRICKS_Y;
        let bz = bi / (WORLD_BRICKS_X * WORLD_BRICKS_Y);
        step_brick_sand_fall(world, bx, by, bz);
    }
}

fn settle_sand(world: &mut World, touched: &mut Vec<u32>) {
    if touched.is_empty() { return; }
    touched.sort_unstable();
    touched.dedup();
    touched.sort_by_key(|&bi| (bi / WORLD_BRICKS_X) % WORLD_BRICKS_Y);
    let mut visited: Vec<u32> = Vec::with_capacity(touched.len() * 2);
    let mut iter = 0;
    while !touched.is_empty() && iter < 8 {
        iter += 1;
        let batch = std::mem::take(touched);
        for bi in batch {
            if visited.contains(&bi) { continue; }
            visited.push(bi);
            if world.movable_mask[bi as usize] == 0 { continue; }
            let bx = bi % WORLD_BRICKS_X;
            let by = (bi / WORLD_BRICKS_X) % WORLD_BRICKS_Y;
            let bz = bi / (WORLD_BRICKS_X * WORLD_BRICKS_Y);
            step_brick_sand_fall(world, bx, by, bz);
            if by > 0 { touched.push(brick_idx(bx, by - 1, bz)); }
        }
    }
}

fn sand_mask(b: &Brick) -> u64 {
    let mut m = 0u64;
    for i in 0..64usize {
        m |= ((b.materials[i] == MAT_SAND) as u64) << i;
    }
    m & b.occupancy
}

fn step_brick_sand_fall(world: &mut World, bx: u32, by: u32, bz: u32) {
    let bi = brick_idx(bx, by, bz);
    if world.movable_mask[bi as usize] == 0 { return; }

    // Multi-pass intra-fall so a 4-deep stack of floaters collapses fully
    // in one tick.
    for _ in 0..3 {
        let occ = world.bricks[bi as usize].occupancy;
        let sand = sand_mask(&world.bricks[bi as usize]);
        if sand == 0 { break; }
        let empty = !occ;
        let falling = sand & (empty << 16);
        if falling == 0 { break; }
        let b = &mut world.bricks[bi as usize];
        b.occupancy ^= falling | (falling >> 16);
        let mut bits = falling;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            b.materials[i - 16] = MAT_SAND;
            b.materials[i] = MAT_AIR;
            bits &= bits - 1;
        }
        world.movable_mask[bi as usize] ^= falling | (falling >> 16);
        world.mark_brick_dirty(bi);
    }

    // Cross-brick fall.
    if by == 0 { return; }
    let cur_occ = world.bricks[bi as usize].occupancy;
    let sand = sand_mask(&world.bricks[bi as usize]);
    let bottom_sand = sand & BOTTOM_LAYER;
    if bottom_sand == 0 { return; }
    let below_bi = brick_idx(bx, by - 1, bz);
    let below_occ = world.bricks[below_bi as usize].occupancy;
    let below_top_empty = !below_occ & TOP_LAYER;
    let cross = bottom_sand & (below_top_empty >> 48);
    if cross == 0 { return; }

    let mut moves: [u32; 16] = [0; 16];
    let mut count = 0usize;
    let mut bits = cross;
    while bits != 0 {
        moves[count] = bits.trailing_zeros();
        count += 1;
        bits &= bits - 1;
    }

    let b = &mut world.bricks[bi as usize];
    b.occupancy ^= cross;
    for k in 0..count { b.materials[moves[k] as usize] = MAT_AIR; }
    let cur_now_empty = b.occupancy == 0;
    world.movable_mask[bi as usize] &= !cross;

    let was_empty = below_occ == 0;
    let was_movable = world.movable_mask[below_bi as usize] != 0;
    let b2 = &mut world.bricks[below_bi as usize];
    b2.occupancy |= cross << 48;
    for k in 0..count { b2.materials[(moves[k] + 48) as usize] = MAT_SAND; }
    world.movable_mask[below_bi as usize] |= cross << 48;
    if !was_movable {
        if let Err(pos) = world.active_bricks.binary_search(&below_bi) {
            world.active_bricks.insert(pos, below_bi);
        }
    }
    if cur_now_empty && cur_occ != 0 { world.refresh_masks_for_brick(bx, by, bz); }
    if was_empty { world.refresh_masks_for_brick(bx, by - 1, bz); }
    world.mark_brick_dirty(bi);
    world.mark_brick_dirty(below_bi);
}

// ---------------- water (DwarfCorp-style level flow) ----------------

fn step_brick_water(world: &mut World, bi: u32, touched: &mut Vec<u32>) {
    // Snapshot the brick once; do every intra-brick computation on the local
    // copy; write back once at the end. Cross-brick transfers touch the
    // neighbour directly (one transfer per cell at most).
    let bx = bi % WORLD_BRICKS_X;
    let by = (bi / WORLD_BRICKS_X) % WORLD_BRICKS_Y;
    let bz = bi / (WORLD_BRICKS_X * WORLD_BRICKS_Y);
    let snap_occ = world.bricks[bi as usize].occupancy;
    let snap_mats: [u8; 64] = world.bricks[bi as usize].materials;
    let snap_movable = world.movable_mask[bi as usize];

    let mut new_occ = snap_occ;
    let mut new_mats = snap_mats;
    let mut new_movable = snap_movable;
    let mut any_change = false;

    // Iterate ONLY the movable bits — sand and water — skipping all stone.
    let mut bits = snap_movable;
    while bits != 0 {
        let i = bits.trailing_zeros() as i32;
        bits &= bits - 1;
        let mat = new_mats[i as usize];
        if !is_water_mat(mat) { continue; }

        let lx = (i & 3) as u32;
        let lz = ((i >> 2) & 3) as u32;
        let ly = ((i >> 4) & 3) as u32;
        let mut remaining = water_level_of(mat) as i32;

        // ---- gravity ----
        // Try the cell directly below: same brick if ly > 0, else top row of
        // brick-below.
        if ly > 0 {
            let below_i = i - 16;
            let below_bit = 1u64 << below_i;
            let below_solid_blocking = (new_occ & below_bit) != 0
                && !is_water_mat(new_mats[below_i as usize]);
            if !below_solid_blocking {
                let below_level = if (new_occ & below_bit) != 0 {
                    water_level_of(new_mats[below_i as usize]) as i32
                } else { 0 };
                let space = MAX_WATER_LEVEL as i32 - below_level;
                let transfer = remaining.min(space);
                if transfer > 0 {
                    let nl = below_level + transfer;
                    new_mats[below_i as usize] = water_mat_for_level(nl as u8);
                    new_occ |= below_bit;
                    new_movable |= below_bit;
                    remaining -= transfer;
                    any_change = true;
                }
            }
        } else if by > 0 {
            // Cross-brick gravity (rare, expensive — still one transfer max).
            let below_bi = brick_idx(bx, by - 1, bz);
            let below_i_in = (lx + lz * 4 + 3 * 16) as usize;
            let below_bit = 1u64 << below_i_in;
            let nb_occ = world.bricks[below_bi as usize].occupancy;
            let nb_solid = (nb_occ & below_bit) != 0
                && !is_water_mat(world.bricks[below_bi as usize].materials[below_i_in]);
            if !nb_solid {
                let below_level = if (nb_occ & below_bit) != 0 {
                    water_level_of(world.bricks[below_bi as usize].materials[below_i_in]) as i32
                } else { 0 };
                let space = MAX_WATER_LEVEL as i32 - below_level;
                let transfer = remaining.min(space);
                if transfer > 0 {
                    cross_apply_water(world, below_bi, below_i_in, (below_level + transfer) as u8, touched);
                    remaining -= transfer;
                }
            }
        }

        if remaining == 0 {
            new_occ &= !(1u64 << i);
            new_movable &= !(1u64 << i);
            new_mats[i as usize] = MAT_AIR;
            any_change = true;
            continue;
        }

        // ---- lateral ----
        if remaining > 1 {
            let gx = bx * BRICK_DIM + lx;
            let gy = by * BRICK_DIM + ly;
            let gz = bz * BRICK_DIM + lz;
            let h = (gx.wrapping_mul(0x9E3779B1)
                  ^ gy.wrapping_mul(0x85EBCA77)
                  ^ gz.wrapping_mul(0xC2B2AE3D)) as usize;
            const DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
            for k in 0..4 {
                if remaining <= 1 { break; }
                let (dx, dz) = DIRS[(h + k) & 3];
                let tlx = lx as i32 + dx;
                let tlz = lz as i32 + dz;
                if tlx >= 0 && tlx < 4 && tlz >= 0 && tlz < 4 {
                    let target = (tlx + tlz * 4 + ly as i32 * 16) as i32;
                    let tbit = 1u64 << target;
                    let occupied = (new_occ & tbit) != 0;
                    let t_mat = new_mats[target as usize];
                    if occupied && !is_water_mat(t_mat) { continue; }
                    let t_level = if occupied { water_level_of(t_mat) as i32 } else { 0 };
                    if remaining > t_level + 1 {
                        let nl = t_level + 1;
                        new_mats[target as usize] = water_mat_for_level(nl as u8);
                        new_occ |= tbit;
                        new_movable |= tbit;
                        remaining -= 1;
                        any_change = true;
                    }
                } else {
                    let nbx = bx as i32 + dx;
                    let nbz = bz as i32 + dz;
                    if nbx < 0 || nbx >= WORLD_BRICKS_X as i32
                    || nbz < 0 || nbz >= WORLD_BRICKS_Z as i32 { continue; }
                    let nbx = nbx as u32;
                    let nbz = nbz as u32;
                    let nb_lx = (tlx & 3) as u32;
                    let nb_lz = (tlz & 3) as u32;
                    let target = (nb_lx + nb_lz * 4 + ly * 16) as usize;
                    let tbit = 1u64 << target;
                    let nb_bi = brick_idx(nbx, by, nbz);
                    let nb_occ = world.bricks[nb_bi as usize].occupancy;
                    let occupied = (nb_occ & tbit) != 0;
                    let t_mat = if occupied { world.bricks[nb_bi as usize].materials[target] } else { MAT_AIR };
                    if occupied && !is_water_mat(t_mat) { continue; }
                    let t_level = if occupied { water_level_of(t_mat) as i32 } else { 0 };
                    if remaining > t_level + 1 {
                        cross_apply_water(world, nb_bi, target, (t_level + 1) as u8, touched);
                        remaining -= 1;
                    }
                }
            }
        }

        // Apply remaining level to current cell if it changed.
        let new_level = remaining as u8;
        let new_self_mat = water_mat_for_level(new_level);
        if new_self_mat != mat {
            new_mats[i as usize] = new_self_mat;
            if new_level == 0 {
                new_occ &= !(1u64 << i);
                new_movable &= !(1u64 << i);
            }
            any_change = true;
        }
    }

    if any_change {
        let was_empty = snap_occ == 0;
        let now_empty = new_occ == 0;
        {
            let b = &mut world.bricks[bi as usize];
            b.occupancy = new_occ;
            b.materials = new_mats;
        }
        world.movable_mask[bi as usize] = new_movable;
        // Maintain active_bricks if this brick just lost all its movable bits.
        if new_movable == 0 {
            if let Ok(pos) = world.active_bricks.binary_search(&bi) {
                world.active_bricks.remove(pos);
            }
        }
        if was_empty != now_empty {
            world.refresh_masks_for_brick(bx, by, bz);
        }
        world.mark_brick_dirty(bi);
        touched.push(bi);
    }
}

// ---------------- smoke (rises, dissipates) ----------------

fn step_brick_smoke(world: &mut World, bi: u32, frame: u64, touched: &mut Vec<u32>) {
    let bx = bi % WORLD_BRICKS_X;
    let by = (bi / WORLD_BRICKS_X) % WORLD_BRICKS_Y;
    let bz = bi / (WORLD_BRICKS_X * WORLD_BRICKS_Y);

    let snap_occ = world.bricks[bi as usize].occupancy;
    let mut new_occ = snap_occ;
    let mut new_mats = world.bricks[bi as usize].materials;
    let mut new_movable = world.movable_mask[bi as usize];
    let mut any_change = false;

    // Walk movable bits — smoke is movable so it's in here.
    let mut bits = new_movable;
    while bits != 0 {
        let i = bits.trailing_zeros() as i32;
        bits &= bits - 1;
        let mat = new_mats[i as usize];
        if mat != MAT_SMOKE { continue; }

        let lx = (i & 3) as u32;
        let lz = ((i >> 2) & 3) as u32;
        let ly = ((i >> 4) & 3) as u32;
        let gx = bx * BRICK_DIM + lx;
        let gy = by * BRICK_DIM + ly;
        let gz = bz * BRICK_DIM + lz;

        // Pseudorandom decision per (cell, frame).
        let h = (gx.wrapping_mul(0x9E3779B1)
              ^ gy.wrapping_mul(0x85EBCA77)
              ^ gz.wrapping_mul(0xC2B2AE3D)
              ^ (frame as u32).wrapping_mul(0xD2B74407)) as u32;

        // 1/40 chance to dissipate every tick — smoke fades out naturally.
        if (h % 40) == 0 {
            new_occ &= !(1u64 << i);
            new_movable &= !(1u64 << i);
            new_mats[i as usize] = MAT_AIR;
            any_change = true;
            continue;
        }

        // Try to rise intra-brick.
        if ly < 3 {
            let up_i = i + 16;
            let up_bit = 1u64 << up_i;
            if (new_occ & up_bit) == 0 {
                new_occ ^= (1u64 << i) | up_bit;
                new_movable ^= (1u64 << i) | up_bit;
                new_mats[up_i as usize] = MAT_SMOKE;
                new_mats[i as usize] = MAT_AIR;
                any_change = true;
                continue;
            }
        } else if by + 1 < WORLD_BRICKS_Y {
            // Cross-brick rise: top row of this brick → y=0 of brick above.
            let up_bi = brick_idx(bx, by + 1, bz);
            let up_i_in = (lx + lz * 4) as usize;
            let up_bit = 1u64 << up_i_in;
            let up_occ = world.bricks[up_bi as usize].occupancy;
            if (up_occ & up_bit) == 0 {
                let nb_was_movable = world.movable_mask[up_bi as usize] != 0;
                let nb_was_empty = up_occ == 0;
                {
                    let nb = &mut world.bricks[up_bi as usize];
                    nb.occupancy |= up_bit;
                    nb.materials[up_i_in] = MAT_SMOKE;
                }
                world.movable_mask[up_bi as usize] |= up_bit;
                if !nb_was_movable {
                    if let Err(pos) = world.active_bricks.binary_search(&up_bi) {
                        world.active_bricks.insert(pos, up_bi);
                    }
                }
                if nb_was_empty {
                    world.refresh_masks_for_brick(bx, by + 1, bz);
                }
                world.mark_brick_dirty(up_bi);
                touched.push(up_bi);
                new_occ &= !(1u64 << i);
                new_movable &= !(1u64 << i);
                new_mats[i as usize] = MAT_AIR;
                any_change = true;
                continue;
            }
        }

        // Can't rise — try lateral within brick.
        const DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
        for k in 0..4 {
            let (dx, dz) = DIRS[((h as usize) + k) & 3];
            let tlx = lx as i32 + dx;
            let tlz = lz as i32 + dz;
            if tlx < 0 || tlx > 3 || tlz < 0 || tlz > 3 { continue; }
            let target = tlx + tlz * 4 + ly as i32 * 16;
            let tbit = 1u64 << target;
            if (new_occ & tbit) == 0 {
                new_occ ^= (1u64 << i) | tbit;
                new_movable ^= (1u64 << i) | tbit;
                new_mats[target as usize] = MAT_SMOKE;
                new_mats[i as usize] = MAT_AIR;
                any_change = true;
                break;
            }
        }
    }

    if any_change {
        let was_empty = snap_occ == 0;
        let now_empty = new_occ == 0;
        {
            let b = &mut world.bricks[bi as usize];
            b.occupancy = new_occ;
            b.materials = new_mats;
        }
        world.movable_mask[bi as usize] = new_movable;
        if new_movable == 0 {
            if let Ok(pos) = world.active_bricks.binary_search(&bi) {
                world.active_bricks.remove(pos);
            }
        }
        if was_empty != now_empty {
            world.refresh_masks_for_brick(bx, by, bz);
        }
        world.mark_brick_dirty(bi);
        touched.push(bi);
    }
}

/// Apply a single cross-brick water set to (nb_bi, voxel index in that brick,
/// new level). Updates occupancy, materials, movable_mask, active list, and
/// emits the dirty/refresh signals.
fn cross_apply_water(
    world: &mut World,
    nb_bi: u32,
    nb_vi: usize,
    new_level: u8,
    touched: &mut Vec<u32>,
) {
    let nb_bit = 1u64 << nb_vi;
    let nb_was_empty = world.bricks[nb_bi as usize].occupancy == 0;
    let nb_was_movable = world.movable_mask[nb_bi as usize] != 0;
    {
        let b = &mut world.bricks[nb_bi as usize];
        if new_level == 0 {
            b.occupancy &= !nb_bit;
            b.materials[nb_vi] = MAT_AIR;
        } else {
            b.occupancy |= nb_bit;
            b.materials[nb_vi] = water_mat_for_level(new_level);
        }
    }
    if new_level == 0 {
        world.movable_mask[nb_bi as usize] &= !nb_bit;
    } else {
        world.movable_mask[nb_bi as usize] |= nb_bit;
    }
    if !nb_was_movable && new_level > 0 {
        if let Err(pos) = world.active_bricks.binary_search(&nb_bi) {
            world.active_bricks.insert(pos, nb_bi);
        }
    }
    let nb_now_empty = world.bricks[nb_bi as usize].occupancy == 0;
    if nb_was_empty != nb_now_empty {
        let nbx = nb_bi % WORLD_BRICKS_X;
        let nby = (nb_bi / WORLD_BRICKS_X) % WORLD_BRICKS_Y;
        let nbz = nb_bi / (WORLD_BRICKS_X * WORLD_BRICKS_Y);
        world.refresh_masks_for_brick(nbx, nby, nbz);
    }
    world.mark_brick_dirty(nb_bi);
    touched.push(nb_bi);
}
