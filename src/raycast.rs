// CPU-side DDA raycast for picking. WALKS WORLD COORDS — must mirror the
// shader exactly because the storage is toroidal (slot_v = pos_mod(world_v,
// WORLD_VOXELS_*)). The previous version walked `camera - origin` linearly,
// which only matched the shader when origin == 0 (player still at spawn).
// As soon as the player moved into a different chunk, the bricks CPU looked
// at ≠ the bricks GPU rendered. That was the "click breaks the wrong voxel"
// bug.

use glam::{IVec3, Vec3};

use crate::voxel::*;

pub struct PickHit {
    /// WORLD voxel coordinate.
    pub voxel: [i32; 3],
    pub normal: [i32; 3],
}

const MAX_STEPS: i32 = 4096;

pub fn raycast(camera_world: Vec3, dir: Vec3, world: &World, world_origin: IVec3) -> Option<PickHit> {
    // Normalize defensively: a near-zero / un-normalized dir would make several
    // axes hit safe_inv's 1e30 clamp at once and the DDA would never advance.
    let dir = dir.normalize_or_zero();
    if dir == Vec3::ZERO {
        return None;
    }
    let win_min = Vec3::new(
        world_origin.x as f32,
        world_origin.y as f32,
        world_origin.z as f32,
    );
    let win_max = win_min
        + Vec3::new(WORLD_VOXELS_X as f32, WORLD_VOXELS_Y as f32, WORLD_VOXELS_Z as f32);
    let inv_dir = Vec3::new(safe_inv(dir.x), safe_inv(dir.y), safe_inv(dir.z));
    let t0 = (win_min - camera_world) * inv_dir;
    let t1 = (win_max - camera_world) * inv_dir;
    let tmin = t0.min(t1);
    let tmax = t0.max(t1);
    let t_enter = tmin.max_element().max(0.0);
    let t_exit = tmax.min_element();
    if t_enter >= t_exit || t_exit < 0.0 {
        return None;
    }

    let mut p = camera_world + dir * (t_enter + 1e-3);
    p = p.clamp(win_min + Vec3::splat(1e-3), win_max - Vec3::splat(1e-3));
    let mut voxel = [p.x.floor() as i32, p.y.floor() as i32, p.z.floor() as i32];
    let step = [sign_i32(dir.x), sign_i32(dir.y), sign_i32(dir.z)];
    let t_delta = [inv_dir.x.abs(), inv_dir.y.abs(), inv_dir.z.abs()];

    let mut t_max = [0.0f32; 3];
    for k in 0..3 {
        let nxt = if step[k] > 0 { (voxel[k] + 1) as f32 } else { voxel[k] as f32 };
        let o = [camera_world.x, camera_world.y, camera_world.z][k];
        let inv = [inv_dir.x, inv_dir.y, inv_dir.z][k];
        t_max[k] = (nxt - o) * inv;
    }

    let in_window = |voxel: [i32; 3]| -> bool {
        let rx = voxel[0] - world_origin.x;
        let ry = voxel[1] - world_origin.y;
        let rz = voxel[2] - world_origin.z;
        rx >= 0 && ry >= 0 && rz >= 0
            && (rx as u32) < WORLD_VOXELS_X
            && (ry as u32) < WORLD_VOXELS_Y
            && (rz as u32) < WORLD_VOXELS_Z
    };

    // World-voxel → storage brick + voxel-in-brick. Matches the shader's
    // `world_to_slot_voxel` + brick_idx.
    let in_solid = |voxel: [i32; 3]| -> bool {
        if !in_window(voxel) { return false; }
        let sx = voxel[0].rem_euclid(WORLD_VOXELS_X as i32) as u32;
        let sy = voxel[1] as u32;
        let sz = voxel[2].rem_euclid(WORLD_VOXELS_Z as i32) as u32;
        let bx = sx / BRICK_DIM;
        let by = sy / BRICK_DIM;
        let bz = sz / BRICK_DIM;
        let bi = brick_idx(bx, by, bz) as usize;
        let b = &world.bricks[bi];
        if b.occupancy == 0 { return false; }
        let lx = sx % BRICK_DIM;
        let ly = sy % BRICK_DIM;
        let lz = sz % BRICK_DIM;
        let vi = brick_voxel_idx(lx, ly, lz);
        if (b.occupancy & (1u64 << vi)) == 0 { return false; }
        // Invisible canopy fringe is never a pick target.
        b.materials[vi as usize] != crate::voxel::MAT_LEAF_FRINGE
    };

    let argmin_step = |t_max: &mut [f32; 3], voxel: &mut [i32; 3]| -> i32 {
        let mut ax: i32 = -1;
        let mut m = f32::INFINITY;
        for k in 0..3 {
            if step[k] != 0 && t_max[k] < m { ax = k as i32; m = t_max[k]; }
        }
        if ax < 0 { return -1; }
        let axu = ax as usize;
        voxel[axu] += step[axu];
        t_max[axu] += t_delta[axu];
        ax
    };

    let mut last_axis: i32 = -1;
    let mut steps_left = MAX_STEPS;

    // Phase 1: skip past any solid voxels we're standing inside.
    while steps_left > 0 && in_solid(voxel) {
        let ax = argmin_step(&mut t_max, &mut voxel);
        if ax < 0 { return None; }
        last_axis = ax;
        steps_left -= 1;
        if !in_window(voxel) { return None; }
    }

    // Phase 2: first solid voxel we hit is the visible surface.
    while steps_left > 0 {
        if !in_window(voxel) { return None; }
        if in_solid(voxel) {
            let mut normal = [0i32; 3];
            if last_axis >= 0 {
                normal[last_axis as usize] = -step[last_axis as usize];
            }
            return Some(PickHit { voxel, normal });
        }
        let ax = argmin_step(&mut t_max, &mut voxel);
        if ax < 0 { return None; }
        last_axis = ax;
        steps_left -= 1;
    }
    None
}

fn sign_i32(x: f32) -> i32 {
    if x > 0.0 { 1 } else if x < 0.0 { -1 } else { 0 }
}

fn safe_inv(x: f32) -> f32 {
    if x.abs() < 1e-8 { 1e30 } else { 1.0 / x }
}
