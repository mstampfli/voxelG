// CPU-side DDA raycast for picking: which voxel is the player clicking on?
// Mirrors the GPU traversal logic but doesn't need the hierarchy because we
// only fire one ray per click, not one per pixel.

use glam::Vec3;

use crate::voxel::*;

pub struct PickHit {
    pub voxel: [i32; 3],
    pub normal: [i32; 3],
}

const MAX_STEPS: i32 = 512;

pub fn raycast(origin: Vec3, dir: Vec3, world: &World) -> Option<PickHit> {
    let world_max = Vec3::new(
        WORLD_VOXELS_X as f32,
        WORLD_VOXELS_Y as f32,
        WORLD_VOXELS_Z as f32,
    );
    let inv_dir = Vec3::new(
        safe_inv(dir.x),
        safe_inv(dir.y),
        safe_inv(dir.z),
    );
    let t0 = (Vec3::ZERO - origin) * inv_dir;
    let t1 = (world_max - origin) * inv_dir;
    let tmin = t0.min(t1);
    let tmax = t0.max(t1);
    let t_enter = tmin.max_element().max(0.0);
    let t_exit = tmax.min_element();
    if t_enter >= t_exit || t_exit < 0.0 {
        return None;
    }

    let mut p = origin + dir * (t_enter + 1e-3);
    p = p.clamp(Vec3::splat(1e-3), world_max - 1e-3);
    let mut voxel = [p.x.floor() as i32, p.y.floor() as i32, p.z.floor() as i32];
    let step = [sign_i32(dir.x), sign_i32(dir.y), sign_i32(dir.z)];
    let t_delta = [inv_dir.x.abs(), inv_dir.y.abs(), inv_dir.z.abs()];

    let mut t_max = [0.0f32; 3];
    for k in 0..3 {
        let nxt = if step[k] > 0 { (voxel[k] + 1) as f32 } else { voxel[k] as f32 };
        let o = [origin.x, origin.y, origin.z][k];
        let inv = [inv_dir.x, inv_dir.y, inv_dir.z][k];
        t_max[k] = (nxt - o) * inv;
    }

    let mut last_axis: i32 = -1;
    for _ in 0..MAX_STEPS {
        if voxel[0] < 0 || voxel[0] >= WORLD_VOXELS_X as i32
         || voxel[1] < 0 || voxel[1] >= WORLD_VOXELS_Y as i32
         || voxel[2] < 0 || voxel[2] >= WORLD_VOXELS_Z as i32 {
            return None;
        }
        let bx = (voxel[0] as u32) / BRICK_DIM;
        let by = (voxel[1] as u32) / BRICK_DIM;
        let bz = (voxel[2] as u32) / BRICK_DIM;
        let bi = brick_idx(bx, by, bz) as usize;
        let b = &world.bricks[bi];
        if b.occupancy != 0 {
            let lx = voxel[0] as u32 - bx * BRICK_DIM;
            let ly = voxel[1] as u32 - by * BRICK_DIM;
            let lz = voxel[2] as u32 - bz * BRICK_DIM;
            let vi = brick_voxel_idx(lx, ly, lz);
            if (b.occupancy & (1u64 << vi)) != 0 {
                let mut normal = [0i32; 3];
                if last_axis >= 0 {
                    normal[last_axis as usize] = -step[last_axis as usize];
                }
                return Some(PickHit { voxel, normal });
            }
        }

        // Argmin tMax
        let mut ax = 0;
        let mut m = t_max[0];
        if t_max[1] < m { ax = 1; m = t_max[1]; }
        if t_max[2] < m { ax = 2; }
        voxel[ax] += step[ax];
        t_max[ax] += t_delta[ax];
        last_axis = ax as i32;
    }
    None
}

fn sign_i32(x: f32) -> i32 {
    if x > 0.0 { 1 } else if x < 0.0 { -1 } else { 0 }
}

fn safe_inv(x: f32) -> f32 {
    if x.abs() < 1e-8 { 1e30 } else { 1.0 / x }
}
