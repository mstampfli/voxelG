// Temporal-differential helpers.
//
// For each dirty brick we project its WORLD-space AABB into screen space and
// mark the 8×8 tiles it covers as dirty. The shader's per-thread early-out
// then skips re-tracing tiles that haven't changed.
//
// Note we project from the world brick coord (slot_world_chunk → world
// brick), NOT from the local slot brick coord — the camera lives in world
// coords now so they must match.

use glam::Vec3;

use crate::camera::Camera;
use crate::voxel::*;

pub fn project_brick_to_tiles(
    bi: u32,
    world: &World,
    camera: &Camera,
    w: u32,
    h: u32,
    tiles_w: u32,
    tiles_h: u32,
    mask: &mut [u32],
) {
    let (bx, by, bz) = brick_coords(bi);

    let slot_cx = bx / STORAGE_CHUNK_BRICKS;
    let slot_cy = by / STORAGE_CHUNK_BRICKS;
    let slot_cz = bz / STORAGE_CHUNK_BRICKS;
    let slot_idx = storage_chunk_idx(slot_cx, slot_cy, slot_cz) as usize;
    let world_chunk = match world.slot_world_chunk[slot_idx] {
        Some(wc) => wc,
        None => return, // stale slot — nothing to project
    };
    let bx_in_chunk = bx % STORAGE_CHUNK_BRICKS;
    let by_in_chunk = by % STORAGE_CHUNK_BRICKS;
    let bz_in_chunk = bz % STORAGE_CHUNK_BRICKS;
    let world_brick_x = world_chunk.x * STORAGE_CHUNK_BRICKS as i32 + bx_in_chunk as i32;
    let world_brick_y = world_chunk.y * STORAGE_CHUNK_BRICKS as i32 + by_in_chunk as i32;
    let world_brick_z = world_chunk.z * STORAGE_CHUNK_BRICKS as i32 + bz_in_chunk as i32;

    let base = Vec3::new(
        (world_brick_x * BRICK_DIM as i32) as f32,
        (world_brick_y * BRICK_DIM as i32) as f32,
        (world_brick_z * BRICK_DIM as i32) as f32,
    );

    let fwd = camera.forward();
    let right = camera.right();
    let up = camera.up();
    let tan_half = (camera.fov_y * 0.5).tan();
    let aspect = w as f32 / h as f32;
    let wf = w as f32;
    let hf = h as f32;

    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;
    let mut any_visible = false;
    let mut any_behind = false;

    let bd = BRICK_DIM as f32;
    for ci in 0..8 {
        let cx = (ci & 1) as f32 * bd;
        let cy = ((ci >> 1) & 1) as f32 * bd;
        let cz = ((ci >> 2) & 1) as f32 * bd;
        let corner = base + Vec3::new(cx, cy, cz);
        let d = corner - camera.pos;
        let z_cam = d.dot(fwd);
        if z_cam <= 0.1 {
            any_behind = true;
            continue;
        }
        let x_cam = d.dot(right);
        let y_cam = d.dot(up);
        let inv_z = 1.0 / (z_cam * tan_half);
        let ndc_x = (x_cam * inv_z) / aspect;
        let ndc_y = y_cam * inv_z;
        let px = ((ndc_x + 1.0) * 0.5 * wf) as i32;
        let py = ((1.0 - ndc_y) * 0.5 * hf) as i32;
        if px < min_x { min_x = px; }
        if py < min_y { min_y = py; }
        if px > max_x { max_x = px; }
        if py > max_y { max_y = py; }
        any_visible = true;
    }

    if !any_visible { return; }
    if any_behind {
        min_x = 0; min_y = 0;
        max_x = w as i32 - 1; max_y = h as i32 - 1;
    }

    if max_x < 0 || max_y < 0 || min_x >= w as i32 || min_y >= h as i32 { return; }
    let tx_min = (min_x.max(0) as u32) / 8;
    let ty_min = (min_y.max(0) as u32) / 8;
    let tx_max = (max_x.min(w as i32 - 1) as u32) / 8;
    let ty_max = (max_y.min(h as i32 - 1) as u32) / 8;
    let tx_max = tx_max.min(tiles_w.saturating_sub(1));
    let ty_max = ty_max.min(tiles_h.saturating_sub(1));

    for ty in ty_min..=ty_max {
        let row = ty * tiles_w;
        for tx in tx_min..=tx_max {
            let idx = row + tx;
            mask[(idx >> 5) as usize] |= 1u32 << (idx & 31);
        }
    }
}
