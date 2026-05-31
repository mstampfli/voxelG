// On-disk chunk cache.
//
// A chunk is 32×32×32 voxels = 512 bricks × 72 bytes = 36 KB raw.
// Once a chunk has been generated (GPU noise + tree paint), we save the
// final brick array to disk under a per-seed directory. On revisit we
// skip both the GPU compute AND the tree paint pass and just `read` the
// file — much faster than regenerating.
//
// File layout: `~/.voxel_cache/<seed_hex>/<wx>_<wy>_<wz>.brk`
// (one file per chunk to keep deletes/regen cheap.)

use std::path::PathBuf;

use crate::voxel::{Brick, STORAGE_CHUNK_BRICKS};

const BRICKS_PER_CHUNK: usize =
    (STORAGE_CHUNK_BRICKS * STORAGE_CHUNK_BRICKS * STORAGE_CHUNK_BRICKS) as usize;
const CHUNK_BYTES: usize = BRICKS_PER_CHUNK * std::mem::size_of::<Brick>();

pub fn cache_root(seed: u64) -> PathBuf {
    let mut p = std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    p.push(".voxel_cache");
    p.push(format!("{:016x}", seed));
    p
}

fn chunk_path(seed: u64, chunk: glam::IVec3) -> PathBuf {
    let mut p = cache_root(seed);
    p.push(format!("{}_{}_{}.brk", chunk.x, chunk.y, chunk.z));
    p
}

/// Try to load a chunk's bricks from disk. Returns None on miss / IO error.
pub fn try_load(seed: u64, chunk: glam::IVec3) -> Option<Vec<Brick>> {
    let path = chunk_path(seed, chunk);
    let bytes = std::fs::read(&path).ok()?;
    if bytes.len() != CHUNK_BYTES { return None; }
    let bricks: Vec<Brick> = bytemuck::cast_slice(&bytes).to_vec();
    Some(bricks)
}

/// Persist a chunk's bricks to disk. Fire-and-forget; errors are logged
/// but never propagate (a failed save just means re-gen next time).
pub fn save(seed: u64, chunk: glam::IVec3, bricks: &[Brick]) {
    debug_assert_eq!(bricks.len(), BRICKS_PER_CHUNK);
    let path = chunk_path(seed, chunk);
    let parent = path.parent().unwrap();
    if let Err(e) = std::fs::create_dir_all(parent) {
        log::warn!("disk cache: mkdir {parent:?} failed: {e}");
        return;
    }
    let bytes: &[u8] = bytemuck::cast_slice(bricks);
    if let Err(e) = std::fs::write(&path, bytes) {
        log::warn!("disk cache: write {path:?} failed: {e}");
    }
}
