// GPU-driven chunk noise generation.
//
// Pumps `world.regen_queue` end-to-end:
//   - drain → batch up to N requests
//   - dispatch compute (one workgroup per brick, 64 threads = 1 voxel each)
//   - copy GPU output → CPU-mappable staging buffer
//   - submit, map_async, poll
//   - on map ready: read bricks, paint trees (CPU post-pass), send into
//     world.gen_result_tx so the existing apply step picks them up.
//
// Single in-flight dispatch (pipeline depth 1). Each frame either submits a
// new batch (if free) or just polls + drains the previous one. The main
// frame thread NEVER blocks on the GPU.

use std::sync::{Arc, Mutex};

use crate::voxel::{self, paint_trees_into_bricks_cached, Brick, World, STORAGE_CHUNK_BRICKS, TreeSpec};

// Fixed-size batching. Adaptive was a bad idea — ping-pong'd between
// extremes and tanked the average. Hand-tuned middle ground:
//   batch=24 dispatches → ~1.5ms GPU gen per frame
//   apply_cap=8 → CPU never bursts more than 8 chunks of apply work
//   rayon CPU workers stay at 2 in main.rs
// Throughput ≈ (24 GPU + 8*2 CPU) per frame ≈ 8000 chunks/sec at 200 FPS.
const BATCH_SIZE: usize = 24;
const APPLY_CAP_PER_FRAME: usize = 8;
const BRICKS_PER_CHUNK: usize =
    (STORAGE_CHUNK_BRICKS * STORAGE_CHUNK_BRICKS * STORAGE_CHUNK_BRICKS) as usize;
const BRICK_BYTES: usize = std::mem::size_of::<Brick>();

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GenRequest {
    world_cx: i32,
    world_cy: i32,
    world_cz: i32,
    lod: i32, // 0 = full octaves, 1 = half (distant chunks)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GenParams {
    seed_x: f32,
    seed_z: f32,
    sea_level: f32,
    _pad: f32,
}

#[derive(Clone, Copy, PartialEq)]
enum MapState {
    Idle,
    Pending,
    Ready,
}

struct InFlight {
    requests: Vec<(u32, glam::IVec3)>,
    map_state: Arc<Mutex<MapState>>,
}

pub struct GpuTerrainGen {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    requests_buf: wgpu::Buffer,
    output_buf: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    inflight: Option<InFlight>,
    seed: u64,
}

impl GpuTerrainGen {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, seed: u64) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain_gen.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/terrain_gen.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain_gen bgl"),
            entries: &[
                // 0: requests (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: out_bricks (storage, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain_gen pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("terrain_gen pipe"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_gen"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let requests_size = (BATCH_SIZE * std::mem::size_of::<GenRequest>()) as u64;
        let requests_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain_gen requests"),
            size: requests_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_size = (BATCH_SIZE * BRICKS_PER_CHUNK * BRICK_BYTES) as u64;
        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain_gen output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain_gen staging"),
            size: output_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let (s_x, s_z) = voxel::seed_offset_xz(seed);
        let params = GenParams {
            seed_x: s_x,
            seed_z: s_z,
            sea_level: 180.0,
            _pad: 0.0,
        };
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain_gen params"),
            size: std::mem::size_of::<GenParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        Self {
            pipeline,
            bind_group_layout,
            requests_buf,
            output_buf,
            staging_buf,
            params_buf,
            inflight: None,
            seed,
        }
    }

    /// Drive one frame's GPU gen work. Non-blocking — either submits a new
    /// batch, or polls + drains a previous one. Result bricks get pushed
    /// into `world.gen_result_tx` so `process_regen_queue` picks them up.
    pub fn tick(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, world: &mut World) {
        // Always poll the device so any pending map_async callback can fire.
        let _ = device.poll(wgpu::Maintain::Poll);

        // 1) If we have an in-flight dispatch, see if it's ready.
        let take_inflight = match &self.inflight {
            Some(inf) => *inf.map_state.lock().unwrap() == MapState::Ready,
            None => false,
        };
        if take_inflight {
            let inf = self.inflight.take().unwrap();
            let n = inf.requests.len();
            let bytes_needed = n * BRICKS_PER_CHUNK * BRICK_BYTES;
            let range = self.staging_buf.slice(0..bytes_needed as u64);
            let data = range.get_mapped_range();
            // Cap how many results we send into world.gen_result_tx per
            // frame — keeps the downstream CPU apply burst small. The
            // unsent ones get pushed back into world.regen_queue (front)
            // so they're re-submitted to the GPU next frame.
            let bricks_per_chunk = BRICKS_PER_CHUNK;
            let send_count = inf.requests.len().min(APPLY_CAP_PER_FRAME);
            // Main thread: just memcpy the readback bytes into owned Vecs
            // (cheap, ~36 KB each). Move them onto a rayon thread which
            // does the expensive paint_trees pass and forwards results to
            // world.gen_result_tx for apply on the main thread.
            // This is what kills the per-frame apply-burst hitch: the
            // ~1-2 ms × N chunks paint work is no longer on the frame
            // path.
            let mut owned: Vec<(u32, glam::IVec3, Vec<Brick>)> = Vec::with_capacity(send_count);
            for i in 0..send_count {
                let (slot, want) = inf.requests[i];
                let off = i * bricks_per_chunk * BRICK_BYTES;
                let chunk_bytes = &data[off..off + bricks_per_chunk * BRICK_BYTES];
                let bricks: &[Brick] = bytemuck::cast_slice(chunk_bytes);
                owned.push((slot, want, bricks.to_vec()));
            }
            let tx = world.gen_result_tx.clone();
            let seed = self.seed;
            rayon::spawn(move || {
                let mut tree_cache: std::collections::HashMap<glam::IVec2, Vec<TreeSpec>> =
                    std::collections::HashMap::with_capacity(64);
                for (slot, want, mut scratch) in owned {
                    paint_trees_into_bricks_cached(want, &mut scratch, seed, &mut tree_cache);
                    // Persist for next visit. Tiny disk write (36 KB),
                    // doesn't block the channel send below.
                    crate::disk_cache::save(seed, want, &scratch);
                    let _ = tx.send((slot, want, scratch));
                }
            });
            // Anything we didn't send gets re-queued. push_front preserves
            // priority — we want the unfinished work next.
            for i in (send_count..inf.requests.len()).rev() {
                world.regen_queue.push_front(inf.requests[i]);
                world.gen_in_flight = world.gen_in_flight.saturating_sub(1);
            }
            let _ = n;
            drop(data);
            self.staging_buf.unmap();
        }

        // 2) If idle and queue has work, submit a new batch.
        if self.inflight.is_none() && !world.regen_queue.is_empty() {
            self.submit_batch(device, queue, world);
        }
    }

    fn submit_batch(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, world: &mut World) {
        let mut requests: Vec<(u32, glam::IVec3)> = Vec::with_capacity(BATCH_SIZE);
        let mut req_buf: Vec<GenRequest> = Vec::with_capacity(BATCH_SIZE);
        while requests.len() < BATCH_SIZE {
            let Some((slot, want)) = world.regen_queue.pop_front() else { break; };
            if world.slot_world_chunk[slot as usize] != Some(want) { continue; }
            // Disk cache hit? Skip GPU + tree paint entirely — just send
            // the saved scratch through the existing channel. CPU apply
            // still runs on the main thread (cheap, ~0.1 ms / chunk).
            if let Some(bricks) = crate::disk_cache::try_load(self.seed, want) {
                let _ = world.gen_result_tx.send((slot, want, bricks));
                continue;
            }
            requests.push((slot, want));
            req_buf.push(GenRequest {
                world_cx: want.x,
                world_cy: want.y,
                world_cz: want.z,
                lod: 0,
            });
            world.gen_in_flight += 1;
        }
        if requests.is_empty() { return; }

        let n = requests.len() as u32;
        // Upload request list (only the bytes used).
        queue.write_buffer(&self.requests_buf, 0, bytemuck::cast_slice(&req_buf));

        // Build bind group.
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain_gen bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.requests_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.output_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("terrain_gen encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("terrain_gen pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            // Workgroup layout (8, 8, 8 * n): brick_x, brick_y, brick_z * batch + chunk_in_batch.
            cpass.dispatch_workgroups(8, 8, 8 * n);
        }
        let bytes = (requests.len() * BRICKS_PER_CHUNK * BRICK_BYTES) as u64;
        encoder.copy_buffer_to_buffer(&self.output_buf, 0, &self.staging_buf, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));

        // Start async map. Callback sets MapState::Ready.
        let map_state = Arc::new(Mutex::new(MapState::Pending));
        let map_state_cb = Arc::clone(&map_state);
        self.staging_buf.slice(0..bytes).map_async(wgpu::MapMode::Read, move |res| {
            let mut s = map_state_cb.lock().unwrap();
            *s = match res {
                Ok(()) => MapState::Ready,
                Err(_) => MapState::Idle,
            };
        });
        self.inflight = Some(InFlight { requests, map_state });
    }
}
