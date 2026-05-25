use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::camera::{Camera, CameraUniform};
use crate::voxel::{
    World, MAT_AIR, MAT_SAND, MAT_GRASS, MAT_DIRT, MAT_STONE,
    MAT_WATER_L1, MAT_WATER_L8,
    MAT_WOOD, MAT_LEAVES, MAT_SNOW, MAT_LAVA, MAT_ICE, MAT_GLASS,
    MAT_COAL, MAT_IRON, MAT_GOLD, MAT_DIAMOND,
    MAT_WOOD_BIRCH, MAT_WOOD_PINE, MAT_LEAVES_BIRCH, MAT_LEAVES_PINE, MAT_LEAVES_AUTUMN,
    MAT_SMOKE, MAT_FIRE,
};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PaletteEntry([f32; 4]);

pub const MAX_REMOTE_PLAYERS: usize = 16;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PlayersBuf {
    count: u32,
    _pad: [u32; 3],
    positions: [[f32; 4]; MAX_REMOTE_PLAYERS],
}

const PALETTE_SIZE: usize = 256;

fn default_palette() -> [PaletteEntry; PALETTE_SIZE] {
    let mut p = [PaletteEntry([0.0; 4]); PALETTE_SIZE];
    p[MAT_AIR   as usize] = PaletteEntry([0.0, 0.0, 0.0, 0.0]);
    p[MAT_SAND  as usize] = PaletteEntry([0.94, 0.85, 0.55, 1.0]);
    p[MAT_GRASS as usize] = PaletteEntry([0.30, 0.65, 0.20, 1.0]);
    p[MAT_DIRT  as usize] = PaletteEntry([0.40, 0.27, 0.15, 1.0]);
    p[MAT_STONE as usize] = PaletteEntry([0.55, 0.55, 0.55, 1.0]);
    let water = [0.10, 0.45, 0.75, 1.0];
    for m in MAT_WATER_L1..=MAT_WATER_L8 {
        p[m as usize] = PaletteEntry(water);
    }
    p[MAT_WOOD as usize]   = PaletteEntry([0.42, 0.27, 0.13, 1.0]);
    p[MAT_LEAVES as usize] = PaletteEntry([0.18, 0.46, 0.16, 1.0]);
    p[MAT_SNOW as usize]   = PaletteEntry([0.95, 0.96, 0.99, 1.0]);
    p[MAT_LAVA as usize]   = PaletteEntry([1.40, 0.40, 0.10, 1.0]);
    p[MAT_ICE as usize]    = PaletteEntry([0.70, 0.85, 0.95, 1.0]);
    p[MAT_GLASS as usize]  = PaletteEntry([0.90, 0.96, 0.99, 1.0]);
    p[MAT_COAL as usize]    = PaletteEntry([0.10, 0.10, 0.12, 1.0]);
    p[MAT_IRON as usize]    = PaletteEntry([0.62, 0.55, 0.45, 1.0]);
    p[MAT_GOLD as usize]    = PaletteEntry([1.20, 0.90, 0.20, 1.0]);
    p[MAT_DIAMOND as usize] = PaletteEntry([0.55, 0.92, 0.98, 1.0]);
    p[MAT_WOOD_BIRCH as usize]    = PaletteEntry([0.85, 0.82, 0.72, 1.0]);
    p[MAT_WOOD_PINE as usize]     = PaletteEntry([0.25, 0.15, 0.08, 1.0]);
    p[MAT_LEAVES_BIRCH as usize]  = PaletteEntry([0.55, 0.78, 0.34, 1.0]);
    p[MAT_LEAVES_PINE as usize]   = PaletteEntry([0.10, 0.32, 0.12, 1.0]);
    p[MAT_LEAVES_AUTUMN as usize] = PaletteEntry([0.90, 0.42, 0.15, 1.0]);
    p[MAT_SMOKE as usize]   = PaletteEntry([0.65, 0.65, 0.70, 1.0]);
    p[MAT_FIRE as usize]    = PaletteEntry([1.60, 0.60, 0.10, 1.0]);
    p
}

pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub size: (u32, u32),

    camera_buf: wgpu::Buffer,
    bricks_buf: wgpu::Buffer,
    tile_mask_buf: wgpu::Buffer,
    chunk_mask_buf: wgpu::Buffer,
    palette_buf: wgpu::Buffer,
    tile_dirty_buf: wgpu::Buffer,
    players_buf: wgpu::Buffer,

    output_tex: wgpu::Texture,
    output_view: wgpu::TextureView,
    beam_tex: wgpu::Texture,
    beam_view: wgpu::TextureView,
    sampler: wgpu::Sampler,

    beam_bgl: wgpu::BindGroupLayout,
    beam_pipeline: wgpu::ComputePipeline,
    beam_bg: wgpu::BindGroup,

    compute_bgl: wgpu::BindGroupLayout,
    compute_pipeline: wgpu::ComputePipeline,
    compute_bg: wgpu::BindGroup,

    blit_bgl: wgpu::BindGroupLayout,
    blit_pipeline: wgpu::RenderPipeline,
    blit_bg: wgpu::BindGroup,
}

impl Renderer {
    pub fn new(window: Arc<Window>, world: &World) -> Self {
        let inner = window.inner_size();
        let (width, height) = (inner.width.max(1), inner.height.max(1));

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance
            .create_surface(window.clone())
            .expect("create_surface");

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("request_adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("voxel device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: 256 << 20, // 256 MB headroom
                    ..wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .expect("request_device");

        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: caps
                .present_modes
                .iter()
                .copied()
                .find(|m| matches!(m, wgpu::PresentMode::Immediate | wgpu::PresentMode::Mailbox))
                .unwrap_or(wgpu::PresentMode::Fifo),
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // -- buffers --
        let camera_init = CameraUniform::from_camera(&Camera::new(), width, height, 0.0, glam::IVec3::ZERO);
        let camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera"),
            contents: bytemuck::bytes_of(&camera_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bricks_bytes: &[u8] = bytemuck::cast_slice(&world.bricks);
        let bricks_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bricks"),
            contents: bricks_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let tile_mask_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tile_mask"),
            contents: bytemuck::cast_slice(&world.tile_mask),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let chunk_mask_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk_mask"),
            contents: bytemuck::cast_slice(&world.chunk_mask),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let palette = default_palette();
        let palette_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("palette"),
            contents: bytemuck::cast_slice(&palette),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Tile-dirty bitmask.
        let tile_dirty_words: u64 = ((3840u64 / 8 + 1) * (2160u64 / 8 + 1) + 31) / 32;
        let tile_dirty_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile_dirty"),
            size: tile_dirty_words * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Remote-player marker positions (multiplayer rendering).
        let players_init = PlayersBuf {
            count: 0,
            _pad: [0; 3],
            positions: [[0.0; 4]; MAX_REMOTE_PLAYERS],
        };
        let players_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("players"),
            contents: bytemuck::bytes_of(&players_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let (output_tex, output_view) = create_output_texture(&device, width, height);
        let (beam_tex, beam_view) = create_beam_texture(&device, width, height);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("blit sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // -- compute pipeline --
        let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Per-tile dirty bitmask for temporal-differential raymarch.
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Remote-player marker positions.
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let compute_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compute pl"),
            bind_group_layouts: &[&compute_bgl],
            push_constant_ranges: &[],
        });
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raymarch shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/raymarch.wgsl").into()),
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("raymarch pipeline"),
            layout: Some(&compute_pl),
            module: &compute_shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let compute_bg = make_compute_bg(
            &device, &compute_bgl, &camera_buf, &bricks_buf,
            &tile_mask_buf, &chunk_mask_buf, &palette_buf, &output_view, &beam_view,
            &tile_dirty_buf, &players_buf,
        );

        // -- beam pipeline (1/8-res coarse pre-pass) --
        let beam_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("beam bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let beam_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("beam pl"),
            bind_group_layouts: &[&beam_bgl],
            push_constant_ranges: &[],
        });
        let beam_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("beam shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/beam.wgsl").into()),
        });
        let beam_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("beam pipeline"),
            layout: Some(&beam_pl),
            module: &beam_shader,
            entry_point: Some("cs_beam"),
            compilation_options: Default::default(),
            cache: None,
        });
        let beam_bg = make_beam_bg(&device, &beam_bgl, &camera_buf, &chunk_mask_buf, &beam_view);

        // -- blit pipeline --
        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        let blit_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit pl"),
            bind_group_layouts: &[&blit_bgl],
            push_constant_ranges: &[],
        });
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/blit.wgsl").into()),
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit pipeline"),
            layout: Some(&blit_pl),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let blit_bg = make_blit_bg(&device, &blit_bgl, &output_view, &sampler);

        Self {
            device, queue, surface, config,
            size: (width, height),
            camera_buf, bricks_buf, tile_mask_buf, chunk_mask_buf, palette_buf,
            tile_dirty_buf, players_buf,
            output_tex, output_view, beam_tex, beam_view, sampler,
            beam_bgl, beam_pipeline, beam_bg,
            compute_bgl, compute_pipeline, compute_bg,
            blit_bgl, blit_pipeline, blit_bg,
        }
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 || (w, h) == self.size { return; }
        self.size = (w, h);
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);

        let (tex, view) = create_output_texture(&self.device, w, h);
        self.output_tex = tex;
        self.output_view = view;
        let (btex, bview) = create_beam_texture(&self.device, w, h);
        self.beam_tex = btex;
        self.beam_view = bview;

        self.compute_bg = make_compute_bg(
            &self.device, &self.compute_bgl, &self.camera_buf, &self.bricks_buf,
            &self.tile_mask_buf, &self.chunk_mask_buf, &self.palette_buf, &self.output_view, &self.beam_view,
            &self.tile_dirty_buf, &self.players_buf,
        );
        self.beam_bg = make_beam_bg(
            &self.device, &self.beam_bgl, &self.camera_buf, &self.chunk_mask_buf, &self.beam_view,
        );
        self.blit_bg = make_blit_bg(&self.device, &self.blit_bgl, &self.output_view, &self.sampler);
    }

    pub fn upload_world(&self, world: &mut World) {
        // Coalesced upload: walk the sorted dirty list into contiguous spans
        // and issue one write_buffer per span. Thousands of 72-byte writes
        // serialise through the wgpu staging path; merging into a handful of
        // multi-KB writes turns "death by a thousand DMAs" into ~free.
        if world.all_dirty {
            self.queue.write_buffer(&self.bricks_buf, 0, bytemuck::cast_slice(&world.bricks));
            self.queue.write_buffer(&self.tile_mask_buf, 0, bytemuck::cast_slice(&world.tile_mask));
            self.queue.write_buffer(&self.chunk_mask_buf, 0, bytemuck::cast_slice(&world.chunk_mask));
            world.all_dirty = false;
            world.dirty_bricks.clear();
            return;
        }
        if world.dirty_bricks.is_empty() { return; }
        world.dirty_bricks.sort_unstable();
        world.dirty_bricks.dedup();

        let stride = std::mem::size_of::<crate::voxel::Brick>() as u64;
        let mut i = 0usize;
        while i < world.dirty_bricks.len() {
            let start = world.dirty_bricks[i];
            let mut end = start;
            let mut j = i + 1;
            while j < world.dirty_bricks.len() && world.dirty_bricks[j] == end + 1 {
                end = world.dirty_bricks[j];
                j += 1;
            }
            let offset = start as u64 * stride;
            let slice = &world.bricks[start as usize ..= end as usize];
            self.queue.write_buffer(&self.bricks_buf, offset, bytemuck::cast_slice(slice));
            i = j;
        }
        self.queue.write_buffer(&self.tile_mask_buf, 0, bytemuck::cast_slice(&world.tile_mask));
        self.queue.write_buffer(&self.chunk_mask_buf, 0, bytemuck::cast_slice(&world.chunk_mask));
        world.dirty_bricks.clear();
    }

    pub fn upload_tile_dirty(&self, mask: &[u32]) {
        if mask.is_empty() { return; }
        self.queue.write_buffer(&self.tile_dirty_buf, 0, bytemuck::cast_slice(mask));
    }

    pub fn upload_players(&self, positions: &[(glam::Vec3, u32)]) {
        let mut data = PlayersBuf {
            count: 0,
            _pad: [0; 3],
            positions: [[0.0; 4]; MAX_REMOTE_PLAYERS],
        };
        let n = positions.len().min(MAX_REMOTE_PLAYERS);
        data.count = n as u32;
        for (i, (p, id)) in positions.iter().take(n).enumerate() {
            data.positions[i] = [p.x, p.y, p.z, *id as f32];
        }
        self.queue.write_buffer(&self.players_buf, 0, bytemuck::bytes_of(&data));
    }

    pub fn update_camera(&self, camera: &Camera, time: f32, world_origin_voxel: glam::IVec3) {
        let u = CameraUniform::from_camera(camera, self.size.0, self.size.1, time, world_origin_voxel);
        self.queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&u));
    }

    pub fn render(&mut self, any_dirty: bool) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let frame_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame"),
        });

        // Temporal-differential: if no tile is dirty, we still blit (so the
        // swapchain stays in sync), but skip the beam + raymarch compute
        // passes entirely. The output_tex from the prior frame is preserved.
        if any_dirty {
            // ---- beam pre-pass at 1/8 resolution ----
            {
                let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("beam"),
                    timestamp_writes: None,
                });
                cp.set_pipeline(&self.beam_pipeline);
                cp.set_bind_group(0, &self.beam_bg, &[]);
                let bx = ((self.size.0 + 7) / 8 + 7) / 8;
                let by = ((self.size.1 + 7) / 8 + 7) / 8;
                cp.dispatch_workgroups(bx, by, 1);
            }
            // ---- main raymarch (skips clean tiles in-shader) ----
            {
                let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("raymarch"),
                    timestamp_writes: None,
                });
                cp.set_pipeline(&self.compute_pipeline);
                cp.set_bind_group(0, &self.compute_bg, &[]);
                let gx = (self.size.0 + 7) / 8;
                let gy = (self.size.1 + 7) / 8;
                cp.dispatch_workgroups(gx, gy, 1);
            }
        }

        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &frame_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0, g: 0.0, b: 0.0, a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rp.set_pipeline(&self.blit_pipeline);
            rp.set_bind_group(0, &self.blit_bg, &[]);
            rp.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        Ok(())
    }
}

fn create_output_texture(device: &wgpu::Device, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("output"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn create_beam_texture(device: &wgpu::Device, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let bw = ((w + 7) / 8).max(1);
    let bh = ((h + 7) / 8).max(1);
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("beam"),
        size: wgpu::Extent3d { width: bw, height: bh, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn make_compute_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    camera_buf: &wgpu::Buffer,
    bricks_buf: &wgpu::Buffer,
    tile_mask_buf: &wgpu::Buffer,
    chunk_mask_buf: &wgpu::Buffer,
    palette_buf: &wgpu::Buffer,
    output_view: &wgpu::TextureView,
    beam_view: &wgpu::TextureView,
    tile_dirty_buf: &wgpu::Buffer,
    players_buf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: bricks_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: tile_mask_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: chunk_mask_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: palette_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(output_view) },
            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(beam_view) },
            wgpu::BindGroupEntry { binding: 7, resource: tile_dirty_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: players_buf.as_entire_binding() },
        ],
    })
}

fn make_beam_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    camera_buf: &wgpu::Buffer,
    chunk_mask_buf: &wgpu::Buffer,
    beam_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("beam bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: chunk_mask_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(beam_view) },
        ],
    })
}

fn make_blit_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("blit bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
        ],
    })
}
