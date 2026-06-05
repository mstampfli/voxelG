// Client application: window creation, input handling and the per-frame
// orchestration loop. Split out of main.rs so the frame loop, the server loop
// and process startup each live in their own module (checklist: hygiene).

use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use glam::{IVec3, Vec3};

use crate::camera::Camera;
use crate::net;
use crate::physics;
use crate::raycast;
use crate::renderer::Renderer;
use crate::temporal;
use crate::voxel::{
    self, World, MAT_STONE, MAT_SAND, MAT_WATER, MAT_WOOD, MAT_LEAVES, MAT_GLASS,
    MAT_LAVA, MAT_ICE, MAT_SNOW, MAT_SMOKE,
};

/// Upper bound on frame rate. The loop sleeps between frames (ControlFlow::
/// WaitUntil) instead of busy-spinning, so an idle scene no longer pins the GPU
/// at ~1800 fps; presentation is additionally vsync-paced by the swapchain.
const FRAME_CAP_HZ: f64 = 144.0;

/// Recenter the streaming window only once the camera has drifted at least this
/// many chunks off-centre. A deadband stops the window thrashing (and
/// regenerating an edge column) when the player walks back and forth across a
/// single chunk boundary (checklist: prefetch with hysteresis).
const STREAM_HYSTERESIS: i32 = 2;

/// Max finished chunks installed per frame. Caps how many bricks get marked
/// dirty (and uploaded) per frame so a chunk cross streams in over a handful of
/// frames instead of one big hitch (checklist: per-frame upload budget).
const CHUNK_INSTALL_BUDGET: u32 = 6;

#[derive(Default)]
struct Keys {
    forward: bool,
    back: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    sprint: bool,
}

pub struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    camera: Camera,
    world: World,
    keys: Keys,
    last_frame: Instant,
    start_time: Instant,
    physics_acc: f32,
    grabbed: bool,
    frames_since_log: u32,
    last_log: Instant,

    // Temporal differential bookkeeping.
    last_camera_pose: Option<(Vec3, f32, f32, f32)>,
    frames_since_full: u32,
    tile_dirty_mask: Vec<u32>,
    first_frame: bool,
    current_material: u8,

    // Multiplayer.
    net: Option<net::NetClient>,
    last_sent_pose: Option<(Vec3, f32, f32)>,
    last_net_send: Instant,
    remote_players: std::collections::HashMap<net::PlayerId, ([f32; 3], f32, f32)>,

    // Click queue. Consumed at the top of a frame after camera state for the
    // about-to-render frame is settled, so the raycast direction matches the
    // rendered crosshair.
    pending_clicks: Vec<winit::event::MouseButton>,
}

impl App {
    pub fn new(net: Option<net::NetClient>) -> Self {
        let mut world = World::new();
        world.fill_demo_terrain();
        // Spawn above the surface at the spawn column. The default y (80) is
        // often *below* the terrain/mountains, which renders as an opaque black
        // screen ("you see nothing"); lift the camera to clear ground + water.
        let mut camera = Camera::new();
        let s = voxel::sample_terrain(camera.pos.x, camera.pos.z, world.seed);
        let surface = s.h.max(s.water_top) as f32;
        camera.pos.y = surface + 10.0;
        Self {
            window: None,
            renderer: None,
            camera,
            world,
            keys: Keys::default(),
            last_frame: Instant::now(),
            start_time: Instant::now(),
            physics_acc: 0.0,
            last_camera_pose: None,
            frames_since_full: 0,
            tile_dirty_mask: Vec::with_capacity(2048),
            first_frame: true,
            current_material: MAT_STONE,
            grabbed: false,
            frames_since_log: 0,
            last_log: Instant::now(),
            net,
            last_sent_pose: None,
            last_net_send: Instant::now(),
            remote_players: std::collections::HashMap::new(),
            pending_clicks: Vec::new(),
        }
    }

    fn world_origin_voxel(&self) -> IVec3 {
        IVec3::new(
            self.world.world_origin_chunk.x * voxel::STORAGE_CHUNK_VOXELS as i32,
            0,
            self.world.world_origin_chunk.y * voxel::STORAGE_CHUNK_VOXELS as i32,
        )
    }

    fn request_redraw(&self) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    fn poll_net(&mut self) {
        let (msgs, my_id) = match self.net.as_mut() {
            Some(net) => (net.drain(), net.my_id),
            None => return,
        };
        for msg in msgs {
            match msg {
                net::Message::PlayerUpdate { id, pos, yaw, pitch } => {
                    if Some(id) != my_id {
                        self.remote_players.insert(id, (pos, yaw, pitch));
                    }
                }
                net::Message::PlayerJoin { id } => {
                    log::info!("player {} joined", id);
                }
                net::Message::PlayerLeave { id } => {
                    log::info!("player {} left", id);
                    self.remote_players.remove(&id);
                }
                net::Message::VoxelEdit { wx, wy, wz, mat } => {
                    self.world.apply_edit(wx, wy, wz, mat);
                }
                net::Message::Explode { cx, cy, cz, radius, mat } => {
                    self.apply_sphere(cx, cy, cz, radius, mat);
                }
                net::Message::JoinAck { your_id, seed } => {
                    log::info!("joined as player {} (seed {})", your_id, seed);
                }
                _ => {}
            }
        }
    }

    fn maybe_send_pose(&mut self, now: Instant) {
        let Some(net) = self.net.as_ref() else { return; };
        let pose = (self.camera.pos, self.camera.yaw, self.camera.pitch);
        let changed = self.last_sent_pose.map_or(true, |p| p != pose);
        // Cap at 20 Hz.
        if changed && (now - self.last_net_send) >= Duration::from_millis(50) {
            net.send(net::Message::PlayerUpdate {
                id: net.my_id.unwrap_or(0),
                pos: pose.0.to_array(),
                yaw: pose.1,
                pitch: pose.2,
            });
            self.last_sent_pose = Some(pose);
            self.last_net_send = now;
        }
    }

    fn broadcast_edit_world(&self, wx: i32, wy: i32, wz: i32, mat: u8) {
        let Some(net) = self.net.as_ref() else { return; };
        net.send(net::Message::VoxelEdit { wx, wy, wz, mat });
    }

    /// Expand a sphere-of-impact into the persistent edit log. Used both for
    /// the local player's left-click and for received `Explode` messages.
    fn apply_sphere(&mut self, cx: i32, cy: i32, cz: i32, radius: u8, mat: u8) {
        let r = radius as i32;
        let r2 = r * r;
        for dy in -r..=r {
            for dx in -r..=r {
                for dz in -r..=r {
                    if dx * dx + dy * dy + dz * dz > r2 { continue; }
                    self.world.apply_edit(cx + dx, cy + dy, cz + dz, mat);
                }
            }
        }
    }

    fn grab(&mut self, on: bool) {
        let Some(win) = &self.window else { return; };
        if on {
            let _ = win
                .set_cursor_grab(CursorGrabMode::Confined)
                .or_else(|_| win.set_cursor_grab(CursorGrabMode::Locked));
            win.set_cursor_visible(false);
        } else {
            let _ = win.set_cursor_grab(CursorGrabMode::None);
            win.set_cursor_visible(true);
        }
        self.grabbed = on;
    }

    fn handle_key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, pressed: bool) {
        match code {
            KeyCode::KeyW => self.keys.forward = pressed,
            KeyCode::KeyS => self.keys.back = pressed,
            KeyCode::KeyA => self.keys.left = pressed,
            KeyCode::KeyD => self.keys.right = pressed,
            KeyCode::Space => self.keys.up = pressed,
            KeyCode::ShiftLeft | KeyCode::ControlLeft => self.keys.down = pressed,
            KeyCode::AltLeft => self.keys.sprint = pressed,
            KeyCode::Digit1 if pressed => self.current_material = MAT_STONE,
            KeyCode::Digit2 if pressed => self.current_material = MAT_SAND,
            KeyCode::Digit3 if pressed => self.current_material = MAT_WATER,
            KeyCode::Digit4 if pressed => self.current_material = MAT_WOOD,
            KeyCode::Digit5 if pressed => self.current_material = MAT_LEAVES,
            KeyCode::Digit6 if pressed => self.current_material = MAT_GLASS,
            KeyCode::Digit7 if pressed => self.current_material = MAT_LAVA,
            KeyCode::Digit8 if pressed => self.current_material = MAT_ICE,
            KeyCode::Digit9 if pressed => self.current_material = MAT_SNOW,
            KeyCode::Digit0 if pressed => self.current_material = MAT_SMOKE,
            // Test-fire: drop a lava block 15 voxels ahead with no raycast.
            KeyCode::KeyT if pressed => {
                let fwd = self.camera.forward();
                let p = self.camera.pos + fwd * 15.0;
                self.world.apply_edit(
                    p.x.floor() as i32, p.y.floor() as i32, p.z.floor() as i32, voxel::MAT_LAVA,
                );
            }
            KeyCode::Escape if pressed => {
                if self.grabbed {
                    self.grab(false);
                } else {
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    /// Run picking for queued clicks against the current (about-to-render)
    /// camera state and apply the resulting edits.
    fn consume_clicks(&mut self) {
        if self.pending_clicks.is_empty() {
            return;
        }
        use winit::event::MouseButton;
        let clicks = std::mem::take(&mut self.pending_clicks);
        let world_origin = self.world_origin_voxel();
        for button in clicks {
            let Some(hit) = raycast::raycast(
                self.camera.pos, self.camera.forward(), &self.world, world_origin,
            ) else { continue; };
            let v = hit.voxel;
            match button {
                MouseButton::Left => {
                    let (cx, cy, cz) = (v[0], v[1], v[2]);
                    let r: u8 = 2;
                    self.apply_sphere(cx, cy, cz, r, voxel::MAT_AIR);
                    if let Some(net) = self.net.as_ref() {
                        net.send(net::Message::Explode {
                            cx, cy, cz, radius: r, mat: voxel::MAT_AIR,
                        });
                    }
                }
                MouseButton::Right => {
                    let wx = v[0] + hit.normal[0];
                    let wy = v[1] + hit.normal[1];
                    let wz = v[2] + hit.normal[2];
                    self.world.apply_edit(wx, wy, wz, self.current_material);
                    self.broadcast_edit_world(wx, wy, wz, self.current_material);
                }
                _ => {}
            }
        }
    }

    /// The per-frame orchestration: input integration, streaming, physics,
    /// temporal-differential tile selection, GPU upload + render.
    fn render_frame(&mut self) {
        if self.renderer.is_none() {
            return;
        }

        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(1.0 / 30.0);
        self.last_frame = now;

        // Pump network in/out before re-borrowing renderer.
        self.poll_net();
        self.maybe_send_pose(now);

        let speed = if self.keys.sprint { 4.0 } else { 1.0 };
        let f = (self.keys.forward as i32 - self.keys.back as i32) as f32;
        let r = (self.keys.right as i32 - self.keys.left as i32) as f32;
        let u = (self.keys.up as i32 - self.keys.down as i32) as f32;
        self.camera.translate_local(dt, f * speed, r * speed, u * speed);

        // Streaming: keep the loaded window centred on the camera, but with a
        // hysteresis deadband so boundary oscillation doesn't thrash regen.
        let cur_origin = self.world.world_origin_chunk;
        let target_origin = voxel::World::target_origin_chunk(self.camera.pos);
        let drift = target_origin - cur_origin;
        if drift.x.abs() >= STREAM_HYSTERESIS || drift.y.abs() >= STREAM_HYSTERESIS {
            self.world.shift_origin(target_origin);
        }
        // Install async-generated chunks under a per-frame budget. Generation
        // itself runs on the worker pool, so the frame never blocks on noise.
        self.world.install_finished_chunks(CHUNK_INSTALL_BUDGET);

        // Raycast queued clicks against the now-settled camera.
        self.consume_clicks();

        // Fixed-step physics.
        self.physics_acc += dt;
        let step = 1.0 / 30.0;
        let mut steps = 0;
        while self.physics_acc >= step && steps < 4 {
            physics::tick(&mut self.world);
            self.physics_acc -= step;
            steps += 1;
        }

        // ---- temporal differential ----
        let cur_pose = (self.camera.pos, self.camera.yaw, self.camera.pitch, self.camera.fov_y);
        let camera_changed = self.last_camera_pose.map_or(true, |p| p != cur_pose);
        let physics_changed = self.world.all_dirty || !self.world.dirty_bricks.is_empty();
        self.frames_since_full += 1;
        let force_full = self.first_frame
            || self.frames_since_full >= 15
            || self.world.all_dirty
            || camera_changed;
        let any_dirty = force_full || physics_changed;

        let (rw, rh) = self.renderer.as_ref().unwrap().size;
        let tiles_w = (rw + 7) / 8;
        let tiles_h = (rh + 7) / 8;
        let word_count = ((tiles_w * tiles_h) as usize + 31) / 32;
        self.tile_dirty_mask.clear();
        self.tile_dirty_mask.resize(word_count, 0);

        if force_full {
            for w in self.tile_dirty_mask.iter_mut() { *w = u32::MAX; }
            self.frames_since_full = 0;
        } else if physics_changed {
            // Project the dirty bricks straight from the world's dirty list —
            // no per-frame clone. upload_world (below) clears the list after.
            let world = &self.world;
            let camera = &self.camera;
            let mask = &mut self.tile_dirty_mask;
            for &bi in &world.dirty_bricks {
                temporal::project_brick_to_tiles(
                    bi, world, camera, rw, rh, tiles_w, tiles_h, mask,
                );
            }
        }

        let renderer = self.renderer.as_mut().unwrap();
        renderer.upload_world(&mut self.world);
        let t = (now - self.start_time).as_secs_f32();
        renderer.update_camera(&self.camera, t, self.world.world_origin_voxel());
        let players_vec: Vec<(Vec3, u32)> = self.remote_players
            .iter()
            .map(|(id, (pos, _yaw, _pitch))| (Vec3::from_array(*pos), *id))
            .collect();
        renderer.upload_players(&players_vec);
        if any_dirty {
            renderer.upload_tile_dirty(&self.tile_dirty_mask);
        }
        match renderer.render(any_dirty) {
            Ok(()) => {}
            // Transient surface states: reconfigure and try again next frame.
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                let (w, h) = renderer.surface_size;
                renderer.resize(w.max(1), h.max(1));
            }
            // The frame simply timed out acquiring the swapchain image; skip it.
            Err(wgpu::SurfaceError::Timeout) => {}
            // Out of GPU memory is unrecoverable — log and bail cleanly.
            Err(wgpu::SurfaceError::OutOfMemory) => {
                log::error!("GPU out of memory — exiting");
                if let Some(w) = &self.window {
                    // Best-effort: ask the loop to exit on the next pump.
                    w.request_redraw();
                }
                self.renderer = None;
            }
        }
        self.last_camera_pose = Some(cur_pose);
        self.first_frame = false;

        self.frames_since_log += 1;
        if (now - self.last_log).as_secs_f32() >= 1.0 {
            log::info!("fps {:>4}  cam {:?}", self.frames_since_log, self.camera.pos);
            self.frames_since_log = 0;
            self.last_log = now;
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("voxel")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                log::error!("failed to create window: {e}");
                event_loop.exit();
                return;
            }
        };
        let renderer = match Renderer::new(window.clone(), &self.world) {
            Ok(r) => r,
            Err(e) => {
                log::error!("failed to initialise renderer: {e}");
                event_loop.exit();
                return;
            }
        };
        self.window = Some(window);
        self.renderer = Some(renderer);
        self.grab(true);
        self.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(r) = self.renderer.as_mut() {
                    r.resize(size.width, size.height);
                }
                self.request_redraw();
            }
            WindowEvent::Focused(false) => self.grab(false),
            WindowEvent::MouseInput { state: ElementState::Pressed, button, .. } => {
                if !self.grabbed {
                    self.grab(true);
                } else {
                    self.pending_clicks.push(button);
                }
                self.request_redraw();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(code) = event.physical_key {
                    self.handle_key(event_loop, code, pressed);
                }
                self.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                self.render_frame();
                if self.renderer.is_none() {
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        if !self.grabbed {
            return;
        }
        if let DeviceEvent::MouseMotion { delta } = event {
            self.camera.rotate(delta.0 as f32, delta.1 as f32);
            // Render promptly on look so mouselook feels responsive rather than
            // waiting for the next paced heartbeat.
            self.request_redraw();
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Explicit frame pacing: instead of busy-spinning (ControlFlow::Poll +
        // unconditional redraw), sleep until the next frame is due. The scene
        // animates continuously (sky / water / foliage / clouds), so we still
        // redraw every paced tick — but capped, and the CPU idles in between.
        if self.window.is_none() {
            return;
        }
        let frame = Duration::from_secs_f64(1.0 / FRAME_CAP_HZ);
        let since = self.last_frame.elapsed();
        if since >= frame {
            self.request_redraw();
            event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + frame));
        } else {
            event_loop.set_control_flow(ControlFlow::WaitUntil(self.last_frame + frame));
        }
    }
}

/// Build the winit event loop and run the client application.
pub fn run_client(net: Option<net::NetClient>) {
    let event_loop = EventLoop::new().expect("event loop");
    // Frame-paced: the loop wakes on input or at the next scheduled frame
    // (see App::about_to_wait), not in a tight Poll spin.
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App::new(net);
    if let Err(e) = event_loop.run_app(&mut app) {
        log::error!("event loop exited with error: {e}");
    }
}
