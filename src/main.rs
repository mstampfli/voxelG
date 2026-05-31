use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

mod camera;
mod disk_cache;
mod gpu_terrain;
mod net;
mod physics;
mod raycast;
mod renderer;
mod temporal;
mod voxel;

use camera::Camera;
use glam::{IVec3, Vec3};
use renderer::Renderer;
use voxel::{
    World, MAT_STONE, MAT_SAND, MAT_WATER, MAT_WOOD, MAT_LEAVES, MAT_GLASS,
    MAT_LAVA, MAT_ICE, MAT_SNOW, MAT_SMOKE,
};

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

struct App {
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

    // Click queue. The MouseInput handler enqueues here; RedrawRequested
    // consumes after all camera state for the about-to-render frame is
    // settled. Guarantees the raycast direction MATCHES the rendered
    // crosshair, so the placed marker lands exactly under the crosshair the
    // player sees.
    pending_clicks: Vec<winit::event::MouseButton>,

    // GPU chunk-noise generator. Created lazily once the renderer (and
    // device + queue) exists.
    gpu_terrain: Option<gpu_terrain::GpuTerrainGen>,
}

impl App {
    fn new(net: Option<net::NetClient>) -> Self {
        let mut world = World::new();
        world.fill_demo_terrain();
        Self {
            window: None,
            renderer: None,
            camera: Camera::new(),
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
            gpu_terrain: None,
        }
    }

    fn world_origin_voxel(&self) -> IVec3 {
        IVec3::new(
            self.world.world_origin_chunk.x * voxel::STORAGE_CHUNK_VOXELS as i32,
            0,
            self.world.world_origin_chunk.y * voxel::STORAGE_CHUNK_VOXELS as i32,
        )
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
        if changed && (now - self.last_net_send) >= std::time::Duration::from_millis(50) {
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
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("voxel")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        let renderer = Renderer::new(window.clone(), &self.world);
        self.gpu_terrain = Some(gpu_terrain::GpuTerrainGen::new(
            &renderer.device, &renderer.queue, self.world.seed,
        ));
        self.window = Some(window);
        self.renderer = Some(renderer);
        self.grab(true);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(r) = self.renderer.as_mut() {
                    r.resize(size.width, size.height);
                }
            }
            WindowEvent::Focused(false) => self.grab(false),
            WindowEvent::MouseInput { state: ElementState::Pressed, button, .. } => {
                if !self.grabbed {
                    self.grab(true);
                } else {
                    // Queue the click — actual raycast runs at the top of
                    // the next RedrawRequested using the same camera state
                    // that's about to be rendered, so the marker lands under
                    // the rendered crosshair regardless of intervening
                    // mouse-motion events.
                    self.pending_clicks.push(button);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(code) = event.physical_key {
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
                        // Test-fire: drop a lava block 15 voxels in front of
                        // the camera with NO raycast. If this lava lands at
                        // the crosshair, raycast/coords are the bug. If it
                        // lands off-crosshair, the camera vectors disagree
                        // between CPU and GPU.
                        KeyCode::KeyT if pressed => {
                            let fwd = self.camera.forward();
                            let p = self.camera.pos + fwd * 15.0;
                            let wx = p.x.floor() as i32;
                            let wy = p.y.floor() as i32;
                            let wz = p.z.floor() as i32;
                            log::info!(
                                "TEST-FIRE cam=({:.2},{:.2},{:.2}) fwd=({:.3},{:.3},{:.3}) -> world=({},{},{})",
                                self.camera.pos.x, self.camera.pos.y, self.camera.pos.z,
                                fwd.x, fwd.y, fwd.z,
                                wx, wy, wz,
                            );
                            self.world.apply_edit(wx, wy, wz, voxel::MAT_LAVA);
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
            }
            WindowEvent::RedrawRequested => {
                if self.renderer.is_none() { return; }

                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32().min(1.0 / 30.0);
                self.last_frame = now;

                // Pump network in/out *before* re-borrowing renderer.
                self.poll_net();
                self.maybe_send_pose(now);

                let speed = if self.keys.sprint { 4.0 } else { 1.0 };
                let f = (self.keys.forward as i32 - self.keys.back as i32) as f32;
                let r = (self.keys.right as i32 - self.keys.left as i32) as f32;
                let u = (self.keys.up as i32 - self.keys.down as i32) as f32;
                self.camera.translate_local(dt, f * speed, r * speed, u * speed);

                // Streaming: shift the world's sliding window so the camera
                // stays near the centre of the loaded region. Crossing a
                // chunk boundary triggers regeneration of edge chunks.
                let target_origin = voxel::World::target_origin_chunk(self.camera.pos);
                if target_origin != self.world.world_origin_chunk {
                    self.world.shift_origin(target_origin);
                }
                if let (Some(gpu), Some(r)) = (self.gpu_terrain.as_mut(), self.renderer.as_ref()) {
                    gpu.tick(&r.device, &r.queue, &mut self.world);
                }
                // Lower CPU rayon worker cap (4 → 2) for stable frametime.
                // GPU does most of the work; CPU helps catch up overflow.
                self.world.process_regen_queue(2);

                // ---- consume queued clicks ----
                // At this point self.camera is in the EXACT state that's
                // about to be rendered: WASD translation applied above,
                // every queued mouse-motion already accounted for. Raycasting
                // here guarantees the marker lands under the rendered
                // crosshair the player sees.
                if !self.pending_clicks.is_empty() {
                    use winit::event::MouseButton;
                    let clicks = std::mem::take(&mut self.pending_clicks);
                    let world_origin = self.world_origin_voxel();
                    for button in clicks {
                        // raycast now walks WORLD coords (matching shader's
                        // toroidal storage lookup). hit.voxel is in WORLD
                        // coords — no translation needed.
                        let hit_opt = raycast::raycast(
                            self.camera.pos, self.camera.forward(), &self.world, world_origin,
                        );
                        log::info!(
                            "click {:?} cam_world={:.1?} fwd={:.3?} -> {:?}",
                            button, self.camera.pos, self.camera.forward(),
                            hit_opt.as_ref().map(|h| h.voxel),
                        );
                        let Some(hit) = hit_opt else { continue; };
                        let v = hit.voxel;
                        match button {
                            MouseButton::Left => {
                                let cx = v[0];
                                let cy = v[1];
                                let cz = v[2];
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
                let dirty_snap = self.world.dirty_bricks.clone();
                self.frames_since_full += 1;
                // Smart cache invalidation handles animations now (shader
                // sets tile_animated bits, next-frame gate ORs them into
                // dirty). The full-refresh threshold here just clears
                // stale animated bits (every 30 frames ≈ 0.5s at 60FPS)
                // so a tile that USED to show water but no longer does
                // stops being re-traced forever.
                let force_full = self.first_frame
                    || self.frames_since_full >= 30
                    || self.world.all_dirty
                    || camera_changed;
                if force_full {
                    if let Some(r) = self.renderer.as_ref() {
                        r.clear_tile_animated();
                    }
                }
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
                    for &bi in &dirty_snap {
                        temporal::project_brick_to_tiles(
                            bi, &self.world, &self.camera, rw, rh, tiles_w, tiles_h, &mut self.tile_dirty_mask,
                        );
                    }
                }

                let renderer = self.renderer.as_mut().unwrap();
                renderer.upload_world(&mut self.world);
                let t = (now - self.start_time).as_secs_f32();
                let world_origin_voxel = glam::IVec3::new(
                    self.world.world_origin_chunk.x * voxel::STORAGE_CHUNK_VOXELS as i32,
                    0,
                    self.world.world_origin_chunk.y * voxel::STORAGE_CHUNK_VOXELS as i32,
                );
                renderer.update_camera(&self.camera, t, world_origin_voxel);
                // Upload remote player markers.
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
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let (w, h) = renderer.size;
                        renderer.resize(w.max(1), h.max(1));
                    }
                    Err(e) => log::error!("render error: {:?}", e),
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
            _ => {}
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        if !self.grabbed {
            return;
        }
        if let DeviceEvent::MouseMotion { delta } = event {
            self.camera.rotate(delta.0 as f32, delta.1 as f32);
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

enum Mode {
    Solo,
    Server(u16),
    Connect(String),
}

fn parse_mode() -> Mode {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--server" => {
                let port = args
                    .get(i + 1)
                    .and_then(|s| s.parse::<u16>().ok())
                    .unwrap_or(7878);
                return Mode::Server(port);
            }
            "--connect" => {
                let addr = args
                    .get(i + 1)
                    .cloned()
                    .unwrap_or_else(|| "127.0.0.1:7878".to_string());
                return Mode::Connect(addr);
            }
            _ => {}
        }
        i += 1;
    }
    Mode::Solo
}

fn run_server(port: u16) -> ! {
    let mut server = net::NetServer::listen(port).expect("listen");
    let seed = 0xC0FFEE_F00D_BEEFu64;
    let mut player_states: std::collections::HashMap<net::PlayerId, ([f32; 3], f32, f32)> =
        std::collections::HashMap::new();
    // Persistent edit log on the server. Each entry overrides the seed-noise
    // generation when a chunk loads. Sent in full to every new client and
    // appended on every VoxelEdit received.
    let mut edits: std::collections::HashMap<(i32, i32, i32), u8> =
        std::collections::HashMap::new();
    const INTEREST_R: f32 = 600.0;
    const INTEREST_R2: f32 = INTEREST_R * INTEREST_R;
    log::info!("server: ready");
    loop {
        let (joined, msgs) = server.poll();
        for id in joined {
            server.send_to(id, net::Message::JoinAck { your_id: id, seed });
            // Replay the full edit log to the joiner so persistent builds
            // appear immediately.
            for (&(wx, wy, wz), &mat) in &edits {
                server.send_to(id, net::Message::VoxelEdit { wx, wy, wz, mat });
            }
            for (pid, st) in &player_states {
                server.send_to(
                    id,
                    net::Message::PlayerUpdate {
                        id: *pid,
                        pos: st.0,
                        yaw: st.1,
                        pitch: st.2,
                    },
                );
            }
            server.broadcast(&net::Message::PlayerJoin { id }, Some(id));
            log::info!("server: player {} joined ({} edits replayed)", id, edits.len());
        }
        for (sender, m) in msgs {
            match m {
                net::Message::Hello => {}
                net::Message::PlayerUpdate { pos, yaw, pitch, .. } => {
                    player_states.insert(sender, (pos, yaw, pitch));
                    let states = player_states.clone();
                    server.broadcast_filter(
                        &net::Message::PlayerUpdate { id: sender, pos, yaw, pitch },
                        |other_id| {
                            if other_id == sender { return false; }
                            states.get(&other_id).map_or(true, |s| {
                                let dx = s.0[0] - pos[0];
                                let dz = s.0[2] - pos[2];
                                (dx * dx + dz * dz) < INTEREST_R2
                            })
                        },
                    );
                }
                net::Message::VoxelEdit { wx, wy, wz, mat } => {
                    edits.insert((wx, wy, wz), mat);
                    let states = player_states.clone();
                    server.broadcast_filter(
                        &net::Message::VoxelEdit { wx, wy, wz, mat },
                        |other_id| {
                            states.get(&other_id).map_or(true, |s| {
                                let dx = s.0[0] - wx as f32;
                                let dz = s.0[2] - wz as f32;
                                (dx * dx + dz * dz) < INTEREST_R2
                            })
                        },
                    );
                }
                net::Message::Explode { cx, cy, cz, radius, mat } => {
                    // Expand on the server-side edit log so joiners see the
                    // destruction. Re-broadcast the *single* Explode message
                    // — clients expand locally, saves ~30x bandwidth.
                    let r = radius as i32;
                    let r2 = r * r;
                    for dy in -r..=r {
                        for dx in -r..=r {
                            for dz in -r..=r {
                                if dx * dx + dy * dy + dz * dz > r2 { continue; }
                                edits.insert((cx + dx, cy + dy, cz + dz), mat);
                            }
                        }
                    }
                    let states = player_states.clone();
                    server.broadcast_filter(
                        &net::Message::Explode { cx, cy, cz, radius, mat },
                        |other_id| {
                            states.get(&other_id).map_or(true, |s| {
                                let dx = s.0[0] - cx as f32;
                                let dz = s.0[2] - cz as f32;
                                (dx * dx + dz * dz) < INTEREST_R2
                            })
                        },
                    );
                }
                net::Message::PlayerLeave { id } => {
                    player_states.remove(&id);
                    server.drop_client(id);
                    server.broadcast(&net::Message::PlayerLeave { id }, None);
                    log::info!("server: player {} left", id);
                }
                _ => {}
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    match parse_mode() {
        Mode::Server(port) => run_server(port),
        mode => {
            let net = match mode {
                Mode::Connect(addr) => match net::NetClient::connect(&addr) {
                    Ok(c) => {
                        log::info!("connected to {}", addr);
                        Some(c)
                    }
                    Err(e) => {
                        log::error!("connect failed: {}", e);
                        None
                    }
                },
                _ => None,
            };
            let event_loop = EventLoop::new().expect("event loop");
            event_loop.set_control_flow(ControlFlow::Poll);
            let mut app = App::new(net);
            event_loop.run_app(&mut app).expect("run_app");
        }
    }
}
