use glam::Vec3;

#[derive(Clone)]
pub struct Camera {
    pub pos: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
    pub move_speed: f32,
    pub look_sensitivity: f32,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            pos: Vec3::new(512.0, 200.0, 512.0),
            yaw: 0.0,
            pitch: -0.5,
            fov_y: 70.0_f32.to_radians(),
            move_speed: 160.0,
            look_sensitivity: 0.0025,
        }
    }

    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.cos() * self.pitch.cos(),
        )
        .normalize()
    }

    pub fn right(&self) -> Vec3 {
        // Forward x world-up, normalized
        let f = self.forward();
        Vec3::new(f.z, 0.0, -f.x).normalize()
    }

    pub fn up(&self) -> Vec3 {
        self.forward().cross(self.right()).normalize()
    }

    pub fn rotate(&mut self, dx: f32, dy: f32) {
        // Mouse right (dx > 0) should turn the view right — i.e. yaw
        // *increases* so forward.x becomes positive.
        self.yaw += dx * self.look_sensitivity;
        self.pitch -= dy * self.look_sensitivity;
        let limit = std::f32::consts::FRAC_PI_2 - 0.01;
        self.pitch = self.pitch.clamp(-limit, limit);
    }

    pub fn translate_local(&mut self, dt: f32, forward: f32, right: f32, up: f32) {
        let f = self.forward();
        let r = self.right();
        let u = Vec3::Y;
        self.pos += (f * forward + r * right + u * up) * self.move_speed * dt;
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub origin: [f32; 3],
    pub _pad0: f32,
    pub forward: [f32; 3],
    pub _pad1: f32,
    pub right: [f32; 3],
    pub _pad2: f32,
    pub up: [f32; 3],
    pub tan_half_fov: f32,
    pub resolution: [f32; 2],
    pub time: f32,
    pub _pad3: f32,
    /// World-voxel offset of the loaded region's lower corner. The shader
    /// uses this to bounds-check rays + mod-fold world voxel coords into the
    /// toroidal slot storage.
    pub world_origin: [i32; 3],
    pub _pad4: i32,
}

impl CameraUniform {
    pub fn from_camera(c: &Camera, width: u32, height: u32, time: f32, world_origin_voxel: glam::IVec3) -> Self {
        Self {
            origin: c.pos.to_array(),
            _pad0: 0.0,
            forward: c.forward().to_array(),
            _pad1: 0.0,
            right: c.right().to_array(),
            _pad2: 0.0,
            up: c.up().to_array(),
            tan_half_fov: (c.fov_y * 0.5).tan(),
            resolution: [width as f32, height as f32],
            time,
            _pad3: 0.0,
            world_origin: [world_origin_voxel.x, world_origin_voxel.y, world_origin_voxel.z],
            _pad4: 0,
        }
    }
}
