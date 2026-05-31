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
            // 1.5× scale: spawn near the center of the 768×384×768 world.
            pos: Vec3::new(384.0, 120.0, 384.0),
            yaw: 0.0,
            pitch: -0.5,
            fov_y: 70.0_f32.to_radians(),
            move_speed: 120.0,
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

/// Orthonormal sun basis + AABB-fitted ortho frustum for shadow-map ray casts.
/// Matches the shader's `sun_dir()` so on-screen + on-shadow agree perfectly.
///
/// Layout (16-byte aligned for uniform buffer):
///   sun_dir         (vec3 + pad)   — unit vector pointing toward the sun
///   sun_basis_x     (vec3 + pad)   — perpendicular to sun_dir, horizontal
///   sun_basis_y     (vec3 + pad)   — perpendicular to sun_dir + sun_basis_x
///   sun_center      (vec3) + half  — world-space center of the loaded AABB,
///                                    plus the ortho half-extent for u/v
///   sun_far         (f32)          — depth range along -sun_dir
///   _pad...
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SunUniform {
    pub sun_dir: [f32; 3],
    pub sun_intensity: f32,
    pub sun_basis_x: [f32; 3],
    pub sun_far: f32,
    pub sun_basis_y: [f32; 3],
    pub sun_half_size: f32,
    pub sun_center: [f32; 3],
    pub _pad0: f32,
    /// sun colour × intensity — what gets multiplied into direct light.
    pub sun_color: [f32; 3],
    pub _pad1: f32,
    /// time-of-day ambient (day or night), already weighted.
    pub ambient_color: [f32; 3],
    pub _pad2: f32,
}

impl SunUniform {
    /// Match the shader's `sun_dir()` exactly.
    pub fn sun_dir_for_time(time: f32) -> Vec3 {
        let a = time * 0.025 + 1.20;
        Vec3::new(a.cos(), a.sin(), 0.30).normalize()
    }

    /// Build sun basis + AABB-fitting ortho frustum that covers the loaded
    /// window from `world_origin` to `world_origin + dims`.
    pub fn fit_to_window(time: f32, world_origin_voxel: glam::IVec3, dims_voxels: glam::UVec3) -> Self {
        let s = Self::sun_dir_for_time(time);
        // Pick a stable horizontal reference for the basis. World-up cross s
        // is non-degenerate as long as s isn't perfectly vertical (the +0.30
        // y-bias in sun_dir guarantees that, but pick a safe fallback just in
        // case future tweaks remove the bias).
        let up = Vec3::Y;
        let bx = up.cross(s);
        let bx = if bx.length_squared() < 1e-4 { Vec3::X } else { bx.normalize() };
        let by = s.cross(bx).normalize();

        let min_w = world_origin_voxel.as_vec3();
        let max_w = min_w + dims_voxels.as_vec3();
        let center = (min_w + max_w) * 0.5;

        // Project all 8 AABB corners into sun-space (u along bx, v along by,
        // depth along s) → find the tightest fitting frustum.
        let corners = [
            Vec3::new(min_w.x, min_w.y, min_w.z),
            Vec3::new(max_w.x, min_w.y, min_w.z),
            Vec3::new(min_w.x, max_w.y, min_w.z),
            Vec3::new(max_w.x, max_w.y, min_w.z),
            Vec3::new(min_w.x, min_w.y, max_w.z),
            Vec3::new(max_w.x, min_w.y, max_w.z),
            Vec3::new(min_w.x, max_w.y, max_w.z),
            Vec3::new(max_w.x, max_w.y, max_w.z),
        ];
        let mut u_min = f32::INFINITY; let mut u_max = f32::NEG_INFINITY;
        let mut v_min = f32::INFINITY; let mut v_max = f32::NEG_INFINITY;
        let mut d_min = f32::INFINITY; let mut d_max = f32::NEG_INFINITY;
        for c in corners {
            let r = c - center;
            let u = r.dot(bx);
            let v = r.dot(by);
            let d = r.dot(s);
            u_min = u_min.min(u); u_max = u_max.max(u);
            v_min = v_min.min(v); v_max = v_max.max(v);
            d_min = d_min.min(d); d_max = d_max.max(d);
        }
        // Use one half-extent for a square ortho frustum (simpler texel math).
        let half_size = (u_max - u_min).max(v_max - v_min) * 0.5;
        let sun_far = d_max - d_min + 4.0;

        // Sun colour ladder mirrors the shader:
        //   intensity = smoothstep(-0.05, 0.10, s.y)
        //   sun_color = mix(warm, mid, smoothstep(0.05, 0.40, s.y)) * intensity
        let h = s.y.clamp(0.0, 1.0);
        let intensity = smoothstep(-0.05, 0.10, s.y);
        let warm = Vec3::new(1.40, 0.60, 0.25);
        let mid  = Vec3::new(1.10, 1.02, 0.92);
        let sun_color = warm.lerp(mid, smoothstep(0.05, 0.40, h)) * intensity;

        // Day/night ambient. Day = pale blue, night = deep blue tint, weighted
        // by intensity so dusk/dawn lerps cleanly.
        let day_amb   = Vec3::new(0.38, 0.42, 0.52);
        let night_amb = Vec3::new(0.04, 0.05, 0.08);
        let ambient_color = night_amb.lerp(day_amb, intensity);

        Self {
            sun_dir: s.to_array(),
            sun_intensity: intensity,
            sun_basis_x: bx.to_array(),
            sun_far,
            sun_basis_y: by.to_array(),
            sun_half_size: half_size,
            sun_center: center.to_array(),
            _pad0: 0.0,
            sun_color: sun_color.to_array(),
            _pad1: 0.0,
            ambient_color: ambient_color.to_array(),
            _pad2: 0.0,
        }
    }
}

#[inline]
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
