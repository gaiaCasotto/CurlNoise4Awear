import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f32)  # fall back to ti.cpu if needed

# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------
NUM_PARTICLES = 60000
DOMAIN = 1.0               # positions in [-DOMAIN, DOMAIN]^3
DT = 0.016                 # timestep
RADIUS = 0.003             # particle radius in world units
NOISE_FREQ = 3.8           # base frequency for noise sampling
FBM_OCTAVES = 4
SPEED = 0.3               # advection speed scale
WRAP = True                # wrap positions at domain boundaries (vs. respawn)

# ---------------------------------------------------------
# Fields
# ---------------------------------------------------------
pos = ti.Vector.field(3, ti.f32, shape=NUM_PARTICLES)
vel = ti.Vector.field(3, ti.f32, shape=NUM_PARTICLES)  # for coloring

vec2 = ti.types.vector(2, ti.f32)
vec3 = ti.types.vector(3, ti.f32)

# ---------------------------------------------------------
# Small math helpers (avoid API pitfalls)
# ---------------------------------------------------------
@ti.func
def fract(x: ti.f32) -> ti.f32:
    return x - ti.floor(x)

@ti.func
def lerp(a: ti.f32, b: ti.f32, t: ti.f32) -> ti.f32:
    return a * (1.0 - t) + b * t

@ti.func
def smoothstep01(t: ti.f32) -> ti.f32:
    return t * t * (3.0 - 2.0 * t)

@ti.func
def saturate1(x: ti.f32) -> ti.f32:
    return ti.max(0.0, ti.min(1.0, x))

# ---------------------------------------------------------
# 3D Value Noise + FBM (hash-based, trilinear)
# ---------------------------------------------------------
@ti.func
def hash31(p: vec3) -> ti.f32:
    # scalar hash in [0,1)
    x = ti.sin(p.dot(vec3(127.1, 311.7, 74.7))) * 43758.5453123
    return fract(x)

@ti.func
def value_noise3(p: vec3) -> ti.f32:
    i = ti.floor(p)
    f = p - i
    # Corners
    c000 = hash31(i + vec3(0, 0, 0))
    c100 = hash31(i + vec3(1, 0, 0))
    c010 = hash31(i + vec3(0, 1, 0))
    c110 = hash31(i + vec3(1, 1, 0))
    c001 = hash31(i + vec3(0, 0, 1))
    c101 = hash31(i + vec3(1, 0, 1))
    c011 = hash31(i + vec3(0, 1, 1))
    c111 = hash31(i + vec3(1, 1, 1))
    u = vec3(smoothstep01(f.x), smoothstep01(f.y), smoothstep01(f.z))
    # Trilinear interpolation
    x00 = lerp(c000, c100, u.x)
    x10 = lerp(c010, c110, u.x)
    x01 = lerp(c001, c101, u.x)
    x11 = lerp(c011, c111, u.x)
    y0  = lerp(x00,  x10,  u.y)
    y1  = lerp(x01,  x11,  u.y)
    return lerp(y0, y1, u.z)

@ti.func
def fbm3(p: vec3, octaves: ti.i32) -> ti.f32:
    amp = 0.5
    freq = 1.0
    s = 0.0
    for _ in range(octaves):
        s += amp * value_noise3(p * freq)
        freq *= 2.0
        amp *= 0.5
    return s

# ---------------------------------------------------------
# Vector Field F(p, t): build 3 independent FBM channels
# ---------------------------------------------------------
@ti.func
def field_F(p: vec3, t: ti.f32) -> vec3:
    # Time-warped coords for gentle animation
    q = p * NOISE_FREQ + vec3(t * 0.15, ti.sin(t * 0.2) * 0.5, ti.cos(t * 0.17) * 0.5)
    # Use shifted inputs so channels are decorrelated
    Fx = fbm3(q + vec3(31.4,  0.0, 19.7), FBM_OCTAVES)
    Fy = fbm3(q + vec3(-12.3, 7.1,  5.9), FBM_OCTAVES)
    Fz = fbm3(q + vec3(4.6,  23.9, -17.2), FBM_OCTAVES)
    # Map from [0,1] to [-1,1] to get zero-mean-ish field
    return vec3(Fx * 2.0 - 1.0, Fy * 2.0 - 1.0, Fz * 2.0 - 1.0)

# ---------------------------------------------------------
# Curl of F via central differences (6 samples)
# curl(F) = (dFz/dy - dFy/dz, dFx/dz - dFz/dx, dFy/dx - dFx/dy)
# ---------------------------------------------------------
@ti.func
def curl_F(p: vec3, t: ti.f32) -> vec3:
    eps = 0.0025
    # Sample F at +/- eps along each axis
    Fx1 = field_F(p + vec3(eps, 0.0, 0.0), t)
    Fx2 = field_F(p - vec3(eps, 0.0, 0.0), t)
    Fy1 = field_F(p + vec3(0.0, eps, 0.0), t)
    Fy2 = field_F(p - vec3(0.0, eps, 0.0), t)
    Fz1 = field_F(p + vec3(0.0, 0.0, eps), t)
    Fz2 = field_F(p - vec3(0.0, 0.0, eps), t)

    dF_dX = (Fx1 - Fx2) / (2.0 * eps)  # each is vec3 of partials w.r.t x
    dF_dY = (Fy1 - Fy2) / (2.0 * eps)  # partials w.r.t y
    dF_dZ = (Fz1 - Fz2) / (2.0 * eps)  # partials w.r.t z

    # Assemble curl from partials
    return vec3(
        dF_dY.z - dF_dZ.y,  # dFz/dy - dFy/dz
        dF_dZ.x - dF_dX.z,  # dFx/dz - dFz/dx
        dF_dX.y - dF_dY.x   # dFy/dx - dFx/dy
    )

# ---------------------------------------------------------
# Simulation kernels
# ---------------------------------------------------------
@ti.kernel
def init_particles(seed: ti.i32):
    for i in range(NUM_PARTICLES):
        # random in [-DOMAIN, DOMAIN]^3
        rx = ti.random(dtype=ti.f32) * 2.0 - 1.0
        ry = ti.random(dtype=ti.f32) * 2.0 - 1.0
        rz = ti.random(dtype=ti.f32) * 2.0 - 1.0
        pos[i] = vec3(rx, ry, rz) * DOMAIN
        vel[i] = vec3(0.0, 0.0, 0.0)

@ti.kernel
def step(t: ti.f32, dt: ti.f32):
    for i in range(NUM_PARTICLES):
        p = pos[i]
        v = curl_F(p, t) * SPEED
        p += v * dt

        # Wrap or respawn
        if WRAP:
            if p.x < -DOMAIN: p.x += 2.0 * DOMAIN
            if p.x >  DOMAIN: p.x -= 2.0 * DOMAIN
            if p.y < -DOMAIN: p.y += 2.0 * DOMAIN
            if p.y >  DOMAIN: p.y -= 2.0 * DOMAIN
            if p.z < -DOMAIN: p.z += 2.0 * DOMAIN
            if p.z >  DOMAIN: p.z -= 2.0 * DOMAIN
        else:
            # respawn near origin if flown too far
            if (abs(p.x) > DOMAIN) or (abs(p.y) > DOMAIN) or (abs(p.z) > DOMAIN):
                p = vec3((ti.random() * 2.0 - 1.0) * 0.2,
                         (ti.random() * 2.0 - 1.0) * 0.2,
                         (ti.random() * 2.0 - 1.0) * 0.2)

        pos[i] = p
        vel[i] = v

# ---------------------------------------------------------
# Rendering (3D, perspective)
# ---------------------------------------------------------
window = ti.ui.Window("3D Curl Noise Particles (Houdini-style)", (1280, 720))
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()

init_particles(0)
t = 0.0

while window.running:
    # Simulation
    step(t, DT)
    t += DT

    # Camera controls (orbit around origin)
    r = 3.2
    eye = vec3(ti.cos(t * 0.2) * r, 1.4, ti.sin(t * 0.2) * r)
    camera.position(eye.x, eye.y, eye.z)
    camera.lookat(0.0, 0.0, 0.0)
    camera.fov(45)  # degrees
    scene.set_camera(camera)

    # Lights
    scene.ambient_light((0.5, 0.5, 0.6))
    scene.point_light(pos=(3, 3, 3), color=(0.7, 0.7, 0.7))

    # Color by speed (magnitude of curl)
    # Map speed -> [0,1] for a simple blue->white ramp
    # (no clamp API; do it manually)
    # compute scale here in Python to keep it simple/fast:
    scene.particles(pos, radius=RADIUS, per_vertex_color=None)  # draw first, then overlay with color below?

    # Simple velocity-based tint (GPU-side: reuse previous vel field via two calls)
    # Faster: one call with a single color; clearer: two calls for highlight.
    # Here, color all particles using a uniform, then draw a faint second pass for "hot" ones:
    scene.particles(pos, radius=RADIUS, color=(0.35, 0.55, 1.0))

   
    canvas.scene(scene)
    window.show()
