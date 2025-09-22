import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f32)  # switch to ti.cpu if needed

# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------
NUM_PARTICLES = 20000
DOMAIN = 2.0
DT = 0.016
SPEED = 0.6
NOISE_FREQ = 1.8
FBM_OCTAVES = 4
TRAIL_STEPS = 20  # number of positions kept per particle
BASE_RADIUS = 0.003


WRAP = False

# ---------------------------------------------------------
# Fields
# ---------------------------------------------------------
vec3 = ti.types.vector(3, ti.f32)

pos   = ti.Vector.field(3, ti.f32, shape=NUM_PARTICLES)                 # current positions
vel   = ti.Vector.field(3, ti.f32, shape=NUM_PARTICLES)                 # current velocities (for future coloring)
trail = ti.Vector.field(3, ti.f32, shape=(NUM_PARTICLES, TRAIL_STEPS))  # ring buffer [i, k]
head  = ti.field(ti.i32, shape=NUM_PARTICLES)                           # write index

# ---------------------------------------------------------
# Helpers
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

# ---------------------------------------------------------
# 3D value noise + FBM
# ---------------------------------------------------------
@ti.func
def hash31(p: vec3) -> ti.f32:
    x = ti.sin(p.dot(vec3(127.1, 311.7, 74.7))) * 43758.5453123
    return fract(x)

@ti.func
def value_noise3(p: vec3) -> ti.f32:
    i = ti.floor(p)
    f = p - i
    c000 = hash31(i + vec3(0, 0, 0))
    c100 = hash31(i + vec3(1, 0, 0))
    c010 = hash31(i + vec3(0, 1, 0))
    c110 = hash31(i + vec3(1, 1, 0))
    c001 = hash31(i + vec3(0, 0, 1))
    c101 = hash31(i + vec3(1, 0, 1))
    c011 = hash31(i + vec3(0, 1, 1))
    c111 = hash31(i + vec3(1, 1, 1))
    u = vec3(smoothstep01(f.x), smoothstep01(f.y), smoothstep01(f.z))
    x00 = lerp(c000, c100, u.x)
    x10 = lerp(c010, c110, u.x)
    x01 = lerp(c001, c101, u.x)
    x11 = lerp(c011, c111, u.x)
    y0 = lerp(x00, x10, u.y)
    y1 = lerp(x01, x11, u.y)
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
# Vector field & curl
# ---------------------------------------------------------
@ti.func
def field_F(p: vec3, t: ti.f32) -> vec3:
    q = p * NOISE_FREQ + vec3(t * 0.15, ti.sin(t * 0.2) * 0.5, ti.cos(t * 0.17) * 0.5)
    Fx = fbm3(q + vec3(31.4, 0.0, 19.7), FBM_OCTAVES)
    Fy = fbm3(q + vec3(-12.3, 7.1, 5.9), FBM_OCTAVES)
    Fz = fbm3(q + vec3(4.6, 23.9, -17.2), FBM_OCTAVES)
    return vec3(Fx * 2 - 1, Fy * 2 - 1, Fz * 2 - 1)

@ti.func
def curl_F(p: vec3, t: ti.f32) -> vec3:
    eps = 0.0025
    Fx1 = field_F(p + vec3(eps, 0, 0), t)
    Fx2 = field_F(p - vec3(eps, 0, 0), t)
    Fy1 = field_F(p + vec3(0, eps, 0), t)
    Fy2 = field_F(p - vec3(0, eps, 0), t)
    Fz1 = field_F(p + vec3(0, 0, eps), t)
    Fz2 = field_F(p - vec3(0, 0, eps), t)
    dFdx = (Fx1 - Fx2) / (2 * eps)
    dFdy = (Fy1 - Fy2) / (2 * eps)
    dFdz = (Fz1 - Fz2) / (2 * eps)
    return vec3(
        dFdy.z - dFdz.y,
        dFdz.x - dFdx.z,
        dFdx.y - dFdy.x
    )

# ---------------------------------------------------------
# Kernels
# ---------------------------------------------------------
@ti.kernel
def init_particles():
    for i in range(NUM_PARTICLES):
        rx = ti.random() * 2 - 1
        ry = ti.random() * 2 - 1
        rz = ti.random() * 2 - 1
        p = vec3(rx, ry, rz) * DOMAIN
        pos[i] = p
        vel[i] = vec3(0, 0, 0)
        head[i] = 0
        for k in range(TRAIL_STEPS):
            trail[i, k] = p  # prefill so first frames show trails

@ti.kernel
def step(t: ti.f32, dt: ti.f32):
    for i in range(NUM_PARTICLES):
        p = pos[i]
        v = curl_F(p, t) * SPEED
        p += v * dt

        if WRAP:
            # wrap #making the particles stay with a confined box
            if p.x < -DOMAIN: p.x += 2 * DOMAIN
            if p.x >  DOMAIN: p.x -= 2 * DOMAIN
            if p.y < -DOMAIN: p.y += 2 * DOMAIN
            if p.y >  DOMAIN: p.y -= 2 * DOMAIN
            if p.z < -DOMAIN: p.z += 2 * DOMAIN
            if p.z >  DOMAIN: p.z -= 2 * DOMAIN
        else: #free roaminfg
            MAX_R = 2.0  # how far particles are allowed to drift
            if ti.sqrt(p.dot(p)) > MAX_R:
                # respawn near origin
                p = vec3((ti.random()*2-1)*0.5, (ti.random()*2-1)*0.5, (ti.random()*2-1)*0.5)

        pos[i] = p
        vel[i] = v

        # push newest into ring buffer
        h = head[i]
        trail[i, h] = p
        head[i] = (h + 1) % TRAIL_STEPS

# === NEW: temp field to hold one trail slice per frame ===
trail_slice = ti.Vector.field(3, ti.f32, shape=NUM_PARTICLES)

@ti.kernel
def extract_trail_slice(k: ti.i32):
    # Copy trail[:, k] into a flat field the renderer can digest
    for i in range(NUM_PARTICLES):
        trail_slice[i] = trail[i, k]

# ---------------------------------------------------------
# Render
# ---------------------------------------------------------
window = ti.ui.Window("3D Curl Noise (Particle Trails)", (1280, 720))
canvas = window.get_canvas()
scene  = ti.ui.Scene()
camera = ti.ui.Camera()

init_particles()
t = 0.0

while window.running:
    step(t, DT)
    t += DT

    # camera orbit (same as before)
    r = 3.0
    eye = vec3(ti.cos(t * 0.2) * r, 1.5, ti.sin(t * 0.2) * r)
    camera.position(eye.x, eye.y, eye.z)
    camera.lookat(0, 0, 0)
    camera.fov(45)
    scene.set_camera(camera)

    scene.ambient_light((0.5, 0.5, 0.6))
    scene.point_light(pos=(3, 3, 3), color=(0.85, 0.85, 0.85))

    # draw bright heads
    scene.particles(pos, radius=BASE_RADIUS, color=(0.9, 0.95, 1.0))

    # draw trail steps as fading dots
    for k in range(TRAIL_STEPS):
        # copy kth slice into a proper field
        extract_trail_slice(k)

        # age factor: older = smaller & dimmer
        age = (k + 1) / TRAIL_STEPS
        rad = BASE_RADIUS * (0.25 + 0.75 * age) * 0.6
        col = (0.35 + 0.55 * age, 0.55 + 0.40 * age, 1.0)

        scene.particles(trail_slice, radius=rad, color=col)

    canvas.scene(scene)
    window.show()