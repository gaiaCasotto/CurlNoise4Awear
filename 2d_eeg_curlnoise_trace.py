# Curl Trace (Houdini-style) in Taichi
# - particles advected by curl noise
# - added trail buffer (ink) with decay + diffusion
# - screen-space accumulation creates smooth streamlines


"""
PARAMETER EXPERIMENTATION:

1) CALM PRESETS
PARTICLE_SPEED = 0.18
DT             = 0.010
DECAY          = 0.990
DIFFUSE        = 0.18
SPLAT_RADIUS   = 2.2
# in curl():
q = p * 0.45  # remove + vec2(t*rate, t*rate) or implement slow rotation
# fbm: fbm(q, 3)


2) Stressed
PARTICLE_SPEED = 0.45
DT             = 0.022
DECAY          = 0.88
DIFFUSE        = 0.06
SPLAT_RADIUS   = 0.9
# in curl():
q = p * 0.7 + vec2(t * 0.06, t * 0.02)  # faster drift
# fbm: fbm(q, 5 or 6)
# jitter: ±0.8–1.0 px in splat

"""



import taichi as ti
import math, random

ti.init(arch=ti.gpu)

# ---------------- config ----------------
WIN            = 900    # window size
IMG_RES        = 1024   # resolution of the trail buffer
NUM_PARTICLES  = 20000
PARTICLE_SPEED = 0.3
DT             = 0.016

DECAY        = 0.885  #how much the ink fades each frame (1.0 = doesnt fade)
DIFFUSE      = 0.06   # small diffusion strength (0..~0.3)
SPLAT_RADIUS = 1.10   # pixel radius of the Gaussian splat each particle writes

# ---------------- window / canvas ----------------
window = ti.ui.Window("Curl Trace (Taichi)", (WIN, WIN))
canvas = window.get_canvas()
gui    = window.get_gui()  

# ---------------- particle buffers ----------------
pos      = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PARTICLES) # to hold the cuurent position of each particle
prev_pos = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PARTICLES) # holds the prvious position to build the trace

vec2 = ti.types.vector(2, ti.f32)

# ---------------- trail buffers ----------------
ink     = ti.field(dtype=ti.f32, shape=(IMG_RES, IMG_RES))   # accumulated trails
ink_tmp = ti.field(dtype=ti.f32, shape=(IMG_RES, IMG_RES))   # tmp buffer for diffusion pass
rgb     = ti.Vector.field(3, dtype=ti.f32, shape=(IMG_RES, IMG_RES))   # the image i actually display

# ---------- helpers ----------
@ti.func
def fract(x: ti.f32) -> ti.f32:
    return x - ti.floor(x)

@ti.func
def lerp(a: ti.f32, b: ti.f32, t: ti.f32) -> ti.f32:
    return a * (1.0 - t) + b * t

@ti.func
def saturate1(x: ti.f32) -> ti.f32: #clamp function
    return ti.max(0.0, ti.min(1.0, x))

@ti.func
def world_to_tex(p: vec2) -> ti.Vector:  # Map world coords to texture pixel space , [-1,1]^2 -> [0,RES)
    q = (p * 0.5 + 0.5) * IMG_RES
    return q

@ti.func
def splat_gaussian(px: ti.i32, py: ti.i32, cx: ti.f32, cy: ti.f32, radius: ti.f32, amp: ti.f32):
    # add a soft, normalized gaussian splat centered at (cx,cy) [in pixel coords]
    x = ti.cast(px, ti.f32)
    y = ti.cast(py, ti.f32)
    dx = x - cx
    dy = y - cy
    r2 = dx * dx + dy * dy
    sig2 = radius * radius
    w = ti.exp(-r2 / (2.0 * sig2))
    ti.atomic_add(ink[px, py], amp * w)

# ---------------- noise functions ----------------
@ti.func
def hash21(p: vec2) -> ti.f32:
    return fract(ti.sin(p.dot(vec2(127.1, 311.7))) * 43758.5453123)

@ti.func
def noise(p: vec2) -> ti.f32:
    i = ti.floor(p)
    f = p - i
    a = hash21(i + vec2(0, 0))
    b = hash21(i + vec2(1, 0))
    c = hash21(i + vec2(0, 1))
    d = hash21(i + vec2(1, 1))
    u = f * f * (3 - 2 * f)
    return lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y)

@ti.func
def fbm(p: vec2, octaves: int) -> ti.f32:
    v = 0.0
    a = 0.5
    freq = 1.0
    for _ in range(octaves):
        v += a * noise(p * freq)
        freq *= 2.0
        a *= 0.5
    return v

# ---------------- curl field ----------------
@ti.func
def curl(p: vec2, t: ti.f32) -> vec2:
    eps = 0.01
    q = p * 0.5 + vec2(t * 0.02, t * 0.02)  #
    dx = (fbm(q + vec2(eps, 0), 6) - fbm(q - vec2(eps, 0), 6)) / (2 * eps)
    dy = (fbm(q + vec2(0, eps), 6) - fbm(q - vec2(0, eps), 6)) / (2 * eps)
    return vec2(dy, -dx)  # divergence-free

# ---------------- kernels ----------------
@ti.kernel
def init_particles():
    for i in range(NUM_PARTICLES):
        p = ti.Vector([ti.random(), ti.random()]) * 2.0 - 1.0  # [-1,1]^2
        pos[i] = p
        prev_pos[i] = p

@ti.kernel
def zero_ink():
    for I in ti.grouped(ink):
        ink[I] = 0.0

@ti.kernel
def advect_and_splat(t: ti.f32, dt: ti.f32, splat_r: ti.f32):
    """
    This kernel does everything per particle: integrate position, then draw a short anti-aliased segment 
    into the ink buffer by laying down small Gaussian discs along the segment. 
    The jitter helps avoid grid patterns
    """
    for i in range(NUM_PARTICLES):
        p = pos[i]
        v = curl(p * 8.0, t)
        prev_pos[i] = p
        p += v * dt * PARTICLE_SPEED

        # wrap-around to keep particles inside domain
        for k in ti.static(range(2)):
            if p[k] < -1.0: p[k] += 2.0
            if p[k] > 1.0:  p[k] -= 2.0
        pos[i] = p

        # --- "trace": draw a short line from prev_pos -> pos into ink texture
        a = world_to_tex(prev_pos[i])
        b = world_to_tex(p)
        ax, ay = a[0], a[1]
        bx, by = b[0], b[1]

        # sample along the segment
        seg_len = ti.sqrt((bx - ax) * (bx - ax) + (by - ay) * (by - ay)) + 1e-6
        #steps   = ti.cast(ti.min(16.0, seg_len / 0.5 + 1.0), ti.i32)  # clamp steps to keep it cheap
        steps   = ti.cast(seg_len, ti.i32)
        amp     = 0.8  # ink amount per sample

        for s in range(steps):
            tt = (ti.cast(s, ti.f32) + 0.5) / ti.cast(steps, ti.f32)
            cx = lerp(ax, bx, tt)
            cy = lerp(ay, by, tt)
            #Randomize splat center slightly each frame to avoid systematic aliasing (grid pattern)
            cx += (ti.random(ti.f32) - 0.5) * 0.5 
            cy += (ti.random(ti.f32) - 0.5) * 0.5

            # compute pixel bounds for splat
            rx = ti.cast(ti.floor(cx - splat_r - 1.0), ti.i32)
            ry = ti.cast(ti.floor(cy - splat_r - 1.0), ti.i32)
            R  = ti.cast(ti.ceil(splat_r + 1.0), ti.i32)
            for dx in range(2 * R + 3):
                for dy in range(2 * R + 3):
                    px = rx + dx
                    py = ry + dy
                    if 0 <= px < IMG_RES and 0 <= py < IMG_RES:
                        splat_gaussian(px, py, cx, cy, splat_r, amp / ti.cast(steps, ti.f32))

@ti.kernel
def decay_and_diffuse(decay: ti.f32):
    # simple decay
    for I in ti.grouped(ink):
        ink[I] *= decay

    # 4-neighbor diffusion (Jacobi step)
    for i, j in ink:
        c = ink[i, j]
        l = ink[max(i - 1, 0), j]
        r = ink[min(i + 1, IMG_RES - 1), j]
        d = ink[i, max(j - 1, 0)]
        u = ink[i, min(j + 1, IMG_RES - 1)]
        ink_tmp[i, j] = c + DIFFUSE * (0.25 * (l + r + u + d) - c)

    for I in ti.grouped(ink):
        ink[I] = ink_tmp[I]

@ti.kernel
def tonemap():
    # simple tone map + colorize to bluish/cyan
    for i, j in ink:
        x = ink[i, j]
        # normalize-ish: soft rolloff
        v = x / (1.0 + 0.75 * x)
        # color ramp: dark->blue->cyan->white
        rgb[i, j] = ti.Vector([0.2 * v, 0.6 * v + 0.2 * v * v, v])

# ---------------- main loop ----------------
init_particles()
zero_ink()

t = 0.0
while window.running:
    #GUI
    with gui.sub_window("Controls", 0.02, 0.02, 0.30, 0.20):
        gui.text("Simulation")
        DT    = gui.slider_float("dt (time step)", DT, 0.001, 0.050)
        SPLAT_RADIUS = gui.slider_float("splat radius", SPLAT_RADIUS, 0.05, 3.00)
        gui.text("larger dt or speed => faster motion")
        DECAY = gui.slider_float("decay", DECAY, 0.05, 1.00)


    advect_and_splat(t, DT, SPLAT_RADIUS)
    decay_and_diffuse(DECAY)
    tonemap()

    canvas.set_image(rgb)                     # draw trail image
    # optional: overlay particles for sparkle
    # canvas.circles(pos, radius=0.0025, color=(1.0, 1.0, 1.0))

    window.show()
    t += 0.01
