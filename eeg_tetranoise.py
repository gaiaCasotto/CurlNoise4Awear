""""-------------- ADDING EEG INPUT --------------- """

# Curl Trace in Taichi,
# - trying out tetranoise function #inspired from https://www.shadertoy.com/view/mlsSWH

import taichi as ti
import numpy as np


ti.init(arch=ti.gpu)

# ---------------- config ----------------
WIN            = 800    # window size
IMG_RES        = 1024   # resolution of the trail buffer
NUM_PARTICLES  = 20000
PARTICLE_SPEED = 0.15 #fix this mess

DECAY        = 0.885  #how much the ink fades each frame (1.0 = doesnt fade)
DIFFUSE      = 0.06   # small diffusion strength (0..~0.3)
SPLAT_RADIUS = 1.10   # pixel radius of the Gaussian splat each particle writes

# ---------------- particle buffers ----------------
pos      = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PARTICLES) # to hold the cuurent position of each particle
prev_pos = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PARTICLES) # holds the prvious position to build the trace

vec2 = ti.types.vector(2, ti.f32)
vec3 = ti.types.vector(3, ti.f32)

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
def clamp1(x: ti.f32, a: ti.f32, b: ti.f32) -> ti.f32:
    return ti.max(a, ti.min(b, x))

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

@ti.func
def step1(edge: ti.f32, x: ti.f32) -> ti.f32:
    # GLSL step(edge, x): 0 if x < edge else 1
    return 0.0 if x < edge else 1.0

@ti.func
def step_vec(edge: vec3, x: vec3) -> vec3:
    return vec3(step1(edge.x, x.x), step1(edge.y, x.y), step1(edge.z, x.z))

# ---------------- noise functions ----------------
@ti.func
def hash21(p: vec2) -> ti.f32:
    return fract(ti.sin(p.dot(vec2(127.1, 311.7))) * 43758.5453123)

@ti.func
def hash33(p: vec3) -> vec3:
    """
    Hash 3D -> 3D. Produces a pseudo-random *unit* vector per lattice point
    (used as gradient dir like in simplex). Deterministic and kernel-safe.
    """
    # mix different dot bases to decorrelate channels
    qx = ti.sin(p.dot(vec3(127.1, 311.7,  74.7))) * 43758.5453
    qy = ti.sin(p.dot(vec3(269.5, 183.3, 246.1))) * 43758.5453
    qz = ti.sin(p.dot(vec3(113.5, 271.9, 124.6))) * 43758.5453
    v  = vec3(fract(qx), fract(qy), fract(qz)) * 2.0 - 1.0  # [-1,1]
    # normalize (avoid div by zero)
    eps = 1e-6
    inv = 1.0 / ti.max(eps, ti.sqrt(v.dot(v)))
    return v * inv

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
def tetranoise(p: vec3) -> ti.f32:
    #Returns ~[0,1]
    # Skew to simplex lattice 
    s = (p.x + p.y + p.z) * (1.0 / 3.0)
    i = vec3(ti.floor(p.x + s), ti.floor(p.y + s), ti.floor(p.z + s))
    t = (i.x + i.y + i.z) * (1.0 / 6.0)
    p = vec3(p.x - (i.x - t), p.y - (i.y - t), p.z - (i.z - t))

    # Determine simplex/tetrahedron via partition planes
    i1 = step_vec(vec3(p.y, p.z, p.x), p)                 # step(p.yzx, p)
    i2 = vec3(ti.max(i1.x, 1.0 - i1.z),
              ti.max(i1.y, 1.0 - i1.x),
              ti.max(i1.z, 1.0 - i1.y))                   # max(i1, 1 - i1.zxy)
    i1 = vec3(ti.min(i1.x, 1.0 - i1.z),
              ti.min(i1.y, 1.0 - i1.x),
              ti.min(i1.z, 1.0 - i1.y))                   # min(i1, 1 - i1.zxy)

    # Offsets to the other corners in skewed space
    p1 = p - i1 + vec3(1.0/6.0, 1.0/6.0, 1.0/6.0)
    p2 = p - i2 + vec3(1.0/3.0, 1.0/3.0, 1.0/3.0)
    p3 = p - vec3(0.5, 0.5, 0.5)

    # Squared distances and falloff
    d0 = p.dot(p)
    d1 = p1.dot(p1)
    d2 = p2.dot(p2)
    d3 = p3.dot(p3)
    v0 = ti.max(0.5 - d0, 0.0)
    v1 = ti.max(0.5 - d1, 0.0)
    v2 = ti.max(0.5 - d2, 0.0)
    v3 = ti.max(0.5 - d3, 0.0)

    # Gradient dot products at corners
    g0 = hash33(i)               # i is a lattice corner in skewed coords
    g1 = hash33(i + i1)
    g2 = hash33(i + i2)
    g3 = hash33(i + vec3(1.0, 1.0, 1.0))
    d = vec3(p.dot(g0), p1.dot(g1), p2.dot(g2))  # first 3
    d4 = p3.dot(g3)

    # Combine (v^3 * 8) like your GLSL; scale and bias into [0,1]
    w0 = v0 * v0 * v0 * 8.0
    w1 = v1 * v1 * v1 * 8.0
    w2 = v2 * v2 * v2 * 8.0
    w3 = v3 * v3 * v3 * 8.0
    n = d.x * w0 + d.y * w1 + d.z * w2 + d4 * w3
    return clamp1(n * 1.732 + 0.5, 0.0, 1.0)


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

# ---------------- fBm (3D) ----------------
@ti.func
def fbm3(p: vec3, octaves: ti.i32) -> ti.f32:
    #  GLSL used 3 octaves with offsets cycling yzx
    n = 0.0
    s = 0.0
    amp = 1.0
    offs = vec3(0.0, 0.23, 0.07)
    # fixed 3 octaves to match the original; you can also loop by octaves
    for k in range(3):
        # freq doubling via exp2(k)
        freq = ti.pow(2.0, ti.cast(k, ti.f32))
        n += tetranoise(p * freq + offs.x) * amp
        s += amp
        amp *= 0.5
        # cycle offsets (yzx)
        offs = vec3(offs.y, offs.z, offs.x)
    return n / s


# ---------------- curl field ----------------
@ti.func
def curl(p: vec2, t: ti.f32) -> vec2:
    eps = 0.01
    q = p * 0.5 + vec2(t * 0.05, 0.02)  #animate input
    dx = (fbm(q + vec2(eps, 0), 4) - fbm(q - vec2(eps, 0), 4)) / (2 * eps)
    dy = (fbm(q + vec2(0, eps), 4) - fbm(q - vec2(0, eps), 4)) / (2 * eps)
    return vec2(dy, -dx)  # divergence-free

@ti.func
def curl_from_tetra(p_xy: vec2, t: ti.f32, scale: ti.f32) -> vec2:
    eps = 0.01
    qx = p_xy * scale
    # scalar field φ(x, y, t) = fbm3([x, y, t], 3)
    f_x1 = fbm3(vec3(qx.x + eps, qx.y, t), 3)
    f_x0 = fbm3(vec3(qx.x - eps, qx.y, t), 3)
    f_y1 = fbm3(vec3(qx.x, qx.y + eps, t), 3)
    f_y0 = fbm3(vec3(qx.x, qx.y - eps, t), 3)
    dfx = (f_x1 - f_x0) / (2.0 * eps)
    dfy = (f_y1 - f_y0) / (2.0 * eps)
    # 2D curl of scalar potential -> divergence-free vector
    return vec2(dfy, -dfx)


# ---------------- flow(p, t) ----------------
@ti.func
def rot2(a: ti.f32) -> ti.Matrix:
    c, s = ti.cos(a), ti.sin(a)
    return ti.Matrix([[c, -s],
                      [s,  c]])

@ti.func
def flow3(p: vec3, t: ti.f32, longer: ti.i32) -> ti.f32:
    """
    Port of your flow() with iTime -> t (seconds), LONGER as a flag.
    Returns fBm(...) value in [0,1].
    """
    # Emulate moving toward sphere surface
    p = vec3(p.x, p.y, p.z - 0.5 * p.dot(p))
    # rotate XY slowly
    R = rot2(t / 16.0)
    xy = R @ vec2(p.x, p.y)
    p = vec3(xy.x, xy.y, p.z)
    # slice speed
    z_add = 0.1 if longer != 0 else 0.15
    p = vec3(p.x, p.y, p.z + z_add * t)
    # sample fBm
    return fbm3(p * 1.5, 3)

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
def advect_and_splat(t: ti.f32, dt: ti.f32, splat_r: ti.f32, speed: ti.f32):
    for i in range(NUM_PARTICLES):
        p = pos[i]
        v = curl_from_tetra(pos[i], t, 3.0)
        prev_pos[i] = p
        p += v * dt * speed #<- per-frame speed mapped from dt

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
        steps   = ti.cast(ti.min(16.0, seg_len / 0.5 + 1.0), ti.i32)  # clamp steps to keep it cheap
        #steps   = ti.cast(seg_len, ti.i32)
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

render_image = ti.Vector.field(3, dtype=ti.f32, shape=(WIN, WIN))

@ti.kernel
def clear_to_blue():
    for i, j in render_image:
        # gentle blue; tweak if you want
        render_image[i, j] = ti.Vector([0.05, 0.10, 0.60])

# ---------------- main loop with EEG input from firebase ----------------
import argparse
import time
from eeg_filereader import LiveArousalClassifier, LiveEEGStreamFeeder      

from flask_app import make_app, start_server


#----------------- for smooth continuous mapping of particle speed -----------------
def smoothstep01(x: float) -> float:
    # cubic smoothstep in [0,1]
    return x*x*(3.0 - 2.0*x)

def map_speed_to_dt(dt: float,
                    dt_min: float = 0.0005,  # your clamp lower bound
                    dt_max: float = 0.050,   # your clamp upper bound
                    v_min: float  = 0.05,    # calm speed multiplier
                    v_max: float  = 0.60,    # stressed speed multiplier
                    use_smoothstep: bool = True) -> float:
    """
    Map the current (smoothed) dt to a particle speed
    Larger dt -> larger speed (monotonic).
    Tune v_min/v_max to taste.
    """
    if not np.isfinite(dt):
        return v_min
    x = (dt - dt_min) / (dt_max - dt_min)
    x = float(np.clip(x, 0.0, 1.0))
    if use_smoothstep:
        x = smoothstep01(x)
    return v_min + (v_max - v_min) * x


def map_ratio_to_dt(ratio: float,
                       r_min: float = 1.0,   # ratio giving min speed
                       r_max: float = 20.0,   # ratio giving max speed (your “EXTREME-STRESS” boundary)
                       dt_min: float = 0.0001, # min speed at r_min
                       dt_max: float = 0.020  # max speed at r_max (matches your dict)
                       ) -> float:
    """
    Log-space mapping with smoothstep:
      - compresses very large ratios (more perceptual)
      - gentle near endpoints
    """
    # guard
    if not np.isfinite(ratio) or ratio <= 0:
        return dt_min
    # normalize in log space
    x = (np.log(ratio) - np.log(r_min)) / (np.log(r_max) - np.log(r_min))
    x = float(np.clip(x, 0.0, 1.0))
    x = smoothstep01(x)
    return dt_min + (dt_max - dt_min) * x


def exp_smooth(current: float, target: float, real_dt: float, tau: float) -> float:
    """One-pole low-pass toward target with time-constant tau (seconds)."""
    if tau <= 0.0:
        return target
    k = 1.0 - np.exp(-max(real_dt, 0.0) / tau)
    return current + (target - current) * k



# ===================== MAIN PROCESS =============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",     type=int, default=5000,      help="Flask server port")
    parser.add_argument("--host",     type=str, default="0.0.0.0", help="Flask bind host")
    
    args = parser.parse_args()

    # ---------- EEG setup ----------
    EEG_FS   = 256.0
    BUFFER_S = 8.0 #buffer_s is the seconds of history to keep in memory
    WIN_S    = 4.0
    feeder = LiveEEGStreamFeeder(fs=EEG_FS, buffer_s=BUFFER_S)
    clf    = LiveArousalClassifier(fs=EEG_FS, lf=(4,12), hf=(13,40), win_s=WIN_S)    

    app = make_app(feeder)
    _   = start_server(app, host=args.host, port=args.port)

    window = ti.ui.Window("Tetranoise Trace", (WIN, WIN))
    canvas = window.get_canvas()
    gui    = window.get_gui()  

    #ADDING GUI CONTROLS TO TUNE THE PARAMETERS
    dt_gui_min = 0.0005
    dt_gui_max = 0.050
    v_min_gui  = 0.05     # calm multiplier (raise if motion too subtle)
    v_max_gui  = 0.30     # stressed multiplier (raise for punchier motion)
    invert_map = False    # larger dt => slower if True

    need_samples    = int(max(1, EEG_FS * WIN_S)) #samples needed to start the CN
    tn_initialiazed = False

    dt_smooth = 0.001   # start calm
    last_time = time.perf_counter()
    tau_rise  = 0.25       # faster response when stress increases
    tau_fall  = 0.60       # slower when relaxing (feels nicer)


    #---------- particle generation process starts here -----------
    t = 0.0
    while window.running:
        #only start particles when i have enough data in the buffer...
        buffered = len(feeder.buf)  # LiveEEGStreamFeeder exposes .buf (deque)
        has_enough = buffered >= need_samples

        if not has_enough:
            # Not enough data -> show blue background, no sim/easing
            gui.text(f"Live stream on :{args.port}")
            gui.text(f"Waiting for data… {buffered}/{need_samples} samples")
            # Paint blue
            clear_to_blue()
            canvas.set_image(render_image)
            window.show()
            continue

        if not tn_initialiazed: #------ this is the first time i have enough data in the buffer -----
            init_particles()
            zero_ink()
            tn_initialiazed = True

        feeder.step_once()
        state, ratio, _ = clf.update(feeder.get_buffer())
        now = time.perf_counter()
        real_dt = now - last_time
        last_time = now


        with gui.sub_window("Controls", 0.02, 0.02, 0.34, 0.30):
            gui.text("Speed mapping (dt -> speed)")
            # Clamp so min <= max
            v_min_gui = gui.slider_float("v_min", v_min_gui, 0.00, 1.50)
            v_max_gui = gui.slider_float("v_max", v_max_gui, 0.01, 2.50)
            if v_max_gui < v_min_gui:
                v_max_gui = v_min_gui + 1e-4

            dt_gui_min = gui.slider_float("dt_min", dt_gui_min, 0.0001, 0.0100)
            dt_gui_max = gui.slider_float("dt_max", dt_gui_max, 0.0100, 0.0800)
            if dt_gui_max <= dt_gui_min:
                dt_gui_max = dt_gui_min + 1e-4

            gui.text("Tip: raise v_min if calm looks too static.")


        target_dt = map_ratio_to_dt(ratio, 
                                    r_min=0.1,    r_max=20.0,
                                    dt_min=0.001, dt_max=0.050)
        tau = tau_rise if target_dt > dt_smooth else tau_fall
        dt_smooth = exp_smooth(dt_smooth, target_dt, real_dt, tau) #smoothly interpolate 
        dt_smooth = float(np.clip(dt_smooth, 0.0005, 0.08)) #safety clamp
        speed = map_speed_to_dt(dt_smooth,
                                dt_min=dt_gui_min, dt_max=dt_gui_max,
                                v_min=v_min_gui,   v_max=v_max_gui)
                
        advect_and_splat(t, dt_smooth , SPLAT_RADIUS, speed)
        decay_and_diffuse(DECAY)
        tonemap()

        canvas.set_image(rgb) # draw trail image
        # experimental: overlay particles for sparkle
        # canvas.circles(pos, radius=0.0025, color=(1.0, 1.0, 1.0))

        window.show()
        t += 0.01

        with gui.sub_window("Readout", 0.70, 0.02, 0.27, 0.12):
            gui.text(f"State: {state}  |  HF/LF: {ratio:.3f}")
            gui.text(f"speed: {speed:.3f}  |  dt:    {dt_smooth:.4f}"  )
            #.text(f"target_speed: {target_speed:.3f}  ->  speed: {speed:.3f}")



if __name__ == "__main__":
    main()