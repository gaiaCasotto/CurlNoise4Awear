#Houdini’s Curl Noise SOP

from eeg_filereader import  OfflineEEGFeeder, LiveArousalClassifier

import taichi as ti
import math, time, random
import numpy as np
ti.init(arch=ti.gpu)

# Window
N = 800

window = ti.ui.Window("Curl Noise Particles (Houdini-style)", (N, N))
canvas = window.get_canvas()
gui    = window.get_gui()


# Particle buffer
NUM_PARTICLES = 20000
pos = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PARTICLES)

vec2 = ti.types.vector(2, ti.f32)

#------- for eeg coloring ------------
TAU_SECS = 3.0 #for alpha calculation
stress_mix    = ti.field(dtype=ti.f32, shape=())  # eased 0..1 used by rendering
stress_target = ti.field(dtype=ti.f32, shape=())  # instantaneous target from classifier
stress_state  = ti.field(dtype=ti.i32,  shape=()) # 0..3 discrete state for UI
stress_mix[None]    = 0.0
stress_target[None] = 0.0
stress_state[None]  = 0



@ti.kernel
def ease_stress(alpha: ti.f32):
    # Exponential smoothing like in your RD:
    # x += alpha * (target - x)
    stress_mix[None] += alpha * (stress_target[None] - stress_mix[None])

# Compute the same “two-palette” blend you used, but on CPU
def compute_particle_color_from_stress(s: float):
    s = max(0.0, min(1.0, float(s)))

    # ---- Stress palette (your RD version) ----
    deep_red   = np.array([0.35, 0.00, 0.05], dtype=np.float32)
    hot_orange = np.array([1.00, 0.35, 0.00], dtype=np.float32)
    bright_yel = np.array([1.00, 0.85, 0.20], dtype=np.float32)

    # emulate their two-step red→orange→yellow via t^0.6
    t = 0.6
    # Use ‘intensity’ proxy; here just set th=0.6 constant so the hue is stable;
    # you can also modulate by overall motion speed if you want dynamics.
    th = 0.6
    col_ext = (1.0 - th) * deep_red + th * hot_orange
    col_ext = (1.0 - th) * col_ext   + th * bright_yel

    # ---- Calm palette (simple blueish ramp as in RD) ----
    # In the RD you used: col_calm = [t, t, 0.8 - t]; here pick a pleasing calm blue
    col_calm = np.array([0.25, 0.45, 0.95], dtype=np.float32)

    # Blend calm→stress by s
    col = (1.0 - s) * col_calm + s * col_ext

    # Optional: small spec/illum feel — brighten a touch as stress grows
    spec = 0.15 + 0.15 * s
    col = np.clip(col * (1.0 + spec), 0.0, 1.0)
    return tuple(col.tolist())




#============================== CURL NOISE =============================
# ---------- helpers ----------
@ti.func
def fract(x: ti.f32) -> ti.f32:
    return x - ti.floor(x)

@ti.func
def lerp(a: ti.f32, b: ti.f32, t: ti.f32) -> ti.f32:
    return a * (1.0 - t) + b * t

@ti.func
def saturate1(x: ti.f32) -> ti.f32:
    return ti.max(0.0, ti.min(1.0, x))

def clamp(x, lo, hi): return max(lo, min(hi, x))



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
    q = p * 0.5 + vec2(t*0.05, 0.0)  # animate input
    dx = (fbm(q + vec2(eps, 0), 4) - fbm(q - vec2(eps, 0), 4)) / (2*eps)
    dy = (fbm(q + vec2(0, eps), 4) - fbm(q - vec2(0, eps), 4)) / (2*eps)
    return vec2(dy, -dx)  # divergence-free

# ---------------- kernels ----------------
@ti.kernel
def init_particles():
    for i in range(NUM_PARTICLES):
        pos[i] = ti.Vector([ti.random(), ti.random()]) * 2.0 - 1.0  # [-1,1]^2

@ti.kernel
def step(t: ti.f32, dt: ti.f32, advect_gain: ti.f32, field_scale: ti.f32, swirl_freq: ti.f32, time_speed: ti.f32):
    for i in range(NUM_PARTICLES):
        p = pos[i]
        # allow a few knobs (kept simple; default to 1.0)
        eps = 0.01
        q = p * (0.5 * swirl_freq) * field_scale + vec2(t * 0.05 * time_speed, 0.0)
        dx = (fbm(q + vec2(eps, 0), 4) - fbm(q - vec2(eps, 0), 4)) / (2*eps)
        dy = (fbm(q + vec2(0, eps), 4) - fbm(q - vec2(0, eps), 4)) / (2*eps)
        v = vec2(dy, -dx)
        p += v * dt * advect_gain
        if p.x < -1.0: p.x += 2.0
        if p.x >  1.0: p.x -= 2.0
        if p.y < -1.0: p.y += 2.0
        if p.y >  1.0: p.y -= 2.0
        pos[i] = p

# ---------------- main loop ----------------
def main():

     # ---------- EEG setup ----------
    EEG_FILES = [
        #"../eeg_files/1_horror_movie_data_filtered.txt",
        #"../eeg_files/2_vipassana_data_filtered.txt",
        #"../eeg_files/3_hot_tub_data_filtered.txt",
        #"../eeg_files/fake_eeg_longblocks.txt" #stressed first
        "../eeg_files/fake_eeg_longblocks_calmfirst.txt"
    ]
    EEG_FS = 256.0
    try:
        feeder = OfflineEEGFeeder(EEG_FILES, fs=EEG_FS, chunk=32, speed=5.0, loop=True, buffer_s=8.0)
        clf    = LiveArousalClassifier(fs=EEG_FS, lf=(4,12), hf=(13,40), win_s=4.0)
        eeg_available = True #if the file exists, if the reader can read it
    except Exception as e:
        print("EEG feeder disabled:", e)
        eeg_available = False
        feeder = None; clf = None
    

    # initialize rendering
    init_particles()
    t0 = time.perf_counter()
    last = t0
    #------- parameters --------
    advect_gain = 0.30
    field_scale = 1.00
    swirl_freq  = 1.00
    time_speed  = 1.00
    radius_base = 0.0028
    t = 0.0
    while window.running:
        now     = time.perf_counter()
        dt_wall = now - last
        last    = now
        t = now - t0

        alpha = 1.0 - math.exp(-dt_wall / TAU_SECS)

        state = "STRESSED"
        ratio = float("nan")
        if eeg_available:
            feeder.step_once()
            state, ratio, changed = clf.update(feeder.get_buffer())

        # Map state → stress_target (exactly your ladder)
        if state == "CALM":
            stress_state[None]  = 0
            stress_target[None] = 0.00
        elif state == "MOD-STRESS":
            stress_state[None]  = 1
            stress_target[None] = 0.33
        elif state == "HIGH-STRESS":
            stress_state[None]  = 2
            stress_target[None] = 0.66
        else:  # "EXTREME-STRESS" (or anything else)
            stress_state[None]  = 3
            stress_target[None] = 1.00

        # Smooth stress like RD
        ease_stress(alpha)
        s = float(stress_mix[None])

        # Optionally modulate motion a bit with stress (subtle)
        advect = advect_gain * (0.8 + 0.6 * s)
        swirl  = swirl_freq  * (0.9 + 0.4 * s)
        scale  = field_scale * (0.9 + 0.4 * s)
        speed  = time_speed  * (0.9 + 0.4 * s)

        # ---- Step simulation ----
        step(t=t, dt=0.016,
             advect_gain=advect,
             field_scale=scale,
             swirl_freq=swirl,
             time_speed=speed)

        # ---- Compute color  ----
        eeg_color = compute_particle_color_from_stress(s)
        eeg_radius = radius_base * (0.9 + 0.6 * (1.0 - s))  # calmer = slightly larger dots

        gui.text(f"EEG: {'ON' if eeg_available else 'OFF'}")
        gui.text(f"HF/LF: {ratio:.3f}" if not math.isnan(ratio) else "HF/LF: n/a")
        states = ["CALM","MOD-STRESS","HIGH-STRESS","EXTREME-STRESS"]
        st = states[min(max(int(stress_state[None]),0),3)]
        gui.text(f"State: {st}  |  stress_mix: {s:.2f}")
        gui.text(f"Color: {tuple(round(c,3) for c in eeg_color)}")


        canvas.circles(pos, radius=eeg_radius, color=eeg_color)
        window.show()

        # after window.show()
        if eeg_available:
            feeder.sleep_dt()   # <- uses feeder’s fs/chunk/speed to throttle



if __name__ == "__main__":
    main()