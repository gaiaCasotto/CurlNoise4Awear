"""ADDING OFFLINE EEG INPUT (from file)"""
# --------- no client needed ------------

import taichi as ti
import math
ti.init(arch=ti.gpu)

# Window
N = 800
window = ti.ui.Window("Curl Noise Particles (Houdini-style)", (N, N))
canvas = window.get_canvas()
gui = window.get_gui()  

# Particle buffer
NUM_PARTICLES = 20000
pos = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PARTICLES)

vec2 = ti.types.vector(2, ti.f32)

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
    # q = p # static motion 
    q = p * 0.5 + vec2(t * 0.02, t * 0.02)  # translate field 
    dx = (fbm(q + vec2(eps, 0), 4) - fbm(q - vec2(eps, 0), 4)) / (2 * eps)
    dy = (fbm(q + vec2(0, eps), 4) - fbm(q - vec2(0, eps), 4)) / (2 * eps)
    return vec2(dy, -dx)  # divergence-free

# ---------------- kernels ----------------
@ti.kernel
def init_particles():
    for i in range(NUM_PARTICLES):
        pos[i] = ti.Vector([ti.random(), ti.random()]) * 2.0 - 1.0  # [-1,1]^2

@ti.kernel
def step_kernel(t: ti.f32, dt: ti.f32, speed: ti.f32):
    for i in range(NUM_PARTICLES):
        v = curl(pos[i] * 3.0, t)
        pos[i] += v * dt * speed
        # wrap-around to keep particles inside domain
        for k in ti.static(range(2)):
            if pos[i][k] < -1.0: pos[i][k] += 2.0
            if pos[i][k] > 1.0: pos[i][k] -= 2.0

# ---------------- main loop with offline EEG input ----------------
import argparse
import time
from eeg_filereader import OfflineEEGFeeder, LiveArousalClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg", action="append", help="Path to EEG .txt file (can repeat)", default=None)
    args = parser.parse_args()

    # ---------- EEG setup ----------
    if args.eeg:
        EEG_FILES = args.eeg                     # from launcher (can be 1+ files)
    else:
        EEG_FILES = ["../eeg_files/fake_eeg_longblocks_calmfirst.txt"]  # fallback

    EEG_FS = 256.0
    try:
        feeder = OfflineEEGFeeder(EEG_FILES, fs=EEG_FS, chunk=32, speed=1.0, loop=True, buffer_s=8.0)
        clf    = LiveArousalClassifier(fs=EEG_FS, lf=(4,12), hf=(13,40), win_s=4.0)
        eeg_available = True
    except Exception as e:
        print("EEG feeder disabled:", e)
        eeg_available = False
        feeder = None; clf = None

    

    init_particles()

    outer_steps = 0
    last_time = time.perf_counter()

    t     = 0.0
    dt    = 0.016     # initial value for the slider
    speed = 0.30    # also expose speed; nice to tweak interactively

    while window.running:
        outer_steps += 1

        # --- GUI panel
        with gui.sub_window("Controls", 0.02, 0.02, 0.30, 0.20):
            gui.text("Simulation")
            dt = gui.slider_float("dt (time step)", dt, 0.001, 0.080)
            speed = gui.slider_float("speed scale", speed, 0.05, 1.00)

        state = "CALM"

        feeder.step_once()
        state, ratio, changed = clf.update(feeder.get_buffer())
        now = time.perf_counter()

        if state == "CALM":
            speed = 0.18
        elif state == "MOD-STRESS":
           
            speed = 0.25
        elif state == "HIGH-STRESS":
            speed = 0.35
        else: #"EXTREME-STRESS"
            speed = 0.5

        #gui.text(f"Live stream on :{args.port}")
        gui.text(f"HF/LF: {ratio:.3f}  |  State: {state}")

        #----- render -----
        step_kernel(t, dt, speed)
        canvas.circles(pos, radius=0.003, color=(0.4, 0.7, 1.0))
        window.show()
        t += 0.01



if __name__ == "__main__":
    main()