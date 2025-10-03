"""ADDING EEG INPUT (from firebase - not live)"""

import taichi as ti
import math
import numpy as np
ti.init(arch=ti.gpu)

N = 800  # Window size
NUM_PARTICLES = 20000 # Particle buffer
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

render_image = ti.Vector.field(3, dtype=ti.f32, shape=(N, N))

@ti.kernel
def clear_to_blue():
    for i, j in render_image:
        # gentle blue; tweak if you want
        render_image[i, j] = ti.Vector([0.05, 0.10, 0.60])

# ---------------- main loop with EEG input from firebase ----------------
import argparse
import time
from eeg_filereader import LiveArousalClassifier, LiveEEGStreamFeeder


# ---------- NEW: Flask ingest server ----------
from flask import Flask, request, jsonify
from threading import Thread

'''===================== FLASK app ===================='''
# ---------- NEW: Flask app factory ----------
def make_app(feeder: LiveEEGStreamFeeder):
    app = Flask(__name__)

    @app.get("/")
    def home():
        return "Awear Test app"

    @app.get("/health")
    def health():
        return jsonify(status="ok", buffered=len(feeder.buf), capacity=feeder.maxlen, fs=feeder.fs)

    @app.post("/ingest")
    def ingest():
        """
        Body: application/json
        {
          "samples": [0.001, -0.002, ...]  # batch of floats
        }
        """
        try:
            payload = request.get_json(force=True, silent=False)
            if not payload or "samples" not in payload:
                return jsonify(error="Missing 'samples' in JSON"), 400
            samples = payload["samples"]
            feeder.push(samples)
            return jsonify(ok=True, received=len(np.asarray(samples).ravel()), buffered=len(feeder.buf))
        except Exception as e:
            return jsonify(error=str(e)), 400

    return app

def start_server(app, host: str, port: int):
    th = Thread(target=lambda: app.run(host=host, port=port, threaded=True, use_reloader=False), daemon=True)
    th.start()
    return th

# ------- end of Flask --------

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

    window = ti.ui.Window("Curl Noise Particles (FIREBASE)", (N, N))
    canvas = window.get_canvas()
    gui = window.get_gui()  

    last_time = time.perf_counter()

    t     = 0.0
    dt    = 0.008         # initial value for the slider
    speed       = 0.30    # also expose speed; nice to tweak interactively
    speed_scale = 1.00
    speed_tau   = 4.0     # Smoothing time constant (seconds) for speed transitions
    
    need_samples    = int(max(1, EEG_FS * WIN_S)) #samples needed to start the CN
    cn_initialiazed = False
    
    
    """
    color_tau (slower/faster fade)
    gamma_col (how “linear” the transition feels)
    """

    # Target speeds per state (before scale)
    SPEED_BY_STATE = {
        "CALM":           0.18,
        "MOD-STRESS":     0.25,
        "HIGH-STRESS":    0.35,
        "EXTREME-STRESS": 0.50,
    }
    
    AGIT_BY_STATE = {
        "CALM":           0.00,
        "MOD-STRESS":     0.35,
        "HIGH-STRESS":    0.70,
        "EXTREME-STRESS": 1.00,
    }

    BLUE_CALM =  (0.4, 0.7, 1.0)
    RED_STRESS = (0.95, 0.10, 0.10)
    # Color smoothing
    agitation = 0.0     # current smoothed agitation (0..1)
    color_tau = 6.0     # seconds; bigger = slower color change
    gamma_col = 1.3     # perceptual shaping (>1 = gentler near calm)

    while window.running:
        # --- GUI panel
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

        if not cn_initialiazed: #------ this is the first time i have enough data in the buffer -----
            init_particles()
            cn_initialiazed = True

        with gui.sub_window("Controls", 0.02, 0.02, 0.30, 0.20):
            gui.text("Simulation")
            dt          = gui.slider_float("dt (time step)", dt,          0.001, 0.080)
            speed_scale = gui.slider_float("speed scale",    speed_scale, 0.20, 2.00)
            speed_tau   = gui.slider_float("speed tau (s)",  speed_tau,   0.5, 10.0)

            gui.text("Color transition")
            color_tau = gui.slider_float("color tau (s)", color_tau, 0.5, 15.0)
            gamma_col = gui.slider_float("color gamma", gamma_col, 0.6, 3.0)


        state = "CALM"

        feeder.step_once()
        state, ratio, changed = clf.update(feeder.get_buffer())
        now = time.perf_counter()

        # --- compute target speed from state ---
        target_base  = SPEED_BY_STATE.get(state, 0.25)
        target_speed = target_base * speed_scale

        now = time.perf_counter()
        dt_wall = now - last_time
        last_time = now
        # alpha = 1 - exp(-dt/tau): robust to varying FPS
        alpha = 1.0 - math.exp(-max(0.0, dt_wall) / max(1e-6, speed_tau))
        speed += (target_speed - speed) * alpha


        # --- color target from state, then time-based smoothing ---
        target_agit = AGIT_BY_STATE.get(state, 0.35)
        alpha_c = 1.0 - math.exp(-max(0.0, dt_wall) / max(1e-6, color_tau))
        agitation += (target_agit - agitation) * alpha_c

        # Perceptual shaping (optional)
        a = agitation ** gamma_col

        # Mix BLUE → RED
        color_rgb = (
            BLUE_CALM[0] * (1 - a) + RED_STRESS[0] * a,
            BLUE_CALM[1] * (1 - a) + RED_STRESS[1] * a,
            BLUE_CALM[2] * (1 - a) + RED_STRESS[2] * a,
        )

        #----- render -----
        step_kernel(t, dt, speed)
        canvas.circles(pos, radius=0.003, color=color_rgb)
        window.show()
        t += 0.01

        # (optional) small readout
        with gui.sub_window("Readout", 0.70, 0.02, 0.27, 0.12):
            gui.text(f"State: {state}  |  HF/LF: {ratio:.3f}" if ratio == ratio else f"State: {state}")
            gui.text(f"target_speed: {target_speed:.3f}  ->  speed: {speed:.3f}")



if __name__ == "__main__":
    main()