"""_______ADDING THE AWEAR DATABASE EEG CONTROLS____"""


#=========NEW CODE============
#from eeg_filereader import OfflineEEGFeeder, LiveArousalClassifier
from eeg_filereader import LiveArousalClassifier, LiveEEGStreamFeeder

from flask import Flask, request, jsonify
from threading import Thread
import argparse
from sys import argv
import math
import time

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


import taichi as ti
ti.init(arch=ti.gpu)


# Window size
N = 800
  
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

# ---------------- main loop with flask CLI----------------
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fs",       type=float, default=256.0,     help="Incoming sample rate (Hz) for the live stream")
    parser.add_argument("--buffer-s", type=float, default=8.0,       help="Seconds of history to keep in memory")
    parser.add_argument("--port",     type=int,   default=5000,      help="Flask server port")
    parser.add_argument("--host",     type=str,   default="0.0.0.0", help="Flask bind host")
    
    args = parser.parse_args()
    
    EEG_FS = float(args.fs)
    WIN_S  = 4.0  #stress-threshold
    feeder = LiveEEGStreamFeeder(fs=EEG_FS, buffer_s=args.buffer_s)
    clf    = LiveArousalClassifier(fs=EEG_FS, lf=(4, 12), hf=(13, 40), win_s=WIN_S)

    app = make_app(feeder)
    _   = start_server(app, host=args.host, port=args.port)

    window = ti.ui.Window("Curl Noise Particles (Houdini-style)", (N, N))
    canvas = window.get_canvas()
    gui    = window.get_gui() 

    init_particles()
    t     = 0.0
    dt    = 0.016     # initial values for the slider
    speed = 0.3      

    while window.running:
        # --- GUI panel
        with gui.sub_window("Controls", 0.02, 0.02, 0.30, 0.20):
            gui.text("Simulation")
            dt    = gui.slider_float("dt (time step)", dt, 0.001, 0.050)
            speed = gui.slider_float("speed scale", speed, 0.05, 1.00)
            gui.text("larger dt or speed => faster motion")

        step_kernel(t, dt, speed)

        canvas.circles(pos, radius=0.003, color=(0.4, 0.7, 1.0))
        window.show()
        t += 0.01


if __name__ == "__main__":
    main()