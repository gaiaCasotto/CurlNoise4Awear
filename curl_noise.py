#simple curl noise implementation in taichi
import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f32)   # switch to ti.cpu if needed

N = 640
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(N, N))

vec2 = ti.types.vector(2, ti.f32)
vec3 = ti.types.vector(3, ti.f32)

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

@ti.func
def saturate3(v: vec3) -> vec3:
    return vec3(saturate1(v.x), saturate1(v.y), saturate1(v.z))

# ---------- noise / fbm ----------
@ti.func
def hash21(p: vec2) -> ti.f32:
    # Cheap hash -> [0,1)
    x = ti.sin(p.dot(vec2(12.9898, 78.233))) * 43758.5453123
    return fract(x)

@ti.func
def noise(p: vec2) -> ti.f32:
    # Value noise with smoothstep interpolation
    i = ti.floor(p)
    f = p - i
    i00 = hash21(i + vec2(0.0, 0.0))
    i10 = hash21(i + vec2(1.0, 0.0))
    i01 = hash21(i + vec2(0.0, 1.0))
    i11 = hash21(i + vec2(1.0, 1.0))
    ux = f.x * f.x * (3.0 - 2.0 * f.x)
    uy = f.y * f.y * (3.0 - 2.0 * f.y)
    a = lerp(i00, i10, ux)
    b = lerp(i01, i11, ux)
    return lerp(a, b, uy)

@ti.func
def fbm(p: vec2, octaves: ti.i32) -> ti.f32:
    amp = 0.5
    freq = 1.0
    s = 0.0
    for _ in range(octaves):
        s += amp * noise(p * freq)
        freq *= 2.0
        amp *= 0.5
    return s

@ti.func
def psi(p: vec2) -> ti.f32:
    return fbm(p, 4)

# ---------- curl field ----------
@ti.func
def curl2(p: vec2, t: ti.f32) -> vec2:
    # time-warped coords for motion
    q = p + vec2(ti.sin(t * 0.7), ti.cos(t * 0.9)) * 2.0
    eps = 0.001
    dpsi_dx = (psi(q + vec2(eps, 0.0)) - psi(q - vec2(eps, 0.0))) / (2.0 * eps)
    dpsi_dy = (psi(q + vec2(0.0, eps)) - psi(q - vec2(0.0, eps))) / (2.0 * eps)
    return vec2(dpsi_dy, -dpsi_dx)  # divergence-free

@ti.kernel
def render(t: ti.f32):
    for i, j in pixels:
        uv = vec2(i, j) / N * 6.0
        v = curl2(uv, t)
        mag = min(v.norm() * 1.5, 1.0)

        # map direction to a color tint (then clamp with saturate3)
        dirx = v.x * 0.5 + 0.5
        diry = v.y * 0.5 + 0.5
        color = vec3(dirx, diry, 1.0 - dirx)
        pixels[i, j] = saturate3(color) * mag

# ---------- UI ----------
window = ti.ui.Window("Curl Noise (2D)", res=(N, N))
canvas = window.get_canvas()
t = 0.0
while window.running:
    render(t)
    canvas.set_image(pixels)
    window.show()
    t += 0.02
