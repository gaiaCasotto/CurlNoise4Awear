# insipired by : https://www.shadertoy.com/view/mlsSWH
# Translation of the two GLSL shaders (BUFFER A + IMAGE) into Python/Taichi.
# - BUFFER A: builds a feedback texture by warping UVs along curl(fBm flow)
# - IMAGE: applies lighting, curvature, and colored highlights on top

import taichi as ti
import math

ti.init(arch=ti.gpu)

# -------------------- Config --------------------
WIN = 900                   # window size (square window)
IMG_RES = 1024              # internal framebuffer resolution
LONGER = False              # corresponds to #define LONGER in GLSL
PINK = False                # corresponds to #define PINK in GLSL

# Feedback blending “frames” (affects spiral length)
N_FRAMES = 32 if LONGER else 16

# -------------------- Window / GUI --------------------
window = ti.ui.Window("Curl Flow (Feedback) + Lighting (Taichi)", (WIN, WIN))
canvas = window.get_canvas()
gui = window.get_gui()

# -------------------- Fields --------------------
# Ping-pong RGB buffers for the feedback pass (BUFFER A)
buf_a0 = ti.Vector.field(3, dtype=ti.f32, shape=(IMG_RES, IMG_RES))
buf_a1 = ti.Vector.field(3, dtype=ti.f32, shape=(IMG_RES, IMG_RES))

# Final shaded image (IMAGE pass)
final_img = ti.Vector.field(3, dtype=ti.f32, shape=(IMG_RES, IMG_RES))

# Time / frame counters
time_s = ti.field(dtype=ti.f32, shape=())
iframe = ti.field(dtype=ti.i32, shape=())


f32  = ti.f32
vec2 = ti.types.vector(2, f32)
vec3 = ti.types.vector(3, f32)
mat2 = ti.types.matrix(2, 2, f32)


# ---- runtime params (host updates these each frame) ----
exposure   = ti.field(ti.f32, shape=())  # tone-mapping exposure
gamma_pow  = ti.field(ti.f32, shape=())  # display gamma
ambient    = ti.field(ti.f32, shape=())  # ambient term in IMAGE pass
hi_scale   = ti.field(ti.f32, shape=())  # highlight strength
curv_boost = ti.field(ti.f32, shape=())  # curvature contrast
shade_gain = ti.field(ti.f32, shape=())  # BUFFER A shading gain
shade_bias = ti.field(ti.f32, shape=())  # BUFFER A shading bias
saturation = ti.field(ti.f32, shape=())  # overall color saturation


# -------------------- Helpers --------------------
@ti.func
def rot2(a: f32) -> ti.Matrix:
    c = ti.cos(a)
    s = ti.sin(a)
    return ti.Matrix([[c, -s], [s, c]])

@ti.func
def fract(x: f32) -> ti.f32:
    return x - ti.floor(x)

@ti.func
def lerp(a: f32, b: f32, t: f32) -> f32:
    return a + (b - a) * t

@ti.func
def lerp3(a: vec3, b: vec3, t: f32) -> vec3:
    return a + (b - a) * t

@ti.func
def fract_v3(v: vec3) -> vec3:
    return vec3([fract(v[0]), fract(v[1]), fract(v[2])])

@ti.func
def dot3(a: vec3, b: vec3) -> f32:
    return a.dot(b)

# Cheap vec3->vec3 hash (as in the GLSL)
@ti.func
def hash33(p: vec3) -> vec3:
    n = ti.sin(dot3(p, vec3([27.0, 57.0, 111.0])))
    v = vec3([2097152.0, 262144.0, 32768.0]) * n
    v = fract_v3(v) * 2.0 - 1.0
    return v


@ti.func
def reinhard_tonemap(c: vec3, expv: f32) -> vec3:
    # classic tonemap: 1 - exp(-c * exposure)
    return vec3([1.0, 1.0, 1.0]) - ti.exp(-c * expv)

@ti.func
def apply_saturation(c: vec3, sat: f32) -> vec3:
    luma = c.dot(vec3([0.299, 0.587, 0.114]))
    g = vec3([luma, luma, luma])
    return lerp3(g, c, sat)

# ----------------------- NOISE ---------------------

# Tetrahedral "Simplex-like" noise 
@ti.func
def tetra_noise(p_in: vec3) -> f32:
    p = p_in
    i = ti.floor(p + (p.sum()) * (1.0 / 3.0))  # dot(p, (1/3))
    p = p - i + (i.sum()) * (1.0 / 6.0)        # p -= i - dot(i, 1/6)

    # partition to determine simplex corners
    i1 = vec3([0.0, 0.0, 0.0])
    i2 = vec3([0.0, 0.0, 0.0])

    # step(p.yzx, p)
    a = vec3([1.0 if p[1] <= p[0] else 0.0,
                   1.0 if p[2] <= p[1] else 0.0,
                   1.0 if p[0] <= p[2] else 0.0])
    i1 = a
    i2 = ti.max(i1, vec3([1.0, 1.0, 1.0]) - vec3([i1[2], i1[0], i1[1]]))
    i1 = ti.min(i1, vec3([1.0, 1.0, 1.0]) - vec3([i1[2], i1[0], i1[1]]))

    p1 = p - i1 + 1.0 / 6.0
    p2 = p - i2 + 1.0 / 3.0
    p3 = p - 0.5

    v0 = ti.max(0.5 - p.dot(p), 0.0)
    v1 = ti.max(0.5 - p1.dot(p1), 0.0)
    v2 = ti.max(0.5 - p2.dot(p2), 0.0)
    v3 = ti.max(0.5 - p3.dot(p3), 0.0)

    d0 = p.dot(hash33(i))
    d1 = p1.dot(hash33(i + i1))
    d2 = p2.dot(hash33(i + i2))
    d3 = p3.dot(hash33(i + vec3([1.0, 1.0, 1.0])))

    v0 = v0 * v0 * v0 * 8.0
    v1 = v1 * v1 * v1 * 8.0
    v2 = v2 * v2 * v2 * 8.0
    v3 = v3 * v3 * v3 * 8.0

    res = (d0 * v0 + d1 * v1 + d2 * v2 + d3 * v3) * 1.732 + 0.5
    return ti.min(ti.max(res, 0.0), 1.0)

@ti.func
def fBm(p: vec3) -> ti.f32:
    # 3 octaves as in GLSL
    n = 0.0
    s = 0.0
    a = 1.0
    offs = vec3([0.0, 0.23, 0.07])
    for _ in range(3):
        n += tetra_noise(p * a + offs) * a
        s += a
        a *= 2.0
        offs = vec3([offs[1], offs[2], offs[0]])
    # weights: first octave *1, second *0.5, third *0.25 — dividing by sum of weights (1+0.5+0.25=1.75)
    return n / (1.0 + 0.5 + 0.25)

@ti.func
def flow(p: vec3, t: ti.f32) -> ti.f32:
    # p is vec3(uva, 0)
    pz = p
    pz[2] -= pz.dot(pz) * 0.5
    # rotate xy
    R = rot2(t / 16.0)
    xy = vec2([pz[0], pz[1]])
    xy = R @ xy
    pz[0], pz[1] = xy[0], xy[1]
    # advance along z
    if LONGER:
        pz[2] += 0.1 * t
    else:
        pz[2] += 0.15 * t
    return fBm(pz * 1.5)

f32  = ti.f32
vec2 = ti.types.vector(2, f32)
vec3 = ti.types.vector(3, f32)

@ti.func
def wrap_idx(i: ti.i32, n: ti.i32) -> ti.i32:
    # robust positive modulo
    return (i % n + n) % n

@ti.func
def sample_rgb(tex: ti.template(), uv: vec2) -> vec3:
    # wrap uv to [0,1)
    u = uv[0] - ti.floor(uv[0])
    v = uv[1] - ti.floor(uv[1])

    # map to texel space (IMG_RES x IMG_RES), centered at texel centers
    W = IMG_RES
    H = IMG_RES
    x = u * W - 0.5
    y = v * H - 0.5

    ix = ti.cast(ti.floor(x), ti.i32)
    iy = ti.cast(ti.floor(y), ti.i32)
    fx = x - ti.cast(ix, f32)
    fy = y - ti.cast(iy, f32)

    ix0 = wrap_idx(ix,   W)
    iy0 = wrap_idx(iy,   H)
    ix1 = wrap_idx(ix+1, W)
    iy1 = wrap_idx(iy+1, H)

    c00 = tex[ix0, iy0]
    c10 = tex[ix1, iy0]
    c01 = tex[ix0, iy1]
    c11 = tex[ix1, iy1]

    c0 = c00 * (1.0 - fx) + c10 * fx
    c1 = c01 * (1.0 - fx) + c11 * fx
    return c0 * (1.0 - fy) + c1 * fy


# -------------------- BUFFER A (feedback / advection) --------------------
@ti.kernel
def buffer_a_step(dst: ti.template(), src: ti.template(), t: ti.f32, frame: int):
    # iResolution.x == iResolution.y == IMG_RES; window is square in this implementation
    res = vec2([IMG_RES * 1.0, IMG_RES * 1.0])

    for i, j in dst:
        # fragCoord style float pixel center (match typical shader behavior)
        frag = vec2([ti.cast(i, ti.f32) + 0.5, ti.cast(j, ti.f32) + 0.5])

        uv = frag / res                    # 0..1
        uva = (frag - res * 0.5) / res[1] # centered, aspect correct

        # curl of scalar field: curl = (df/dy, -df/dx)
        e = 0.005
        p = vec3([uva[0], uva[1], 0.0])

        dx = (flow(p + vec3([e, 0.0, 0.0]), t) - flow(p - vec3([e, 0.0, 0.0]), t)) / (2.0 * e)
        dy = (flow(p + vec3([0.0, e, 0.0]), t) - flow(p - vec3([0.0, e, 0.0]), t)) / (2.0 * e)
        curl = vec2([dy, -dx])

        # update UV by curl (scale by aspect correction)
        uv += curl * 0.006 * vec2([res[1] / res[0], 1.0])

        # Base transcendental color pattern
        snNs = ti.sin(uv[0] * 8.0 - ti.cos(uv[1] * 12.0)) * 0.25 \
             + ti.sin(uv[1] * 8.0 - ti.cos(uv[0] * 12.0)) * 0.25 \
             + 0.5  # rough match to dot(vec2(.25))
        col = 0.5 + 0.45 * vec3([
            ti.cos(6.2831 * snNs / 6.0 + 0.0 * 0.8),
            ti.cos(6.2831 * snNs / 6.0 + 1.2 * 0.8),
            ti.cos(6.2831 * snNs / 6.0 + 2.0 * 0.8),
        ])

        # shading via flow (uncurled variant in original)
        # adjustable shading to avoid blowout
        shade = flow(p, t) * shade_gain[None] + shade_bias[None]   # e.g. 1.4 * f + (-0.2)
        shade = ti.max(shade, 0.0)                                  # no negative dark crush
        col *= shade

        # control saturation before blending
        col = apply_saturation(col, saturation[None])

        # blend with previous frame sample
        prev = sample_rgb(src, uv)
        if frame > 0:
            col = prev * (1.0 - 1.0 / N_FRAMES) + col * (1.0 / N_FRAMES)

        # clamp non-negative like GLSL
        dst[i, j] = ti.max(col, vec3([0.0, 0.0, 0.0]))

# -------------------- IMAGE (lighting / curvature / highlights) --------------------
        
# Luma from the feedback texture at uv (wrap + bilinear already handled by sample_rgb)
@ti.func
def height_luma(src: ti.template(), uv: vec2) -> f32:
    col = sample_rgb(src, uv)
    return col.dot(vec3([0.299, 0.587, 0.114]))

@ti.kernel
def image_pass(dst: ti.template(), src: ti.template()):
    res = vec2([float(IMG_RES), float(IMG_RES)])

    for i, j in dst:
        frag = vec2([ti.cast(i, f32) + 0.5, ti.cast(j, f32) + 0.5])
        uv   = frag / res

        col = src[i, j]

        # base height (luma)
        height = col.dot(vec3([0.299, 0.587, 0.114]))

        # directional derivative along normalize(vec2(1,2))
        step_vec = vec2([1.0, 2.0])
        step_vec = step_vec / ti.sqrt(step_vec.dot(step_vec))
        step_uv  = uv - step_vec * 0.001 * vec2([res[0] / res[1], 1.0])
        sample_off = height_luma(src, step_uv)

        b  = ti.max(sample_off - height, 0.0) / 0.001
        b2 = ti.max(height - sample_off, 0.0) / 0.001
        hi_col = (vec3([0.02, 0.20, 1.0]) * (b * 0.8) + vec3([1.0, 0.2, 0.1]) * (b2 * 0.3)) * hi_scale[None]

        # gradient/curvature taps (use a proper Taichi vec, not a Python tuple)
        e = vec2([0.0045, 0.0])

        t_mx = height_luma(src, uv - e)                       # left
        t_px = height_luma(src, uv + e)                       # right
        t_my = height_luma(src, uv - vec2([e[1], e[0]]))      # down
        t_py = height_luma(src, uv + vec2([e[1], e[0]]))      # up

        # surface normal from two differential vectors
        vx = vec3([e[0] * 2.0, 0.0, t_mx - t_px])
        vy = vec3([0.0, -e[0] * 2.0, t_py - t_my])
        sn = vx.cross(vy).normalized()

        # curvature term
        amp  = 0.7
        curv = (height * 4.0 - (t_mx + t_px + t_my + t_py)) / e[0] / 2.0 * amp + 0.5
        curv = ti.min(ti.max(curv, 0.0), 1.0)

        # lighting
        ld     = vec3([-0.5, 1.0, -1.0]).normalized()
        ndotl  = ti.max(sn.dot(ld), 0.0)

        # optional PINK variant as a compile-time branch
        if ti.static(PINK):
            # mix(col.xzy, col, ndotl)
            col = col * ndotl + vec3([col[0], col[2], col[1]]) * (1.0 - ndotl)

        col = col * ndotl + hi_col + ambient[None]
        col = col * lerp(1.0, curv + 0.2, curv_boost[None])  # gentler curvature


        # tone map and gamma
        col = reinhard_tonemap(col, exposure[None])
        col = ti.pow(ti.max(col, 0.0), 1.0 / gamma_pow[None])
        # final saturation
        col = apply_saturation(col, saturation[None])
        dst[i, j] = col


# -------------------- Init --------------------
@ti.kernel
def init_buffers():
    for i, j in buf_a0:
        # start with a gentle colorful base (same transcendental color as in BUFFER A)
        u = (i + 0.5) / IMG_RES
        v = (j + 0.5) / IMG_RES
        snNs = ti.sin(u * 8.0 - ti.cos(v * 12.0)) * 0.25 \
             + ti.sin(v * 8.0 - ti.cos(u * 12.0)) * 0.25 \
             + 0.5
        col = 0.5 + 0.45 * vec3([
            ti.cos(6.2831 * snNs / 6.0 + 0.0 * 0.8),
            ti.cos(6.2831 * snNs / 6.0 + 1.2 * 0.8),
            ti.cos(6.2831 * snNs / 6.0 + 2.0 * 0.8),
        ])
        buf_a0[i, j] = col
        buf_a1[i, j] = col
    time_s[None] = 0.0
    iframe[None] = 0

# -------------------- Main Loop --------------------
init_buffers()
exposure[None]   = 1.0   # start neutral
gamma_pow[None]  = 2.2   # real display gamma
ambient[None]    = 0.15  # less ambient than 0.4
hi_scale[None]   = 0.6   # reduce highlight punch
curv_boost[None] = 0.6   # less curvature contrast
shade_gain[None] = 1.4   # was effectively 2.0
shade_bias[None] = -0.2  # was -0.5 -> brighter base without clipping
saturation[None] = 0.9   # slightly desaturate to prevent blooming


use_a0_as_src = True
prev_t = 0.0

while window.running:
    # simple fixed dt for visual stability
    dt = 1.0 / 60.0
    time_s[None] += dt
    iframe[None] += 1

    t = time_s[None]
    f = iframe[None]

    if use_a0_as_src:
        buffer_a_step(buf_a1, buf_a0, t, f)
        image_pass(final_img, buf_a1)
    else:
        buffer_a_step(buf_a0, buf_a1, t, f)
        image_pass(final_img, buf_a0)

    use_a0_as_src = not use_a0_as_src

    # Draw
    canvas.set_image(final_img)
    # UI toggles
    with gui.sub_window("Controls", 0.02, 0.02, 0.34, 0.36):
        gui.text("Tone / Light")
        exposure[None]   = gui.slider_float("Exposure", exposure[None],   0.2, 4.0)
        gamma_pow[None]  = gui.slider_float("Gamma",    gamma_pow[None],  1.6, 2.6)
        ambient[None]    = gui.slider_float("Ambient",  ambient[None],    0.00, 0.40)
        hi_scale[None]   = gui.slider_float("Hi Scale", hi_scale[None],   0.0, 1.5)
        curv_boost[None] = gui.slider_float("CurvBoost",curv_boost[None], 0.0, 1.5)
        gui.text("Buffer A Shading")
        shade_gain[None] = gui.slider_float("Gain",     shade_gain[None], 0.6, 2.2)
        shade_bias[None] = gui.slider_float("Bias",     shade_bias[None], -0.6, 0.2)
        saturation[None] = gui.slider_float("Saturation", saturation[None], 0.5, 1.2)


    window.show()
