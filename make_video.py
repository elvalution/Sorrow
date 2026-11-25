#!/usr/bin/env python3
"""Sorrow Valentine lyric video.

usage: python make_video.py song.mp3 out.mp4 [--preview SECONDS] [--fps 24]
Needs: ffmpeg on PATH, pip install pillow numpy
Lyrics come from lyrics.txt (blank line = pause between sections).
Line timings come from timings.json ([[start, end, "text"], ...]); edit it to
fine-tune sync, then re-run. Delete it to regenerate the automatic estimate.
"""
import argparse, json, math, os, random, re, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1280, 720
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------- timing ----------
def syllables(s):
    return sum(max(1, len(re.findall(r"[aeiouy]+", w.lower()))) for w in re.findall(r"[A-Za-z']+", s)) or 1

def load_lines(path):
    lines, groups = [], []
    for raw in open(path, encoding="utf-8").read().splitlines():
        if raw.strip():
            lines.append(raw.strip())
    return lines

def estimate_timings(lines, dur, vocal_start=17.0, last_chorus=132.2, outro=178.7):
    """Anchor-based estimate: main body, final chorus (5 lines), outro (3 lines)."""
    vocal_start = min(vocal_start, dur * 0.1)
    last_chorus, outro = min(last_chorus, dur * 0.69), min(outro, dur * 0.93)
    segs = [(lines[:-8], vocal_start, last_chorus),
            (lines[-8:-3], last_chorus, last_chorus + (outro - last_chorus) * 0.8),
            (lines[-3:], outro, dur - 1.0)]
    out = []
    for seg, a, b in segs:
        wts = np.array([syllables(l) for l in seg], float)
        edges = a + (b - a) * np.concatenate([[0], np.cumsum(wts)]) / wts.sum()
        for l, s, e in zip(seg, edges[:-1], edges[1:]):
            out.append([round(s, 2), round(min(e - 0.15, s + 6.0), 2), l])
    return out

# ---------- drawing helpers ----------
def vgrad(top, bot):
    t = np.linspace(0, 1, H)[:, None, None]
    a = np.array(top, float)[None, None, :] * (1 - t) + np.array(bot, float)[None, None, :] * t
    return Image.fromarray(np.repeat(a, W, 1).astype(np.uint8))

def person(d, x, base, h, col, coat=None, flip=1, step=0.0):
    s = h / 100.0
    sw = math.sin(step) * 8 * s
    d.ellipse([x - 9 * s, base - h, x + 9 * s, base - h + 18 * s], fill=col)
    d.polygon([(x - 13 * s, base - h + 22 * s), (x + 13 * s, base - h + 22 * s),
               (x + 16 * s, base - 38 * s), (x - 16 * s, base - 38 * s)], fill=col)
    d.line([(x - 5 * s, base - 38 * s), (x - 5 * s + sw, base)], fill=col, width=int(9 * s))
    d.line([(x + 5 * s, base - 38 * s), (x + 5 * s - sw, base)], fill=col, width=int(9 * s))
    d.line([(x - 14 * s, base - h + 26 * s), (x - 16 * s, base - 50 * s)], fill=col, width=int(6 * s))
    if coat:
        d.line([(x + 14 * s, base - h + 26 * s), (x + 22 * s * flip, base - 40 * s)], fill=col, width=int(6 * s))
        d.polygon([(x + 18 * s * flip, base - 42 * s), (x + 34 * s * flip, base - 42 * s),
                   (x + 36 * s * flip, base - 8 * s), (x + 20 * s * flip, base - 8 * s)], fill=coat)

def heart(d, cx, cy, r, fill, half=None):
    pts = []
    for i in range(0, 360, 6):
        t = math.radians(i)
        x = 16 * math.sin(t) ** 3
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        pts.append((x, y))
    if half == "L":
        pts = [(x, y) for x, y in pts if x <= 0.3]
    elif half == "R":
        pts = [(x, y) for x, y in pts if x >= -0.3]
    k = r / 16.0
    d.polygon([(cx + x * k, cy + y * k) for x, y in pts], fill=fill)

def petal(d, x, y, a, s, col):
    c, sn = math.cos(a), math.sin(a)
    pts = [(0, -s), (s * .7, 0), (0, s * 1.2), (-s * .7, 0)]
    d.polygon([(x + px * c - py * sn, y + px * sn + py * c) for px, py in pts], fill=col)

# ---------- scenes: f(t, p) -> RGB image; t = seconds in scene, p = 0..1 ----------
RND = random.Random(7)
BOKEH = [(RND.uniform(0, W), RND.uniform(0, H * .75), RND.uniform(18, 60),
          RND.choice([(200, 70, 90), (230, 150, 90), (90, 120, 200), (210, 200, 160)]),
          RND.uniform(0, 6)) for _ in range(26)]
DROPS = [(RND.uniform(0, W), RND.uniform(0, H), RND.uniform(250, 520), RND.uniform(14, 34)) for _ in range(170)]
GLASS = [(RND.uniform(0, W), RND.uniform(-200, 0), RND.uniform(20, 60), RND.uniform(18, 60)) for _ in range(26)]
BG = {}

def bg(key, top, bot):
    if key not in BG:
        BG[key] = vgrad(top, bot)
    return BG[key].copy()

def scene_rain(t, p):
    im = bg("rain", (14, 18, 38), (30, 34, 58))
    ov = Image.new("RGB", (W, H), (0, 0, 0)); d = ImageDraw.Draw(ov)
    for x, y, r, c, ph in BOKEH:
        k = .55 + .35 * math.sin(t * .8 + ph)
        d.ellipse([x - r, y - r, x + r, y + r], fill=tuple(int(v * k * .55) for v in c))
    ov = ov.filter(ImageFilter.GaussianBlur(9))
    im = Image.blend(im, Image.eval(Image.blend(im, ov, .8), lambda v: v), .9)
    im = Image.fromarray(np.clip(np.asarray(im, float) + np.asarray(ov, float) * .6, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for x, y0, v, ln in DROPS:
        y = (y0 + t * v) % (H + 40) - 20
        d.line([(x, y), (x - 3, y + ln)], fill=(150, 170, 210), width=1)
    for x, y0, v, ln in GLASS:
        y = y0 + ((t * v * .35) % (H + 220))
        d.ellipse([x - 3, y, x + 3, y + 7], fill=(190, 205, 235))
        d.line([(x, y - ln * 3), (x, y)], fill=(95, 110, 150), width=1)
    return im

def scene_parting(t, p):
    im = bg("part", (38, 32, 60), (120, 80, 90))
    d = ImageDraw.Draw(im)
    d.ellipse([W / 2 - 70, 330, W / 2 + 70, 470], fill=(210, 150, 130))
    d.rectangle([0, 560, W, H], fill=(16, 14, 24))
    for i in range(7):
        x = 90 + i * 190
        d.rectangle([x, 410, x + 5, 560], fill=(20, 18, 30))
    gap = 60 + 460 * p ** 0.8
    person(d, W / 2 - gap, 600, 215, (10, 8, 16), coat=(120, 25, 45), flip=-1, step=t * 4)
    person(d, W / 2 + gap, 600, 205, (10, 8, 16), step=t * 4 + 1.5)
    hy = 380 - (t * 18) % 300
    heart(d, W / 2 + 4 * math.sin(t), hy, 16, (170, 35, 60))
    d.line([(W / 2 + 4 * math.sin(t), hy + 14), (W / 2, hy + 90)], fill=(80, 70, 90), width=1)
    return im

def scene_rose(t, p):
    im = bg("rose", (16, 16, 24), (36, 30, 40))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 600, W, H], fill=(22, 18, 24))
    sway = math.sin(t * .6) * 5
    sx, hx, hy = W / 2, W / 2 + 120 + sway, 330 + 40 * min(1, p * 3)
    pts = [(sx + (hx - sx) * u ** 2.2, 640 - (640 - hy) * u) for u in np.linspace(0, 1, 30)]
    d.line(pts, fill=(40, 70, 50), width=8)
    d.polygon([(sx + 12, 560), (sx + 70, 520), (sx + 30, 580)], fill=(34, 60, 44))
    for r, col in [(60, (90, 18, 36)), (46, (120, 24, 46)), (30, (70, 12, 30)), (16, (50, 8, 22))]:
        d.ellipse([hx - r, hy - r * .9, hx + r, hy + r * .9], fill=col)
    for i in range(14):
        ph = (t * .35 + i / 14) % 1.0
        petal(d, hx + 120 * math.sin(i * 2.1) * ph + 30 * math.sin(t + i), hy + 20 + (640 - hy) * ph,
              t * 1.3 + i, 14, (140, 28, 55))
    for i in range(10):
        petal(d, 160 + i * 110 + 20 * (i % 3), 628 + (i % 3) * 14, i * 1.3, 12, (100, 20, 42))
    return im

def scene_table(t, p):
    im = bg("table", (12, 12, 20), (28, 22, 30))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 470, W, H], fill=(44, 30, 30))
    d.rectangle([0, 470, W, 480], fill=(70, 48, 44))
    d.rectangle([W - 330, 260, W - 150, 640], fill=(30, 22, 26))
    d.rectangle([W - 350, 250, W - 130, 275], fill=(36, 26, 30))
    fl = 1 + .08 * math.sin(t * 9) + .05 * math.sin(t * 23)
    glow = Image.new("RGB", (W, H), (0, 0, 0)); g = ImageDraw.Draw(glow)
    g.ellipse([W / 2 - 260, 150, W / 2 + 260, 640], fill=(int(120 * fl), int(70 * fl), 30))
    glow = glow.filter(ImageFilter.GaussianBlur(70))
    im = Image.fromarray(np.clip(np.asarray(im, float) + np.asarray(glow, float) * .6, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.rectangle([W / 2 - 8, 380, W / 2 + 8, 470], fill=(220, 210, 190))
    d.ellipse([W / 2 - 7 * fl, 345, W / 2 + 7 * fl, 385], fill=(255, 190, 90))
    for gx in (W / 2 - 230, W / 2 + 190):
        d.polygon([(gx, 400), (gx + 50, 400), (gx + 38, 465), (gx + 12, 465)], outline=(170, 180, 205), fill=(40, 44, 60))
        d.line([(gx + 25, 465), (gx + 25, 485)], fill=(170, 180, 205), width=2)
    d.polygon([(W / 2 + 40, 440), (W / 2 + 120, 436), (W / 2 + 124, 470), (W / 2 + 44, 474)], fill=(150, 40, 60))
    heart(d, W / 2 + 82, 455, 9, (230, 200, 205))
    return im

def scene_door(t, p):
    im = bg("door", (10, 10, 16), (20, 18, 24))
    d = ImageDraw.Draw(im)
    d.rectangle([W / 2 - 170, 120, W / 2 + 170, 640], fill=(46, 34, 30))
    d.rectangle([W / 2 - 140, 150, W / 2 + 140, 640], fill=(26, 20, 20))
    d.line([(W / 2, 150), (W / 2, 640)], fill=(18, 14, 14), width=3)
    glow = Image.new("RGB", (W, H), (0, 0, 0)); g = ImageDraw.Draw(glow)
    g.polygon([(W / 2 - 140, 640), (W / 2 + 140, 640), (W / 2 + 520, 720), (W / 2 - 520, 720)], fill=(150, 100, 40))
    g.rectangle([W / 2 - 4, 150, W / 2 + 4, 640], fill=(255, 190, 100))
    glow = glow.filter(ImageFilter.GaussianBlur(26))
    im = Image.fromarray(np.clip(np.asarray(im, float) + np.asarray(glow, float) * (.75 + .1 * math.sin(t * 2)), 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 640, W, H], fill=(20, 16, 20))
    d.ellipse([W / 2 + 100, 400, W / 2 + 118, 418], fill=(190, 150, 70))
    d.ellipse([W / 2 + 180, 655, W / 2 + 196, 671], outline=(200, 170, 90), width=3)
    d.line([(W / 2 + 196, 663), (W / 2 + 230, 663)], fill=(200, 170, 90), width=3)
    return im

def scene_heart(t, p):
    im = bg("heart", (18, 14, 30), (46, 26, 44))
    d = ImageDraw.Draw(im)
    sep = 6 + 150 * min(1, p * 1.4)
    for sgn, half in ((-1, "L"), (1, "R")):
        cx = W / 2 + sgn * sep; cy = 330 + sgn * 10 * math.sin(t * .8)
        heart(d, cx, cy, 190, (150, 30, 58), half)
    d.line([(W / 2 + 6 * math.sin(t * 3), 150), (W / 2 - 6, 330), (W / 2 + 8, 420), (W / 2, 520)], fill=(20, 14, 26), width=4)
    for i in range(60):
        x = (i * 211 + t * (14 + i % 7)) % W; y = (i * 97 + t * (30 + i % 11)) % H
        d.ellipse([x, y, x + 3, y + 3], fill=(210, 190, 220))
    return im

SCENES = [scene_rain, scene_parting, scene_rose, scene_table, scene_door, scene_heart]

def make_vignette():
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    return np.clip(1.15 - .55 * r ** 1.8, .25, 1)[:, :, None].astype(np.float32)

# ---------- text ----------
def find_font(size, override=None):
    cands = [override] if override else []
    cands += ["/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "C:/Windows/Fonts/georgiab.ttf", "C:/Windows/Fonts/arialbd.ttf",
              "/system/fonts/Roboto-Bold.ttf", "/system/fonts/DroidSerif-Bold.ttf",
              "/System/Library/Fonts/Supplemental/Georgia Bold.ttf"]
    for c in cands:
        if c and os.path.exists(c):
            return ImageFont.truetype(c, size)
    return ImageFont.load_default(size)

def draw_text(im, text, alpha, font, dy=0):
    if alpha <= 0.01:
        return im
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    words, rows, cur = text.split(), [], ""
    for w in words:
        t2 = (cur + " " + w).strip()
        if d.textlength(t2, font=font) > W - 160 and cur:
            rows.append(cur); cur = w
        else:
            cur = t2
    rows.append(cur)
    lh = int(font.size * 1.3); y0 = H - 150 - lh * (len(rows) - 1) / 2 + dy
    band = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ImageDraw.Draw(band).rectangle(
        [0, y0 - lh, W, y0 + lh * len(rows)], fill=(0, 0, 0, int(110 * alpha)))
    lay = Image.alpha_composite(lay, band.filter(ImageFilter.GaussianBlur(12))); d = ImageDraw.Draw(lay)
    for i, r in enumerate(rows):
        x = (W - d.textlength(r, font=font)) / 2; y = y0 + i * lh - lh / 2
        d.text((x + 2, y + 3), r, font=font, fill=(0, 0, 0, int(200 * alpha)))
        d.text((x, y), r, font=font, fill=(250, 236, 240, int(255 * alpha)))
    return Image.alpha_composite(im.convert("RGBA"), lay).convert("RGB")

def title_card(im, t, dur_total, font_big, font_small):
    a = min(1, t / 1.5) * min(1, max(0, (6 - t) / 1.5)) if t < 6 else 0
    if a <= 0:
        return im
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    for txt, f, y in (("Sorrow Valentine Day", font_big, 270), ("a Valentine's Day breakup song", font_small, 360)):
        x = (W - d.textlength(txt, font=f)) / 2
        d.text((x + 2, y + 3), txt, font=f, fill=(0, 0, 0, int(200 * a)))
        d.text((x, y), txt, font=f, fill=(250, 220, 228, int(255 * a)))
    return Image.alpha_composite(im.convert("RGBA"), lay).convert("RGB")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio"); ap.add_argument("out")
    ap.add_argument("--lyrics", default=os.path.join(HERE, "lyrics.txt"))
    ap.add_argument("--timings", default=os.path.join(HERE, "timings.json"))
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--scene-len", type=float, default=16.0)
    ap.add_argument("--preview", type=float, default=0, help="render only first N seconds")
    ap.add_argument("--start", type=float, default=0, help="preview start offset")
    ap.add_argument("--font")
    a = ap.parse_args()

    dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                         "-of", "csv=p=0", a.audio]).decode().strip())
    if os.path.exists(a.timings):
        tim = json.load(open(a.timings, encoding="utf-8"))
    else:
        tim = estimate_timings(load_lines(a.lyrics), dur)
        json.dump(tim, open(a.timings, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print("wrote", a.timings, "(edit start/end seconds to fine-tune)")

    font = find_font(54, a.font); fbig = find_font(76, a.font); fsm = find_font(32, a.font)
    vig = make_vignette(); rng = np.random.default_rng(1)
    total = a.preview or (dur - a.start)
    n = int(total * a.fps)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(a.fps),
           "-i", "-", "-ss", str(a.start), "-t", str(total), "-i", a.audio, "-c:v", "libx264", "-preset", "medium",
           "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-af",
           f"afade=t=out:st={max(0, total - 2)}:d=2", "-shortest", a.out]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    fade = 1.5
    for i in range(n):
        T = a.start + i / a.fps
        k = int(T // a.scene_len); lt = T - k * a.scene_len
        order = [0, 1, 2, 3, 4, 5]
        sc = SCENES[order[k % 6]]
        im = sc(T, min(1, lt / a.scene_len))
        if lt > a.scene_len - fade:  # crossfade into the next scene
            nxt = SCENES[order[(k + 1) % 6]](T, 0)
            im = Image.blend(im, nxt, (lt - (a.scene_len - fade)) / fade)
        arr = np.asarray(im, np.float32)
        arr = arr * .78 + arr.mean(2, keepdims=True) * .22  # muted, sad palette
        arr[:, :, 2] *= 1.06
        arr = arr * vig + rng.normal(0, 4, (H, W, 1)).astype(np.float32)
        arr *= min(1, T / 2.0) * min(1, max(0, (dur - T) / 2.0))
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        im = title_card(im, T, dur, fbig, fsm)
        for s, e, text in tim:
            if s - .4 <= T <= e + .5:
                al = min(1, (T - (s - .4)) / .5, (e + .5 - T) / .5)
                im = draw_text(im, text, max(0, al), font, dy=int(8 * (1 - min(1, (T - s + .4) / .5))))
        ff.stdin.write(im.tobytes())
        if i % 240 == 0:
            print(f"{i}/{n} frames", flush=True)
    ff.stdin.close(); ff.wait()
    print("done ->", a.out)

if __name__ == "__main__":
    main()
