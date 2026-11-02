"""Synthesized soundtrack for the TBL promo video (60 s, 120 BPM, D major).
Soft but driving: warm pads, felt keys, plucked arps, gentle drums, UI foley."""
import numpy as np
from scipy import signal
from scipy.io import wavfile
import timeline as TL

SR = 48000
DUR = TL.DURATION
N = int(SR * DUR)
BEAT = TL.BEAT
rng = np.random.default_rng(42)

music = np.zeros((2, N))
sfx = np.zeros((2, N))
send = np.zeros((2, N))     # reverb send
dsend = np.zeros((2, N))    # delay send


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def tt(n):
    return np.arange(n) / SR


def place(bus, t0, x, pan=0.0, gain=1.0, rev=0.0, dly=0.0):
    """Mix mono signal x at time t0 with equal-power pan into bus (+ sends)."""
    i0 = int(round(t0 * SR))
    if i0 >= N:
        return
    if i0 < 0:
        x = x[-i0:]
        i0 = 0
    x = x[: N - i0] * gain
    a = (pan + 1) * np.pi / 4
    l, r = np.cos(a) * x, np.sin(a) * x
    bus[0, i0:i0 + len(x)] += l
    bus[1, i0:i0 + len(x)] += r
    if rev:
        send[0, i0:i0 + len(x)] += l * rev
        send[1, i0:i0 + len(x)] += r * rev
    if dly:
        dsend[0, i0:i0 + len(x)] += l * dly
        dsend[1, i0:i0 + len(x)] += r * dly


def sos(kind, f, order=2):
    if isinstance(f, (list, tuple)):
        return signal.butter(order, [f[0] / (SR / 2), f[1] / (SR / 2)], btype=kind, output='sos')
    return signal.butter(order, f / (SR / 2), btype=kind, output='sos')


def filt(x, kind, f, order=2):
    return signal.sosfilt(sos(kind, f, order), x)


def tv_filter(x, fc_fn, kind='lowpass', block=256, q_bw=None):
    """Time-varying Butterworth filter, coefficients updated per block."""
    y = np.zeros_like(x)
    zi = None
    for i in range(0, len(x), block):
        t = i / SR
        f = float(np.clip(fc_fn(t), 30, SR / 2 * 0.95))
        if kind == 'bandpass':
            bw = q_bw or 0.6
            lo, hi = f / (1 + bw), min(f * (1 + bw), SR / 2 * 0.95)
            s = signal.butter(1, [lo / (SR / 2), hi / (SR / 2)], btype='bandpass', output='sos')
        else:
            s = signal.butter(2, f / (SR / 2), btype=kind, output='sos')
        if zi is None:
            zi = np.zeros((s.shape[0], 2))
        y[i:i + block], zi = signal.sosfilt(s, x[i:i + block], zi=zi)
    return y


# ------------------------------------------------------------------ harmony
CH = {
    'D': dict(pad=[57, 62, 66, 69], bass=38, arp=[62, 66, 69, 74, 76]),
    'A': dict(pad=[57, 61, 64, 69], bass=33, arp=[61, 64, 69, 73, 76]),
    'Bm': dict(pad=[59, 62, 66, 71], bass=35, arp=[62, 66, 71, 74, 78]),
    'G': dict(pad=[59, 62, 67, 71], bass=31, arp=[62, 67, 71, 74, 79]),
}
PROG = ['D', 'A', 'Bm', 'G']


def chord_at(t):
    bar = int(t // TL.BAR)
    if bar == 26:
        return 'G' if t < 53.0 else 'A'
    if bar >= 27:
        return 'D'
    return PROG[bar % 4]


def segments():
    segs, t = [], 0.0
    cur, start = chord_at(0), 0.0
    while t < DUR:
        c = chord_at(t)
        if c != cur:
            segs.append((start, t, cur))
            cur, start = c, t
        t += 0.25
    segs.append((start, DUR, cur))
    return segs


# ------------------------------------------------------------------ instruments
def saw_additive(f, n, detune_cents=0.0, fmax=5000):
    t = tt(n)
    ff = f * 2 ** (detune_cents / 1200)
    y = np.zeros(n)
    k = 1
    ph = rng.uniform(0, 2 * np.pi)
    while k * ff < fmax and k < 40:
        y += np.sin(2 * np.pi * k * ff * t + ph * k) / k
        k += 1
    return y


def pad_note(m, dur, tail=1.2):
    n = int((dur + tail) * SR)
    f = mtof(m)
    y = sum(saw_additive(f, n, d) for d in (-8, 0, 8)) / 3
    t = tt(n)
    env = np.minimum(1, t / 0.35) * np.where(t < dur, 1, np.exp(-(t - dur) / 0.35))
    return y * env


def keys(m, dur=2.5, vel=1.0):
    """Felt piano: inharmonic partials with frequency-dependent decay."""
    n = int(dur * SR)
    t = tt(n)
    f = mtof(m)
    y = np.zeros(n)
    for k in range(1, 8):
        fk = k * f * np.sqrt(1 + 0.0004 * k * k)
        if fk > 9000:
            break
        y += np.sin(2 * np.pi * fk * t) * (1 / k ** 1.6) * np.exp(-t * (1.2 + 1.1 * k))
    y *= np.minimum(1, t / 0.004)
    y += filt(rng.standard_normal(n), 'lowpass', 1200) * np.exp(-t / 0.01) * 0.05  # felt thump
    return filt(y, 'lowpass', 3500) * vel


def pluck(m, dur=0.45, bright=1.0):
    n = int(dur * SR)
    t = tt(n)
    f = mtof(m)
    y = np.zeros(n)
    for k in range(1, 12):
        if k * f > 12000:
            break
        y += np.sin(2 * np.pi * k * f * t) / k * np.exp(-t * (5 + 3.2 * k / bright))
    return y * np.minimum(1, t / 0.002)


def bell(m, dur=3.0, index=2.5):
    n = int(dur * SR)
    t = tt(n)
    f = mtof(m)
    I = index * np.exp(-t / 0.6)
    y = np.sin(2 * np.pi * f * t + I * np.sin(2 * np.pi * f * 3.5 * t))
    y += 0.4 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / 0.4)
    return y * np.exp(-t / (dur / 3.5)) * np.minimum(1, t / 0.002)


def kick(gain=1.0, dur=0.45):
    n = int(dur * SR)
    t = tt(n)
    f = 44 + 96 * np.exp(-t / 0.035)
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = np.sin(ph) * np.exp(-t / 0.2)
    y += filt(rng.standard_normal(n), 'highpass', 2500) * np.exp(-t / 0.003) * 0.15
    return np.tanh(y * 1.4) * gain


def clap(dur=0.35):
    n = int(dur * SR)
    t = tt(n)
    nz = filt(rng.standard_normal(n), 'bandpass', (900, 5000))
    env = np.zeros(n)
    for d in (0, 0.009, 0.018):
        env += np.where(t >= d, np.exp(-(t - d) / 0.012), 0) * 0.6
    env += np.where(t >= 0.018, np.exp(-(t - 0.018) / 0.11), 0)
    body = np.sin(2 * np.pi * 185 * t) * np.exp(-t / 0.05) * 0.3
    return nz * env * 0.5 + body


def hat(dur=0.06, decay=0.022):
    n = int(dur * SR)
    t = tt(n)
    return filt(rng.standard_normal(n), 'highpass', 7500) * np.exp(-t / decay)


def tick(m, vel=1.0, decay=0.05):
    n = int(0.18 * SR)
    t = tt(n)
    f = mtof(m)
    y = np.sin(2 * np.pi * f * t) * np.exp(-t / decay) + 0.25 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / (decay / 2))
    return y * np.minimum(1, t / 0.001) * vel


def click(freq=2200):
    n = int(0.05 * SR)
    t = tt(n)
    y = filt(rng.standard_normal(n), 'highpass', 1800) * np.exp(-t / 0.002) * 0.6
    y += np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.012) * 0.5
    return y


def thud():
    n = int(0.4 * SR)
    t = tt(n)
    f = 70 + 80 * np.exp(-t / 0.03)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.09) + filt(rng.standard_normal(n), 'lowpass', 900) * np.exp(-t / 0.03) * 0.3


def whoosh(dur=0.6, f0=300, f1=4000, rev=False):
    n = int(dur * SR)
    x = rng.standard_normal(n)
    fc = lambda t: f0 * (f1 / f0) ** (t / dur)
    y = tv_filter(x, fc, 'bandpass', q_bw=0.9)
    t = tt(n)
    env = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 2
    if rev:
        env = (t / dur) ** 2.5
    return y * env


def riser(dur, f0=250, f1=6000):
    n = int(dur * SR)
    t = tt(n)
    x = rng.standard_normal(n)
    y = tv_filter(x, lambda tx: f0 * (f1 / f0) ** (tx / dur) ** 1.6, 'bandpass', q_bw=0.5)
    ph = 2 * np.pi * np.cumsum(180 * (6 ** ((t / dur) ** 2))) / SR
    tone = np.sin(ph) * 0.12
    env = (t / dur) ** 2.2
    return (y + tone) * env


def boom(dur=2.2):
    n = int(dur * SR)
    t = tt(n)
    f = 34 + 70 * np.exp(-t / 0.08)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.7)
    y += filt(rng.standard_normal(n), 'lowpass', 1500) * np.exp(-t / 0.25) * 0.35
    return np.tanh(y * 1.3)


def pop_snd(m):
    n = int(0.16 * SR)
    t = tt(n)
    f = mtof(m) * (1 + 0.6 * np.exp(-t / 0.012))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.05)


# ------------------------------------------------------------------ sections
def in_range(t, *ranges):
    return any(a <= t < b for a, b in ranges)


DRUMS = [(12.0, 47.75), (48.0, 53.75)]
GROOVE_LIGHT = (18.0, 26.0)
PLUCK = [(12.0, 18.0), (26.0, 42.0), (48.0, 53.75)]

# ---- pads (with cutoff automation)
pad_bus = np.zeros(N)
for a, b, c in segments():
    if a >= 53.75:
        continue
    b = min(b, 53.75)
    for m in CH[c]['pad']:
        y = pad_note(m, b - a, tail=0.6)
        i0 = int(a * SR)
        seg = y[: N - i0]
        pad_bus[i0:i0 + len(seg)] += seg * 0.09


def pad_cut(t):
    if t < 12:
        return 500 + 1300 * (t / 12) ** 2
    if t < 46:
        return 2400
    if t < 53.75:
        return 2400 + 3000 * ((t - 46) / 7.75) ** 2
    return 1500


pad_bus = tv_filter(pad_bus, pad_cut)
# sidechain from the kick pattern
duck = np.ones(N)
t_all = tt(N)
for b0 in np.arange(12.0, 53.75, BEAT):
    if in_range(b0, *DRUMS):
        i0 = int(b0 * SR)
        k = np.exp(-np.arange(int(0.4 * SR)) / SR / 0.13)
        seg = 1 - 0.55 * k
        duck[i0:i0 + len(seg)] = np.minimum(duck[i0:i0 + len(seg)], seg[: N - i0])
pad_st = np.vstack([pad_bus, np.roll(pad_bus, int(0.011 * SR))]) * duck
music += pad_st * 0.9
send += pad_st * 0.35

# ---- final chord at 54 (big and warm)
for m in [38, 50, 57, 62, 66, 69, 74]:
    y = pad_note(m, 3.2, tail=2.8) * (0.07 if m > 40 else 0.1)
    y = filt(y, 'lowpass', 2600)
    place(music, 54.0, y, pan=rng.uniform(-.3, .3), rev=0.6)

# ---- bass
for a, b, c in segments():
    if not in_range(a, (12.0, 53.75)) and a < 12:
        # intro: sustained soft sub from bar 3
        if a >= 4.0:
            n = int((b - a + 0.3) * SR)
            t = tt(n)
            f = mtof(CH[c]['bass'])
            y = (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t)) * np.minimum(1, t / 0.4) * np.where(t < b - a, 1, np.exp(-(t - (b - a)) / 0.1))
            place(music, a, y, gain=0.10 * min(1, (a - 4) / 4 + 0.3))
        continue
for b0 in np.arange(12.0, 53.75, BEAT / 2):
    if not in_range(b0, *DRUMS):
        continue
    c = chord_at(b0)
    f = mtof(CH[c]['bass'])
    n = int(BEAT / 2 * SR)
    t = tt(n)
    off = (round(b0 / (BEAT / 2)) % 2) == 1
    env = np.minimum(1, t / 0.006) * np.exp(-t / (0.22 if off else 0.3))
    y = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) + 0.12 * np.sin(6 * np.pi * f * t)
    place(music, b0, y * env, gain=0.2 if off else 0.23)

# ---- intro felt keys (0 - 12)
for b0 in np.arange(0.0, 11.5, BEAT):
    c = chord_at(b0)
    notes = [m + 12 for m in CH[c]['pad']]
    step = int(round(b0 / BEAT))
    if b0 < 4.0 and step % 2:
        continue
    m = notes[step % len(notes)]
    vel = 0.26 if b0 < 6 else 0.3
    place(music, b0, keys(m, 3.0, vel), pan=(-.35 if step % 2 else .35), rev=0.5)
    if b0 >= 6.0:
        m2 = notes[(step + 2) % len(notes)] + 12
        place(music, b0 + BEAT / 2, keys(m2, 2.0, 0.14), pan=(.4 if step % 2 else -.4), rev=0.5, dly=0.3)
    if b0 % TL.BAR == 0:
        place(music, b0, keys(CH[c]['pad'][0], 3.5, 0.2), rev=0.4)

# ---- light section (18-26): keys chords on the off-beats instead of plucks
for b0 in np.arange(18.0, 26.0, BEAT):
    c = chord_at(b0)
    for m in CH[c]['pad'][1:]:
        place(music, b0 + BEAT / 2, keys(m + 12, 1.2, 0.09), pan=rng.uniform(-.4, .4), rev=0.35)

# ---- plucked 16th arps
PAT = [0, 2, 3, 1, 2, 4, 3, 1, 0, 2, 3, 4, 3, 2, 1, 2]
for i, b0 in enumerate(np.arange(12.0, 53.75, BEAT / 4)):
    if not in_range(b0, *PLUCK):
        continue
    c = chord_at(b0)
    step = int(round((b0 - 12.0) / (BEAT / 4)))
    m = CH[c]['arp'][PAT[step % 16]] + (12 if b0 >= 48 else 0)
    acc = 1.0 if step % 4 == 0 else (0.7 if step % 2 == 0 else 0.55)
    lvl = 0.085 * acc
    if 48 <= b0:
        lvl *= 0.85 + 0.4 * (b0 - 48) / 5.75
    place(music, b0, pluck(m, 0.4, 1.0 + (0.8 if b0 >= 48 else 0)), pan=(-.45 if step % 2 else .45), gain=lvl, rev=0.2, dly=0.35)

# ---- drums
for b0 in np.arange(0.0, 53.75, BEAT / 4):
    step = int(round(b0 / (BEAT / 4)))
    beat_pos = step % 16
    if in_range(b0, *DRUMS):
        light = in_range(b0, GROOVE_LIGHT)
        if beat_pos % 4 == 0:
            place(music, b0, kick(), gain=0.42 if not light else 0.36)
        if beat_pos in (4, 12):
            place(music, b0, clap(), gain=0.16, pan=0.05, rev=0.25)
        if beat_pos % 4 == 2:
            place(music, b0, hat(0.08, 0.035), gain=0.07, pan=0.25)
        if b0 >= 46.0 or b0 >= 48.0:
            if beat_pos % 2 == 1:
                place(music, b0, hat(), gain=0.035, pan=-0.25)
        elif not light and beat_pos % 2 == 1 and beat_pos not in (3,):
            place(music, b0, hat(0.05, 0.015), gain=0.025, pan=-0.3)
    # intro: soft hats from 6 s, heartbeat kick from 8 s
    elif 6.0 <= b0 < 11.5:
        if beat_pos % 4 == 2:
            place(music, b0, hat(0.08, 0.03), gain=0.035 + 0.02 * (b0 - 6) / 5.5, pan=0.2)
        if b0 >= 8.0 and beat_pos % 4 == 0:
            place(music, b0, filt(kick(), 'lowpass', 400), gain=0.22 * (b0 - 7) / 4.5)

# snare roll 46 - 47.75
roll_t = list(np.arange(46.0, 47.0, BEAT / 4)) + list(np.arange(47.0, 47.75, BEAT / 8))
for k, b0 in enumerate(roll_t):
    place(music, b0, clap(0.2), gain=0.03 + 0.12 * k / len(roll_t), pan=0.1 * np.sin(k), rev=0.2)

# ---- risers and impacts
place(sfx, 8.5, riser(3.45), gain=0.16, rev=0.3)
place(sfx, 50.0, riser(3.75, 300, 9000), gain=0.18, rev=0.3)
place(sfx, 45.5, riser(2.25, 400, 7000), gain=0.08)
# reverse swell into the drop
swell = sum(keys(m + 12, 1.0, 0.25) for m in CH['Bm']['pad'])
swell = signal.fftconvolve(swell, np.exp(-tt(int(1.2 * SR)) / 0.35) * rng.standard_normal(int(1.2 * SR)) * 0.05)[: int(1.0 * SR)]
place(sfx, 11.0, swell[::-1] * np.linspace(0, 1, len(swell)) ** 2, gain=0.8)
place(sfx, 12.0, boom(), gain=0.3, rev=0.4)
place(sfx, 48.0, boom(1.6), gain=0.28, rev=0.3)
place(sfx, 54.0, boom(2.8), gain=0.3, rev=0.5)

# ---- logo chimes
for d, m, g in [(0.0, 74, 0.12), (0.3, 81, 0.08), (0.45, 78, 0.08), (0.8, 86, 0.05)]:
    place(sfx, 12.0 + d, bell(m, 3.0), gain=g, pan=(d - 0.4), rev=0.6)
for i, d in enumerate((0.75, 0.87, 0.99)):
    place(sfx, 12.0 + d, tick(74 + [0, 4, 7][i], 0.5, 0.04), gain=0.15, pan=(i - 1) * 0.3, rev=0.3)
place(sfx, 12.3, pop_snd(76), gain=0.12, pan=-0.3)
place(sfx, 12.45, pop_snd(81), gain=0.12, pan=0.3)
for d, m, g in [(0.0, 74, 0.13), (0.35, 78, 0.08), (0.5, 81, 0.08), (0.9, 86, 0.06), (2.0, 90, 0.035)]:
    place(sfx, 54.0 + d, bell(m, 4.0), gain=g, pan=(d - 0.5) * 0.6, rev=0.7)
for i, m in enumerate([62, 66, 69, 74]):
    place(music, 55.5 + i * 0.25, keys(m + 12, 3.5, 0.18), pan=(i - 1.5) * 0.25, rev=0.6)
place(music, 57.5, keys(78, 3.0, 0.12), rev=0.7)

# ---- word cascade ticks (pitched on chord tones, ascending)
def cascade(times, t_end, base_gain):
    for i, t0 in enumerate(times):
        if t0 >= t_end:
            break
        c = chord_at(t0)
        tones = CH[c]['arp']
        m = tones[i % len(tones)] + 12 * ((i // len(tones)) % 2)
        nxt = times[i + 1] - t0 if i + 1 < len(times) else 0.5
        dec = float(np.clip(nxt * 0.35, 0.012, 0.07))
        g = base_gain * (0.55 + 0.45 * min(1, nxt / 0.25))
        place(sfx, t0, tick(m, 1.0, dec), gain=g, pan=0.35 * np.sin(i * 1.3), rev=0.25)


cascade(TL.INTRO_TIMES[1:], TL.INTRO_END, 0.16)
place(sfx, 1.0, tick(74, 1.0, 0.12), gain=0.14, rev=0.5)
cascade(TL.OUTRO_TIMES, TL.OUTRO_END, 0.13)
# counter ticks 48.1 - 49.25: accelerating rattle, then a chime on the infinity sign
t0 = 48.1
while t0 < 49.25:
    k = (t0 - 48.1) / 1.15
    place(sfx, t0, tick(81 + int(k * 12), 1.0, 0.01), gain=0.035 + 0.03 * k, pan=rng.uniform(-.3, .3))
    t0 += 0.09 * (1 - k) + 0.018
for d, m, g in [(0.0, 86, 0.09), (0.12, 90, 0.06), (0.24, 93, 0.05)]:
    place(sfx, 49.25 + d, bell(m, 2.2, 1.6), gain=g, pan=(d - 0.12) * 3, rev=0.6)
place(sfx, 49.25, pop_snd(74), gain=0.1)

# ---- transitions
for t0, d, pan in [(17.55, 0.55, -0.5), (18.0, 0.5, 0.6), (25.55, 0.5, -0.6), (26.3, 0.45, -0.4), (26.95, 0.45, 0.4),
                   (33.55, 0.5, 0.0), (34.0, 0.4, 0.3), (35.72, 0.45, -0.3), (37.72, 0.45, 0.3), (39.72, 0.45, -0.3),
                   (41.7, 0.5, 0.2), (47.4, 0.6, 0.0)]:
    place(sfx, t0, whoosh(d, 350, 5000), gain=0.12, pan=pan, rev=0.2)
place(sfx, 47.35, whoosh(0.65, 200, 6000, rev=True), gain=0.14)

# ---- UI foley (scene 3)
place(sfx, 18.3, whoosh(0.35, 800, 3000), gain=0.05, pan=0.6)
place(sfx, 19.02, thud(), gain=0.22)
place(sfx, 19.08, pop_snd(81), gain=0.08)
for t0 in (20.1, 20.9):
    place(sfx, t0, click(), gain=0.18, pan=0.3)
for i, t0 in enumerate([20.28, 20.39, 20.48, 20.57, 20.67]):
    place(sfx, t0, tick(93 + (i % 2) * 2, 1.0, 0.008), gain=0.05, pan=0.35)
place(sfx, 21.6, click(1700), gain=0.2, pan=0.1)
place(sfx, 21.62, pop_snd(74), gain=0.07)
# progress shimmer: faint rising sparkles while translating
for i, t0 in enumerate(np.arange(21.9, 24.6, BEAT / 4)):
    c = chord_at(t0)
    m = CH[c]['arp'][(i * 3) % 5] + 24
    place(sfx, t0, tick(m, 1.0, 0.03), gain=0.018 + 0.02 * i / 22, pan=np.sin(i) * 0.6, rev=0.3)
for d, m in [(0.0, 74), (0.1, 78), (0.2, 81), (0.3, 86)]:
    place(sfx, 24.72 + d, bell(m, 1.8, 1.2), gain=0.07, pan=(d - 0.15) * 2, rev=0.5)

# ---- chip grid: ascending melody on eighth notes, one note per chip
SCALE = [62, 64, 66, 69, 71, 74, 76, 78, 81, 83, 86, 88, 90, 93, 95, 98]
for i, t0 in enumerate(TL.cues()['chips']):
    c = chord_at(t0)
    m = SCALE[i]
    place(sfx, t0, pluck(m, 0.6, 1.5), gain=0.09, pan=((i % 4) - 1.5) * 0.3, rev=0.3, dly=0.25)
    place(sfx, t0, pop_snd(m - 12), gain=0.04)
# feature cards pops (scene 5)
for base in (34.0, 36.0, 38.0, 40.0):
    for j in range(4):
        place(sfx, base + 0.3 + j * 0.125, tick(81 + j * 2, 1.0, 0.02), gain=0.035, pan=0.4)
# provider chips
for a in (26.9, 27.55):
    for j in range(7):
        place(sfx, a + j * 0.125, tick(86 + (j % 3) * 3, 1.0, 0.015), gain=0.028, pan=0.5 if a > 27 else -0.5)

# ------------------------------------------------------------------ fx + master
# ping-pong delay (dotted eighth)
dt = int(BEAT * 0.75 * SR)
dl = np.zeros((2, N))
for k in range(1, 6):
    g = 0.42 ** k
    src = dsend[(k + 1) % 2]
    dl[k % 2, dt * k:] += src[: N - dt * k] * g
dl = np.vstack([filt(dl[0], 'lowpass', 4500), filt(dl[1], 'lowpass', 4500)])
send += dl * 0.4

# reverb: decorrelated exponential noise IR
ir_n = int(3.0 * SR)
t_ir = tt(ir_n)
irs = []
for ch in range(2):
    ir = rng.standard_normal(ir_n) * np.exp(-6.9 * t_ir / 2.6)
    ir = filt(ir, 'lowpass', 5500)
    ir[: int(0.015 * SR)] = 0
    irs.append(ir / np.sqrt(np.sum(ir ** 2)))
rev = np.vstack([signal.fftconvolve(send[0], irs[0])[:N], signal.fftconvolve(send[1], irs[1])[:N]])
rev = np.vstack([filt(rev[0], 'highpass', 180), filt(rev[1], 'highpass', 180)])

# trailer dynamics: the intro breathes, the drop at 12 s opens up
sec = np.ones(N)
i12 = int(12.0 * SR)
sec[:i12] = np.linspace(0.42, 0.62, i12)
sec_st = np.vstack([sec, sec])
mix = (music + dl * 0.5 + rev * 0.55) * sec_st + sfx * np.vstack([np.maximum(sec, 0.8)] * 2)
# hard silence break before the final hit (keep the final chord untouched)
i_cut, i_hit = int(53.75 * SR), int(54.0 * SR)
fade = int(0.012 * SR)
mix[:, i_cut:i_cut + fade] *= np.linspace(1, 0, fade)
mix[:, i_cut + fade:i_hit] = 0
# gentle master bus: low shelf warmth + soft clip + fades
mix = np.vstack([filt(mix[0], 'highpass', 28), filt(mix[1], 'highpass', 28)])
mix /= np.max(np.abs(mix)) + 1e-9
mix = np.tanh(mix * 1.6) / np.tanh(1.6)
fi = int(0.02 * SR)
mix[:, :fi] *= np.linspace(0, 1, fi)
fo = int(2.0 * SR)
mix[:, -fo:] *= np.linspace(1, 0, fo) ** 1.5
mix *= 0.89 / np.max(np.abs(mix))
wavfile.write('soundtrack.wav', SR, mix.T.astype(np.float32))
print('written soundtrack.wav', mix.shape)
