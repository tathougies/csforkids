from mido import MidiFile, bpm2tempo, tick2second
from dataclasses import dataclass
from enum import Enum
import itertools
import mido
import random
import math
import numba
import sounddevice
import numpy as np
import threading

DROPOFF_CUTOFF = 0.0000001
DROPOFF_TIME = 50.0 / 1000.0 # 50 ms
DROPOFF_COEFF = math.log(DROPOFF_CUTOFF, math.e)

@numba.njit(cache=True)
def convert_pitch_to_hz(pitch):
    base_frequencies = {
        "C": 16.35, "C#": 17.32, "D": 18.35, "D#": 19.45, "E": 20.60,
        "F": 21.83, "F#": 23.12, "G": 24.50, "G#": 25.96, "A": 27.50,
        "A#": 29.14, "B": 30.87
    }

    def calculate_frequency(tone, octave):
        return base_frequencies[tone.upper()] * (2 ** octave)

    if isinstance(pitch, str):
        tone = pitch[:-1]
        octave = int(pitch[-1])
        frequency = calculate_frequency(tone, octave)
        if frequency is None:
            raise ValueError(f"Unknown pitch: {pitch}")
        return frequency
    else: #if isinstance(pitch, (int, float, np.ndarray)):
        return pitch

def soft_clip(x):
    return np.tanh(x)

def hard_clip(x):
    return np.minimum(np.maximum(x, -0.8), 0.8)

def drive(x, factor=3):
    return soft_clip(factor * x)

@numba.njit(cache=True)
def tone(frequency, t):
    return np.sin(2 * np.pi * frequency * t).astype(np.float32)

def rectified(frequency, t):
    return rectify(drive(tone(frequency, t)))

@numba.njit(cache=True)
def tone_with_vibrato(frequency, t, r=6.0, depth=0.5):
    return np.sin(2 * np.pi * (frequency * t + (depth / r) * (1 - np.cos(2 * np.pi * r * t)))).astype(np.float32)

def vibrato(pitch, t):
    v = (1.0/np.maximum(t,1e-9)) * (1 + 0.001 * tone(6, t + pitch))
    n = tone(pitch + v, t)
    return n

def instrument1(pitch, t):
    return tone(pitch, t) * 0.5 + tone(pitch / 2, t) * 0.25

def voice_attack_env(t, attack=0.02):
    """
    Simple attack envelope.
    t: time in seconds (absolute or relative)
    """
    t0 = t - t.min()
    env = np.clip(t0 / max(attack, 1e-6), 0.0, 1.0)
    return env.astype(np.float32)

def vowel_formants(vowel="a"):
    return {
        "a": (800, 1150, 2900),  # ah
        "e": (500, 1700, 2500),  # eh
        "i": (300, 2200, 3000),  # ee
        "o": (500,  900, 2400),  # oh
        "u": (350,  600, 2400),  # oo
    }.get(vowel, (800, 1150, 2900))

voice_state = None
def voice(pitch, t, sr=44100, vowel="a", attack=0.02,
          vibrato_hz=5.5, vibrato_cents=20,
          brightness=0.7, breath=0.01):
    """
    pitch: Hz (scalar or array)
    t: time array in seconds
    state: dict (kept across callbacks)
    """
    global voice_state
    if voice_state is None:
        voice_state = {}

    t = np.asarray(t, np.float32)
    pitch = np.asarray(pitch, np.float32)

    # ---- Vibrato ----
    vib = np.sin(2*np.pi*vibrato_hz*t)
    pitch_v = pitch * (2 ** ((vib * vibrato_cents) / 1200.0))

    # ---- Phase accumulator (kept across blocks) ----
    phase = voice_state.get("phase", 0.0)
    phase_inc = 2*np.pi * pitch_v / sr
    phase_arr = phase + np.cumsum(phase_inc)
    voice_state["phase"] = phase_arr[-1] % (2*np.pi)

    # ---- Harmonic source (additive saw-ish) ----
    max_h = int(min(40, (sr * 0.45) / max(pitch_v.max(), 1e-6)))
    src = np.zeros_like(t)
    for k in range(1, max_h + 1):
        amp = (1.0 / k) ** (1.0 - brightness)
        src += amp * np.sin(k * phase_arr)

    src /= np.max(np.abs(src) + 1e-9)

    # Breath noise
    src += breath * np.random.randn(len(src)).astype(np.float32)

    # ---- Formant filter bank ----
    F1, F2, F3 = vowel_formants(vowel)
    Qs = (8.0, 10.0, 12.0)

    y = np.zeros_like(src)
    for i, (F, Q) in enumerate(zip((F1, F2, F3), Qs)):
        b, a = biquad_bandpass(sr, F, Q)
        st = voice_state.get(f"bp{i}")
        yf, st2 = biquad_process(src, b, a, st)
        voice_state[f"bp{i}"] = st2
        y += yf

    # ---- Attack envelope ONLY ----
    env = voice_attack_env(t, attack)
    out = y * env

    # Safety
    peak = np.max(np.abs(out)) + 1e-9
    if peak > 1.0:
        out /= peak

    return out #.astype(np.float32)

def biquad_process(x, b, a, state=None):
    if state is None:
        x1 = x2 = y1 = y2 = 0.0
    else:
        x1, x2, y1, y2 = state

    y = np.empty_like(x)
    b0, b1, b2 = b
    _, a1, a2 = a

    for i, x0 in enumerate(x):
        y0 = b0*x0 + b1*x1 + b2*x2 - a1*y1 - a2*y2
        y[i] = y0
        x2, x1 = x1, x0
        y2, y1 = y1, y0

    return y, (x1, x2, y1, y2)

def biquad_bandpass(sr, f0, Q):
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2 * Q)
    c = np.cos(w0)

    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1 + alpha
    a1 = -2 * c
    a2 = 1 - alpha

    b = np.array([b0, b1, b2], np.float32) / a0
    a = np.array([1.0, a1 / a0, a2 / a0], np.float32)
    return b, a

def church_organ(pitch, t):
    return organ(pitch, t, registration='strings8')

def organ(pitch, t, registration='principal8'):
    # --- Registration (mixtures / stops) ---
    # Each entry: (foot_ratio, gain, tilt_adjust)
    # foot_ratio multiplies fundamental: 8' -> 1.0, 4' -> 2.0, 16' -> 0.5, 2' -> 4.0, etc.
    regs = {
        "principal8": [(1.0, 1.00, 0.0)],
        "flute8":     [(1.0, 1.00, +0.4)],          # more rolloff (rounder)
        "strings8":   [(1.0, 0.90, -0.2), (2.0, 0.25, -0.1)],  # brighter + slight 4'
        "full":       [(0.5, 0.35, -0.1), (1.0, 1.00, -0.1), (2.0, 0.55, -0.2), (4.0, 0.25, -0.2)],
        "plenum":     [(1.0, 1.00, -0.2), (2.0, 0.70, -0.2), (4.0, 0.35, -0.15)],  # big principal chorus
        "bourdon16":  [(0.5, 1.00, +0.5), (1.0, 0.35, +0.5)],  # very mellow
    }
    stops = regs.get(registration, regs["principal8"])
    return organ_(pitch, t, stops)

@numba.njit(cache=True)
def organ_(pitch, t, stops):
    """
    Simple pipe/church organ-ish synth (attack + settle + sustain; NO release).
    Uses additive synthesis with near-harmonic partials (low inharmonicity),
    stable phases, light tremulant, and selectable "registration" mixtures.

    Requires:
      - convert_pitch_to_hz(pitch) -> float
      - tone(hz, t) -> ndarray
    """
    t = np.asarray(t, dtype=np.float32)
    f0 = np.float32(convert_pitch_to_hz(pitch))
    if f0 <= 0:
        return np.zeros_like(t, dtype=np.float32)

    tt = np.maximum(t, 0.0)

    # --- Envelope: organ is steady-state, slow-ish attack, tiny "speech" transient, then flat sustain ---
    f_ref = 440.0
    slow = min(max((f_ref / f0) ** 0.20, 0.7), 1.6)  # lower pipes speak a bit slower

    attack = 0.020 + 0.060 * slow                   # ~20–120 ms
    settle = 0.080 + 0.120 * slow                   # time for chiff / brightness to mellow

    # Smooth rise
    env_a = 1.0 - np.exp(-tt / max(attack, 1e-6))

    # Chiff / speech brightness that fades after onset
    chiff_env = np.exp(-tt / max(settle, 1e-6))

    # --- Very small inharmonicity (pipes are mostly harmonic) ---
    # Keep it extremely small to avoid piano-like stretch.
    B = min(max(2e-6 * (f0 / f_ref) ** 0.7, 0.0), 2e-5)

    # --- Tremulant (amplitude modulation) ---
    # Church organs often have tremulant; keep subtle.
    trem_rate = 5.5
    trem_depth = 0.03
    trem = 1.0 + trem_depth * tone(trem_rate, tt)

    # --- Pitch-dependent partial count ---
    n_partials = int(min(max(28 * (f_ref / max(f0, 1e-9)) ** 0.10, 14), 8))

    # Deterministic RNG for slight de-correlation between stops
    two_pi = np.float32(2.0 * np.pi)
    # Base deterministic seed derived from f0
    base_seed = np.uint64(int(f0 * 100.0))  # deterministic across runs
    wave = np.zeros_like(tt, dtype=np.float32)

    for stop_i, (foot_ratio, gain, tilt_adj) in enumerate(stops):
        f_base = f0 * foot_ratio

        # Organ spectra depend heavily on stop type; use tilt and a couple of "formant" bumps.
        # Lower tilt => brighter.
        tilt_body = 1.15 + tilt_adj
        tilt_chiff = 0.65 + 0.5 * tilt_adj  # chiff is brighter (smaller tilt)

        # Small deterministic per-partial phase (and per-stop) to avoid sterile buzzing
        phases = np.empty(n_partials, dtype=np.float32)
        stop_seed = splitmix64(base_seed + np.uint64(0xD1B54A32D192ED03) * np.uint64(stop_i))

        for k in range(n_partials):
            phases[k] = two_pi * np.float32(uniform01(stop_seed, k))

        s = np.zeros_like(tt, dtype=np.float32)
        for n in range(1, n_partials + 1):
            fn = (n * f_base) * np.sqrt(1.0 + B * (n**2))

            # Time-varying tilt: brighter at onset (chiff), then mellows
            tilt_t = tilt_body + (tilt_chiff - tilt_body) * chiff_env

            # Base rolloff + gentle voicing bumps:
            # - principals: strong 2nd/3rd/4th-ish
            # - flutes: suppress upper partials via tilt_adj above
            bump = 1.0 + 0.18 * np.exp(-((n - 3.0) / 2.2) ** 2) + 0.10 * np.exp(-((n - 6.0) / 2.8) ** 2)
            amp = bump / (n ** np.clip(tilt_t, 0.6, 2.2))

            # Pipe steady-state is very stable: extremely slow partial damping
            # (still include a tiny amount to avoid infinite ringing artifacts)
            tau_n = 8.0 + 2.5 * slow + 0.6 * (1.0 / (0.2 + 0.06 * n))
            partial_env = np.exp(-tt / max(tau_n, 1e-6))

            # Add subtle "speech" (a little bit of noisy-ish edge) by
            # slightly boosting upper partials during chiff window
            speech_boost = 1.0 + 0.25 * chiff_env * (n / max(n_partials, 1)) ** 1.6

            # Phase via time shift (keeps tone() usage)
            t_shift = phases[n-1] / (2.0 * np.pi * max(fn, 1e-9))
            s += (amp * speech_boost) * partial_env * tone(fn, tt + t_shift)

        wave += gain * s

    # Apply envelope and tremulant
    wave *= env_a * trem

    # Soft normalize
    peak = np.float32(np.max(np.abs(wave)) if wave.size else 0.0)
    if peak > 1.0:
        wave = wave / peak

    return wave

@numba.njit(cache=True)
def piano(f0, t, velocity=1.0):
    # Inharmonicity factor (string stiffness)
    B = (1 * np.exp(-t * 3)) * 0.0004 * (f0 / 440.0)

    sample = np.zeros_like(t, dtype=np.float32)

    # Number of partials
    num_partials = 12
    if f0 < 90:
        shimmer = 1
    elif f0 < 250:
        shimmer = 2
    else:
        shimmer = 3

    base_seed = np.uint64(int(f0 * 100.0))  # deterministic across runs
    SHIMMER_SIZE = 0.002 * f0
    for n in range(1, num_partials + 1):
        for s in range(0, shimmer):
            # Inharmonic frequency
            d_shimmer = SHIMMER_SIZE * 2 * uniform01(base_seed, s) - (SHIMMER_SIZE/2)
            fn = n * (f0 +d_shimmer) * np.sqrt(1 + B * n * n)

            # Amplitude roll-off (rough piano spectral tilt)
            amp = (1.0 / n) * np.exp(-0.15 * n)
            amp_t = piano_attack_env(t * n/3) * amp
            t_shift = np.float32(uniform01(base_seed, shimmer + n))
            sample += (amp_t * tone(fn, t + t_shift)/shimmer)

    # Gentle normalization
    return sample * 0.3

@numba.njit(cache=True)
def piano_attack_env(t, velocity=1.0):
    """
    Piano-like attack+early-settle envelope.
    t: seconds since note-on
    velocity: 0..1 (use MIDI vel/127)
    """

    # Attack time: faster when hit harder
    atk = 0.0025 - 0.0015 * velocity     # ~1–2.5 ms
    atk = max(atk, 0.0008)

    # "Settle" time: how quickly it falls from the initial ping
    settle = 0.020 + 0.020 * (1.0 - velocity)  # ~20–40 ms

    # Sustain level after the initial ping (still before real decay)
    sustain = 0.55 + 0.25 * velocity  # harder hit -> higher early level

    # Exponential decay to fade to zero eventually
    decay_time = 1.0  # Decay time in seconds to reach near zero
    decay = np.exp(-t / decay_time)

    # 1) Fast rise (exponential-ish)
    rise = 1.0 - np.exp(-t / atk)

    # 2) Small overshoot "ping" right after onset
    bump_center = 0.004 + 0.002 * (1.0 - velocity)  # 4–6 ms
    bump_width  = 0.0025                             # ~2.5 ms
    bump = np.exp(-((t - bump_center) / bump_width) ** 2)

    overshoot = 1.0 + (0.12 + 0.10 * velocity) * bump  # 12–22% overshoot

    # 3) Settle from 1.0 down toward sustain
    settle_curve = sustain + (1.0 - sustain) * np.exp(-t / settle)

    # Final envelope, includes decay to zero
    envelope = rise * overshoot * settle_curve * decay

    return envelope

def violin(pitch, t):
    # Vibrato: ~5.5–6.5 Hz, depth ~0.3–0.6% depending on taste
    vib_rate = 6.0
    vib_depth = 0.004  # 0.4%
    f0 = convert_pitch_to_hz(pitch)
    f = f0

    # Additive spectrum: violin is bright with many partials
    s = 0.0
    num_partials = 32  # Increased to ensure richness and less wavy sound over time
    base_seed = np.uint64(int(f0 * 100.0))  # deterministic across runs
    for n in range(1, num_partials + 1):
        fn = n * f
        t_shift = np.float32(uniform01(base_seed, 2 * n)) / max(fn, 1e-9)
        n_shift = (0.01 * np.float32(uniform01(base_seed, 2 * n))) - 0.005
        fn_ = f * (n + n_shift)
        tt = t + t_shift
        # Spectral rolloff, but not too steep
        amp = 1.0 / (n ** 1.1)

        # A simple “body” emphasis around a few harmonics (rough resonance)
        body = 1.0 + 0.25 * np.exp(-((n - 4.0) / 2.0) ** 2) + 0.15 * np.exp(-((n - 9.0) / 3.0) ** 2)

        s += (amp * body) * np.sin(2 * np.pi * (fn_ * tt + (1.0 / vib_rate) * (1 - np.cos(2 * np.pi * vib_rate * tt))))

    # Bow noise: eliminate time-dependent randomness to avoid wavy sound
    bow_intensity = 0.02
    random.seed(0)  # Fixed seed to eliminate time-dependent randomness
    bow = (random.uniform(-1.0, 1.0) * bow_intensity)

    # Mild saturation to keep it from sounding too “organ”
    out = np.tanh((s * 0.5) + bow)

    return out * 0.35 * piano_attack_env(t)

def detuned(f, t):
    return (
        np.exp(-t * 0.3) * (tone(12, t)*0.5+0.9) * tone(f * 0.999, t) +
        np.exp(-t * 1.1) * (tone(3, t) *0.8 + 0.9) * tone(f * 1.003, t + 3) +
        (1 - np.exp(-t * 1.1)) * tone_with_vibrato(f * 1.001, t + 2)+
        piano_attack_env(t) * 0.2 * tone(f * 1.999, t - 2) +
        piano_attack_env(t) * 0.3 * tone_with_vibrato(f * 2.001, t - 1, r=6.3, depth=1) +
        np.exp(-t * 0.5) * 0.2 * tone(f * 3.002, t) +
        piano_attack_env(t, velocity=0.5) * 0.1 * tone_with_vibrato(f * 3.99, t - 1, r=7.0, depth=0.5) +
        piano_attack_env(t, velocity=0.5) * 0.2 * tone(f * 4.10, t + 2) +
        np.exp(-t * 5) * tone(f * 7.99, t - 1))/4.0

@numba.njit(cache=True)
def trumpet(pitch, t):
    # Convert the MIDI pitch to frequency in Hz
    base_freq = convert_pitch_to_hz(pitch)

    # Define the amplitudes for the harmonics; simulating a trumpet sound
    amplitudes = [0.6, 0.3, 0.15, 0.1, 0.05]
    harmonics = [1, 2, 3, 4, 5]

    # Define ADS envelope parameters
    attack_duration = 0.1  # 10% of the tone
    decay_duration = 0.2   # 20% of the tone
    sustain_level = 0.7    # Sustain level at 70% amplitude

    # Calculate the envelope value based on time t
    envelope = np.where(
        t < attack_duration,
        t / attack_duration,
        np.where(
            t < attack_duration + decay_duration,
            1.0 - ((t - attack_duration) / decay_duration) * (1.0 - sustain_level),
            sustain_level
        )
    )

    # Initialize the signal with the fundamental tone and apply the envelope
    signal = amplitudes[0] * tone(base_freq, t)

    # Add harmonics to the signal
    for harmonic, amplitude in zip(harmonics[1:], amplitudes[1:]):
        signal += amplitude * tone(base_freq * harmonic, t)

    return signal * envelope

# Deterministic randomness

@numba.njit(cache=True)
def splitmix64(x):
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = x
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return z ^ (z >> np.uint64(31))

@numba.njit(cache=True)
def u01_from_u64(x):
    # 53-bit float in [0,1)
    return (x >> np.uint64(11)) * (1.0 / (1 << 53))

@numba.njit(cache=True)
def uniform01(seed_u64, k):
    # deterministic uniform in [0,1) for index k
    return u01_from_u64(splitmix64(seed_u64 + np.uint64(k)))

class Phase(Enum):
    PRESS = 'PRESS'
    RELEASE = 'RELEASE'

@dataclass
class Note:
    message: mido.Message
    time: int
    phase: Phase
    amp: float

def midi_note_to_hz(note: int, a4: float = 440.0) -> float:
    '''Translate a MIDI pitch into hertz'''
    return a4 * (2 ** ((note - 69) / 12.0))

def dropoff(t_since_release):
    return np.exp((t_since_release/DROPOFF_TIME) * DROPOFF_COEFF)

def play_midi(filename, instrument, samplerate=44100, volume=0.8):
    # Load midi, and read first track
    m = MidiFile(filename)
    t = mido.merge_tracks(m.tracks)
    tempo = bpm2tempo(120)

    msgs = iter(t)

    last_time = None
    midi_time = 0
    notes = []
    releasing_notes = []
    seconds_per_frame = 1.0 / float(samplerate)
    next_msg = None
    t_last = 0
    done = threading.Event()

    def find_note_index(notes, note, is_new=False):
        for i, n in enumerate(notes):
            if is_new:
                this = n[0]
            else:
                this = n
            if this.message.note == note:
                return i
        return None

    def sound_cb(outdata, frames, time, status):
        nonlocal last_time, notes, releasing_notes, next_msg, tempo, midi_time, t_last

        dac_time = time.outputBufferDacTime

        if last_time is None:
            last_time = dac_time

        t = (dac_time - last_time) + midi_time
        midi_time = t
        last_time = dac_time

        t_final = t + frames * seconds_per_frame
        t_all = np.arange(0, frames, dtype=np.float32) * seconds_per_frame + t

        # process midi events
        t_cur = t_last
        new_notes = []
        released_notes = []
        while True:
            if next_msg is not None:
                msg = next_msg
            else:
                try:
                    msg = next(msgs)
                except StopIteration:
                    done.set()
                    raise sounddevice.CallbackStop

            dt = tick2second(msg.time, m.ticks_per_beat, tempo)
            t_cur += dt
            if t_cur < t_final:
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                elif msg.type == 'note_on':
                    amp = math.pow(msg.velocity / 127.0, 1.5)
                    new_notes.append((Note(msg, t_cur, Phase.PRESS, amp), t_cur, t_all > t_cur))
                elif msg.type == 'control_change' and msg.control == 64:
                    print("CONTROL")
                    if msg.value < 64: # Sustain off, release all notes
                        released_notes.extend((n, t_cur, t_all < t_cur) for n in notes)
                        notes = []
                elif msg.type == 'note_off':
                    i = find_note_index(notes, msg.note)
                    if i is not None:
                        released_note = notes[i]
                        del notes[i]
                        released_notes.append((released_note, t_cur, t_all < t_cur))
                    else:
                        i = find_note_index(new_notes, msg.note, is_new=True)
                        if i is not None:
                            released_note, _, t_note = new_notes[i]
                            del new_notes[i]
                            released_notes.append((released_note, t_cur, t_note & (t_all < t_cur)))
                t_last = t_cur
                next_msg = None
            else:
                next_msg = msg
                break

        for n, treleased, _ in released_notes:
            releasing_notes.append((n, treleased))
        outdata[:, 0] = sum(instrument(midi_note_to_hz(n.message.note), t_all - n.time) * n.amp for n in notes) +\
            sum(instrument(midi_note_to_hz(n.message.note), t_all - n.time) * t_note * n.amp\
                for n, _, t_note in itertools.chain(new_notes, released_notes)) + \
            sum(instrument(midi_note_to_hz(n.message.note), t_all - n.time) * dropoff(t_all - treleased) * (t_all > treleased) * n.amp \
                for n, treleased in releasing_notes)
        outdata *= volume

        # Move notes over
        for n, _, _ in new_notes:
            notes.append(n)
        releasing_notes = [(n, treleased) for n, treleased in releasing_notes if dropoff(t_final - treleased) > DROPOFF_CUTOFF]

    with sounddevice.OutputStream(samplerate=samplerate, callback=sound_cb, channels=1):
        
        done.wait()
#        stream.stop()

if __name__ == "__main__":
    import sys
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("midi", help="MIDI file to load")
    p.add_argument("--voice", help="Voice to use", type=str, default='tone')
    p.add_argument("--volume", help="Volume", type=float, default=0.8)
    o = p.parse_args()

    VOICES = { 'vibrato': vibrato, 'tone': tone, 'instrument1': instrument1, 'piano': piano, 'organ': organ, 'church_organ': church_organ, 'violin': violin, 'voice': voice, 'brass': trumpet, 'test': detuned, 'rectified':rectified  }

    f = o.midi
    play_midi(f, VOICES[o.voice], volume=o.volume)
    import time
    time.sleep(120)

    
