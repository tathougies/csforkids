# Make your own instruments here

from synth import *

# MODIFY THIS ONE!
@instrument
def my_voice(frequency, t):
    return tone(frequency, t)

# EXAMPLES FROM THE TEXT

@instrument
def example1(note, t):
  t1 = tone(note, t)
  t2 = tone(note * 2, t)
  t3 = tone(note * 3, t)
  # Multiplying t1 and t2 change the volume of the tones
  return (8/10) * t1 + (2/10) * t2 + (1/20) * t3

@instrument
def simple1(note, t):
  # Produce tones that are the fundamental frequency, but with some adjustment so they're off pitch
  t1 = tone(note * 0.997, t)
  t2 = tone(note * 1.001, t)
  fundamental = t1 + t2

  harmonic1 = tone(note * 2.01, t)
  harmonic2 = tone(note * 3.05, t)
  harmonic3 = tone(note * 4.001, t)

  # Adding the waves together, the peaks and troughs can sum up to a max of 5
  return (fundamental + harmonic1 + harmonic2 + harmonic3) / 5

@instrument
def simple2(note, t):
  t1 = tone(note * 0.997, t)
  t2 = tone(note * 1.001, t)
  fundamental = t1 + t2
  harmonic1 = tone(note * 2.01, t)
  harmonic2 = tone(note * 3.05, t)
  harmonic3 = tone(note * 4.001, t)
  return (fundamental + # weight 2
      0.4 * harmonic1 + # weight 0.4
      0.1 * harmonic2 + # weight 0.1
      0.08 * harmonic3 # weight 0.08
      ) / 2.58 # divide by total weight

@instrument
def simple3(note, t):
  # Example of decay
  t1 = tone(note * 0.997, t)
  t2 = tone(note * 1.001, t)
  fundamental = t1 + t2
  harmonic1 = tone(note * 2.01, t)
  harmonic2 = tone(note * 3.05, t)
  harmonic3 = tone(note * 4.001, t)
  sustained_note = (fundamental + # weight 2
      0.4 * harmonic1 + # weight 0.4
      0.1 * harmonic2 + # weight 0.1
      0.08 * harmonic3 # weight 0.08
      ) / 2.58 # divide by total weight
  return fade_out(t) * sustained_note

@instrument
def simple4(note, t):
  # Example of decay
  t1 = tone(note * 0.997, t)
  t2 = tone(note * 1.001, t)
  fundamental = (t1 + t2) * fade_out(t)
  harmonic1 = tone(note * 2.01, t)  * fade_out(1.5*t)
  harmonic2 = tone(note * 3.05, t)  * fade_out(2 * t)
  harmonic3 = tone(note * 4.001, t)  * fade_out(10 * t)
  return (fundamental + # weight 2
      0.4 * harmonic1 + # weight 0.4
      0.1 * harmonic2 + # weight 0.1
      0.08 * harmonic3 # weight 0.08
      ) / 2.58 # divide by total weight

@instrument
def simple5(note, t):
  # Example of decay
  t1 = tone(note * 0.997, t)
  t2 = tone(note * 1.001, t)
  fundamental = (t1 + t2) * fade_out(t)
  harmonic1 = tone(note * 2.01, t)  * fade_out(1.5*t)
  harmonic2 = tone(note * 3.05, t)  * fade_out(2 * t)
  harmonic3 = tone(note * 4.001, t)  * fade_out(10 * t)
  summed = (fundamental + # weight 2
      0.4 * harmonic1 + # weight 0.4
      0.1 * harmonic2 + # weight 0.1
      0.08 * harmonic3 # weight 0.08
      ) / 2.58
  return summed * adsr(0.001, 0.002, t)

@instrument
def tremolo(note, t):
  tremolo = tone(10, t)
  volume = 1 + tremolo * 0.1
  return volume * tone(note, t)

@instrument
def funny_vibrato(note, t):
  vibrato = tone(6, t)
  return tone(note + vibrato * 0.2 , t)

@instrument
def vibrato(note, t):
  vibrato = vibrato_phase(6, t)
  return tone_from_phase(note * t + vibrato * 0.2)

@instrument
def simple_vibrato(note, t):
    # Another way to do vibrato
    return tone_with_vibrato(note, t, 6, 0.2)

@instrument
def rectify_example(note, t):
    return rectify(tone(note, t))

@instrument
def hardclip_example(note, t):
    return hard_clip(tone(note, t))

@instrument
def softclip_example(note, t):
    return soft_clip(tone(note, t))

# SOME EXAMPLES. You can change these, but it's better to copy / paste

@instrument
def simple(frequency, t):
    return tone(frequency, t)

@instrument
def detuned_piano(frequency, t):
    return (
        pluck(t * 0.3) * (tone(12, t) * 0.5 + 0.9) * tone(frequency * 0.999, t) +
        pluck(t * 1.1) * (tone(3, t) * 0.8 + 0.9) * tone(frequency * 1.003, t + 3) +
        (1 - pluck(t * 1.1)) * tone_with_vibrato(frequency * 1.001, t + 2) +
        piano_attack_env(t) * 0.2 * tone(frequency * 1.999, t - 2) +
        piano_attack_env(t) * 0.3 * tone_with_vibrato(frequency * 2.001, t - 1, r=6.3, depth=1) +
        pluck(t * 0.5) * 0.2 * tone(frequency * 3.002, t) +
        piano_attack_env(t, velocity=0.5) * 0.1 * tone_with_vibrato(frequency * 3.99, t - 1, r=7.0, depth=0.5) +
        piano_attack_env(t, velocity=0.5) * 0.2 * tone(frequency * 4.10, t + 2) +
        pluck(t * 5) * tone(frequency * 7.99, t - 1))/4.0

@instrument
def dulcimer(f, t):
    return (pluck(t * 10) * tone(f, t) + \
            pluck(t * 9.9) * tone(f * 0.99, t) + \
            pluck(t * 10.1) * tone(f * 1.002, t) + \
            (1/4) * pluck(t * 12) * tone (f * 2.001, t) + \
            (1/8) * pluck(t * 11) * tone(f * 2.999, t) + \
            (1/9) * pluck(t * 11.5) * tone(7 * 3.01, t)) / 6

# THESE INSTRUMENTS HAVE PRETTY COMPLICATED DEFINITIONS, BUT YOU CAN SEE THEM IN synth.py

@instrument
def complex_piano(f, t):
    return piano(f, t)

USER_INSTRUMENTS['complex_church_organ'] = church_organ
USER_INSTRUMENTS['complex_organ'] = pipe_organ
USER_INSTRUMENTS['complex_violin'] = violin

# DO NOT CHANGE ANYTHING PAST HERE
ui()
