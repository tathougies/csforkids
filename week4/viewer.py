import turtle
import math
import numpy as np

def normal_logpdf(x, mean, std):
    return -0.5 * np.log(2 * np.pi * std ** 2) - 0.5*((x - mean)/std)**2

class Waveform(object):
    def __init__(self, w, samplerate=44100):
        self.wave = np.asarray(w)
        self.samplerate = samplerate

    def autocorrelate(self, freq_min=20, freq_max=20000):
        min_lag = self.samplerate / freq_max
        max_lag = self.samplerate / freq_min

        lag_count = math.ceil(max_lag) - math.floor(min_lag)
        acf = np.zeros((lag_count,))

        three_seconds_samples = self.samplerate * 3
        three_seconds = self.wave[:three_seconds_samples]

        for i in range(0, lag_count):
            lag = 1 + i

            acf[i] = np.correlate(three_seconds, self.wave[lag:three_seconds_samples+lag])

        # Score
        lag_max = np.argmax(acf) # Find the highest lag
        print("Max lag is ", lag_max, " corresponding to frequency of ", self.samplerate / lag_max)

        mean = np.mean(acf)
        stdev = np.std(acf)
        max_freq = self.samplerate / lag_max

        max_score = None
        for possible in range(lag_max - 1, 300, -1):
            close_enough_score = -normal_logpdf(possible, mean, stdev)

            possible_freq = self.samplerate / possible
            freq_ratio = possible_freq / max_freq
            harmonic_score = np.exp(-(lag_max/4)) * np.cos(2 * np.pi * freq_ratio)

            score = harmonic_score * close_enough_score * acf[possible]
            if max_score is None:
                max_score = (possible, score)
            else:
                last_possible, last_score = max_score
                if score >= last_score:
                    max_score = (possible, score)

        best_lag = lag_max
        if max_score is not None:
            (best_lag, _) = max_score

        print("Best lag is " , best_lag, " freq: ", self.samplerate/best_lag, " max_scare: ", max_score)
        return best_lag

def view(*waveforms, samplerate=44100):
    waveforms = [Waveform(w, samplerate=samplerate) for w in waveforms]
    max_offset = min(w.wave.shape[0] for w in waveforms)
    start_lag = max(w.autocorrelate() for w in waveforms)
    offset = 0

    t = turtle.Turtle(visible=False)
    t.speed(0)
    screen = t.screen
    screen.tracer(0)
    hw, hh = screen.screensize()
    w = hw * 2
    h = hh * 2

    h_max = (h/2) * 0.9
    h_min = -h_max

    # Number of samples / pixel

    def time_per_view_ms():
        nonlocal start_lag
        return (1.0 / start_lag) * 1000

    def sample(x):
        where = x / w

    def draw():
        nonlocal offset, start_lag, screen
        t.clear()
        t.speed(0)
        screen.tracer(0)
        for wave in waveforms:
            t.penup()
            t.goto(-w/2, 0)
            t.pendown()
            d = wave.wave[offset:offset + start_lag]
            for x in range(0, w):
                y = np.interp((x/w) * (start_lag - 1), np.arange(start_lag), d)
                t.goto(x - (w/2), y * h_max)
        screen.update()

    def left():
        nonlocal offset
        offset = max(0, offset - 10)
        draw()
    def right():
        nonlocal offset
        offset = min(max_offset, offset + 10)
        draw()
    def quit():
        exit(0)

    screen.listen()
    screen.onkey(quit, "q")
    screen.onkey(left, "Left")
    screen.onkey(right, "Right")
    screen.mainloop()

if __name__ == "__main__":
    import sys
    _, f = sys.argv
    t = np.linspace(0, 4, 44100 * 4)
    w = np.sin(2 * np.pi * float(f) * t)
    view(w)
