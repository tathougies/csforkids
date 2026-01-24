# FIx broken midi files
import mido

def fix_track(t):
    if any(e.type == 'note_on' and e.velocity == 0 for e in t):
        print("Fix zero velocity notes -> note_off")
        es = mido.MidiTrack()
        for e in t:
            if e.type == 'note_on' and e.velocity == 0:
                es.append(mido.Message('note_off', note=e.note, time=e.time, channel=e.channel))
            else:
                es.append(mido.Message(e))
        return es
    else:
        return t

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("Fix MIDI Files")
    p.add_argument("file", help="File to fix")

    o = p.parse_args()
    m = mido.MidiFile(o.file)
    tracks = [fix_track(t) for t in m.tracks]
    m.tracks = tracks
    m.save("output.mid")
