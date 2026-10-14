# fewkeys

**Say more with fewer keys.** fewkeys is a small, offline helper for people who type slowly, for example with eye gaze, a switch or a head pointer, as many people living with ALS do. You type only the first letter of each word, and fewkeys offers whole sentences to choose from:

```
iwsw        ->  I want some water
cwtatl      ->  Can we talk about this later
ishpitb/bed ->  I saw him play in the bedroom
```

Everything runs on your own computer. What you say is never sent anywhere.

## Why

In Google's SpeakFaster study, a language model expanded word-initial abbreviations like these into full sentences. It saved **57% of motor actions** compared with a standard predictive keyboard in simulation. Two eye-gaze typists living with ALS typed **29% to 60% faster** with it (Cai et al., *Nature Communications* 15, 9449, 2024: https://www.nature.com/articles/s41467-024-53873-3).

That system ran on a large cloud model. fewkeys tries the same idea with small open models that run on an ordinary laptop or AAC tablet. That matters for three reasons:

- **Privacy.** A person's conversations with family, nurses and doctors stay on their machine.
- **No internet needed.** It keeps working in hospitals, on the road and during outages.
- **Free.** It's open source (MIT), with nothing to subscribe to.

## Where suggestions come from

Suggestions are shown in this order, and each one says where it came from:

1. **Your own phrases** ("said before"). Every sentence you choose is counted in a phrase file on your machine (`~/.fewkeys/phrases.json`), and the ones you use most often come first.
2. **Everyday phrases** ("common"). A short built-in list ("I am too hot", "Please call the nurse", ...), so fewkeys helps from the first minute, even with no model running.
3. **A local language model** ("suggested"). This uses any small instruction model served by [Ollama](https://ollama.com). fewkeys checks every model guess: a sentence is shown only if it has exactly one word per letter, in order, and contains any word you added after a slash.

## Quick start

You need Python 3.8+ and nothing else. To get model suggestions, also install Ollama and pull a small model:

```
ollama pull llama3.2:3b
python fewkeys.py serve
```

Open http://localhost:8765/ in your browser. Type your letters, press Enter, and choose a sentence. The browser speaks it aloud.

To use your own offline [Piper](https://github.com/rhasspy/piper) voice instead of the browser's voice:

```
python fewkeys.py serve --piper-model en_US-lessac-medium.onnx
```

From the command line:

```
python fewkeys.py expand iwsw
python fewkeys.py expand ishpitb/bedroom --context "Them: Where did you see him?"
python fewkeys.py add "Please put the game on"
python fewkeys.py --model "" expand pwiat          # no model: your phrases + common phrases only
python fewkeys.py --model qwen2.5:3b bench          # measure a model on 40 built-in sentences
```

## Tips

- **Add the other person's words** in the "What they just said" box. The model uses the conversation to guess better.
- **Pin down an important word** by typing it, or just its start, after a slash: `cyctd/doc`.
- **The page is built for any input method.** It has large targets, a visible focus ring, and announcements for screen readers. It works with eye-gaze mouse emulation, switch scanning and plain keyboards.
- **Privacy defaults:** fewkeys listens only on `127.0.0.1`, keeps nothing in logs, and refuses requests from other web pages.

## Measuring

`fewkeys bench` takes 40 everyday sentences (`BENCH_PHRASES`) and gives the model only their initials. It reports:

- **top1** and **top5:** how often the intended sentence is the first suggestion, or among the first five.
- **mean_seconds:** the average time per expansion.
- **keystroke_saving:** a hit costs the letters plus one selection. A miss costs the letters plus typing the whole sentence.

Your own phrase history isn't included, so real use, where people repeat themselves, should do better than the benchmark.

## Status and limits

This is an early prototype (0.1.0). It has **not yet been tested with people who use AAC**. That testing is the next step, and it should be done with speech-language pathologists and people living with ALS. fewkeys is not a medical device and is not a replacement for a person's AAC system. It is meant to sit alongside one.

Roadmap:
- Send the chosen sentence straight into existing AAC software and eye-gaze keyboards (OptiKey, Grid, Windows Eye Control), not just this page.
- Personal voices: build a Piper voice from a person's banked recordings, offline.
- An evaluation with AAC users: words per minute and motor actions against their usual setup.

If you use AAC, or support someone who does, and want to try fewkeys or tell us what's missing, please open an issue or email munimversion2@gmail.com.

## Development

```
python -m unittest test_fewkeys.py
```

Made by MV2 of Munim, Inc. with AI agents. MIT license.
