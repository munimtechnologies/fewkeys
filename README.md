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

1. **Your own phrases** ("said before"). Every sentence you choose is counted in a phrase file on your machine (`~/.fewkeys/phrases.json`), and the ones you use most often come first. Your most used sentences are also shown to the model as examples of how you talk.
2. **Everyday phrases** ("common"). A short built-in list ("I am too hot", "Please call the nurse", ...), so fewkeys helps from the first minute, even with no model running.
3. **A local language model** ("suggested"). This uses any small instruction model served by [Ollama](https://ollama.com). fewkeys checks every model guess: a sentence is shown only if it has exactly one word per letter, in order, and contains any word you added after a slash.

### Holding small models to the letters (0.2)

Small models are bad at "exactly one word per letter". In our first test on a laptop, a 3B model answered `cwtatl` with things like "could talk later" and "want to walk", and it recovered **none** of the 40 benchmark sentences.

So since 0.2, fewkeys doesn't just ask nicely. It sends Ollama a JSON schema whose `pattern` allows only sentences with one word per letter, each starting with that letter (for `iwsw`: `^[iI][A-Za-z0-9']*,? [wW]...$`). Ollama turns that pattern into a grammar, so the model *cannot* write a word that starts with the wrong letter. It still chooses the words, so the sentences are its best guesses.

If your Ollama is too old to take a schema, fewkeys falls back to a plain prompt and filters the answers. Use `--plain` to compare the two.

### Ranking by agreement (new in 0.3)

fewkeys asks the model more than once (`--rounds`, default 2), the second time a little more freely. A sentence the model proposes in more rounds is shown first; ties keep the model's own order. SpeakFaster drew 128 samples per abbreviation from its model (temperature 1.0); a laptop can't afford that, so fewkeys uses a few rounds and trusts the sentences the model agrees with itself on. The model also sees six worked examples now, from three-word requests to eight-word questions.

## Quick start

You need Python 3.8+ and nothing else. To get model suggestions, also install Ollama and pull a small model (any one is fine; fewkeys picks one it finds):

```
ollama pull llama3.2:3b
python fewkeys.py serve
```

Open http://localhost:8765/ in your browser. Type your letters, press Enter, and choose a sentence. The browser speaks it aloud. Tick "Also copy the sentence I choose" to paste it into your AAC software, email or chat.

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
python fewkeys.py --model qwen2.5:3b --plain bench  # the same without the grammar, to compare
python fewkeys.py --rounds 1 serve                  # one model call per expansion: faster, less sure
```

`--model auto` (the default) uses the first installed model from a short list of small, fast ones (llama3.2:3b, qwen2.5:3b, gemma3:4b, phi4-mini, ...), or else the smallest model you have.

## Tips

- **Add the other person's words** in the "What they just said" box. The model uses the conversation to guess better.
- **Pin down an important word** by typing it, or just its start, after a slash: `cyctd/doc`.
- **The page is built for any input method.** It has large targets, a visible focus ring, and announcements for screen readers. It works with eye-gaze mouse emulation, switch scanning and plain keyboards.
- **Privacy defaults:** fewkeys listens only on `127.0.0.1`, keeps nothing in logs, and refuses requests from other web pages.

## Measuring

`fewkeys bench` takes 40 everyday sentences (`BENCH_PHRASES`) and gives the model only their initials. It reports:

- **top1** and **top5:** how often the intended sentence is the first suggestion, or among the first five.
- **word_accuracy:** for each sentence, the best share of words in the right place among the suggestions, averaged. A near miss ("... up a bit" for "... up a little") still counts 7/8, and in real use the person can fix it by pinning the missing word with a slash.
- **fitting_per_phrase:** how many suggestions fitted the letters, on average.
- **mean_seconds:** the average time per expansion.
- **keystroke_saving:** a hit costs the letters plus one selection. A miss costs the letters plus typing the whole sentence.
- **constrained:** the share of expansions where Ollama accepted the grammar.

The printed summary also records the model, the number of rounds and whether the grammar was on. `--json results.json` also saves every sentence and what the model offered for it.

Your own phrase history isn't included, so real use, where people repeat themselves, should do better than the benchmark.

Results so far:

| fewkeys | model | mode | top1 | top5 | keystroke saving | notes |
|---|---|---|---|---|---|---|
| 0.1.0 | qwen2.5:3b | plain prompt | 0.0 | 0.0 | (none) | answers were short phrases, not one word per letter |
| 0.3.0 | qwen2.5:3b | grammar, 2 rounds | | | | to be measured on a laptop CPU |

## Status and limits

This is an early prototype (0.3.0). It has **not yet been tested with people who use AAC**. That testing is the next step, and it should be done with speech-language pathologists and people living with ALS. fewkeys is not a medical device and is not a replacement for a person's AAC system. It is meant to sit alongside one.

Roadmap:
- Send the chosen sentence straight into existing AAC software and eye-gaze keyboards (OptiKey, Grid, Windows Eye Control), not just through the clipboard.
- Personal voices: build a Piper voice from a person's banked recordings, offline.
- An evaluation with AAC users: words per minute and motor actions against their usual setup.

If you use AAC, or support someone who does, and want to try fewkeys or tell us what's missing, please open an issue or email munimversion2@gmail.com.

## Development

```
python -m unittest test_fewkeys.py
```

Made by MV2 of Munim, Inc. with AI agents. MIT license.
