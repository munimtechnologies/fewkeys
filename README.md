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
- **Free.** It's open source (Apache-2.0), with nothing to subscribe to.

## Where suggestions come from

Suggestions are shown in this order, and each one says where it came from:

1. **Your own phrases** ("said before"). Every sentence you choose is counted in a phrase file on your machine (`~/.fewkeys/phrases.json`), and the ones you use most often come first. Your most used sentences are also shown to the model as examples of how you talk.
2. **Everyday phrases** ("common"). A short built-in list ("I am too hot", "Please call the nurse", ...), so fewkeys helps from the first minute, even with no model running.
3. **A local language model** ("suggested"). This uses any small instruction model served by [Ollama](https://ollama.com). fewkeys checks every model guess: a sentence is shown only if it has exactly one word per letter, in order, and contains any word you added after a slash.

### Holding small models to the letters (0.2)

Small models are bad at "exactly one word per letter". In our first test on a laptop, a 3B model answered `cwtatl` with things like "could talk later" and "want to walk", and it recovered **none** of the 40 benchmark sentences.

So since 0.2, fewkeys doesn't just ask nicely. It sends Ollama a JSON schema whose `pattern` allows only sentences with one word per letter, each starting with that letter (for `iwsw`: `^[iI][A-Za-z0-9']*,? [wW]...$`). Ollama turns that pattern into a grammar, so the model *cannot* write a word that starts with the wrong letter. It still chooses the words, so the sentences are its best guesses.

If your Ollama is too old to take a schema, fewkeys falls back to a plain prompt and filters the answers. Use `--plain` to compare the two.

### Ranking by agreement (0.3)

fewkeys asks the model more than once (`--rounds`, default 2), the second time a little more freely. A sentence the model proposes in more rounds is shown first; ties keep the model's own order. SpeakFaster drew 128 samples per abbreviation from its model (temperature 1.0); a laptop can't afford that, so fewkeys uses a few rounds and trusts the sentences the model agrees with itself on. The model also sees six worked examples, from three-word requests to eight-word questions.

## Quick start

You need Python 3.8+ and nothing else. To get model suggestions, also install Ollama and pull a small model (any one is fine; fewkeys picks one it finds):

```
ollama pull llama3.2:3b
python fewkeys.py serve
```

Open http://localhost:8765/ in your browser. If port 8765 is already taken by another program, fewkeys uses the next free port and prints the address to open. Type your letters, press Enter, and choose a sentence. The browser speaks it aloud. Tick "Also copy the sentence I choose" to paste it into your AAC software, email or chat.

To use your own offline [Piper](https://github.com/rhasspy/piper) voice instead of the browser's voice:

```
python fewkeys.py serve --piper-model en_US-lessac-medium.onnx
```

From the command line:

```
python fewkeys.py expand iwsw
python fewkeys.py expand ishpitb/bedroom --context "Them: Where did you see him?"
python fewkeys.py add "Please put the game on"
python fewkeys.py --model "" expand pwiat                  # no model: your phrases + common phrases only
python fewkeys.py --model qwen2.5:3b bench                  # measure a model on 40 built-in sentences
python fewkeys.py --model qwen2.5:3b --plain bench          # the same without the grammar, to compare
python fewkeys.py --model qwen2.5:3b bench --context        # the same, after the other person's words
python fewkeys.py --model qwen2.5:3b bench --dialogues turk-dialogues.txt --context   # a public AAC dialogue set
python fewkeys.py --rounds 1 serve                          # one model call per expansion: faster, less sure
```

`--model auto` (the default) uses the first installed model from a short list of small, fast ones (llama3.2:3b, qwen2.5:3b, gemma3:4b, phi4-mini, ...), or else the smallest model you have.

## Tips

- **Add the other person's words** in the "What they just said" box. The model uses the conversation to guess better.
- **Pin down an important word** by typing it, or just its start, after a slash: `cyctd/doc`.
- **The page is built for any input method.** It has large targets, a visible focus ring, and announcements for screen readers. It works with eye-gaze mouse emulation, switch scanning and plain keyboards.
- **Privacy defaults:** fewkeys listens only on `127.0.0.1`, keeps nothing in logs, and refuses requests from other web pages.

## Measuring

`fewkeys bench` takes 40 everyday sentences (`BENCH_PHRASES`) and gives the model only their initials. It reports:

- **top1** and **top5:** how often the intended sentence is the first suggestion, or among the first five. Case and punctuation are ignored.
- **word_accuracy:** for each sentence, the best share of words in the right place among the suggestions, averaged. A near miss ("... up a bit" for "... up a little") still counts 7/8, and in real use the person can fix it by pinning the missing word with a slash.
- **fitting_per_phrase:** how many suggestions fitted the letters, on average.
- **mean_seconds:** the average time per expansion.
- **keystroke_saving:** a hit costs the letters plus one selection. A miss costs the letters plus typing the whole sentence.
- **constrained:** the share of expansions where Ollama accepted the grammar.

The printed summary also records the model, the number of rounds, whether the grammar and the conversation context were on, and which sentence set was used. `--json results.json` also saves every sentence and what the model offered for it.

### With conversation context (0.4)

In real conversations, people mostly answer something that was just said. The SpeakFaster team found that showing the model one previous turn more than doubled exact expansion accuracy (Cai et al., NAACL 2022, https://arxiv.org/abs/2205.03767). `bench --context` measures that effect for your model. Each of the 40 sentences comes with something the other person might have just said (`BENCH_DIALOGUES`), for example "How's the soup?" before "I think it needs more salt". The partner's turn never contains the answer itself.

These partner turns were written for this benchmark, not taken from real conversations. Treat the difference between `bench` and `bench --context` as a rough sign of how much context helps, not a measurement from AAC users.

### On a public AAC dialogue set (0.5)

Our 40 sentences are our own, so their numbers can't be compared with anyone else's. `bench --dialogues FILE` uses a public set instead: the **Turk Dialogues** by Keith Vertanen (ASSETS 2017), 1,420 six-turn conversations written for AAC research, with train, dev and test splits.

1. Download `turk-dialogues.txt` from https://aactext.org/turk/ (the original file, with the split column; the filtered file has no splits). It is not included here; please cite Vertanen's paper if you publish results.
2. Run `python fewkeys.py --model qwen2.5:3b bench --dialogues turk-dialogues.txt --context`.

This follows the main abbreviation-expansion test in SpeakFaster (Cai et al., arXiv:2312.01532). Each item is the **second turn** of a **test-split** dialogue, kept if it has at most 10 words and mid-sentence punctuation marks. With `--context`, the model sees the first turn. Only the initials are typed, with no keywords. `--limit` sets how many dialogues are used (default 100, in file order; `--limit 0` for all), because each one takes a few seconds on a laptop CPU.

On that test, SpeakFaster's fine-tuned 64-billion-parameter cloud model put the exact sentence among its top 5 for **72.6% to 76.3%** of turns, drawing 128 samples per abbreviation. The comparison is rough, and here is why:

- SpeakFaster used a corrected version of the dialogues, with typos fixed. The original file keeps them, and a typo makes an exact match impossible.
- fewkeys uses a small, general model that wasn't trained for this, and asks it only 2 times.
- With the default `--limit` it uses the first 100 test dialogues, not all of them.

So expect fewkeys to score lower. The gap is what fine-tuning a small model (on the train split) would have to close.

Your own phrase history isn't included in any benchmark, so real use, where people repeat themselves, should do better.

### Long benchmark runs and a no-model baseline (`turkbench.py`)

`turkbench.py` runs the same benchmark for long, unattended jobs, for example on a server CPU. It calls `fewkeys.bench` one sentence at a time, so the rows and metrics are the same, and it adds a few things:

- **Saved after every sentence.** It can stop at a time limit (`--max-minutes`, `--deadline`) and still keep every result so far, marked `"complete": false`.
- **reachable:** the share of target sentences the grammar can write at all. The grammar allows letters, digits and straight apostrophes inside a word, so a target with a hyphen or a curly apostrophe ("well-known", "don’t") can never be matched exactly. **top5_of_reachable** scores only those that can.
- **by_words:** top-5 accuracy for sentences of 1-3, 4-6 and 7-10 words.
- **`lookup`:** a baseline with no model at all. Every turn of the train-split dialogues goes into a phrase list. For each test item, it offers the five most frequent listed sentences with the same initials. This shows how much a phrase list alone gets, and **in_train** shows how many test sentences appear word for word in the train split.
- **`report`:** a Markdown table of every result file in a folder.

```
python turkbench.py lookup --dialogues turk-dialogues.txt --out lookup.json
python turkbench.py bench --model qwen2.5:3b --set turk --dialogues turk-dialogues.txt --context --max-minutes 75 --out qwen-turk-context.json
python turkbench.py bench --model qwen2.5:3b --set builtin --out qwen-builtin.json
python turkbench.py report --dir . --out results.md
```

Results so far:

| fewkeys | model | sentences | mode | top1 | top5 | keystroke saving | notes |
|---|---|---|---|---|---|---|---|
| 0.1.0 | qwen2.5:3b | built-in 40 | plain prompt | 0.0 | 0.0 | (none) | answers were short phrases, not one word per letter |
| 0.5.0 | qwen2.5:3b | built-in 40 | grammar, 2 rounds | | | | being measured on CPU |
| 0.5.0 | qwen2.5:3b | built-in 40 | grammar, 2 rounds, context | | | | being measured on CPU |
| 0.5.0 | qwen2.5:3b | Turk Dialogues test, first 100 | grammar, 2 rounds, context | | | | being measured; SpeakFaster 64B: top5 0.726 to 0.763 |
| 0.5.0 | llama3.2:3b | Turk Dialogues test, first 100 | grammar, 2 rounds, context | | | | being measured |
| (none) | train-split lookup | Turk Dialogues test | phrase list only | | | | being measured |

## Status and limits

This is an early prototype (0.5). It has **not yet been tested with people who use AAC**. That testing is the next step, and it should be done with speech-language pathologists and people living with ALS. fewkeys is not a medical device and is not a replacement for a person's AAC system. It is meant to sit alongside one.

Roadmap:
- Send the chosen sentence straight into existing AAC software and eye-gaze keyboards. For now the clipboard is the way in. From their public documentation, OptiKey's suggestions come from its built-in engine or Presage, and AsTeRICS Grid's HTTP action sends requests but does not show the reply, so a deeper link needs changes on their side, which we would like to contribute.
- A small model fine-tuned for abbreviation expansion, trained on public dialogue data whose licence allows it, and measured on the Turk Dialogues test split.
- Personal voices: build a Piper voice from a person's banked recordings, offline.
- An evaluation with AAC users: words per minute and motor actions against their usual setup.

If you use AAC, or support someone who does, and want to try fewkeys or tell us what's missing, please open an issue or email munimversion2@gmail.com.

## Development

```
python -m unittest test_fewkeys.py test_turkbench.py
```

Made by MV2 of Munim, Inc. with AI agents. Apache-2.0 license.
