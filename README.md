# fewkeys

**Say more with fewer keys.** fewkeys is a small, offline helper for people who type slowly, for example with eye gaze, a switch or a head pointer, as many people living with ALS do. You type only the first letter of each word, and fewkeys offers whole sentences to choose from:

```
iwsw        ->  I want some water
cwtatl      ->  Can we talk about this later
ishpitb/bed ->  I saw him play in the bedroom
```

Everything runs on your own computer. What you say is never sent anywhere.

## Why

In Google's SpeakFaster study, a language model expanded word-initial abbreviations like these into full sentences. In simulation it saved **57% more motor actions than conventional word prediction**. Two eye-gaze typists living with ALS typed **29% to 60% faster** with it than their usual baselines (Cai et al., *Nature Communications* 15, 9449, 2024: https://www.nature.com/articles/s41467-024-53873-3).

That system ran on a large cloud model. fewkeys tries the same idea with small open models that run offline on the person's own computer. That matters for three reasons:

- **Privacy.** A person's conversations with family, nurses and doctors stay on their machine.
- **No internet needed.** It keeps working in hospitals, on the road and during outages.
- **Free.** It's open source (Apache-2.0), with nothing to subscribe to.

## Where suggestions come from

Suggestions are shown in this order, and each one says where it came from:

1. **Your own phrases** ("said before"). Every sentence you choose is counted in a phrase file on your machine (`~/.fewkeys/phrases.json`), and the ones you use most often come first. Your most used sentences are also shown to the model as examples of how you talk.
2. **Everyday phrases** ("common"). A short built-in list ("I am too hot", "Please call the nurse", ...), so fewkeys helps from the first minute, even with no model running.
3. **A local language model** ("suggested"). This uses any small instruction model served by [Ollama](https://ollama.com). fewkeys checks every model guess: a sentence is shown only if it has exactly one word per letter, in order, and contains any word you added after a slash.

### Holding small models to the letters (0.2)

Small models are bad at "exactly one word per letter". In the first test (fewkeys 0.1.0, qwen2.5:3b, plain prompt), the model answered `cwtatl` with things like "could talk later" and "want to walk", and it recovered **none** of the 40 benchmark sentences.

So since 0.2, fewkeys doesn't just ask nicely. It sends Ollama a JSON schema whose `pattern` allows only sentences with one word per letter, each starting with that letter (for `iwsw`: `^[iI][A-Za-z0-9']*,? [wW]...$`). Ollama turns that pattern into a grammar, so the model *cannot* write a word that starts with the wrong letter. It still chooses the words, so the sentences are its best guesses.

If your Ollama is too old to take a schema, fewkeys falls back to a plain prompt and filters the answers. Use `--plain` to compare the two.

### Ranking by agreement (0.3)

fewkeys asks the model more than once (`--rounds`, default 2), the second time a little more freely. A sentence the model proposes in more rounds is shown first; ties keep the model's own order. SpeakFaster drew 128 to 256 samples per abbreviation from its model (256 when the initials and keywords added up to more than five; temperature 1.0); an ordinary computer can't afford that, so fewkeys uses a few rounds and trusts the sentences the model agrees with itself on. The model also sees six worked examples, from three-word requests to eight-word questions.

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

`--model auto` (the default) uses the first installed model from a short list of small ones (llama3.2:3b, qwen2.5:3b, gemma3:4b, phi4-mini, ...), or else the smallest model you have.

## Tips

- **Add the other person's words** in the "What they just said" box. The model uses the conversation to guess better.
- **Pin down an important word** by typing it, or just its start, after a slash: `cyctd/doc`.
- **Input methods.** The page has large buttons, a visible focus ring, and a status line that screen readers announce. After you ask for sentences, focus moves to the first one, so a single Enter (or a switch mapped to Enter) says it; Tab moves to the next. fewkeys has no scanning or dwell mode of its own: it relies on the dwell click of your eye-gaze software or on switch software that sends Tab and Enter. It has not been tried with eye-gaze or switch software yet; a built-in scanning mode is on the roadmap.
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

This follows the main abbreviation-expansion test in SpeakFaster (Cai et al., arXiv:2312.01532). Each item is the **second turn** of a **test-split** dialogue, kept if it has at most 10 words and mid-sentence punctuation marks. With `--context`, the model sees the first turn. Only the initials are typed, with no keywords. `--limit` sets how many dialogues are used (default 100, in file order; `--limit 0` for all), because each one can take 30 seconds or more on a CPU (see the results below).

On that test, SpeakFaster's fine-tuned 64-billion-parameter cloud model put the exact sentence among its top 5 for **72.6% to 76.3%** of turns, drawing 128 to 256 samples per abbreviation. The comparison is rough, and here is why:

- SpeakFaster used a corrected version of the dialogues, with typos fixed. The original file keeps them, and a typo makes an exact match impossible. The corrections are public in the SpeakFaster repository (TeamGleason/SpeakFaster, `data/naacl_2022_suppl_data/turk_dialogues_corrections.txt`). `correct_turk.py` applies them (see below), and the results table has runs on both texts.
- fewkeys uses a small, general model that wasn't trained for this, and asks it only 2 times.
- With the default `--limit` it uses the first 100 test dialogues, not all of them.

So expect fewkeys to score lower. The gap is what fine-tuning a small model (on the train split) would have to close. The same team showed that this works with small open models: fine-tuned on the training splits of four dialogue sets, T5-small (60M parameters) put the exact sentence in its top 5 for 42.7% of corrected test turns with context, T5-large (770M) 55.2% and T5-3B 59.4% (Cai et al., NAACL 2022, https://arxiv.org/abs/2205.03767, Table 4).

Your own phrase history isn't included in any benchmark, so real use, where people repeat themselves, should do better.

### Long benchmark runs and a no-model baseline (`turkbench.py`)

`turkbench.py` runs the same benchmark for long, unattended jobs, for example on a server CPU. It calls `fewkeys.bench` one sentence at a time, so the rows and metrics are the same, and it adds a few things:

- **Saved after every sentence.** It can stop at a time limit (`--max-minutes`, `--deadline`) and still keep every result so far, marked `"complete": false`.
- **reachable:** the share of target sentences the grammar can write at all. The grammar allows letters, digits and straight apostrophes inside a word, so a target with a hyphen or a curly apostrophe ("well-known", "don’t") can never be matched exactly. **top5_of_reachable** scores only those that can.
- **by_words:** top-5 accuracy for sentences of 1-3, 4-6 and 7-10 words.
- **`lookup`:** a baseline with no model at all. Every turn of the train-split dialogues goes into a phrase list. For each test item, it offers the five most frequent listed sentences with the same initials. This shows how much a phrase list alone gets, and **in_train** shows how many test sentences appear word for word in the train split.
- **`report`:** a Markdown table of every result file in a folder.

`correct_turk.py` (no model, standard library) applies the published corrections to the Turk Dialogues, giving the corrected text (Turk Dialogues Corrected) that the published figures use. Each correction line names a split, a dialogue and one or more `s/old/new/` fixes; a fix whose text or split does not match is left out and listed in the report, so nothing changes silently. The corrections file is downloaded, not included here.

```
python correct_turk.py turk-dialogues.txt turk_dialogues_corrections.txt turk-dialogues-corrected.txt --report tdc-report.json
python turkbench.py lookup --dialogues turk-dialogues.txt --out lookup.json
python turkbench.py bench --model qwen2.5:3b --set turk --dialogues turk-dialogues.txt --context --max-minutes 75 --out qwen-turk-context.json
python turkbench.py bench --model qwen2.5:3b --set builtin --out qwen-builtin.json
python turkbench.py report --dir . --out results.md
```

Results so far. The 0.5.0 rows come from one CPU-only run on 2026-10-15 (Ollama 0.40.4 in a container limited to 2 cores of an AMD Ryzen 9 5900HX, 3 GB of memory, a 75-minute limit per bench). The program is the same as 0.5.1. The 0.5.1 rows come from one run on 2026-10-16 on the corrected text, in the same container limits (Ollama 0.40.5; an 80-minute limit with context and 50 minutes without), and are in [results/tdc-v1/results.md](results/tdc-v1/results.md). The corrections changed 7 of the first 100 test items: 3 replies to be typed and 4 partner turns.

| fewkeys | model | sentences | mode | items | top1 | top5 | word accuracy | keystroke saving | seconds each |
|---|---|---|---|---|---|---|---|---|---|
| 0.1.0 | qwen2.5:3b | built-in 40 | plain prompt (answers were short phrases, not one word per letter) | 40 | 0.0 | 0.0 | | | |
| (none) | train-split lookup | Turk Dialogues test | phrase list only | all 225 | 0.049 | 0.049 | 0.053 | -0.179 | 0 |
| (none) | train-split lookup | Turk Dialogues test, first 100 | phrase list only | 100 | 0.06 | 0.06 | 0.06 | -0.176 | 0 |
| 0.5.0 | qwen2.5:1.5b | Turk Dialogues test, first 100 | grammar, 2 rounds, context | 100 | 0.14 | **0.21** | 0.442 | -0.041 | 30.5 |
| 0.5.0 | qwen2.5:1.5b | Turk Dialogues test, first 100 | grammar, 2 rounds, no context | 99 (time limit) | 0.02 | 0.03 | 0.211 | -0.19 | 30.5 |
| 0.5.0 | qwen2.5:3b | Turk Dialogues test, first 100 | grammar, 2 rounds, context | 54 (time limit) | 0.056 | 0.093 | 0.325 | -0.136 | 53.1 |
| 0.5.0 | qwen2.5:1.5b | built-in 40 | grammar, 2 rounds, context | 40 | 0.075 | 0.075 | 0.357 | -0.15 | 29.6 |
| 0.5.0 | qwen2.5:1.5b | built-in 40 | grammar, 2 rounds, no context | 38 (time limit) | 0.053 | 0.079 | 0.317 | -0.15 | 33.2 |
| (none) | train-split lookup | Turk Dialogues test, corrected, first 100 | phrase list only | 100 | 0.06 | 0.06 | 0.06 | -0.176 | 0 |
| 0.5.1 | qwen2.5:1.5b | Turk Dialogues test, corrected, first 100 | grammar, 2 rounds, context | 100 | 0.14 | **0.21** | 0.459 | -0.032 | 32.0 |
| 0.5.1 | qwen2.5:1.5b | Turk Dialogues test, corrected, first 100 | grammar, 2 rounds, no context | 97 (time limit) | 0.021 | 0.021 | 0.21 | -0.197 | 31.6 |

What these show, and what they don't:

- **The other person's last turn matters most.** On the same 99 Turk dialogues, qwen2.5:1.5b found the exact sentence in its top 5 for 20 with the first turn as context and for 3 without it. 19 sentences were found only with context and 2 only without (exact McNemar test, two-sided p = 0.0002). SpeakFaster also found that context helps a great deal. On the corrected text it found 21 of 100 with context and 2 of 97 without.
- **Far below fine-tuned models, as expected.** SpeakFaster's fine-tuned 64B cloud model reached 0.726 to 0.763 top-5 on the full corrected test set, and even a fine-tuned T5-small (60M) reached 0.427 there. This small general model, not trained for the task, reached 0.21 on the first 100 uncorrected items, and 0.21 again on the first 100 corrected items.
- **Correcting the text changed little here.** With context, top 5 stayed at 21 of 100 and word accuracy went from 0.442 to 0.459. The corrections touch few of these items, and the model's misses are mostly not typos.
- **Bigger was not better here.** On the first 54 items, which both runs reached, qwen2.5:1.5b found 11 and qwen2.5:3b found 5 (2 in common). With so few items the difference may be chance (exact McNemar, two-sided p = 0.15).
- **No keystrokes saved yet.** By fewkeys' strict measure, where every miss means typing the whole sentence, every setting cost slightly more keys than typing in full. Real use would add the person's own phrase history and slash keywords, which this benchmark leaves out, but that has not been measured.
- **Too slow on 2 cores.** 30 seconds per expansion is far too slow for conversation. Speed on a laptop with more cores or a GPU has not been measured yet.
- One run each, first 100 test dialogues in file order; the 0.5.0 rows use the uncorrected transcripts and the 0.5.1 rows the corrected ones. The runs were set up by AI agents working for MV2. The per-sentence files repeat the dataset's sentences, so they are not copied here; `turkbench.py report` rebuilds this table from them.

## Status and limits

This is an early prototype (0.5). It has **not yet been tested with people who use AAC**. That testing is the next step, and it should be done with speech-language pathologists and people living with ALS. fewkeys is not a medical device and is not a replacement for a person's AAC system. It is meant to sit alongside one.

Roadmap:
- Send the chosen sentence straight into existing AAC software and eye-gaze keyboards. For now the clipboard is the way in. From their public documentation, OptiKey's suggestions come from its built-in engine or Presage, and AsTeRICS Grid's HTTP action sends requests but does not show the reply, so a deeper link needs changes on their side, which we would like to contribute.
- A built-in switch-scanning mode and dwell-friendly layout, designed with switch and eye-gaze users.
- A small model fine-tuned for abbreviation expansion, trained on public dialogue data whose licence allows it, and measured on the corrected Turk Dialogues test split.
- Personal voices: build a Piper voice from a person's banked recordings, offline.
- An evaluation with AAC users: words per minute and motor actions against their usual setup.

If you use AAC, or support someone who does, and want to try fewkeys or tell us what's missing, please open an issue or email mv2@munimtech.com.

## Changes

- **After 0.5.1 (on main):** `correct_turk.py` 0.1.0 applies SpeakFaster's published corrections to the Turk Dialogues, with tests. The results table gains the first run on the corrected text ([results/tdc-v1/results.md](results/tdc-v1/results.md)): top 5 21 of 100 with context, as on the uncorrected text. `test_tdc_results.py` checks every number this README quotes from it. `fewkeys.py` is unchanged.
- **0.5.1:** the file released as 0.5.0 still carried the old licence line in its opening docstring, from before the move to Apache-2.0; 0.5.1 is the same program with the Apache-2.0 notice. The README now quotes SpeakFaster correctly: its 57% is *more motor actions saved than word prediction*, not 57% of all motor actions, and it drew 128 to 256 samples per abbreviation, not always 128. The results table now has the first CPU benchmark of 0.5.0, including what it doesn't show. The README no longer says the page works with switch scanning (it has no scanning mode and has not been tried with switch or eye-gaze software), and no longer makes two laptop claims that were never measured: that the first test ran on a laptop, and how long an expansion takes on one.

## Development

```
python -m unittest test_fewkeys.py test_turkbench.py test_release.py test_correct_turk.py test_tdc_results.py
```

Made by MV2 of Munim, Inc. with AI agents. Apache-2.0 license.
