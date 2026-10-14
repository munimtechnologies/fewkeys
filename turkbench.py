"""turkbench: run fewkeys' bench on a CPU, saving the results after every item.

The rows and metrics are those of `fewkeys bench` (fewkeys.bench is called one
item at a time), plus a time budget, a check of whether the grammar fewkeys
gives the model can write the target sentence at all, and accuracy by
sentence length (SpeakFaster reports accuracy against abbreviation length).
Standard library only. Apache-2.0 licensed.
Made by MV2 of Munim, Inc. with AI agents.
"""

import argparse
import collections
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fewkeys as fk  # noqa: E402

LENGTH_BANDS = ((1, 3), (4, 6), (7, 10))
TABLE_COLUMNS = ("model", "set", "context", "rounds", "phrases", "top1", "top5", "word_accuracy",
                 "keystroke_saving", "reachable", "mean_seconds", "complete")


def reachable(phrase):
    """True if the grammar fewkeys gives the model can write this sentence.

    The grammar allows letters, digits and straight apostrophes in a word, so
    targets with hyphens or curly apostrophes ("well-known", "don’t") can
    never be matched exactly.
    """
    ws = fk.words(phrase)
    if not ws:
        return False
    return re.match(fk.sentence_pattern(fk.initials_of(phrase)), " ".join(ws)) is not None


def load_items(set_name, path=None, limit=fk.DEFAULT_DIALOGUE_LIMIT):
    """(partner turn, reply) pairs: the built-in 40, or the Turk Dialogues test split."""
    if set_name == "builtin":
        return list(fk.BENCH_DIALOGUES)
    return fk.load_dialogues(path, limit=limit)


def summarize(rows):
    """fewkeys.bench's summary over the rows, plus reachability and length bands."""
    n = len(rows) or 1
    full = sum(len(r["phrase"]) for r in rows)
    keys = sum(len(r["letters"]) + (1 if r["rank"] else len(r["phrase"])) for r in rows)
    summary = {
        "phrases": len(rows),
        "top1": round(sum(1 for r in rows if r["rank"] == 1) / n, 3),
        "top5": round(sum(1 for r in rows if r["rank"]) / n, 3),
        "word_accuracy": round(sum(r["words_right"] for r in rows) / n, 3),
        "fitting_per_phrase": round(sum(len(r["suggestions"]) for r in rows) / n, 2),
        "mean_seconds": round(sum(r["seconds"] for r in rows) / n, 2),
        "keystroke_saving": round(1 - keys / full, 3) if full else 0.0,
        "constrained": round(sum(1 for r in rows if r["constrained"]) / n, 3),
        "errors": sum(1 for r in rows if r["error"]),
        "reachable": round(sum(1 for r in rows if r["reachable"]) / n, 3),
    }
    reach = [r for r in rows if r["reachable"]]
    summary["top5_of_reachable"] = round(sum(1 for r in reach if r["rank"]) / (len(reach) or 1), 3)
    bands = {}
    for lo, hi in LENGTH_BANDS:
        band = [r for r in rows if lo <= len(r["letters"]) <= hi]
        bands["%d-%d" % (lo, hi)] = {
            "phrases": len(band),
            "top5": round(sum(1 for r in band if r["rank"]) / (len(band) or 1), 3),
        }
    summary["by_words"] = bands
    return summary


def write(path, rows, info):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"summary": dict(summarize(rows), **info), "rows": rows}, f, indent=1)
    os.replace(tmp, path)


def run(items, client, out_path, info, rounds=fk.DEFAULT_ROUNDS, context=False,
        max_minutes=0, deadline=0, clock=time.time, log=None):
    """Bench the items in order, saving after each; stop early at a time limit."""
    rows, start, stopped = [], clock(), None
    info = dict(info, rounds=rounds, context=context, complete=False, stopped=None)
    for i, (turn, reply) in enumerate(items):
        now = clock()
        if max_minutes and now - start > max_minutes * 60:
            stopped = "max_minutes"
            break
        if deadline and now > deadline:
            stopped = "deadline"
            break
        contexts = [["Them: " + turn]] if context else None
        row = fk.bench([reply], client, rounds=rounds, contexts=contexts)["rows"][0]
        row["reachable"] = reachable(reply)
        rows.append(row)
        write(out_path, rows, dict(info, minutes=round((clock() - start) / 60, 1)))
        if log and (i + 1) % 10 == 0:
            log("%d/%d done, %d in top 5" % (i + 1, len(items), sum(1 for r in rows if r["rank"])))
    info.update(complete=stopped is None, stopped=stopped,
                minutes=round((clock() - start) / 60, 1))
    write(out_path, rows, info)
    return dict(summarize(rows), **info)


def lookup_rows(path, items, limit=5):
    """No-model baseline: the most frequent train-split turns with the same initials.

    Every turn of every train dialogue goes into a phrase list; for each item
    the list offers its `limit` most frequent sentences with the item's
    initials. This is what a phrase bank alone gets, before any model.
    """
    counts, first = collections.Counter(), {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            fields = [" ".join(x.split()) for x in line.split("\t")]
            if len(fields) < 3 or fields[0] != "train":
                continue
            for turn in fields[2:]:
                key = fk.normalize(turn)
                if key:
                    counts[key] += 1
                    first.setdefault(key, turn)
    by_initials = collections.defaultdict(list)
    for key in counts:  # first-seen order, so ties keep file order
        by_initials[fk.initials_of(key)].append(key)
    rows = []
    for _, reply in items:
        letters = fk.initials_of(reply)
        keys = sorted(by_initials.get(letters, []), key=lambda k: -counts[k])[:limit]
        suggestions = [first[k] for k in keys]
        target = fk.normalize(reply)
        rows.append({"phrase": reply, "letters": letters, "context": [],
                     "rank": keys.index(target) + 1 if target in keys else None,
                     "words_right": round(fk.word_accuracy(reply, suggestions), 3),
                     "seconds": 0.0, "error": None, "constrained": False,
                     "suggestions": suggestions, "reachable": reachable(reply),
                     "in_train": target in counts})
    return rows


def table(summaries):
    """A Markdown table of bench summaries, one row each."""
    lines = ["| " + " | ".join(TABLE_COLUMNS) + " |",
             "|" + "---|" * len(TABLE_COLUMNS)]
    for s in summaries:
        lines.append("| " + " | ".join(str(s.get(c, "")) for c in TABLE_COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def report(folder, out_path):
    summaries = []
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and isinstance(data.get("summary"), dict) and "rows" in data:
            summaries.append(dict(data["summary"], file=os.path.basename(path)))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(table(summaries))
    return summaries


def ollama_version(client):
    try:
        return client._call("/api/version").get("version", "")
    except (OSError, ValueError):
        return ""


def main(argv=None):
    p = argparse.ArgumentParser(prog="turkbench")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--model", required=True)
    b.add_argument("--set", choices=("turk", "builtin"), default="turk")
    b.add_argument("--dialogues", help="turk-dialogues.txt")
    b.add_argument("--context", action="store_true")
    b.add_argument("--limit", type=int, default=fk.DEFAULT_DIALOGUE_LIMIT)
    b.add_argument("--rounds", type=int, default=fk.DEFAULT_ROUNDS)
    b.add_argument("--max-minutes", type=float, default=0)
    b.add_argument("--deadline", type=float, default=0, help="stop after this Unix time")
    b.add_argument("--timeout", type=float, default=300, help="seconds per model call")
    b.add_argument("--ollama", default=fk.DEFAULT_OLLAMA)
    b.add_argument("--out", required=True)
    lk = sub.add_parser("lookup", help="the no-model phrase-list baseline")
    lk.add_argument("--dialogues", required=True)
    lk.add_argument("--limit", type=int, default=fk.DEFAULT_DIALOGUE_LIMIT)
    lk.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--dir", required=True)
    r.add_argument("--out", required=True)
    args = p.parse_args(argv)

    if args.cmd == "report":
        summaries = report(args.dir, args.out)
        print(table(summaries))
        return 0
    if args.cmd == "lookup":
        items = load_items("turk", args.dialogues, max(0, args.limit))
        rows = lookup_rows(args.dialogues, items)
        info = {"model": "train-split lookup (no model)", "fewkeys": fk.__version__,
                "grammar": False, "set": "turk-dialogues test", "limit": args.limit,
                "rounds": 0, "context": False, "complete": True, "stopped": None,
                "in_train": round(sum(1 for r in rows if r["in_train"]) / (len(rows) or 1), 3)}
        write(args.out, rows, info)
        print(json.dumps(dict(summarize(rows), **info), indent=1))
        return 0
    items = load_items(args.set, args.dialogues, max(0, args.limit))
    if not items:
        print("turkbench: no items", file=sys.stderr)
        return 2
    client = fk.OllamaClient(args.ollama, args.model, timeout=args.timeout)
    warm = time.time()
    try:
        client.generate("Say OK.")
    except (OSError, ValueError) as e:
        print("turkbench: warm-up failed: %s" % e, file=sys.stderr)
    info = {"model": args.model, "fewkeys": fk.__version__, "grammar": True,
            "set": "built-in" if args.set == "builtin" else "turk-dialogues test",
            "limit": args.limit if args.set == "turk" else len(items),
            "ollama": ollama_version(client), "warmup_seconds": round(time.time() - warm, 1)}
    summary = run(items, client, args.out, info, rounds=max(1, args.rounds), context=args.context,
                  max_minutes=args.max_minutes, deadline=args.deadline,
                  log=lambda msg: print(msg, flush=True))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
