"""correct_turk: apply the published Turk Dialogues corrections (TDC).

Cai et al. (NAACL 2022, arXiv:2205.03767) fixed spelling and grammar slips in
Vertanen's Turk Dialogues and scored their abbreviation-expansion models on the
corrected text ("Turk Dialogues Corrected", TDC). The corrections are public in
TeamGleason/SpeakFaster (MIT), data/naacl_2022_suppl_data/
turk_dialogues_corrections.txt, one line per fix:

    test chain2_1_2_3_4_5_6 s/your ready/you're ready/ s/Ohh/Oh/

that is, the split, the dialogue id and one or more sed-style substitutions.
Each substitution replaces every occurrence in that dialogue's line. This is a
fresh implementation of that format (no code copied); the corrections file
itself is downloaded at run time, not shipped here.

Usage:
    python correct_turk.py turk-dialogues.txt turk_dialogues_corrections.txt \
        turk-dialogues-corrected.txt --report tdc-report.json

Standard library only. Apache-2.0 licensed.
Made by MV2 of Munim, Inc. with AI agents.
"""

import argparse
import json
import re
import sys

__version__ = "0.1.0"

LINE_RE = re.compile(r"^(\S+)\s+(\S+)\s+(.*\S)\s*$")


def parse_commands(text):
    """Read "s/old/new/ s/old/new/ ..." left to right into [(old, new), ...].

    Scanning slash by slash (not splitting on " s/") keeps a fix such as
    "s/it s/it's/" whole. Text may not contain a slash, as in sed.
    """
    out, i = [], 0
    while i < len(text):
        if not text.startswith("s/", i):
            raise ValueError("expected s/ at %r" % text[i:])
        middle = text.find("/", i + 2)
        end = text.find("/", middle + 1) if middle >= 0 else -1
        if end < 0:
            raise ValueError("unterminated substitution: %r" % text[i:])
        old, new = text[i + 2:middle], text[middle + 1:end]
        if not old or old == new:
            raise ValueError("empty or no-op substitution: %r" % text[i:end + 1])
        out.append((old, new))
        i = end + 1
        while i < len(text) and text[i].isspace():
            i += 1
    return out


def parse_corrections(lines):
    """Read correction lines into {dialogue id: [(split, old, new), ...]}.

    An id can appear on several lines; its substitutions are kept in file
    order. Blank lines are skipped; a malformed line raises ValueError.
    """
    out = {}
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        m = LINE_RE.match(line.rstrip("\r\n"))
        if not m:
            raise ValueError("line %d: expected split, id and s/old/new/" % number)
        split, chain, rest = m.groups()
        try:
            commands = parse_commands(rest)
        except ValueError as e:
            raise ValueError("line %d: %s" % (number, e))
        for old, new in commands:
            out.setdefault(chain, []).append((split, old, new))
    return out


def correct(dialogue_lines, corrections):
    """Apply corrections to turk-dialogues.txt lines (split, id, turns, by tabs).

    Returns (corrected lines, report). A substitution whose old text is not
    in its dialogue, or whose split does not match, is left out and listed
    in the report, so nothing is changed silently.
    """
    out = []
    applied, missing, wrong_split = [], [], []
    seen = set()
    for line in dialogue_lines:
        fields = line.split("\t")
        chain = fields[1].strip() if len(fields) > 1 else ""
        for split, old, new in corrections.get(chain, ()):
            seen.add(chain)
            if fields[0].strip() != split:
                wrong_split.append({"id": chain, "split": split, "old": old})
            elif old in line:
                line = line.replace(old, new)
                applied.append({"id": chain, "split": split, "old": old, "new": new})
            else:
                missing.append({"id": chain, "split": split, "old": old})
        out.append(line)
    unknown = sorted(set(corrections) - seen)
    report = {
        "substitutions": sum(len(v) for v in corrections.values()),
        "applied": len(applied),
        "applied_by_split": {s: sum(1 for a in applied if a["split"] == s)
                             for s in sorted({a["split"] for a in applied})},
        "dialogues_changed": len({a["id"] for a in applied}),
        "missing": missing,
        "wrong_split": wrong_split,
        "unknown_ids": unknown,
        "changes": applied,
    }
    return out, report


def main(argv=None):
    p = argparse.ArgumentParser(prog="correct_turk", description=__doc__.splitlines()[0])
    p.add_argument("dialogues", help="turk-dialogues.txt (Vertanen, https://aactext.org/turk/)")
    p.add_argument("corrections", help="turk_dialogues_corrections.txt (TeamGleason/SpeakFaster)")
    p.add_argument("out", help="where to write the corrected dialogues")
    p.add_argument("--report", help="also write a JSON report of every change")
    p.add_argument("--version", action="version", version="correct_turk " + __version__)
    args = p.parse_args(argv)
    if args.out in (args.dialogues, args.corrections):
        print("correct_turk: the output must be a new file", file=sys.stderr)
        return 2
    with open(args.corrections, encoding="utf-8") as f:
        corrections = parse_corrections(f)
    with open(args.dialogues, encoding="utf-8", errors="replace", newline="") as f:
        lines = f.readlines()
    fixed, report = correct(lines, corrections)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        f.writelines(fixed)
    summary = {k: report[k] for k in ("substitutions", "applied", "applied_by_split",
                                      "dialogues_changed")}
    summary.update(missing=len(report["missing"]), wrong_split=len(report["wrong_split"]),
                   unknown_ids=len(report["unknown_ids"]))
    print(json.dumps(summary, indent=1))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
