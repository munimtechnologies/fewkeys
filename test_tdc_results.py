import json
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "results", "tdc-v1")


def read(*parts):
    with open(os.path.join(HERE, *parts), encoding="utf-8") as f:
        return f.read()


def cells(line):
    return [c.strip().strip("*") for c in line.strip().strip("|").split("|")]


def results_rows():
    """The rows of turkbench.py report for the corrected-text run, keyed by (model, phrases)."""
    rows = {}
    for line in read("results", "tdc-v1", "results.md").splitlines()[2:]:
        c = cells(line)
        rows[(c[0], int(c[4]))] = {"context": c[2] == "True", "top1": float(c[5]),
                                   "top5": float(c[6]), "word_accuracy": float(c[7]),
                                   "keystroke_saving": float(c[8]), "mean_seconds": float(c[10]),
                                   "complete": c[11] == "True"}
    return rows


def readme_rows():
    """The README table rows on the corrected text."""
    return [cells(line) for line in read("README.md").splitlines()
            if line.startswith("| ") and "corrected" in line]


class TdcResultsTest(unittest.TestCase):
    """Every number the README quotes from the run on Turk Dialogues Corrected."""

    def test_results_file(self):
        rows = results_rows()
        self.assertEqual(len(rows), 4)
        self.assertTrue(rows[("qwen2.5:1.5b", 100)]["context"])
        self.assertTrue(rows[("qwen2.5:1.5b", 100)]["complete"])
        self.assertFalse(rows[("qwen2.5:1.5b", 97)]["context"])
        self.assertFalse(rows[("qwen2.5:1.5b", 97)]["complete"])

    def test_readme_table_rows(self):
        rows, quoted = results_rows(), readme_rows()
        self.assertEqual(len(quoted), 3)
        for c in quoted:
            items = int(c[4].split()[0])
            model = "train-split lookup (no model)" if c[0] == "(none)" else c[1]
            r = rows[(model, items)]
            if c[0] != "(none)":
                self.assertEqual(c[0], "0.5.1")
                self.assertEqual("no context" not in c[3], r["context"], c)
                self.assertEqual("time limit" not in c[4], r["complete"], c)
            for i, key in ((5, "top1"), (6, "top5"), (7, "word_accuracy"), (8, "keystroke_saving")):
                self.assertEqual(float(c[i]), r[key], (c, key))
            self.assertLessEqual(abs(float(c[9]) - r["mean_seconds"]), 0.05 + 1e-9, c)

    def test_quoted_counts(self):
        readme, rows = read("README.md"), results_rows()
        with_context, without = rows[("qwen2.5:1.5b", 100)], rows[("qwen2.5:1.5b", 97)]
        found = (round(with_context["top5"] * 100), round(without["top5"] * 97))
        self.assertEqual(found, (21, 2))
        self.assertIn("On the corrected text it found 21 of 100 with context and 2 of 97 without", readme)
        self.assertIn("top 5 stayed at 21 of 100 and word accuracy went from 0.442 to 0.459", readme)
        self.assertEqual(with_context["word_accuracy"], 0.459)
        self.assertIn("top 5 21 of 100 with context", readme)

    def test_items_changed(self):
        counts = json.loads(read("results", "tdc-v1", "items-changed-counts.json"))["first_100"]
        self.assertEqual(counts["reply_changed"] + counts["partner_turn_changed"], counts["changed"])
        self.assertIn("The corrections changed %d of the first 100 test items: %d replies to be typed "
                      "and %d partner turns" % (counts["changed"], counts["reply_changed"],
                                                counts["partner_turn_changed"]), read("README.md"))

    def test_ollama_version(self):
        self.assertIn('{"version":"0.40.5"}', read("results", "tdc-v1", "machine.txt"))
        self.assertIn("(Ollama 0.40.5;", read("README.md"))


if __name__ == "__main__":
    unittest.main()
