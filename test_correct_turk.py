import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import correct_turk as ct  # noqa: E402

# Lines in the layout of turk-dialogues.txt (split, id, six turns), written
# for the tests, and correction lines in the layout of the published file.
DIALOGUES = (
    "set\tid\tturn1\tturn2\tturn3\tturn4\tturn5\tturn6\n"
    "train\tc1\tOhh, hi\tHi, your ready?\tYes\tOk\tGo\tBye\n"
    "test\tc2\tWhat now?\tI ll call it s fine\tSure\tOk\tThen\tBye\n"
    "test\tc3\tDid you see Drogon?\tYes, he is a pit bill\tNice\tOk\tYes\tBye\n"
)
CORRECTIONS = (
    "train c1 s/Ohh/Oh/ s/your ready/you're ready/\n"
    "\n"
    "test c2 s/I ll/I'll/\n"
    "test c2 s/it s/it's/\n"
    "test\tc3 s/Drogon/Dragon/ s/pit bill/pitbull/ s/kyte/kite/\n"
    "dev c2 s/Then/Than/\n"
    "test c9 s/gone/went/\n"
)


class ParseTests(unittest.TestCase):
    def test_commands_are_read_left_to_right(self):
        self.assertEqual(ct.parse_commands("s/Ohh/Oh/ s/your ready/you're ready/"),
                         [("Ohh", "Oh"), ("your ready", "you're ready")])

    def test_text_containing_space_s_stays_whole(self):
        self.assertEqual(ct.parse_commands("s/it s/it's/"), [("it s", "it's")])
        self.assertEqual(ct.parse_commands("s/Your  welcome/You're welcome/"),
                         [("Your  welcome", "You're welcome")])

    def test_odd_but_valid_commands(self):
        self.assertEqual(ct.parse_commands("s/OH</OH,/"), [("OH<", "OH,")])

    def test_bad_commands_raise(self):
        for bad in ("x/a/b/", "s/a/b", "s//b/", "s/a/a/", "s/a/b/ junk"):
            with self.assertRaises(ValueError, msg=bad):
                ct.parse_commands(bad)

    def test_corrections_keep_every_line_of_an_id_in_order(self):
        got = ct.parse_corrections(io.StringIO(CORRECTIONS))
        self.assertEqual(got["c2"], [("test", "I ll", "I'll"), ("test", "it s", "it's"),
                                     ("dev", "Then", "Than")])
        self.assertEqual(len(got["c3"]), 3)
        self.assertEqual(sum(len(v) for v in got.values()), 9)

    def test_malformed_line_names_its_number(self):
        with self.assertRaisesRegex(ValueError, "line 2"):
            ct.parse_corrections(["test c1 s/a/b/\n", "test c2\n"])


class CorrectTests(unittest.TestCase):
    def setUp(self):
        self.lines = io.StringIO(DIALOGUES).readlines()
        self.fixed, self.report = ct.correct(self.lines, ct.parse_corrections(
            io.StringIO(CORRECTIONS)))

    def test_lines_are_corrected_in_place(self):
        self.assertEqual(self.fixed[0], self.lines[0])
        self.assertEqual(self.fixed[1], "train\tc1\tOh, hi\tHi, you're ready?\tYes\tOk\tGo\tBye\n")
        self.assertEqual(self.fixed[2], "test\tc2\tWhat now?\tI'll call it's fine\tSure\tOk\tThen\tBye\n")
        self.assertEqual(self.fixed[3], "test\tc3\tDid you see Dragon?\tYes, he is a pitbull"
                                        "\tNice\tOk\tYes\tBye\n")

    def test_report_lists_what_was_not_applied(self):
        r = self.report
        self.assertEqual(r["substitutions"], 9)
        self.assertEqual(r["applied"], 6)
        self.assertEqual(r["applied_by_split"], {"test": 4, "train": 2})
        self.assertEqual(r["dialogues_changed"], 3)
        self.assertEqual(r["missing"], [{"id": "c3", "split": "test", "old": "kyte"}])
        self.assertEqual(r["wrong_split"], [{"id": "c2", "split": "dev", "old": "Then"}])
        self.assertEqual(r["unknown_ids"], ["c9"])
        self.assertEqual(r["applied"] + len(r["missing"]) + len(r["wrong_split"]) + 1,
                         r["substitutions"])


class CliTests(unittest.TestCase):
    def test_writes_corrected_file_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "turk-dialogues.txt")
            fixes = os.path.join(tmp, "fixes.txt")
            out = os.path.join(tmp, "corrected.txt")
            report = os.path.join(tmp, "report.json")
            for path, text in ((src, DIALOGUES), (fixes, CORRECTIONS)):
                with open(path, "w", encoding="utf-8", newline="") as f:
                    f.write(text)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(ct.main([src, fixes, out, "--report", report]), 0)
            summary = json.loads(buf.getvalue())
            self.assertEqual((summary["applied"], summary["missing"], summary["unknown_ids"]),
                             (6, 1, 1))
            with open(out, encoding="utf-8", newline="") as f:
                self.assertIn("I'll call it's fine", f.read())
            with open(report, encoding="utf-8") as f:
                self.assertEqual(len(json.load(f)["changes"]), 6)

    def test_refuses_to_overwrite_its_input(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(ct.main(["a.txt", "b.txt", "a.txt"]), 2)


if __name__ == "__main__":
    unittest.main()
