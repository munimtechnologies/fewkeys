import json
import os
import tempfile
import unittest

import turkbench as tb


class FakeClient:
    model = "fake"

    def __init__(self, sentences):
        self.sentences = sentences
        self.calls = 0

    def generate(self, prompt, temperature=0.2, schema=None):
        self.calls += 1
        return json.dumps({"sentences": self.sentences})


def fake_clock(values):
    values = list(values)

    def clock():
        return values.pop(0) if len(values) > 1 else values[0]
    return clock


ITEMS = [("How are you?", "I am fine"), ("Want some tea?", "Yes please")]


class TestReachable(unittest.TestCase):
    def test_plain_and_straight_apostrophe(self):
        self.assertTrue(tb.reachable("I'm fine, thanks."))
        self.assertTrue(tb.reachable("Meet me at 5 then"))

    def test_hyphen_and_curly_apostrophe_are_not(self):
        self.assertFalse(tb.reachable("It is a well-known place"))
        self.assertFalse(tb.reachable("I don’t know"))
        self.assertFalse(tb.reachable("..."))


class TestRun(unittest.TestCase):
    def test_full_run_saves_rows_and_summary(self):
        client = FakeClient(["I am fine", "Is anyone free"])
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "b.json")
            summary = tb.run(ITEMS, client, out, {"model": "fake"}, rounds=1, context=True)
            with open(out, encoding="utf-8") as f:
                data = json.load(f)
        self.assertTrue(summary["complete"])
        self.assertIsNone(summary["stopped"])
        self.assertEqual(summary["phrases"], 2)
        self.assertEqual(summary["top1"], 0.5)
        self.assertEqual(summary["top5"], 0.5)
        self.assertEqual(summary["reachable"], 1.0)
        self.assertEqual(summary["by_words"]["1-3"], {"phrases": 2, "top5": 0.5})
        self.assertEqual(summary["by_words"]["4-6"]["phrases"], 0)
        self.assertEqual(data["rows"][0]["context"], ["Them: How are you?"])
        self.assertEqual(data["rows"][0]["rank"], 1)
        self.assertIsNone(data["rows"][1]["rank"])
        self.assertEqual(data["summary"]["model"], "fake")
        self.assertEqual(client.calls, 2)

    def test_no_context(self):
        client = FakeClient(["I am fine"])
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "b.json")
            tb.run(ITEMS[:1], client, out, {}, rounds=1)
            with open(out, encoding="utf-8") as f:
                data = json.load(f)
        self.assertEqual(data["rows"][0]["context"], [])
        self.assertFalse(data["summary"]["context"])

    def test_stops_at_max_minutes_and_keeps_partial_rows(self):
        client = FakeClient(["I am fine"])
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "b.json")
            summary = tb.run(ITEMS, client, out, {}, rounds=1, max_minutes=1,
                             clock=fake_clock([0, 0, 0, 100]))
            with open(out, encoding="utf-8") as f:
                data = json.load(f)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["stopped"], "max_minutes")
        self.assertEqual(summary["phrases"], 1)
        self.assertEqual(len(data["rows"]), 1)
        self.assertFalse(data["summary"]["complete"])

    def test_stops_at_deadline(self):
        client = FakeClient(["I am fine"])
        with tempfile.TemporaryDirectory() as tmp:
            summary = tb.run(ITEMS, client, os.path.join(tmp, "b.json"), {}, rounds=1,
                             deadline=50, clock=fake_clock([0, 60]))
        self.assertEqual(summary["stopped"], "deadline")
        self.assertEqual(summary["phrases"], 0)
        self.assertEqual(client.calls, 0)


TURK = "\n".join([
    "set\tid\tturn1\tturn2\tturn3\tturn4\tturn5\tturn6",
    "train\ta\tHow are you?\tI am fine\tGood\tYes\tOk\tBye",
    "train\tb\tHi there\tIs anyone free\tI am fine.\tNo\tSure\tLater",
    "dev\tc\tHello\tI am free\tYes please\tNo\tSure\tLater",
    "test\td\tHow are you today?\tI am fine\tGood\tYes\tOk\tBye",
    "test\te\tWant some tea?\tYes please\tOk\tThanks\tSure\tBye",
]) + "\n"


class TestLookup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "turk-dialogues.txt")
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(TURK)

    def tearDown(self):
        self.tmp.cleanup()

    def test_most_frequent_train_turn_first_and_dev_ignored(self):
        items = tb.load_items("turk", self.path)
        self.assertEqual(items, [("How are you today?", "I am fine"), ("Want some tea?", "Yes please")])
        rows = tb.lookup_rows(self.path, items)
        self.assertEqual(rows[0]["suggestions"], ["I am fine", "Is anyone free"])
        self.assertEqual(rows[0]["rank"], 1)
        self.assertTrue(rows[0]["in_train"])
        self.assertEqual(rows[1]["suggestions"], [])
        self.assertIsNone(rows[1]["rank"])
        self.assertFalse(rows[1]["in_train"])

    def test_cli_writes_summary(self):
        out = os.path.join(self.tmp.name, "lookup.json")
        self.assertEqual(tb.main(["lookup", "--dialogues", self.path, "--out", out]), 0)
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["summary"]["top5"], 0.5)
        self.assertEqual(data["summary"]["in_train"], 0.5)
        self.assertEqual(data["summary"]["rounds"], 0)
        self.assertEqual(len(data["rows"]), 2)


class TestReport(unittest.TestCase):
    def test_table_lists_each_bench(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("a", "b"):
                tb.run(ITEMS[:1], FakeClient(["I am fine"]), os.path.join(tmp, name + ".json"),
                       {"model": "m-" + name, "set": "built-in"}, rounds=1)
            with open(os.path.join(tmp, "other.json"), "w", encoding="utf-8") as f:
                json.dump({"version": "0.1"}, f)
            out = os.path.join(tmp, "results.md")
            summaries = tb.report(tmp, out)
            with open(out, encoding="utf-8") as f:
                text = f.read()
        self.assertEqual([s["model"] for s in summaries], ["m-a", "m-b"])
        self.assertIn("| m-a | built-in | False | 1 | 1 | 1.0 | 1.0 |", text)
        self.assertEqual(len(text.strip().splitlines()), 4)

    def test_builtin_items(self):
        self.assertEqual(len(tb.load_items("builtin")), 40)


if __name__ == "__main__":
    unittest.main()
