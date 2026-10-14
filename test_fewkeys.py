import contextlib
import io
import json
import os
import re
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fewkeys as fk  # noqa: E402

# A few lines in the layout of turk-dialogues.txt (split, id, six turns),
# written for the tests, plus one line in the filtered file's layout.
TURK_SAMPLE = (
    "set\tid\tturn1\tturn2\tturn3\tturn4\tturn5\tturn6\n"
    "train\tc1\tWould you like to sit down?\tNo, I'm fine standing up\tOkay\tThanks\tSure\tBye\n"
    "test\tc2\tWhat's the weather forecast?\tHot and  humid.\tUgh.\tI know\tStay in\tYes\n"
    "test\tc3\tHow was it?\tIt was one of the longest and most boring days I have ever had"
    "\tOh\tNo\tYes\tOk\n"
    "test\tc4\tAre you hungry?\tYes, a little\tPizza?\tSure\tOk\tGood\n"
    "c5\tIs this the filtered file?\tYes it is\tOh\tOk\tFine\tBye\n"
)


def write_file(folder, name, text):
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.schemas = []
        self.temperatures = []

    def generate(self, prompt, temperature=0.2, schema=None):
        self.prompts.append(prompt)
        self.schemas.append(schema)
        self.temperatures.append(temperature)
        return self.replies.pop(0) if self.replies else ""


class EchoClient:
    """Always gives the same reply, whatever the prompt."""

    def __init__(self, reply):
        self.reply = reply

    def generate(self, prompt, temperature=0.2, schema=None):
        return self.reply


class DownClient:
    def generate(self, prompt, temperature=0.2, schema=None):
        raise OSError("connection refused")


class NoGrammarClient:
    """Like an old Ollama: refuses a format schema, answers plain prompts."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def generate(self, prompt, temperature=0.2, schema=None):
        self.calls += 1
        if schema is not None:
            raise fk.ModelError("invalid JSON schema in format")
        return self.reply


class MissingModelClient:
    def generate(self, prompt, temperature=0.2, schema=None):
        raise fk.ModelError('model "llama3.2:3b" not found, try pulling it first')


class ParseTests(unittest.TestCase):
    def test_parse_abbreviation(self):
        ab = fk.parse_abbreviation("ishpitb/Bedroom")
        self.assertEqual(ab.initials, "ishpitb")
        self.assertEqual(ab.keywords, ["bedroom"])
        self.assertEqual(fk.parse_abbreviation("I S H").initials, "ish")
        self.assertEqual(fk.parse_abbreviation("iat//").keywords, [])

    def test_empty_abbreviation_is_an_error(self):
        with self.assertRaises(ValueError):
            fk.parse_abbreviation("")
        with self.assertRaises(ValueError):
            fk.parse_abbreviation("/water")

    def test_matches(self):
        self.assertTrue(fk.matches("I saw him play in the bedroom.", "ishpitb"))
        self.assertTrue(fk.matches("I saw him play in the bedroom", "ishpitb/bed"))
        self.assertFalse(fk.matches("I saw him play in the bedroom", "ishpitb/kitchen"))
        self.assertFalse(fk.matches("I saw him playing", "ishpitb"))
        self.assertFalse(fk.matches("You saw him play in the bedroom", "ishpitb"))

    def test_contractions_are_one_word(self):
        self.assertTrue(fk.matches("I'm too hot", "ith"))
        self.assertTrue(fk.matches("Let's watch the game!", "lwtg"))
        self.assertEqual(fk.initials_of("Please wait, I am typing"), "pwiat")

    def test_turn_length_counts_mid_sentence_punctuation(self):
        self.assertEqual(fk.turn_length("No, I'm fine standing up"), 6)
        self.assertEqual(fk.turn_length("Hot and humid."), 3)
        self.assertEqual(fk.turn_length("Yes. Dr. Lee said so!"), 7)
        self.assertEqual(fk.turn_length("Wait... what?"), 3)

    def test_word_accuracy(self):
        target = "Can you turn the volume up a little"
        self.assertEqual(fk.word_accuracy(target, [target.upper() + "."]), 1.0)
        self.assertEqual(fk.word_accuracy(target, ["Can you turn the volume up a bit"]), 7 / 8)
        self.assertEqual(fk.word_accuracy("See you soon", ["Sure you sing", "See you Sunday"]),
                         2 / 3)
        self.assertEqual(fk.word_accuracy("See you soon", ["See you"]), 0.0)
        self.assertEqual(fk.word_accuracy("See you soon", []), 0.0)

    def test_parse_model_output(self):
        reply = '1. I want water\n2) "Is it wet"\n- Hello there\n\n•  Go now\n'
        self.assertEqual(fk.parse_model_output(reply),
                         ["I want water", "Is it wet", "Hello there", "Go now"])

    def test_parse_model_output_drops_thinking_and_echoed_letters(self):
        reply = "<think>they want to rest</think>\niat -> I am tired\nI ate toast"
        self.assertEqual(fk.parse_model_output(reply, "iat"), ["I am tired", "I ate toast"])

    def test_parse_json_sentences(self):
        self.assertEqual(fk.parse_json_sentences('{"sentences": ["I am tired", " I  ate toast "]}'),
                         ["I am tired", "I ate toast"])
        self.assertEqual(fk.parse_json_sentences('["Go now"]'), ["Go now"])
        # Cut off by the token limit: keep the finished sentences.
        self.assertEqual(fk.parse_json_sentences('{"sentences": ["I want water", "I wa'),
                         ["I want water"])
        # Not JSON at all: read it line by line.
        self.assertEqual(fk.parse_json_sentences("I am tired\nI ate toast"),
                         ["I am tired", "I ate toast"])

    def test_sentence_pattern(self):
        pattern = fk.sentence_pattern("iwsw")
        self.assertTrue(pattern.startswith("^") and pattern.endswith("$"))
        self.assertTrue(re.match(pattern, "I want some water"))
        self.assertTrue(re.match(pattern, "I want, some water?"))
        self.assertTrue(re.match(pattern, "i wish she'd write."))
        self.assertFalse(re.match(pattern, "I want water"))
        self.assertFalse(re.match(pattern, "You want some water"))
        self.assertFalse(re.match(pattern, "I want some water now"))
        self.assertTrue(re.match(fk.sentence_pattern("i2"), "I 2nd"))
        for sentence in fk.CORE_PHRASES + fk.BENCH_PHRASES:
            if "'" not in sentence:
                self.assertTrue(re.match(fk.sentence_pattern(fk.initials_of(sentence)), sentence),
                                sentence)

    def test_sentence_schema(self):
        schema = fk.sentence_schema(fk.parse_abbreviation("iat/tired"), n=4)
        items = schema["properties"]["sentences"]
        self.assertEqual(items["items"]["pattern"], fk.sentence_pattern("iat"))
        self.assertEqual(items["maxItems"], 4)
        self.assertEqual(schema["required"], ["sentences"])
        json.dumps(schema)

    def test_build_prompt(self):
        ab = fk.parse_abbreviation("ishpitb/bedroom")
        prompt = fk.build_prompt(ab, ["Them: Where was he?"])
        self.assertIn("Letters: ishpitb (7 words)", prompt)
        self.assertIn("The words start with: i, s, h, p, i, t, b", prompt)
        self.assertIn("Them: Where was he?", prompt)
        self.assertIn("bedroom", prompt)
        self.assertIn("one per line", prompt)
        for letters, sentence in fk.EXAMPLES:
            self.assertEqual(fk.initials_of(sentence), letters)
            self.assertIn("%s -> %s" % (letters, sentence), prompt)

    def test_examples_are_not_bench_phrases(self):
        bench = {fk.normalize(p) for p in fk.BENCH_PHRASES}
        for _, sentence in fk.EXAMPLES:
            self.assertNotIn(fk.normalize(sentence), bench)

    def test_bench_dialogues(self):
        self.assertEqual(len(fk.BENCH_DIALOGUES), 40)
        self.assertEqual(fk.BENCH_PHRASES, tuple(reply for _, reply in fk.BENCH_DIALOGUES))
        self.assertEqual(len({fk.normalize(p) for p in fk.BENCH_PHRASES}), 40)
        for turn, reply in fk.BENCH_DIALOGUES:
            # The other person's words never give the whole answer away.
            self.assertNotIn(fk.normalize(reply), fk.normalize(turn))
            self.assertFalse(turn.startswith("Them:"), turn)

    def test_build_prompt_with_own_examples_and_json(self):
        ab = fk.parse_abbreviation("pwt")
        prompt = fk.build_prompt(ab, examples=[("ptko", "Put the kettle on")], as_json=True)
        self.assertIn("Things this person often says:\nptko -> Put the kettle on", prompt)
        self.assertIn('{"sentences": [...]}', prompt)
        self.assertNotIn("one per line", prompt)

    def test_choose_model(self):
        models = [{"name": "qwen3:8b", "size": 5200000000},
                  {"name": "nomic-embed-text:latest", "size": 274000000},
                  {"name": "qwen2.5:3b", "size": 1900000000}]
        self.assertEqual(fk.choose_model(models), "qwen2.5:3b")
        models = [{"name": "mistral:7b", "size": 4100000000},
                  {"name": "tinyllama:latest", "size": 637000000},
                  {"name": "nomic-embed-text:latest", "size": 274000000}]
        self.assertEqual(fk.choose_model(models), "tinyllama:latest")
        self.assertEqual(fk.choose_model([{"name": "phi4-mini:latest", "size": 2}]),
                         "phi4-mini:latest")
        self.assertIsNone(fk.choose_model([]))


class DialogueFileTests(unittest.TestCase):
    def test_load_dialogues_takes_short_second_turns_of_the_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_file(tmp, "turk-dialogues.txt", TURK_SAMPLE)
            self.assertEqual(fk.load_dialogues(path),
                             [("What's the weather forecast?", "Hot and humid."),
                              ("Are you hungry?", "Yes, a little")])
            self.assertEqual(fk.load_dialogues(path, limit=1),
                             [("What's the weather forecast?", "Hot and humid.")])
            self.assertEqual(fk.load_dialogues(path, split="train"),
                             [("Would you like to sit down?", "No, I'm fine standing up")])
            self.assertEqual(fk.load_dialogues(path, max_length=3),
                             [("What's the weather forecast?", "Hot and humid.")])

    def test_filtered_file_has_no_split_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_file(tmp, "turk-dialogues-filtered.txt",
                              "id\tturn1\tturn2\tturn3\tturn4\tturn5\tturn6\n"
                              "c5\tIs this the filtered file?\tYes it is\tOh\tOk\tFine\tBye\n")
            self.assertEqual(fk.load_dialogues(path), [])


class ExpandTests(unittest.TestCase):
    def test_common_phrases_work_without_a_model(self):
        result = fk.expand("iwsw")
        self.assertIn({"text": "I want some water", "source": "common"}, result["suggestions"])
        self.assertIsNone(result["model_error"])
        self.assertFalse(result["constrained"])

    def test_model_output_is_filtered_and_deduplicated(self):
        client = FakeClient([
            "1. I saw him play in the bedroom\n2. I saw her playing\n3. Is she playing in the bath\n",
            "I saw him play in the bedroom.\nI should have put it to bed",
        ])
        result = fk.expand("ishpitb/bed", client=client)
        self.assertEqual([s["text"] for s in result["suggestions"]],
                         ["I saw him play in the bedroom", "I should have put it to bed"])
        self.assertEqual({s["source"] for s in result["suggestions"]}, {"model"})
        self.assertEqual(len(client.prompts), 2)

    def test_constrained_expand_sends_schema_and_reads_json(self):
        client = FakeClient(['{"sentences": ["I want some water", "I wish she would", '
                             '"I want some water.", "Ice was so"]}'])
        result = fk.expand("iwsw", client=client, use_common=False)
        self.assertEqual([s["text"] for s in result["suggestions"]],
                         ["I want some water", "I wish she would"])
        self.assertTrue(result["constrained"])
        self.assertEqual(client.schemas[0]["properties"]["sentences"]["items"]["pattern"],
                         fk.sentence_pattern("iwsw"))
        self.assertIn('{"sentences": [...]}', client.prompts[0])

    def test_sentences_proposed_in_more_rounds_rank_first(self):
        client = FakeClient(['{"sentences": ["I wish she would", "I want some water"]}',
                             '{"sentences": ["Ice was so wet", "I want some water"]}',
                             '{"sentences": ["I want some water", "I wish she would"]}'])
        result = fk.expand("iwsw", client=client, use_common=False, rounds=3)
        self.assertEqual([s["text"] for s in result["suggestions"]],
                         ["I want some water", "I wish she would", "Ice was so wet"])
        self.assertEqual(len(client.prompts), 3)
        self.assertEqual([round(t, 2) for t in client.temperatures], [0.2, 0.7, 1.2])

    def test_a_repeat_within_one_round_is_one_vote(self):
        client = FakeClient(['{"sentences": ["Ice was so wet", "ice was so wet.", '
                             '"I want some water"]}',
                             '{"sentences": ["I want some water", "Ice was so wet"]}'])
        result = fk.expand("iwsw", client=client, use_common=False)
        self.assertEqual([s["text"] for s in result["suggestions"]],
                         ["Ice was so wet", "I want some water"])

    def test_full_list_skips_the_model(self):
        client = FakeClient(["I am tired"])
        result = fk.expand("iat", client=client, limit=1)
        self.assertEqual(result["suggestions"], [{"text": "I am thirsty", "source": "common"}])
        self.assertEqual(client.prompts, [])

    def test_plain_mode_sends_no_schema(self):
        client = FakeClient(["I am tired"])
        result = fk.expand("iat", client=client, use_common=False, constrain=False)
        self.assertEqual(client.schemas[0], None)
        self.assertFalse(result["constrained"])
        self.assertEqual(result["suggestions"], [{"text": "I am tired", "source": "model"}])

    def test_refused_grammar_falls_back_to_plain(self):
        client = NoGrammarClient("I am tired\nI ate toast")
        result = fk.expand("iat", client=client, use_common=False)
        self.assertEqual([s["text"] for s in result["suggestions"]], ["I am tired", "I ate toast"])
        self.assertFalse(result["constrained"])
        self.assertIsNone(result["model_error"])
        self.assertEqual(client.calls, 3)

    def test_missing_model_is_reported(self):
        result = fk.expand("iat", client=MissingModelClient())
        self.assertIn("not found", result["model_error"])
        self.assertEqual({s["source"] for s in result["suggestions"]}, {"common"})
        self.assertIn({"text": "I am tired", "source": "common"}, result["suggestions"])

    def test_model_failure_keeps_own_phrases(self):
        bank = fk.PhraseBank()
        bank.add("I am tired")
        result = fk.expand("iat", bank=bank, client=DownClient())
        self.assertEqual(result["suggestions"][0], {"text": "I am tired", "source": "mine"})
        self.assertIn("refused", result["model_error"])
        texts = [fk.normalize(s["text"]) for s in result["suggestions"]]
        self.assertEqual(len(texts), len(set(texts)))

    def test_own_phrases_become_prompt_examples(self):
        bank = fk.PhraseBank()
        bank.add("Put the kettle on", now=1)
        client = FakeClient([])
        fk.expand("wdyt", bank=bank, client=client, rounds=1)
        self.assertIn("ptko -> Put the kettle on", client.prompts[0])

    def test_phrase_bank_round_trip_and_ranking(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "phrases.json")
            bank = fk.PhraseBank(path)
            bank.add("Sit up", now=1)
            bank.add("See you tomorrow", now=2)
            bank.add("Sell your truck", now=3)
            bank.add("See you tomorrow", now=4)
            bank.add("   ", now=5)
            bank.save()
            again = fk.PhraseBank(path)
            self.assertEqual(again.suggest(fk.parse_abbreviation("syt")),
                             ["See you tomorrow", "Sell your truck"])
            self.assertEqual(again.phrases["See you tomorrow"]["count"], 2)
            self.assertEqual(len(again.phrases), 3)

    def test_phrase_bank_examples(self):
        bank = fk.PhraseBank()
        bank.add("Yes", now=1)
        bank.add("Put the kettle on", now=2)
        bank.add("Put the kettle on", now=3)
        bank.add("Call me later", now=4)
        self.assertEqual(bank.examples(),
                         [("ptko", "Put the kettle on"), ("cml", "Call me later")])
        self.assertEqual(bank.examples(k=1), [("ptko", "Put the kettle on")])

    def test_bench(self):
        client = EchoClient("I am going out\nSure you sing")
        report = fk.bench(["I am going out", "See you soon"], client)
        summary = report["summary"]
        self.assertEqual(summary["phrases"], 2)
        self.assertEqual(summary["top1"], 0.5)
        self.assertEqual(summary["top5"], 0.5)
        self.assertEqual(summary["word_accuracy"], 0.667)
        self.assertEqual(summary["constrained"], 1.0)
        self.assertEqual(report["rows"][0]["rank"], 1)
        self.assertEqual(report["rows"][0]["words_right"], 1.0)
        self.assertEqual(report["rows"][0]["context"], [])
        self.assertEqual(report["rows"][1]["suggestions"], ["Sure you sing"])
        self.assertEqual(report["rows"][1]["words_right"], 0.333)
        self.assertIsNone(report["rows"][1]["rank"])
        self.assertGreater(summary["keystroke_saving"], 0)

    def test_bench_with_context(self):
        client = FakeClient(['{"sentences": ["I am going out"]}'])
        report = fk.bench(["I am going out"], client, rounds=1,
                          contexts=[["Them: Where are you off to?"]])
        self.assertIn("The conversation so far:\nThem: Where are you off to?", client.prompts[0])
        self.assertEqual(report["rows"][0]["context"], ["Them: Where are you off to?"])
        self.assertEqual(report["rows"][0]["rank"], 1)
        self.assertEqual(report["summary"]["top1"], 1.0)

    def test_bench_phrases_are_not_core_phrases(self):
        core = {fk.normalize(p) for p in fk.CORE_PHRASES}
        for phrase in fk.BENCH_PHRASES:
            self.assertNotIn(fk.normalize(phrase), core)


class CliTests(unittest.TestCase):
    def test_add_then_expand(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "phrases.json")
            self.assertEqual(fk.main(["--bank", path, "--model", "", "add", "Good", "game"]), 0)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = fk.main(["--bank", path, "--model", "", "--rounds", "3", "expand", "gg"])
            self.assertEqual(code, 0)
            self.assertIn("1. Good game  [mine]", out.getvalue())

    def test_bench_needs_a_model(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = fk.main(["--bank", "", "--model", "", "bench", "--context"])
        self.assertEqual(code, 2)
        self.assertIn("needs a model", err.getvalue())

    def test_bench_on_dialogues(self):
        client = FakeClient(['{"sentences": ["Hot and humid."]}',
                             '{"sentences": ["Yes, a lot"]}'])
        client.model = "fake:1b"
        with tempfile.TemporaryDirectory() as tmp:
            path = write_file(tmp, "turk-dialogues.txt", TURK_SAMPLE)
            report_path = os.path.join(tmp, "report.json")
            out = io.StringIO()
            with mock.patch.object(fk, "OllamaClient", lambda *args: client), \
                    contextlib.redirect_stdout(out):
                code = fk.main(["--bank", "", "--model", "fake:1b", "--rounds", "1", "bench",
                                "--dialogues", path, "--context", "--json", report_path])
            self.assertEqual(code, 0)
            summary = json.loads(out.getvalue())
            with open(report_path, encoding="utf-8") as f:
                report = json.load(f)
        self.assertEqual(summary["phrases"], 2)
        self.assertEqual(summary["top1"], 0.5)
        self.assertEqual(summary["set"], "turk-dialogues test")
        self.assertTrue(summary["context"])
        self.assertEqual(summary["model"], "fake:1b")
        self.assertEqual(summary["fewkeys"], fk.__version__)
        self.assertIn("The conversation so far:\nThem: What's the weather forecast?",
                      client.prompts[0])
        self.assertEqual(report["rows"][1]["phrase"], "Yes, a little")
        self.assertEqual(report["rows"][1]["context"], ["Them: Are you hungry?"])
        self.assertEqual(report["rows"][1]["words_right"], 0.667)
        self.assertEqual(report["set"], "turk-dialogues test")

    def test_bench_dialogues_file_problems(self):
        with tempfile.TemporaryDirectory() as tmp:
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = fk.main(["--bank", "", "--model", "x", "bench", "--dialogues",
                                os.path.join(tmp, "missing.txt")])
            self.assertEqual(code, 2)
            self.assertIn("cannot read", err.getvalue())
            path = write_file(tmp, "turk-dialogues-filtered.txt",
                              "id\tturn1\tturn2\tturn3\tturn4\tturn5\tturn6\n"
                              "c5\tIs this the filtered file?\tYes it is\tOh\tOk\tFine\tBye\n")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = fk.main(["--bank", "", "--model", "x", "bench", "--dialogues", path])
            self.assertEqual(code, 2)
            self.assertIn("no test dialogues", err.getvalue())


class ServerTests(unittest.TestCase):
    def test_server(self):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "phrases.json")
            app = fk.App(fk.PhraseBank(path))
            server = fk.make_server(app, "127.0.0.1", 0)
            port = server.server_address[1]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = "http://127.0.0.1:%d" % port

            def post(route, obj, ctype="application/json"):
                req = urllib.request.Request(base + route, data=json.dumps(obj).encode("utf-8"),
                                             headers={"Content-Type": ctype}, method="POST")
                with opener.open(req, timeout=10) as resp:
                    return resp.status, resp.read()

            try:
                status, _ = post("/said", {"text": "Let's watch the game"})
                self.assertEqual(status, 200)
                self.assertTrue(os.path.exists(path))
                status, body = post("/expand", {"text": "lwtg", "context": ["Them: What now?"]})
                data = json.loads(body.decode("utf-8"))
                self.assertEqual(data["suggestions"][0],
                                 {"text": "Let's watch the game", "source": "mine"})
                status, body = post("/speak", {"text": "hello"})
                self.assertEqual(status, 204)
                with self.assertRaises(urllib.error.HTTPError) as cm:
                    post("/said", {"text": "x"}, ctype="text/plain")
                self.assertEqual(cm.exception.code, 415)
                cm.exception.close()
                with self.assertRaises(urllib.error.HTTPError) as cm:
                    post("/expand", {"text": ""})
                self.assertEqual(cm.exception.code, 400)
                cm.exception.close()
                with opener.open(base + "/", timeout=10) as resp:
                    page = resp.read().decode("utf-8")
                self.assertIn('<html lang="en">', page)
                self.assertIn('aria-live="polite"', page)
                self.assertIn('id="copy"', page)
            finally:
                server.shutdown()
                server.server_close()

    def test_busy_port_moves_to_a_later_one(self):
        app = fk.App(fk.PhraseBank())
        first = fk.make_server(app, "127.0.0.1", 0)
        try:
            port = first.server_address[1]
            second = fk.make_server_near(app, "127.0.0.1", port)
            try:
                self.assertGreater(second.server_address[1], port)
                self.assertLess(second.server_address[1], port + fk.PORT_TRIES)
            finally:
                second.server_close()
        finally:
            first.server_close()


if __name__ == "__main__":
    unittest.main()
