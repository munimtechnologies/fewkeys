"""fewkeys: say more with fewer keys.

Offline abbreviation expansion for people who type slowly (eye gaze, switches,
head pointers). Type the first letter of each word, for example "iwsw", and
fewkeys offers whole sentences: "I want some water". Suggestions come from the
person's own phrase history, a small set of everyday phrases and, if one is
running, a local language model served by Ollama. Nothing leaves the machine.

The idea follows Google's SpeakFaster study (Cai et al., Nature Communications
2024), which used a cloud model; fewkeys does the same with small open models
on an ordinary laptop. Small models rarely keep to "one word per letter" on
their own, so fewkeys holds them to the letters with a JSON-schema grammar, and
ranks the sentences the model proposes again and again first.
Standard library only. MIT licensed.
Made by MV2 of Munim, Inc. with AI agents.
"""

import argparse
import collections
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

__version__ = "0.5.0"

DEFAULT_MODEL = "auto"
DEFAULT_OLLAMA = "http://127.0.0.1:11434"
DEFAULT_ROUNDS = 2
DEFAULT_PORT = 8765
PORT_TRIES = 10
MAX_BODY = 64 * 1024
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
# `bench --dialogues`: how many dialogues to take, and the longest turn kept
# (SpeakFaster's limit: 10 words and mid-sentence punctuation marks).
DEFAULT_DIALOGUE_LIMIT = 100
MAX_TURN_LENGTH = 10

# Small instruction models that answer quickly on a laptop CPU, best first.
# "auto" picks the first one installed, else the smallest installed model.
PREFERRED_MODELS = (
    "llama3.2:3b",
    "qwen2.5:3b",
    "gemma3:4b",
    "phi4-mini",
    "qwen2.5:1.5b",
    "llama3.2:1b",
    "gemma3:1b",
)

WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*")
NUMBERING_RE = re.compile(r"^\s*(?:\(?\d{1,2}[.):]\s*|[-*•–]\s+)")
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
JSON_STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"')
MID_PUNCT_RE = re.compile(r"[,.;:!?]+(?=.*[A-Za-z0-9])")

Abbreviation = collections.namedtuple("Abbreviation", "initials keywords")

# Everyday phrases, so fewkeys is useful before it has learned anything and
# when no model is running.
CORE_PHRASES = (
    "Yes",
    "No",
    "Thank you",
    "Please",
    "I need help",
    "Please wait, I am typing",
    "Give me a minute",
    "I am in pain",
    "I am uncomfortable",
    "I am too hot",
    "I am too cold",
    "I want some water",
    "I am thirsty",
    "I am hungry",
    "I am tired",
    "I want to sleep",
    "I need the bathroom",
    "I am short of breath",
    "Please call the nurse",
    "Please call my family",
    "Can you move my head",
    "Can you adjust my pillow",
    "Can you scratch my nose",
    "Please wipe my eyes",
    "Please turn me over",
    "Please turn on the TV",
    "Please turn off the light",
    "Please open the window",
    "Please charge my device",
    "Can you read it to me",
    "Something is wrong",
    "I love you",
    "How are you",
    "How was your day",
    "I am fine",
    "I don't know",
    "I don't understand",
    "Say that again please",
    "Not now",
    "Maybe later",
    "That's funny",
    "See you tomorrow",
    "Good morning",
    "Good night",
)

# Few-shot examples for the model prompt: short and long, statements and
# questions, so small models see the shape of the task.
EXAMPLES = (
    ("iwsw", "I want some water"),
    ("cyctd", "Can you call the doctor"),
    ("tfc", "Thanks for coming"),
    ("mbh", "My back hurts"),
    ("wifdt", "What is for dinner tonight"),
    ("dywtgfaw", "Do you want to go for a walk"),
)

# Conversational sentences for `fewkeys bench`, each after something the other
# person might just have said (used with `bench --context`). Kept apart from
# CORE_PHRASES so the benchmark measures the model, not the built-in list. The
# partner turns were written for this benchmark, not taken from real talk.
BENCH_DIALOGUES = (
    ("Can you hear the TV okay?", "Can you turn the volume up a little"),
    ("It's sunny out. Any plans?", "I would like to go outside today"),
    ("Don't forget you see Dr. Lee tomorrow.", "What time is the appointment tomorrow"),
    ("I'm back from the store.", "Did you remember to feed the dog"),
    ("Here's the newspaper you wanted.", "Please put my glasses on"),
    ("What should we put on TV?", "I want to watch the news"),
    ("Your brother was shouting at the TV all evening.", "Who won the game last night"),
    ("I'm going to see Aunt Ruth this afternoon.", "Tell her I said hello"),
    ("Are you comfortable?", "My left foot is cold"),
    ("We need to decide about the house.", "Can we talk about this later"),
    ("How was your day?", "I had a good day today"),
    ("The sun is right in your eyes.", "Please close the curtains"),
    ("I spoke to Sarah this morning.", "When are the kids coming over"),
    ("Is the pain back?", "I need my medicine now"),
    ("So, what did you think?", "That was a great movie"),
    ("Is there anyone you'd like to talk to?", "Could you call my sister"),
    ("You seem quiet. What's on your mind?", "I am worried about the bills"),
    ("I don't feel like cooking.", "Let's order pizza tonight"),
    ("It's so quiet in here.", "Put on some music please"),
    ("Why are you squinting?", "The light is too bright"),
    ("I have to run to the pharmacy.", "How long will you be gone"),
    ("Mark from the office sent a card.", "I miss going to work"),
    ("You got an email from your cousin.", "Can you read me the email"),
    ("The chair seems slow today.", "Please check the battery on my chair"),
    ("Do you want to come to the party tonight?", "I would rather stay home"),
    ("Take your time, I'm not going anywhere.", "Thank you for being so patient"),
    ("I'm going to hang the washing out.", "Is it raining outside"),
    ("How does that feel?", "Move my arm a little higher"),
    ("Where would you like to sit?", "I want to sit by the window"),
    ("I just got off the phone with the clinic.", "What did the doctor say"),
    ("Come on, we're going to be late.", "Please do not rush me"),
    ("Sorry it took me so long to visit.", "I am happy you came"),
    ("Do you want anything for dessert?", "Can I have some ice cream"),
    ("It's getting chilly in here.", "Turn the fan off please"),
    ("The kids are off school in the spring.", "We should plan a trip"),
    ("Anything I should remember for tomorrow?", "Remind me to call mom"),
    ("How's the soup?", "I think it needs more salt"),
    ("Are you looking for something?", "Where did you put my phone"),
    ("The neighbors are at the door.", "Ask them to wait outside"),
    ("How are you feeling this morning?", "I feel much better today"),
)
BENCH_PHRASES = tuple(reply for _, reply in BENCH_DIALOGUES)


class ModelError(ValueError):
    """Ollama answered, but refused the request (unknown model, bad format...)."""


def words(phrase):
    """Split a phrase into words; punctuation is ignored, "I'm" is one word."""
    return WORD_RE.findall(phrase)


def normalize(phrase):
    return " ".join(w.casefold() for w in words(phrase))


def initials_of(phrase):
    return "".join(w[0].lower() for w in words(phrase))


def turn_length(phrase):
    """Words plus mid-sentence punctuation marks, the length SpeakFaster limited."""
    return len(words(phrase)) + len(MID_PUNCT_RE.findall(phrase))


def parse_abbreviation(text):
    """Parse "ishpitb/bedroom" into initials "ishpitb" and keywords ["bedroom"].

    Each letter or digit before the first slash is the first character of one
    word. Each part after a slash is a word (or the start of a word) that must
    appear in the sentence.
    """
    head, _, tail = (text or "").partition("/")
    initials = "".join(ch.lower() for ch in head if ch.isalnum())
    if not initials:
        raise ValueError("type at least one letter")
    keywords = [k.strip().lower() for k in tail.split("/") if k.strip()]
    return Abbreviation(initials, keywords)


def matches(phrase, abbrev):
    """True if the phrase has one word per initial, in order, and every keyword."""
    if isinstance(abbrev, str):
        abbrev = parse_abbreviation(abbrev)
    ws = words(phrase)
    if len(ws) != len(abbrev.initials):
        return False
    for word, initial in zip(ws, abbrev.initials):
        if word[0].lower() != initial:
            return False
    lowered = [w.lower() for w in ws]
    return all(any(w.startswith(k) for w in lowered) for k in abbrev.keywords)


def word_accuracy(target, suggestions):
    """The best share of words in the right place among the suggestions, 0 to 1.

    A near miss such as "Can you turn the volume up a bit" for "... a little"
    scores 7/8: the person can pin the missing word with a slash and try again.
    """
    want = normalize(target).split()
    best = 0.0
    for suggestion in suggestions:
        got = normalize(suggestion).split()
        if want and len(got) == len(want):
            best = max(best, sum(1 for a, b in zip(want, got) if a == b) / len(want))
    return best


def sentence_pattern(initials):
    """A fully anchored regex for one sentence with one word per initial.

    Ollama turns it into a grammar, so the model can only write words that
    start with the typed letters, in order.
    """
    parts = []
    for ch in initials:
        first = "[%s%s]" % (ch.lower(), ch.upper()) if ch.lower() != ch.upper() else "[%s]" % ch
        parts.append(first + "[A-Za-z0-9']*")
    return "^" + ",? ".join(parts) + "[.?!]?$"


def sentence_schema(abbrev, n=5):
    return {
        "type": "object",
        "properties": {
            "sentences": {
                "type": "array",
                "items": {"type": "string", "pattern": sentence_pattern(abbrev.initials)},
                "minItems": 1,
                "maxItems": n,
            },
        },
        "required": ["sentences"],
    }


def choose_model(models):
    """Pick a model name from Ollama's /api/tags list, or None if there is none."""
    sizes = {}
    for m in models:
        name = m.get("name") or m.get("model") or ""
        if name and "embed" not in name:
            sizes[name] = m.get("size") or 0
    for want in PREFERRED_MODELS:
        for name in sizes:
            if name == want or name == want + ":latest":
                return name
    if not sizes:
        return None
    return min(sizes, key=lambda name: (sizes[name], name))


def clean_candidate(line, initials=None):
    line = NUMBERING_RE.sub("", line.strip(), count=1)
    if initials:
        m = re.match(r"(\S+)\s*(?:->|=>|:)\s*(.+)$", line)
        if m and m.group(1).lower() == initials:
            line = m.group(2)
    line = line.strip().strip('"“”`')
    if len(line) > 1 and line[0] == "'" and line[-1] == "'":
        line = line[1:-1]
    return " ".join(line.split())


def parse_model_output(text, initials=None):
    """Turn a model reply into candidate sentences, one per line."""
    text = THINK_RE.sub("", text or "")
    out = []
    for line in text.splitlines():
        candidate = clean_candidate(line, initials)
        if candidate:
            out.append(candidate)
    return out


def parse_json_sentences(text):
    """Read {"sentences": [...]} from a model reply, even if it was cut short."""
    text = THINK_RE.sub("", text or "")
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if isinstance(data, dict):
        data = data.get("sentences")
    if isinstance(data, list):
        return [" ".join(s.split()) for s in data if isinstance(s, str) and s.strip()]
    found = []
    for m in JSON_STRING_RE.finditer(text):
        try:
            s = json.loads(m.group(0))
        except ValueError:
            continue
        if s != "sentences" and s.strip():
            found.append(" ".join(s.split()))
    return found or parse_model_output(text)


def build_prompt(abbrev, context=(), n=5, examples=(), as_json=False):
    lines = [
        "You help a person with a motor disability talk faster. "
        "They type only the first letter of each word.",
        "Guess what they want to say. Each letter is the first letter of one word, "
        "in order, so the sentence has exactly as many words as there are letters.",
        "Write natural, everyday sentences in the first person, the way people really talk.",
    ]
    if examples:
        lines += ["", "Things this person often says:"]
        for letters, sentence in examples:
            lines.append("%s -> %s" % (letters, sentence))
    lines += ["", "Examples:"]
    for letters, sentence in EXAMPLES:
        lines.append("%s -> %s" % (letters, sentence))
    turns = [c.strip() for c in context if c and c.strip()]
    if turns:
        lines += ["", "The conversation so far:"] + turns[-8:]
    lines += ["", "Letters: %s (%d words)" % (abbrev.initials, len(abbrev.initials)),
              "The words start with: " + ", ".join(abbrev.initials)]
    if abbrev.keywords:
        lines.append("Use these words (or words starting with them): " + ", ".join(abbrev.keywords))
    if as_json:
        lines.append('Answer in JSON as {"sentences": [...]} with up to %d different '
                     "possible sentences, most likely first." % n)
    else:
        lines.append("Write %d different possible sentences, most likely first, one per line, "
                     "with no numbers and nothing else." % n)
    return "\n".join(lines)


class OllamaClient:
    """Minimal client for a local Ollama server (https://ollama.com)."""

    def __init__(self, base_url=DEFAULT_OLLAMA, model=DEFAULT_MODEL, timeout=60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        # Talk to the local server directly, never through a configured proxy.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _call(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.base_url + path, data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = json.loads(e.read().decode("utf-8")).get("error") or ""
            except (OSError, ValueError, AttributeError):
                pass
            finally:
                e.close()
            raise ModelError(detail or "Ollama answered HTTP %d" % e.code)

    def resolve_model(self):
        if self.model == "auto":
            tags = self._call("/api/tags")
            name = choose_model(tags.get("models") or [])
            if not name:
                raise ModelError("no Ollama model is installed; try: ollama pull "
                                 + PREFERRED_MODELS[0])
            self.model = name
        return self.model

    def generate(self, prompt, temperature=0.2, schema=None):
        payload = {
            "model": self.resolve_model(),
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 256},
        }
        if schema is not None:
            payload["format"] = schema
        data = self._call("/api/generate", payload)
        if "error" in data:
            raise ModelError(data["error"])
        return data.get("response", "")


class PhraseBank:
    """The person's own phrases, counted, kept in one JSON file on their machine."""

    def __init__(self, path=None):
        self.path = path
        self.phrases = {}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for phrase, rec in data.get("phrases", {}).items():
                self.phrases[phrase] = {"count": int(rec.get("count", 1)),
                                        "last": float(rec.get("last", 0))}

    def add(self, phrase, now=None):
        phrase = " ".join(phrase.split())
        if not words(phrase):
            return
        rec = self.phrases.setdefault(phrase, {"count": 0, "last": 0.0})
        rec["count"] += 1
        rec["last"] = time.time() if now is None else now

    def _ranked(self):
        return sorted(self.phrases, key=lambda p: (-self.phrases[p]["count"], -self.phrases[p]["last"]))

    def suggest(self, abbrev, limit=8):
        return [p for p in self._ranked() if matches(p, abbrev)][:limit]

    def examples(self, k=3):
        """The person's most used sentences, as (letters, sentence) for the prompt."""
        out = []
        for phrase in self._ranked():
            if len(out) >= k:
                break
            if 2 <= len(words(phrase)) <= 12:
                out.append((initials_of(phrase), phrase))
        return out

    def save(self):
        if not self.path:
            return
        folder = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(folder, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "phrases": self.phrases}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)


def expand(text, context=(), bank=None, client=None, limit=8, rounds=DEFAULT_ROUNDS,
           use_common=True, constrain=True):
    """Return sentences that fit the typed letters, best first.

    Order: the person's own phrases, then everyday phrases, then model
    guesses. The model is asked `rounds` times, each time a little more
    freely; a sentence it proposes in more rounds ranks higher (a cheap
    stand-in for the 128 samples per abbreviation SpeakFaster drew). Model
    guesses that do not fit the letters are dropped. With constrain, the
    model is held to the letters by a grammar; an Ollama that refuses the
    grammar is asked again in plain text.
    """
    abbrev = parse_abbreviation(text)
    out, seen = [], set()

    def take(phrase, source):
        key = normalize(phrase)
        if key and key not in seen and len(out) < limit:
            seen.add(key)
            out.append({"text": phrase, "source": source})

    if bank is not None:
        for phrase in bank.suggest(abbrev, limit):
            take(phrase, "mine")
    if use_common:
        for phrase in CORE_PHRASES:
            if matches(phrase, abbrev):
                take(phrase, "common")
    error = None
    constrained = False
    if client is not None and len(out) < limit:
        examples = bank.examples() if bank is not None else ()
        schema = sentence_schema(abbrev) if constrain else None
        votes = {}  # normalized sentence -> {"text", "rounds", "best"}, in first-seen order
        r = 0
        while r < rounds:
            prompt = build_prompt(abbrev, context, examples=examples, as_json=schema is not None)
            try:
                reply = client.generate(prompt, temperature=0.2 + 0.5 * r, schema=schema)
            except ModelError as e:
                if schema is not None:
                    # Older Ollama versions cannot take a schema; ask in plain text.
                    schema = None
                    continue
                error = str(e) or e.__class__.__name__
                break
            except (OSError, ValueError) as e:
                error = str(e) or e.__class__.__name__
                break
            if schema is not None:
                constrained = True
                candidates = parse_json_sentences(reply)
            else:
                candidates = parse_model_output(reply, abbrev.initials)
            this_round = []
            for candidate in candidates:
                key = normalize(candidate)
                if key not in this_round and matches(candidate, abbrev):
                    this_round.append(key)
                    vote = votes.setdefault(key, {"text": candidate, "rounds": 0,
                                                  "best": len(this_round) - 1})
                    vote["rounds"] += 1
                    vote["best"] = min(vote["best"], len(this_round) - 1)
            r += 1
        # sorted() is stable, so ties keep the order the model first gave them.
        for vote in sorted(votes.values(), key=lambda v: (-v["rounds"], v["best"])):
            take(vote["text"], "model")
    return {"abbreviation": abbrev.initials, "keywords": list(abbrev.keywords),
            "suggestions": out, "model_error": error, "constrained": constrained}


def load_dialogues(path, split="test", limit=DEFAULT_DIALOGUE_LIMIT, max_length=MAX_TURN_LENGTH):
    """Read bench items from Vertanen's Turk Dialogues (https://aactext.org/turk/).

    turk-dialogues.txt has one dialogue per line: the split (train, dev or
    test), an id and six turns, separated by tabs. Like SpeakFaster's main
    test, each item is a dialogue's second turn with its first turn as
    context, kept if it has at most max_length words and mid-sentence
    punctuation marks. Returns (first turn, second turn) pairs in file order;
    limit 0 keeps them all.
    """
    items = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            fields = [" ".join(x.split()) for x in line.split("\t")]
            if len(fields) < 4 or fields[0] != split:
                continue
            turn, reply = fields[2], fields[3]
            if turn and words(reply) and turn_length(reply) <= max_length:
                items.append((turn, reply))
                if limit and len(items) >= limit:
                    break
    return items


def bench(phrases, client, limit=5, rounds=DEFAULT_ROUNDS, constrain=True, contexts=None):
    """Measure how often the model recovers a sentence from its initials alone.

    contexts, if given, holds the conversation turns the model sees before
    each phrase, in the same order. Keystroke saving counts the letters plus
    one selection for a hit, and the letters plus typing the whole sentence
    for a miss.
    """
    rows, accuracies = [], []
    for i, phrase in enumerate(phrases):
        letters = initials_of(phrase)
        context = list(contexts[i]) if contexts else []
        start = time.monotonic()
        result = expand(letters, context, bank=None, client=client, limit=limit,
                        rounds=rounds, use_common=False, constrain=constrain)
        seconds = time.monotonic() - start
        suggestions = [s["text"] for s in result["suggestions"]]
        texts = [normalize(s) for s in suggestions]
        target = normalize(phrase)
        rank = texts.index(target) + 1 if target in texts else None
        accuracies.append(word_accuracy(phrase, suggestions))
        rows.append({"phrase": phrase, "letters": letters, "context": context, "rank": rank,
                     "words_right": round(accuracies[-1], 3),
                     "seconds": round(seconds, 2), "error": result["model_error"],
                     "constrained": result["constrained"], "suggestions": suggestions})
    n = len(rows) or 1
    full = sum(len(r["phrase"]) for r in rows)
    keys = sum(len(r["letters"]) + (1 if r["rank"] else len(r["phrase"])) for r in rows)
    summary = {
        "phrases": len(rows),
        "top1": round(sum(1 for r in rows if r["rank"] == 1) / n, 3),
        "top5": round(sum(1 for r in rows if r["rank"]) / n, 3),
        "word_accuracy": round(sum(accuracies) / n, 3),
        "fitting_per_phrase": round(sum(len(r["suggestions"]) for r in rows) / n, 2),
        "mean_seconds": round(sum(r["seconds"] for r in rows) / n, 2),
        "keystroke_saving": round(1 - keys / full, 3) if full else 0.0,
        "constrained": round(sum(1 for r in rows if r["constrained"]) / n, 3),
        "errors": sum(1 for r in rows if r["error"]),
    }
    return {"summary": summary, "rows": rows}


class Piper:
    """Speak with a Piper voice (https://github.com/rhasspy/piper), offline."""

    def __init__(self, model, binary="piper"):
        self.model = model
        self.binary = binary

    def synthesize(self, text):
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "say.wav")
                subprocess.run([self.binary, "--model", self.model, "--output_file", out],
                               input=text.encode("utf-8"), capture_output=True,
                               timeout=60, check=True)
                with open(out, "rb") as f:
                    return f.read()
        except (OSError, subprocess.SubprocessError):
            return None


class App:
    def __init__(self, bank, client=None, piper=None, limit=8, allow_any_host=False,
                 constrain=True, rounds=DEFAULT_ROUNDS):
        self.bank = bank
        self.client = client
        self.piper = piper
        self.limit = limit
        self.allow_any_host = allow_any_host
        self.constrain = constrain
        self.rounds = rounds
        self.lock = threading.Lock()

    def expand(self, text, context):
        with self.lock:
            return expand(text, context, bank=self.bank, client=self.client, limit=self.limit,
                          rounds=self.rounds, constrain=self.constrain)

    def said(self, text):
        with self.lock:
            self.bank.add(text)
            self.bank.save()

    def speak(self, text):
        if self.piper is None or not text:
            return None
        return self.piper.synthesize(text)


class _Handler(BaseHTTPRequestHandler):
    app = None
    server_version = "fewkeys/" + __version__

    def log_message(self, fmt, *args):
        # What the person says stays out of logs.
        pass

    def _send(self, status, body, ctype):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, obj):
        self._send(status, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def _host_ok(self):
        # Refuse requests addressed to other host names (DNS rebinding).
        if self.app.allow_any_host:
            return True
        host = (self.headers.get("Host") or "").strip().lower()
        name = host.split("]")[0] + "]" if host.startswith("[") else host.split(":")[0]
        return name in LOCAL_HOSTS

    def _read_body(self):
        # Read the body before any reply, so closing the socket never resets it.
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        return self.rfile.read(length) if 0 < length <= MAX_BODY else b""

    def do_GET(self):
        if not self._host_ok():
            return self._json(403, {"error": "forbidden host"})
        if self.path in ("/", "/index.html"):
            return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        raw = self._read_body()
        if not self._host_ok():
            return self._json(403, {"error": "forbidden host"})
        # Only JSON, so other web pages cannot post here without a preflight.
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            return self._json(415, {"error": "send application/json"})
        try:
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("expected a JSON object")
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        text = str(data.get("text", "")).strip()
        if self.path == "/expand":
            context = data.get("context") or []
            if not isinstance(context, list):
                context = []
            try:
                result = self.app.expand(text, [str(c) for c in context][-8:])
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            return self._json(200, result)
        if self.path == "/said":
            if text:
                self.app.said(text)
            return self._json(200, {"ok": True})
        if self.path == "/speak":
            audio = self.app.speak(text)
            if audio is None:
                self.send_response(204)
                self.end_headers()
                return None
            return self._send(200, audio, "audio/wav")
        return self._json(404, {"error": "not found"})


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR lets a second server share a busy port silently.
    allow_reuse_address = os.name != "nt"


def make_server(app, host="127.0.0.1", port=DEFAULT_PORT):
    handler = type("FewkeysHandler", (_Handler,), {"app": app})
    return _Server((host, port), handler)


def make_server_near(app, host="127.0.0.1", port=DEFAULT_PORT, tries=PORT_TRIES):
    """Listen on port, or on the first free one of the next few ports."""
    error = None
    for p in range(port, port + tries if port else 1):
        try:
            return make_server(app, host, p)
        except (OSError, OverflowError) as e:
            error = e
    raise error


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>fewkeys</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 1.4rem/1.4 system-ui, sans-serif; margin: 0 auto; max-width: 60rem; padding: 1rem; }
  label { display: block; font-weight: 600; margin-top: 1rem; }
  label.inline { display: flex; align-items: center; gap: .6rem; font-weight: 400; }
  input { font: inherit; width: 100%; box-sizing: border-box; padding: .6rem; border: 2px solid; border-radius: .4rem; }
  input[type=checkbox] { width: 2rem; height: 2rem; flex: none; }
  button { font: inherit; min-height: 3.5rem; padding: .5rem 1rem; margin: .3rem 0; border: 2px solid; border-radius: .4rem; cursor: pointer; }
  #choices button { display: block; width: 100%; text-align: left; }
  :focus-visible { outline: 4px solid #1a5fb4; outline-offset: 2px; }
  .source { font-size: 1rem; }
  #turns { font-size: 1.1rem; }
</style>
</head>
<body>
<main>
<h1>fewkeys</h1>
<p>Type the first letter of each word, then press Enter. You can add a whole word after a slash, like <kbd>ishpitb/bedroom</kbd>.</p>
<form id="ask">
  <label for="partner">What they just said (optional)</label>
  <input id="partner" autocomplete="off">
  <label for="letters">Your letters</label>
  <input id="letters" autocomplete="off" autocapitalize="off" spellcheck="false" required>
  <button type="submit">Show sentences</button>
  <label class="inline"><input type="checkbox" id="copy"> Also copy the sentence I choose, to paste into another app</label>
</form>
<p id="status" role="status" aria-live="polite"></p>
<h2 id="choices-heading">Sentences</h2>
<div id="choices" role="group" aria-labelledby="choices-heading"></div>
<h2>Conversation</h2>
<ol id="turns"></ol>
</main>
<script>
const turns = [];
const $ = (id) => document.getElementById(id);

$("copy").checked = localStorage.getItem("fewkeys-copy") === "1";
$("copy").addEventListener("change", () => localStorage.setItem("fewkeys-copy", $("copy").checked ? "1" : "0"));

function post(path, body) {
  return fetch(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
}

function addTurn(who, text) {
  turns.push(who + ": " + text);
  const li = document.createElement("li");
  li.textContent = who + ": " + text;
  $("turns").appendChild(li);
}

async function say(text) {
  let copied = false;
  if ($("copy").checked && navigator.clipboard) {
    try { await navigator.clipboard.writeText(text); copied = true; } catch (e) {}
  }
  addTurn("Me", text);
  post("/said", {text});
  $("choices").replaceChildren();
  $("letters").value = "";
  let spoken = false;
  try {
    const r = await post("/speak", {text});
    if (r.status === 200) {
      const audio = new Audio(URL.createObjectURL(await r.blob()));
      await audio.play();
      spoken = true;
    }
  } catch (e) {}
  if (!spoken && "speechSynthesis" in window) {
    speechSynthesis.speak(new SpeechSynthesisUtterance(text));
  }
  $("status").textContent = "Said: " + text + (copied ? " (copied)" : "");
  $("letters").focus();
}

$("ask").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const partner = $("partner").value.trim();
  if (partner) { addTurn("Them", partner); $("partner").value = ""; }
  const text = $("letters").value.trim();
  if (!text) return;
  $("status").textContent = "Thinking…";
  let data;
  try {
    const r = await post("/expand", {text, context: turns.slice(-8)});
    data = await r.json();
    if (!r.ok) { $("status").textContent = data.error || "Something went wrong."; return; }
  } catch (e) {
    $("status").textContent = "fewkeys is not running.";
    return;
  }
  const box = $("choices");
  box.replaceChildren();
  const labels = {mine: "(said before)", common: "(common)", model: "(suggested)"};
  data.suggestions.forEach((s, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = (i + 1) + ". " + s.text + " ";
    const tag = document.createElement("span");
    tag.className = "source";
    tag.textContent = labels[s.source] || "";
    b.appendChild(tag);
    b.addEventListener("click", () => say(s.text));
    box.appendChild(b);
  });
  let msg = data.suggestions.length + " sentences.";
  if (!data.suggestions.length) msg = "No sentences yet. Try adding a word after a slash.";
  if (data.model_error) msg += " The local model did not answer (" + data.model_error + "), so only saved and common phrases are shown.";
  $("status").textContent = msg;
  if (data.suggestions.length) box.querySelector("button").focus();
});
</script>
</body>
</html>
"""


def default_bank_path():
    return os.path.join(os.path.expanduser("~"), ".fewkeys", "phrases.json")


def main(argv=None):
    p = argparse.ArgumentParser(prog="fewkeys", description="Say more with fewer keys: "
                                "offline abbreviation expansion for slow typists.")
    p.add_argument("--bank", default=default_bank_path(), help="phrase history file")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help='Ollama model name; "auto" (default) picks a small installed model; '
                        "pass an empty string to work without a model")
    p.add_argument("--ollama", default=DEFAULT_OLLAMA, help="Ollama server URL")
    p.add_argument("--plain", action="store_true",
                   help="do not hold the model to the letters with a grammar")
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                   help="model calls per expansion (default %(default)s); sentences the "
                        "model proposes in more calls rank higher")
    p.add_argument("--version", action="version", version="fewkeys " + __version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("expand", help="print sentences for some letters")
    e.add_argument("letters")
    e.add_argument("--context", action="append", default=[],
                   help='a conversation turn such as "Them: where were you?" (repeatable)')
    s = sub.add_parser("serve", help="run the local web page")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help="port to listen on (default %(default)s); if it is busy, "
                        "fewkeys uses the next free port")
    s.add_argument("--piper-model", help="Piper .onnx voice to speak with (default: browser voice)")
    s.add_argument("--piper", default="piper", help="path to the piper program")
    a = sub.add_parser("add", help="add a phrase to the history")
    a.add_argument("phrase", nargs="+")
    b = sub.add_parser("bench", help="measure the model on built-in sentences")
    b.add_argument("--json", help="also write full results to this file")
    b.add_argument("--context", action="store_true",
                   help="show the model what the other person just said before each sentence")
    b.add_argument("--dialogues", metavar="FILE",
                   help="use the test split of Vertanen's Turk Dialogues instead of the built-in "
                        "sentences (turk-dialogues.txt from https://aactext.org/turk/)")
    b.add_argument("--limit", type=int, default=DEFAULT_DIALOGUE_LIMIT,
                   help="dialogues to take from --dialogues, 0 for all (default %(default)s)")
    args = p.parse_args(argv)

    bank = PhraseBank(args.bank)
    client = OllamaClient(args.ollama, args.model) if args.model else None
    constrain = not args.plain
    rounds = max(1, args.rounds)

    if args.cmd == "add":
        bank.add(" ".join(args.phrase))
        bank.save()
        return 0
    if args.cmd == "expand":
        try:
            result = expand(args.letters, args.context, bank=bank, client=client,
                            rounds=rounds, constrain=constrain)
        except ValueError as err:
            print("fewkeys: %s" % err, file=sys.stderr)
            return 2
        for i, sug in enumerate(result["suggestions"], 1):
            print("%d. %s  [%s]" % (i, sug["text"], sug["source"]))
        if result["model_error"]:
            print("model unavailable: " + result["model_error"], file=sys.stderr)
        return 0 if result["suggestions"] else 1
    if args.cmd == "bench":
        if client is None:
            print("fewkeys: bench needs a model (--model)", file=sys.stderr)
            return 2
        dialogues, source = BENCH_DIALOGUES, "built-in"
        if args.dialogues:
            try:
                dialogues = load_dialogues(args.dialogues, limit=max(0, args.limit))
            except OSError as err:
                print("fewkeys: cannot read %s (%s)" % (args.dialogues, err), file=sys.stderr)
                return 2
            if not dialogues:
                print("fewkeys: no test dialogues in %s; use turk-dialogues.txt, "
                      "not the filtered file" % args.dialogues, file=sys.stderr)
                return 2
            source = "turk-dialogues test"
        contexts = [["Them: " + turn] for turn, _ in dialogues] if args.context else None
        report = bench([reply for _, reply in dialogues], client, rounds=rounds,
                       constrain=constrain, contexts=contexts)
        info = {"model": client.model, "fewkeys": __version__, "rounds": rounds,
                "grammar": constrain, "context": args.context, "set": source}
        print(json.dumps(dict(report["summary"], **info), indent=1))
        if args.json:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(dict(report, **info), f, indent=1)
        return 0
    piper = Piper(args.piper_model, args.piper) if args.piper_model else None
    loopback = args.host in ("127.0.0.1", "localhost", "::1")
    app = App(bank, client, piper, allow_any_host=not loopback, constrain=constrain,
              rounds=rounds)
    try:
        server = make_server_near(app, args.host, args.port)
    except (OSError, OverflowError) as err:
        print("fewkeys: cannot listen on port %d or the %d after it (%s); try another --port"
              % (args.port, PORT_TRIES - 1, err), file=sys.stderr)
        return 2
    port = server.server_address[1]
    if args.port and port != args.port:
        print("fewkeys: port %d is busy, so using %d" % (args.port, port))
    print("fewkeys %s is running on http://%s:%d/  (Ctrl+C to stop)"
          % (__version__, "localhost" if loopback else args.host, port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
