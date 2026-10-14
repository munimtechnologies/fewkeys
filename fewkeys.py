"""fewkeys: say more with fewer keys.

Offline abbreviation expansion for people who type slowly (eye gaze, switches,
head pointers). Type the first letter of each word, for example "iwsw", and
fewkeys offers whole sentences: "I want some water". Suggestions come from the
person's own phrase history, a small set of everyday phrases and, if one is
running, a local language model served by Ollama. Nothing leaves the machine.

The idea follows Google's SpeakFaster study (Cai et al., Nature Communications
2024), which used a cloud model; fewkeys does the same with small open models
on an ordinary laptop. Standard library only. MIT licensed.
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
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

__version__ = "0.1.0"

DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA = "http://127.0.0.1:11434"
MAX_BODY = 64 * 1024
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}

WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*")
NUMBERING_RE = re.compile(r"^\s*(?:\(?\d{1,2}[.):]\s*|[-*•–]\s+)")
THINK_RE = re.compile(r"<think>.*?</think>", re.S)

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

# Few-shot examples for the model prompt.
EXAMPLES = (
    ("iwsw", "I want some water"),
    ("cyctd", "Can you call the doctor"),
    ("tfc", "Thanks for coming"),
)

# Conversational sentences for `fewkeys bench`, kept apart from CORE_PHRASES
# so the benchmark measures the model, not the built-in list.
BENCH_PHRASES = (
    "Can you turn the volume up a little",
    "I would like to go outside today",
    "What time is the appointment tomorrow",
    "Did you remember to feed the dog",
    "Please put my glasses on",
    "I want to watch the news",
    "Who won the game last night",
    "Tell her I said hello",
    "My left foot is cold",
    "Can we talk about this later",
    "I had a good day today",
    "Please close the curtains",
    "When are the kids coming over",
    "I need my medicine now",
    "That was a great movie",
    "Could you call my sister",
    "I am worried about the bills",
    "Let's order pizza tonight",
    "Put on some music please",
    "The light is too bright",
    "How long will you be gone",
    "I miss going to work",
    "Can you read me the email",
    "Please check the battery on my chair",
    "I would rather stay home",
    "Thank you for being so patient",
    "Is it raining outside",
    "Move my arm a little higher",
    "I want to sit by the window",
    "What did the doctor say",
    "Please do not rush me",
    "I am happy you came",
    "Can I have some ice cream",
    "Turn the fan off please",
    "We should plan a trip",
    "Remind me to call mom",
    "I think it needs more salt",
    "Where did you put my phone",
    "Ask them to wait outside",
    "I feel much better today",
)


def words(phrase):
    """Split a phrase into words; punctuation is ignored, "I'm" is one word."""
    return WORD_RE.findall(phrase)


def normalize(phrase):
    return " ".join(w.casefold() for w in words(phrase))


def initials_of(phrase):
    return "".join(w[0].lower() for w in words(phrase))


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


def build_prompt(abbrev, context=(), n=5):
    lines = [
        "You help a person with a motor disability talk faster. "
        "They type only the first letter of each word.",
        "Guess what they want to say. Each letter is the first letter of one word, "
        "in order, so the sentence has exactly as many words as there are letters.",
        "Write natural, everyday sentences in the first person, the way people really talk.",
        "",
        "Examples:",
    ]
    for letters, sentence in EXAMPLES:
        lines.append("%s -> %s" % (letters, sentence))
    turns = [c.strip() for c in context if c and c.strip()]
    if turns:
        lines += ["", "The conversation so far:"] + turns[-8:]
    lines += ["", "Letters: %s (%d words)" % (abbrev.initials, len(abbrev.initials))]
    if abbrev.keywords:
        lines.append("Use these words (or words starting with them): " + ", ".join(abbrev.keywords))
    lines.append("Write %d different possible sentences, one per line, "
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

    def generate(self, prompt, temperature=0.2):
        body = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 200},
        }).encode("utf-8")
        req = urllib.request.Request(self.base_url + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with self.opener.open(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if "error" in data:
            raise ValueError(data["error"])
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

    def suggest(self, abbrev, limit=8):
        hits = [p for p in self.phrases if matches(p, abbrev)]
        hits.sort(key=lambda p: (-self.phrases[p]["count"], -self.phrases[p]["last"]))
        return hits[:limit]

    def save(self):
        if not self.path:
            return
        folder = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(folder, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "phrases": self.phrases}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)


def expand(text, context=(), bank=None, client=None, limit=8, rounds=2, use_common=True):
    """Return sentences that fit the typed letters, best first.

    Order: the person's own phrases, then everyday phrases, then model
    guesses. Model guesses that do not fit the letters are dropped.
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
    if client is not None:
        prompt = build_prompt(abbrev, context)
        for r in range(rounds):
            if len(out) >= limit:
                break
            try:
                reply = client.generate(prompt, temperature=0.2 + 0.5 * r)
            except (OSError, ValueError) as e:
                error = str(e) or e.__class__.__name__
                break
            for candidate in parse_model_output(reply, abbrev.initials):
                if matches(candidate, abbrev):
                    take(candidate, "model")
    return {"abbreviation": abbrev.initials, "keywords": list(abbrev.keywords),
            "suggestions": out, "model_error": error}


def bench(phrases, client, limit=5, rounds=2):
    """Measure how often the model recovers a sentence from its initials alone.

    Keystroke saving counts the letters plus one selection for a hit, and the
    letters plus typing the whole sentence for a miss.
    """
    rows = []
    for phrase in phrases:
        letters = initials_of(phrase)
        start = time.monotonic()
        result = expand(letters, (), bank=None, client=client, limit=limit,
                        rounds=rounds, use_common=False)
        seconds = time.monotonic() - start
        texts = [normalize(s["text"]) for s in result["suggestions"]]
        target = normalize(phrase)
        rank = texts.index(target) + 1 if target in texts else None
        rows.append({"phrase": phrase, "letters": letters, "rank": rank,
                     "seconds": round(seconds, 2), "error": result["model_error"]})
    n = len(rows) or 1
    full = sum(len(r["phrase"]) for r in rows)
    keys = sum(len(r["letters"]) + (1 if r["rank"] else len(r["phrase"])) for r in rows)
    summary = {
        "phrases": len(rows),
        "top1": round(sum(1 for r in rows if r["rank"] == 1) / n, 3),
        "top5": round(sum(1 for r in rows if r["rank"]) / n, 3),
        "mean_seconds": round(sum(r["seconds"] for r in rows) / n, 2),
        "keystroke_saving": round(1 - keys / full, 3) if full else 0.0,
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
    def __init__(self, bank, client=None, piper=None, limit=8, allow_any_host=False):
        self.bank = bank
        self.client = client
        self.piper = piper
        self.limit = limit
        self.allow_any_host = allow_any_host
        self.lock = threading.Lock()

    def expand(self, text, context):
        with self.lock:
            return expand(text, context, bank=self.bank, client=self.client, limit=self.limit)

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


def make_server(app, host="127.0.0.1", port=8765):
    handler = type("FewkeysHandler", (_Handler,), {"app": app})
    return ThreadingHTTPServer((host, port), handler)


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
  input { font: inherit; width: 100%; box-sizing: border-box; padding: .6rem; border: 2px solid; border-radius: .4rem; }
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
  $("status").textContent = "Said: " + text;
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
  if (data.model_error) msg += " The local model did not answer, so only saved and common phrases are shown.";
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
                   help="Ollama model name; pass an empty string to work without a model")
    p.add_argument("--ollama", default=DEFAULT_OLLAMA, help="Ollama server URL")
    p.add_argument("--version", action="version", version="fewkeys " + __version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("expand", help="print sentences for some letters")
    e.add_argument("letters")
    e.add_argument("--context", action="append", default=[],
                   help='a conversation turn such as "Them: where were you?" (repeatable)')
    s = sub.add_parser("serve", help="run the local web page")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--piper-model", help="Piper .onnx voice to speak with (default: browser voice)")
    s.add_argument("--piper", default="piper", help="path to the piper program")
    a = sub.add_parser("add", help="add a phrase to the history")
    a.add_argument("phrase", nargs="+")
    b = sub.add_parser("bench", help="measure the model on built-in sentences")
    b.add_argument("--json", help="also write full results to this file")
    args = p.parse_args(argv)

    bank = PhraseBank(args.bank)
    client = OllamaClient(args.ollama, args.model) if args.model else None

    if args.cmd == "add":
        bank.add(" ".join(args.phrase))
        bank.save()
        return 0
    if args.cmd == "expand":
        try:
            result = expand(args.letters, args.context, bank=bank, client=client)
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
        report = bench(BENCH_PHRASES, client)
        print(json.dumps(dict(report["summary"], model=args.model), indent=1))
        if args.json:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(dict(report, model=args.model), f, indent=1)
        return 0
    piper = Piper(args.piper_model, args.piper) if args.piper_model else None
    loopback = args.host in ("127.0.0.1", "localhost", "::1")
    app = App(bank, client, piper, allow_any_host=not loopback)
    server = make_server(app, args.host, args.port)
    print("fewkeys %s is running on http://%s:%d/  (Ctrl+C to stop)"
          % (__version__, "localhost" if loopback else args.host, server.server_address[1]))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
