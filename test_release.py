import math
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fewkeys as fk  # noqa: E402


def read(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return f.read()


class ReleaseTextTest(unittest.TestCase):
    """The licence notice and the figures the README quotes from SpeakFaster."""

    def test_version(self):
        self.assertEqual(fk.__version__, "0.5.1")

    def test_licence_is_apache(self):
        self.assertIn("Apache-2.0 licensed", fk.__doc__)
        for name in ("fewkeys.py", "README.md"):
            self.assertIsNone(re.search(r"\bMIT\b", read(name)), name)

    def test_speakfaster_figures(self):
        readme = read("README.md")
        # 57% is relative to word prediction (arXiv:2312.01532 abstract).
        self.assertIn("57% more motor actions than conventional word prediction", readme)
        self.assertNotIn("57% of motor actions", readme)
        # 128 samples up to five initials plus keywords, 256 above that.
        self.assertNotRegex(readme, r"\b128 samples")
        self.assertNotIn("the 128 samples", fk.expand.__doc__)
        self.assertIn("72.6% to 76.3%", readme)

    def test_mcnemar_figures(self):
        # Exact two-sided McNemar p from the discordant pairs the README quotes.
        def p(a, b):
            n, k = a + b, min(a, b)
            tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
            return min(1.0, 2 * tail)
        readme = read("README.md")
        self.assertEqual(round(p(19, 2), 4), 0.0002)
        self.assertIn("19 sentences were found only with context and 2 only without", readme)
        self.assertIn("p = 0.0002", readme)
        self.assertEqual(round(p(9, 3), 2), 0.15)
        self.assertIn("found 11 and qwen2.5:3b found 5 (2 in common)", readme)
        self.assertIn("p = 0.15", readme)

    def test_no_stale_status_or_contact(self):
        readme = read("README.md")
        self.assertNotIn("being measured", readme)
        self.assertNotIn("munimversion2", readme)
        self.assertNotIn("munimv2/", readme)
        self.assertIn("mv2@munimtech.com", readme)
        self.assertNotRegex(readme, r"\ban AI\b")

    def test_no_untested_claims(self):
        # The page has no scanning or dwell mode, and no run was timed on a laptop.
        readme = read("README.md")
        self.assertNotIn("switch scanning and plain keyboards", readme)
        self.assertNotIn("first test on a laptop", readme)
        self.assertNotIn("few seconds on a laptop CPU", readme)
        self.assertIn("It has not been tried with eye-gaze or switch software yet", readme)
        self.assertNotIn("dwell", fk.PAGE)
        self.assertNotIn("scan", fk.PAGE)


if __name__ == "__main__":
    unittest.main()
