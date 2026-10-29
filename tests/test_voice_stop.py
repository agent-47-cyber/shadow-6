import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.textproc import is_stop_command


class VoiceStopCommandTests(unittest.TestCase):
    def test_common_whisper_spellings_stop(self):
        for phrase in (
            "S-6 stop",
            "s6 stop",
            "S six stop",
            "shadow 6 stop",
            "SHADOW-6 stop.",
            "please, Shadow Six stop!",
        ):
            with self.subTest(phrase=phrase):
                self.assertTrue(is_stop_command(phrase))

    def test_phrase_must_end_with_stop_command(self):
        self.assertFalse(is_stop_command("shadow 6 is the name"))
        self.assertFalse(is_stop_command("please stop"))
        self.assertFalse(is_stop_command("S-6 stop now"))


if __name__ == "__main__":
    unittest.main()
