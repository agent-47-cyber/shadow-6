import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hinglish import romanize_hinglish


class RomanHinglishTests(unittest.TestCase):
    def test_hindi_is_romanized_and_english_is_preserved(self):
        self.assertEqual(
            romanize_hinglish("कल मैं office जाऊंगा"),
            "kal main office jaunga",
        )

    def test_mixed_sentence_keeps_english_and_punctuation(self):
        self.assertEqual(
            romanize_hinglish("Can you send me वह file जो कल बनाई थी?"),
            "Can you send me vah file jo kal banai thi?",
        )

    def test_latin_only_input_is_unchanged(self):
        text = "Yaar, I will be late, traffic bahut hai."
        self.assertEqual(romanize_hinglish(text), text)


if __name__ == "__main__":
    unittest.main()
