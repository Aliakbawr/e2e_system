import unittest

from src.tts.numbers import integer_to_words, verbalize_numbers


class PersianNumberVerbalizationTests(unittest.TestCase):
    def test_integer_uses_persian_place_values(self):
        self.assertEqual(integer_to_words(330), "سیصد و سی")
        self.assertEqual(integer_to_words(12045), "دوازده هزار و چهل و پنج")

    def test_decimal_uses_fractional_place_value(self):
        self.assertEqual(verbalize_numbers("1.02"), "یک و دو صدم")
        self.assertEqual(verbalize_numbers("0.125"), "صفر و صد و بیست و پنج هزارم")

    def test_accepts_persian_digits_and_decimal_separator(self):
        self.assertEqual(verbalize_numbers("۱٫۰۲"), "یک و دو صدم")
        self.assertEqual(verbalize_numbers("قیمت ۳۳۰ تومان است"), "قیمت سیصد و سی تومان است")

    def test_accepts_grouping_and_negative_numbers(self):
        self.assertEqual(verbalize_numbers("-1,250"), "منفی یک هزار و دویست و پنجاه")

    def test_trailing_decimal_zero_reduces_to_its_spoken_value(self):
        self.assertEqual(verbalize_numbers("1.20"), "یک و دو دهم")


if __name__ == "__main__":
    unittest.main()
