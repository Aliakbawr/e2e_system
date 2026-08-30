"""Convert numeric literals to words suitable for Persian speech synthesis."""

import re


_DIGIT_TRANSLATION = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
    "01234567890123456789",
)
_ONES = (
    "صفر",
    "یک",
    "دو",
    "سه",
    "چهار",
    "پنج",
    "شش",
    "هفت",
    "هشت",
    "نه",
    "ده",
    "یازده",
    "دوازده",
    "سیزده",
    "چهارده",
    "پانزده",
    "شانزده",
    "هفده",
    "هجده",
    "نوزده",
)
_TENS = ("", "", "بیست", "سی", "چهل", "پنجاه", "شصت", "هفتاد", "هشتاد", "نود")
_HUNDREDS = ("", "صد", "دویست", "سیصد", "چهارصد", "پانصد", "ششصد", "هفتصد", "هشتصد", "نهصد")
_SCALES = (
    "",
    "هزار",
    "میلیون",
    "میلیارد",
    "تریلیون",
    "کوادریلیون",
    "کوینتیلیون",
)
_DECIMAL_DENOMINATORS = {
    1: "دهم",
    2: "صدم",
    3: "هزارم",
    4: "ده هزارم",
    5: "صد هزارم",
    6: "میلیونیم",
}
_NUMBER_PATTERN = re.compile(
    r"(?<![\d۰-۹٠-٩])"
    r"(?P<sign>[+-]?)"
    r"(?P<integer>[\d۰-۹٠-٩]+(?:[,٬][\d۰-۹٠-٩]{3})*)"
    r"(?:[.٫](?P<fraction>[\d۰-۹٠-٩]+))?"
    r"(?![\d۰-۹٠-٩])"
)


def _join(parts: list[str]) -> str:
    return " و ".join(part for part in parts if part)


def _three_digits_to_words(number: int) -> str:
    parts = []
    hundreds, remainder = divmod(number, 100)
    if hundreds:
        parts.append(_HUNDREDS[hundreds])
    if remainder < 20:
        if remainder:
            parts.append(_ONES[remainder])
    else:
        tens, ones = divmod(remainder, 10)
        parts.append(_TENS[tens])
        if ones:
            parts.append(_ONES[ones])
    return _join(parts)


def integer_to_words(number: int) -> str:
    """Return the cardinal Persian words for an integer."""
    if number == 0:
        return _ONES[0]
    if number < 0:
        return f"منفی {integer_to_words(-number)}"

    groups = []
    remaining = number
    scale_index = 0
    while remaining:
        remaining, group = divmod(remaining, 1000)
        if group:
            if scale_index >= len(_SCALES):
                return " ".join(_ONES[int(digit)] for digit in str(number))
            words = _three_digits_to_words(group)
            scale = _SCALES[scale_index]
            groups.append(f"{words} {scale}".strip())
        scale_index += 1
    return _join(list(reversed(groups)))


def _decimal_to_words(integer: str, fraction: str | None, sign: str) -> str:
    integer_value = int(integer.translate(_DIGIT_TRANSLATION).replace(",", "").replace("٬", ""))
    whole_words = integer_to_words(integer_value)

    if fraction is not None:
        fraction = fraction.translate(_DIGIT_TRANSLATION).rstrip("0")
    if fraction:
        fraction_value = int(fraction)
        denominator = _DECIMAL_DENOMINATORS.get(len(fraction))
        if denominator is None:
            spoken_digits = " ".join(_ONES[int(digit)] for digit in fraction)
            result = f"{whole_words} ممیز {spoken_digits}"
        else:
            result = (
                f"{whole_words} و {integer_to_words(fraction_value)} {denominator}"
            )
    else:
        result = whole_words

    return f"منفی {result}" if sign == "-" and (integer_value or fraction) else result


def verbalize_numbers(text: str) -> str:
    """Replace Latin, Persian, and Arabic-Indic numeric literals with words."""
    return _NUMBER_PATTERN.sub(
        lambda match: _decimal_to_words(
            match.group("integer"),
            match.group("fraction"),
            match.group("sign"),
        ),
        str(text),
    )
