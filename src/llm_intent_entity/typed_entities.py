"""Deterministic canonicalization and scoring for critical ASR entities.

The LLM is intentionally limited to locating spans and assigning an entity type.
Equality, metrics, and abstention are handled here so that critical identifiers do
not depend on a subjective model-generated score.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from enum import Enum
from functools import lru_cache
import re
import unicodedata

from indic_numtowords import num2words, supported_langs
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Entity types whose matching policy affects meaning preservation."""

    PHONE_NUMBER = "PHONE_NUMBER"
    ACCOUNT_ID = "ACCOUNT_ID"
    AMOUNT = "AMOUNT"
    DATE = "DATE"
    PERSON = "PERSON"
    LOCATION = "LOCATION"


class TypedEntityCandidate(BaseModel):
    """A raw entity span and type extracted by the LLM."""

    entity_type: EntityType
    raw: str = Field(min_length=1)


class CanonicalEntity(BaseModel):
    """A typed entity after deterministic normalization."""

    entity_type: EntityType
    raw: str
    canonical_value: str | None = None
    currency: str | None = None
    normalized_forms: list[str] = Field(default_factory=list)
    review_required: bool = False
    review_reason: str | None = None

    @property
    def display_value(self) -> str:
        if self.currency and self.canonical_value is not None:
            return f"{self.currency} {self.canonical_value}"
        if self.canonical_value is not None:
            return self.canonical_value
        return " / ".join(self.normalized_forms) or self.raw

    def typed_payload(self) -> dict[str, object]:
        """Serialize with entity-specific canonical field names."""

        payload: dict[str, object] = {
            "entity_type": self.entity_type.value,
            "raw": self.raw,
        }
        if self.entity_type in {EntityType.PHONE_NUMBER, EntityType.ACCOUNT_ID}:
            payload["canonical_digits"] = self.canonical_value
        elif self.entity_type == EntityType.AMOUNT:
            payload["currency"] = self.currency
            payload["numeric_value"] = self.canonical_value
        elif self.entity_type == EntityType.DATE:
            payload["iso_value"] = self.canonical_value
        else:
            payload["normalized_forms"] = self.normalized_forms
        if self.review_required:
            payload["review_required"] = True
            payload["review_reason"] = self.review_reason
        return payload


class EntityMatch(BaseModel):
    """A one-to-one deterministic reference/hypothesis match."""

    entity_type: EntityType
    reference: CanonicalEntity
    hypothesis: CanonicalEntity


class TypedEntityScore(BaseModel):
    """Strict entity metrics plus an explicit manual-review state."""

    score: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    named_entity_error_rate: float | None
    matches: list[EntityMatch] = Field(default_factory=list)
    missing: list[CanonicalEntity] = Field(default_factory=list)
    spurious: list[CanonicalEntity] = Field(default_factory=list)
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    review_required: bool = False
    review_reasons: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class MatchPolicy:
    """Configuration for deterministic and normalized-form matching."""

    person_mode: str = "exact"
    location_mode: str = "exact"
    fuzzy_threshold: float = 0.92
    phone_min_digits: int = 7
    phone_max_digits: int = 15
    default_currency: str | None = "INR"
    date_order: str | None = None

    def __post_init__(self) -> None:
        if self.person_mode not in {"exact", "fuzzy"}:
            raise ValueError("person_mode must be 'exact' or 'fuzzy'")
        if self.location_mode not in {"exact", "fuzzy"}:
            raise ValueError("location_mode must be 'exact' or 'fuzzy'")
        if not 0 <= self.fuzzy_threshold <= 1:
            raise ValueError("fuzzy_threshold must be between 0 and 1")
        if self.date_order not in {None, "DMY", "MDY"}:
            raise ValueError("date_order must be None, 'DMY', or 'MDY'")


@dataclass(frozen=True)
class NumberParse:
    value: str | None
    review_reason: str | None = None

    @property
    def review_required(self) -> bool:
        return self.value is None


LANGUAGE_ALIASES = {
    "assamese": "as",
    "bengali": "bn",
    "bodo": "brx",
    "dogri": "doi",
    "english": "en",
    "gujarati": "gu",
    "hindi": "hi",
    "kannada": "kn",
    "kashmiri": "ks",
    "konkani": "kok",
    "maithili": "mai",
    "malayalam": "ml",
    "manipuri": "mni",
    "marathi": "mr",
    "nepali": "ne",
    "odia": "or",
    "oriya": "or",
    "punjabi": "pa",
    "sanskrit": "sa",
    "santali": "sat",
    "sindhi": "sd",
    "tamil": "ta",
    "telugu": "te",
    "urdu": "ur",
}

EXTENDED_LANGS = {"doi", "sat", "mai", "sd", "brx", "ne", "mni", "kok", "ks", "sa"}
SUPPORTED_LANGS = set(supported_langs) | EXTENDED_LANGS

REPEAT_WORDS = {
    "double": 2,
    "triple": 3,
    "डबल": 2,
    "ट्रिपल": 3,
    "دبل": 2,
    "ٹرپل": 3,
}

CONNECTOR_WORDS = {"and", "और", "aur"}

TRANSLITERATED_HINDI = {
    "zero": 0,
    "shunya": 0,
    "sunya": 0,
    "ek": 1,
    "do": 2,
    "teen": 3,
    "tin": 3,
    "char": 4,
    "chaar": 4,
    "panch": 5,
    "paanch": 5,
    "chhe": 6,
    "cheh": 6,
    "saat": 7,
    "aath": 8,
    "nau": 9,
    "das": 10,
    "bees": 20,
    "tis": 30,
    "tees": 30,
    "chalis": 40,
    "chaalis": 40,
    "pachas": 50,
    "saath": 60,
    "sattar": 70,
    "assi": 80,
    "nabbe": 90,
}

CURRENCY_MARKERS = {
    "₹": "INR",
    "rs": "INR",
    "rs.": "INR",
    "inr": "INR",
    "rupee": "INR",
    "rupees": "INR",
    "rupaye": "INR",
    "rupay": "INR",
    "रुपया": "INR",
    "रुपये": "INR",
    "रुपए": "INR",
    "রুপি": "INR",
    "ரூபாய்": "INR",
    "రూపాయి": "INR",
    "ರೂಪಾಯಿ": "INR",
    "രൂപ": "INR",
    "રૂપિયા": "INR",
    "ਰੁਪਏ": "INR",
    "ଟଙ୍କା": "INR",
    "روپے": "INR",
    "$": "USD",
    "usd": "USD",
    "dollar": "USD",
    "dollars": "USD",
    "€": "EUR",
    "eur": "EUR",
    "euro": "EUR",
    "euros": "EUR",
    "£": "GBP",
    "gbp": "GBP",
    "pound": "GBP",
    "pounds": "GBP",
}

DECIMAL_WORDS = {
    "point",
    "decimal",
    "dot",
    "nuqta",
    "दशमलव",
    "दशांश",
    "দশমিক",
    "દશાંશ",
    "புள்ளி",
    "దశాంశం",
    "ದಶಮಾಂಶ",
    "ദശാംശം",
    "اعشاریہ",
    "ਦਸ਼ਮਲਵ",
    "ଦଶମିକ",
}

MONTHS = {
    "january": 1,
    "jan": 1,
    "जनवरी": 1,
    "february": 2,
    "feb": 2,
    "फरवरी": 2,
    "march": 3,
    "mar": 3,
    "मार्च": 3,
    "april": 4,
    "apr": 4,
    "अप्रैल": 4,
    "may": 5,
    "मई": 5,
    "june": 6,
    "jun": 6,
    "जून": 6,
    "july": 7,
    "jul": 7,
    "जुलाई": 7,
    "august": 8,
    "aug": 8,
    "अगस्त": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "सितंबर": 9,
    "october": 10,
    "oct": 10,
    "अक्टूबर": 10,
    "november": 11,
    "nov": 11,
    "नवंबर": 11,
    "december": 12,
    "dec": 12,
    "दिसंबर": 12,
}


def _ascii_digits(text: str) -> str:
    """Translate decimal digits from any Unicode script to ASCII."""

    translated: list[str] = []
    for char in unicodedata.normalize("NFKC", text):
        if char.isdecimal():
            try:
                translated.append(str(unicodedata.decimal(char)))
            except (TypeError, ValueError):
                translated.append(char)
        else:
            translated.append(char)
    return "".join(translated)


def _normalize_lookup(text: str) -> str:
    text = _ascii_digits(text).casefold()
    text = text.replace("\u200c", "").replace("\u200d", "").replace("\u200b", "")
    text = re.sub(r"[-‐‑‒–—]", " ", text)
    normalized: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if char.isspace() or char == "." or category[0] in {"L", "M", "N"}:
            normalized.append(char)
        else:
            normalized.append(" ")
    return " ".join("".join(normalized).split())


def _language_code(language: str | None) -> str:
    if not language:
        return "hi"
    normalized = language.strip().casefold().replace("_", "-")
    normalized = LANGUAGE_ALIASES.get(normalized, normalized.split("-", 1)[0])
    return normalized if normalized in SUPPORTED_LANGS else "hi"


def _word_variations(number: int, language: str) -> list[str]:
    try:
        result = num2words(number, lang=language, variations=True)
    except (KeyError, TypeError, ValueError):
        return []
    return result if isinstance(result, list) else [result]


@lru_cache(maxsize=64)
def _number_lexicon(language: str) -> dict[tuple[str, ...], int]:
    """Invert indic-numtowords for atomic 0-999 number phrases."""

    lexicon: dict[tuple[str, ...], int] = {}
    for number in range(1000):
        for phrase in _word_variations(number, language):
            normalized = _normalize_lookup(phrase)
            if normalized:
                lexicon[tuple(normalized.split())] = number

    if language in {"hi", "en"}:
        for word, number in TRANSLITERATED_HINDI.items():
            lexicon[(word,)] = number

    # Common ASR spellings missing from the upstream variation tables.
    lexicon[("oh",)] = 0
    lexicon[("o",)] = 0
    lexicon[("पांच",)] = 5
    lexicon[("छह",)] = 6
    return lexicon


@lru_cache(maxsize=64)
def _scale_lexicon(language: str) -> dict[tuple[str, ...], int]:
    scales: dict[tuple[str, ...], int] = {}
    one_forms = {
        tuple(_normalize_lookup(value).split())
        for value in _word_variations(1, language)
    }
    for scale in (100, 1_000, 100_000, 10_000_000):
        for phrase in _word_variations(scale, language):
            tokens = tuple(_normalize_lookup(phrase).split())
            if not tokens:
                continue
            scales[tokens] = scale
            for one in one_forms:
                if tokens[: len(one)] == one and len(tokens) > len(one):
                    scales[tokens[len(one) :]] = scale

    manual = {
        ("hundred",): 100,
        ("sau",): 100,
        ("हजार",): 1_000,
        ("hazaar",): 1_000,
        ("hazar",): 1_000,
        ("lakh",): 100_000,
        ("lac",): 100_000,
        ("crore",): 10_000_000,
        ("karod",): 10_000_000,
        ("करोड",): 10_000_000,
    }
    scales.update(manual)
    return scales


@lru_cache(maxsize=64)
def _amount_lexicon(language: str) -> dict[tuple[str, ...], int]:
    """Build direct forms for agglutinative numbers and common Indian scales."""

    values = set(range(10_000))
    for scale in (1_000, 100_000, 10_000_000):
        values.update(multiplier * scale for multiplier in range(1, 100))

    lexicon: dict[tuple[str, ...], int] = {}
    for number in values:
        for phrase in _word_variations(number, language):
            normalized = _normalize_lookup(phrase)
            if normalized:
                lexicon[tuple(normalized.split())] = number
    return lexicon


def _combined_lexicons(
    language: str | None,
) -> tuple[dict[tuple[str, ...], int], dict[tuple[str, ...], int]]:
    languages = {"en", "hi", _language_code(language)}
    numbers: dict[tuple[str, ...], int] = {}
    scales: dict[tuple[str, ...], int] = {}
    for code in languages:
        numbers.update(_number_lexicon(code))
        scales.update(_scale_lexicon(code))
    return numbers, scales


def _combined_amount_lexicon(language: str | None) -> dict[tuple[str, ...], int]:
    amounts: dict[tuple[str, ...], int] = {}
    for code in {"en", "hi", _language_code(language)}:
        amounts.update(_amount_lexicon(code))
    return amounts


def _longest_match(
    tokens: list[str],
    position: int,
    lexicon: dict[tuple[str, ...], int],
    max_tokens: int = 8,
) -> tuple[int, int] | None:
    for size in range(min(max_tokens, len(tokens) - position), 0, -1):
        key = tuple(tokens[position : position + size])
        if key in lexicon:
            return lexicon[key], size
    return None


def _validate_grouped_number(text: str) -> bool:
    if "," not in text:
        return True
    western = re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", text)
    indian = re.fullmatch(r"\d{1,3}(?:,\d{2})*,\d{3}(?:\.\d+)?", text)
    return bool(western or indian)


def parse_identifier(raw: str, language: str | None = None) -> NumberParse:
    """Parse an identifier as a digit sequence, preserving spoken chunks.

    Unlike amount parsing, ``fifty eight seven twenty`` becomes ``58720``;
    ``double`` and ``triple`` repeat only a single following digit.
    """

    source = _ascii_digits(raw).strip()
    compact = re.sub(r"[\s()+.\-/]", "", source)
    if compact and compact.isdigit():
        return NumberParse(compact)

    normalized = _normalize_lookup(source)
    if not normalized:
        return NumberParse(None, "identifier contains no digits or number words")

    tokens = normalized.split()
    numbers, _ = _combined_lexicons(language)
    digits: list[str] = []
    position = 0
    while position < len(tokens):
        token = tokens[position]
        if token in CONNECTOR_WORDS:
            position += 1
            continue

        repeat = REPEAT_WORDS.get(token)
        if repeat:
            match = _longest_match(tokens, position + 1, numbers)
            if not match or match[0] > 9:
                return NumberParse(
                    None, f"'{token}' must be followed by exactly one digit"
                )
            value, consumed = match
            digits.append(str(value) * repeat)
            position += consumed + 1
            continue

        if token.isdigit():
            digits.append(token)
            position += 1
            continue

        match = _longest_match(tokens, position, numbers)
        if match:
            value, consumed = match
            digits.append(str(value))
            position += consumed
            continue

        return NumberParse(None, f"unrecognized identifier token '{token}'")

    if not digits:
        return NumberParse(None, "identifier contains no digits or number words")
    return NumberParse("".join(digits))


def _extract_currency(
    raw: str, default_currency: str | None
) -> tuple[str, str | None, bool]:
    normalized = _ascii_digits(raw)
    found: set[str] = set()
    for marker, code in CURRENCY_MARKERS.items():
        if marker in {"₹", "$", "€", "£"}:
            if marker in normalized:
                found.add(code)
                normalized = normalized.replace(marker, " ")
            continue
        pattern = rf"(?<!\w){re.escape(marker)}(?!\w)"
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            found.add(code)
            normalized = re.sub(pattern, " ", normalized, flags=re.IGNORECASE)
    if len(found) > 1:
        return normalized, None, True
    return normalized, next(iter(found), default_currency), False


def _parse_integer_words(raw: str, language: str | None) -> NumberParse:
    tokens = _normalize_lookup(raw).split()
    if not tokens:
        return NumberParse(None, "amount contains no numeric value")

    direct_value = _combined_amount_lexicon(language).get(tuple(tokens))
    if direct_value is not None:
        return NumberParse(str(direct_value))

    numbers, scales = _combined_lexicons(language)
    total = 0
    current = 0
    found_number = False
    position = 0
    while position < len(tokens):
        if tokens[position] in CONNECTOR_WORDS:
            position += 1
            continue

        scale_match = _longest_match(tokens, position, scales)
        number_match = _longest_match(tokens, position, numbers)
        if scale_match and (not number_match or scale_match[1] >= number_match[1]):
            scale, consumed = scale_match
            found_number = True
            if scale == 100:
                current = max(current, 1) * scale
            else:
                total += max(current, 1) * scale
                current = 0
            position += consumed
            continue

        if number_match:
            number, consumed = number_match
            current += number
            found_number = True
            position += consumed
            continue

        if tokens[position].isdigit():
            current += int(tokens[position])
            found_number = True
            position += 1
            continue

        return NumberParse(None, f"unrecognized amount token '{tokens[position]}'")

    return (
        NumberParse(str(total + current))
        if found_number
        else NumberParse(None, "amount contains no numeric value")
    )


def parse_amount(
    raw: str,
    language: str | None = None,
    default_currency: str | None = "INR",
) -> tuple[NumberParse, str | None]:
    """Parse currency, Indian digit grouping, scales, and decimal amounts."""

    without_currency, currency, conflicting_currencies = _extract_currency(
        raw, default_currency
    )
    if conflicting_currencies:
        return NumberParse(None, "amount contains conflicting currencies"), None

    stripped = without_currency.strip()
    if re.fullmatch(r"[+\-]?[\d,]+(?:\.\d+)?", stripped):
        unsigned = stripped.lstrip("+-")
        if not _validate_grouped_number(unsigned):
            return NumberParse(None, "invalid numeric grouping"), currency
        try:
            value = Decimal(stripped.replace(",", ""))
        except InvalidOperation:
            return NumberParse(None, "invalid decimal amount"), currency
        return NumberParse(format(value.normalize(), "f")), currency

    normalized = _normalize_lookup(stripped)
    decimal_matches = [
        word
        for word in DECIMAL_WORDS
        if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", normalized)
    ]
    if decimal_matches:
        if len(decimal_matches) != 1:
            return NumberParse(None, "amount has multiple decimal markers"), currency
        left, right = re.split(
            rf"(?<!\w){re.escape(decimal_matches[0])}(?!\w)", normalized, maxsplit=1
        )
        whole = _parse_integer_words(left, language)
        fraction = parse_identifier(right, language)
        if whole.review_required or fraction.review_required:
            reason = whole.review_reason or fraction.review_reason
            return NumberParse(None, reason), currency
        return NumberParse(f"{whole.value}.{fraction.value}"), currency

    return _parse_integer_words(normalized, language), currency


def parse_date(raw: str, policy: MatchPolicy) -> NumberParse:
    """Parse explicit dates and abstain on unresolved day/month order."""

    normalized = _normalize_lookup(raw)
    iso_match = re.fullmatch(r"(\d{4})[./ ](\d{1,2})[./ ](\d{1,2})", normalized)
    if iso_match:
        year, month, day = map(int, iso_match.groups())
        try:
            return NumberParse(date(year, month, day).isoformat())
        except ValueError:
            return NumberParse(None, "invalid calendar date")

    numeric = re.fullmatch(r"(\d{1,2})[./ ](\d{1,2})[./ ](\d{4})", normalized)
    if numeric:
        first, second, year = map(int, numeric.groups())
        if policy.date_order == "DMY":
            day, month = first, second
        elif policy.date_order == "MDY":
            month, day = first, second
        elif first > 12 >= second:
            day, month = first, second
        elif second > 12 >= first:
            month, day = first, second
        else:
            return NumberParse(None, "ambiguous day/month order")
        try:
            return NumberParse(date(year, month, day).isoformat())
        except ValueError:
            return NumberParse(None, "invalid calendar date")

    month_pattern = "|".join(
        sorted((re.escape(month) for month in MONTHS), key=len, reverse=True)
    )
    words = re.fullmatch(rf"(\d{{1,2}})\s+({month_pattern})\s+(\d{{4}})", normalized)
    if words:
        day, month_word, year = words.groups()
        try:
            return NumberParse(
                date(int(year), MONTHS[month_word], int(day)).isoformat()
            )
        except ValueError:
            return NumberParse(None, "invalid calendar date")

    return NumberParse(None, "unsupported or context-dependent date")


def canonicalize_entity(
    candidate: TypedEntityCandidate | dict[str, str],
    language: str | None = None,
    policy: MatchPolicy | None = None,
) -> CanonicalEntity:
    """Canonicalize one typed entity without guessing through ambiguity."""

    policy = policy or MatchPolicy()
    if not isinstance(candidate, TypedEntityCandidate):
        candidate = TypedEntityCandidate.model_validate(candidate)

    entity_type = candidate.entity_type
    if entity_type in {EntityType.PHONE_NUMBER, EntityType.ACCOUNT_ID}:
        parsed = parse_identifier(candidate.raw, language)
        reason = parsed.review_reason
        if (
            entity_type == EntityType.PHONE_NUMBER
            and parsed.value is not None
            and not policy.phone_min_digits
            <= len(parsed.value)
            <= policy.phone_max_digits
        ):
            parsed = NumberParse(
                None,
                f"phone number must contain {policy.phone_min_digits}-{policy.phone_max_digits} digits",
            )
            reason = parsed.review_reason
        return CanonicalEntity(
            entity_type=entity_type,
            raw=candidate.raw,
            canonical_value=parsed.value,
            review_required=parsed.review_required,
            review_reason=reason,
        )

    if entity_type == EntityType.AMOUNT:
        parsed, currency = parse_amount(
            candidate.raw, language, policy.default_currency
        )
        return CanonicalEntity(
            entity_type=entity_type,
            raw=candidate.raw,
            canonical_value=parsed.value,
            currency=currency,
            review_required=parsed.review_required,
            review_reason=parsed.review_reason,
        )

    if entity_type == EntityType.DATE:
        parsed = parse_date(candidate.raw, policy)
        return CanonicalEntity(
            entity_type=entity_type,
            raw=candidate.raw,
            canonical_value=parsed.value,
            review_required=parsed.review_required,
            review_reason=parsed.review_reason,
        )

    normalized = _normalize_lookup(candidate.raw)
    if not normalized:
        return CanonicalEntity(
            entity_type=entity_type,
            raw=candidate.raw,
            review_required=True,
            review_reason="entity has no normalizable content",
        )
    return CanonicalEntity(
        entity_type=entity_type,
        raw=candidate.raw,
        canonical_value=normalized,
        normalized_forms=[normalized],
    )


def _entities_equal(
    reference: CanonicalEntity, hypothesis: CanonicalEntity, policy: MatchPolicy
) -> bool:
    if reference.entity_type != hypothesis.entity_type:
        return False
    if reference.review_required or hypothesis.review_required:
        return False
    if reference.entity_type == EntityType.AMOUNT:
        return (
            reference.canonical_value == hypothesis.canonical_value
            and reference.currency == hypothesis.currency
        )
    if reference.entity_type in {EntityType.PERSON, EntityType.LOCATION}:
        mode = (
            policy.person_mode
            if reference.entity_type == EntityType.PERSON
            else policy.location_mode
        )
        if reference.canonical_value == hypothesis.canonical_value:
            return True
        if mode == "fuzzy":
            return (
                SequenceMatcher(
                    None,
                    reference.canonical_value or "",
                    hypothesis.canonical_value or "",
                ).ratio()
                >= policy.fuzzy_threshold
            )
        return False
    return reference.canonical_value == hypothesis.canonical_value


def score_typed_entities(
    references: list[TypedEntityCandidate | dict[str, str]],
    hypotheses: list[TypedEntityCandidate | dict[str, str]],
    language: str | None = None,
    policy: MatchPolicy | None = None,
) -> TypedEntityScore:
    """Canonicalize, strictly match, and compute precision/recall/F1/NEER."""

    policy = policy or MatchPolicy()
    canonical_references = [
        canonicalize_entity(item, language, policy) for item in references
    ]
    canonical_hypotheses = [
        canonicalize_entity(item, language, policy) for item in hypotheses
    ]

    review_reasons = [
        f"{entity.entity_type.value} '{entity.raw}': {entity.review_reason}"
        for entity in canonical_references + canonical_hypotheses
        if entity.review_required
    ]
    if review_reasons:
        return TypedEntityScore(
            score=None,
            precision=None,
            recall=None,
            f1=None,
            named_entity_error_rate=None,
            missing=canonical_references,
            spurious=canonical_hypotheses,
            review_required=True,
            review_reasons=review_reasons,
        )

    adjacency = [
        [
            hypothesis_index
            for hypothesis_index, hypothesis in enumerate(canonical_hypotheses)
            if _entities_equal(reference, hypothesis, policy)
        ]
        for reference in canonical_references
    ]
    hypothesis_to_reference: dict[int, int] = {}

    def find_augmenting_match(reference_index: int, visited: set[int]) -> bool:
        for hypothesis_index in adjacency[reference_index]:
            if hypothesis_index in visited:
                continue
            visited.add(hypothesis_index)
            previous_reference = hypothesis_to_reference.get(hypothesis_index)
            if previous_reference is None or find_augmenting_match(
                previous_reference, visited
            ):
                hypothesis_to_reference[hypothesis_index] = reference_index
                return True
        return False

    for reference_index in range(len(canonical_references)):
        find_augmenting_match(reference_index, set())

    reference_to_hypothesis = {
        reference_index: hypothesis_index
        for hypothesis_index, reference_index in hypothesis_to_reference.items()
    }
    matches = [
        EntityMatch(
            entity_type=canonical_references[reference_index].entity_type,
            reference=canonical_references[reference_index],
            hypothesis=canonical_hypotheses[hypothesis_index],
        )
        for reference_index, hypothesis_index in sorted(reference_to_hypothesis.items())
    ]
    missing = [
        reference
        for reference_index, reference in enumerate(canonical_references)
        if reference_index not in reference_to_hypothesis
    ]
    matched_hypotheses = set(hypothesis_to_reference)

    spurious = [
        entity
        for index, entity in enumerate(canonical_hypotheses)
        if index not in matched_hypotheses
    ]
    true_positives = len(matches)
    false_negatives = len(missing)
    false_positives = len(spurious)
    precision = (
        true_positives / (true_positives + false_positives)
        if canonical_hypotheses
        else 1.0
    )
    recall = true_positives / len(canonical_references) if canonical_references else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    missing_by_type: dict[EntityType, int] = defaultdict(int)
    spurious_by_type: dict[EntityType, int] = defaultdict(int)
    for entity in missing:
        missing_by_type[entity.entity_type] += 1
    for entity in spurious:
        spurious_by_type[entity.entity_type] += 1
    substitutions = sum(
        min(count, spurious_by_type[entity_type])
        for entity_type, count in missing_by_type.items()
    )
    deletions = false_negatives - substitutions
    insertions = false_positives - substitutions
    denominator = max(len(canonical_references), 1)
    neer = (substitutions + deletions + insertions) / denominator

    return TypedEntityScore(
        score=f1,
        precision=precision,
        recall=recall,
        f1=f1,
        named_entity_error_rate=neer,
        matches=matches,
        missing=missing,
        spurious=spurious,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
    )


def entity_aware_text_signature(text: str, entities: list[CanonicalEntity]) -> str:
    """Replace raw spans with canonical typed placeholders for intent checks."""

    signature = _normalize_lookup(text)
    replacements: list[tuple[str, str]] = []
    for entity in entities:
        raw = _normalize_lookup(entity.raw)
        if raw and not entity.review_required:
            marker = f"{entity.entity_type.value.casefold()} {entity.display_value.casefold()}"
            replacements.append((raw, marker))
    for raw, marker in sorted(
        replacements, key=lambda pair: len(pair[0]), reverse=True
    ):
        signature = signature.replace(raw, marker)
    return " ".join(signature.split())


def typed_score_explanation(score: TypedEntityScore) -> str:
    """Create a concise audit trail suitable for the legacy explanation column."""

    if score.review_required:
        return "Deterministic scorer abstained: " + "; ".join(score.review_reasons)
    return (
        "Deterministic typed-entity score: "
        f"{len(score.matches)} matched, {len(score.missing)} missing, "
        f"{len(score.spurious)} spurious; F1={score.f1:.3f}, "
        f"NEER={score.named_entity_error_rate:.3f}."
    )
