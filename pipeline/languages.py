"""Shared language definitions for EU5 pipeline.

Each entry: (url_code, game_loc_code, display_name)
- url_code: used in site URLs and output directory names
- game_loc_code: used in Paradox localization filenames (*_l_{code}.yml)
- display_name: native language name for UI display
"""

LANGUAGES = [
    ("de",      "german",       "Deutsch"),
    ("en",      "english",      "English"),
    ("es",      "spanish",      "Español"),
    ("fr",      "french",       "Français"),
    ("ja",      "japanese",     "日本語"),
    ("ko",      "korean",       "한국어"),
    ("pl",      "polish",       "Polski"),
    ("pt-br",   "braz_por",     "Português (BR)"),
    ("ru",      "russian",      "Русский"),
    ("tr",      "turkish",      "Türkçe"),
    ("zh-hans", "simp_chinese", "简体中文"),
]

# Quick lookups
GAME_CODE_TO_URL = {game: url for url, game, _ in LANGUAGES}
URL_TO_GAME_CODE = {url: game for url, game, _ in LANGUAGES}
