"""Internationalization (i18n) - Multi-language support.

This module provides multi-language support for ai-mem with 28+ languages.
Uses a simple JSON-based translation system with inheritance.

Configuration:
    AI_MEM_LANGUAGE=es  # Set language (default: en)
    AI_MEM_FALLBACK_LANGUAGE=en  # Fallback if translation missing

Supported languages:
    en (English), es (Spanish), zh (Chinese), ja (Japanese),
    ko (Korean), fr (French), de (German), it (Italian),
    pt (Portuguese), ru (Russian), ar (Arabic), hi (Hindi),
    and more...
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..logging_config import get_logger

logger = get_logger("i18n")

# i18n directory
I18N_DIR = Path(__file__).parent

# Default settings
DEFAULT_LANGUAGE = "en"
FALLBACK_LANGUAGE = "en"

# Configuration via environment
CURRENT_LANGUAGE = os.environ.get("AI_MEM_LANGUAGE", DEFAULT_LANGUAGE).lower()
CONFIGURED_FALLBACK = os.environ.get("AI_MEM_FALLBACK_LANGUAGE", FALLBACK_LANGUAGE).lower()

# Supported languages with their native names
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Español",
    "zh": "中文",
    "zh-tw": "繁體中文",
    "ja": "日本語",
    "ko": "한국어",
    "fr": "Français",
    "de": "Deutsch",
    "it": "Italiano",
    "pt": "Português",
    "pt-br": "Português (Brasil)",
    "ru": "Русский",
    "ar": "العربية",
    "hi": "हिन्दी",
    "bn": "বাংলা",
    "pa": "ਪੰਜਾਬੀ",
    "vi": "Tiếng Việt",
    "th": "ไทย",
    "id": "Bahasa Indonesia",
    "ms": "Bahasa Melayu",
    "tr": "Türkçe",
    "pl": "Polski",
    "nl": "Nederlands",
    "sv": "Svenska",
    "da": "Dansk",
    "no": "Norsk",
    "fi": "Suomi",
    "uk": "Українська",
}

# Translation cache
_translations: Dict[str, Dict[str, str]] = {}
_loaded_languages: set = set()


def get_language() -> str:
    """Get the current language.

    Returns:
        Language code (e.g., 'en', 'es')
    """
    return CURRENT_LANGUAGE


def set_language(lang: str) -> bool:
    """Set the current language.

    Args:
        lang: Language code

    Returns:
        True if language is supported
    """
    global CURRENT_LANGUAGE

    lang = lang.lower()
    if lang not in SUPPORTED_LANGUAGES:
        logger.warning(f"Unsupported language: {lang}")
        return False

    CURRENT_LANGUAGE = lang
    os.environ["AI_MEM_LANGUAGE"] = lang
    logger.info(f"Language set to: {lang}")
    return True


def list_languages() -> List[Dict[str, str]]:
    """List all supported languages.

    Returns:
        List of language info dicts
    """
    return [
        {"code": code, "name": name}
        for code, name in SUPPORTED_LANGUAGES.items()
    ]


def _load_translations(lang: str) -> Dict[str, str]:
    """Load translations for a language.

    Args:
        lang: Language code

    Returns:
        Translation dictionary
    """
    if lang in _loaded_languages:
        return _translations.get(lang, {})

    # Try to load from file
    lang_file = I18N_DIR / f"{lang}.json"

    if not lang_file.exists():
        # Try base language (e.g., 'zh' for 'zh-tw')
        base_lang = lang.split("-")[0]
        lang_file = I18N_DIR / f"{base_lang}.json"

    if lang_file.exists():
        try:
            with open(lang_file, encoding="utf-8") as f:
                _translations[lang] = json.load(f)
            _loaded_languages.add(lang)
            logger.debug(f"Loaded translations for: {lang}")
        except Exception as e:
            logger.warning(f"Failed to load translations for {lang}: {e}")
            _translations[lang] = {}
    else:
        _translations[lang] = {}

    _loaded_languages.add(lang)
    return _translations.get(lang, {})


def t(key: str, **kwargs) -> str:
    """Translate a key to the current language.

    Args:
        key: Translation key (e.g., 'cli.add.success')
        **kwargs: Variables to interpolate

    Returns:
        Translated string (or key if not found)
    """
    lang = get_language()
    translations = _load_translations(lang)

    # Look up key (supports dot notation)
    value = translations
    for part in key.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            value = None
            break

    # Fallback to fallback language
    if value is None and lang != CONFIGURED_FALLBACK:
        fallback_translations = _load_translations(CONFIGURED_FALLBACK)
        value = fallback_translations
        for part in key.split("."):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = None
                break

    # If still not found, return key
    if value is None:
        return key

    # Interpolate variables
    if kwargs and isinstance(value, str):
        try:
            return value.format(**kwargs)
        except KeyError:
            return value

    return value if isinstance(value, str) else key


def translate(key: str, lang: Optional[str] = None, **kwargs) -> str:
    """Translate a key to a specific language.

    Args:
        key: Translation key
        lang: Target language (default: current language)
        **kwargs: Variables to interpolate

    Returns:
        Translated string
    """
    if lang is None:
        return t(key, **kwargs)

    # Temporarily switch language
    original = get_language()
    set_language(lang)
    result = t(key, **kwargs)
    set_language(original)
    return result


def get_translations(lang: Optional[str] = None) -> Dict[str, Any]:
    """Get all translations for a language.

    Args:
        lang: Language code (default: current)

    Returns:
        Translation dictionary
    """
    return _load_translations(lang or get_language())


# Shorthand alias
_ = t
