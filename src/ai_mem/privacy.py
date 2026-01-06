import re
from typing import Tuple

MAX_TAG_COUNT = 100
PRIVATE_TAG = "private"
CONTEXT_TAGS = ("ai-mem-context", "claude-mem-context")


def _count_tags(content: str) -> int:
    lower_content = content.lower()
    return sum(lower_content.count(f"<{tag}>") for tag in (PRIVATE_TAG, *CONTEXT_TAGS))


def strip_memory_tags(content: str) -> Tuple[str, bool]:
    if not content:
        return content, False

    text = str(content)
    tag_count = _count_tags(text)
    stripped = False
    if tag_count > MAX_TAG_COUNT:
        raise ValueError(f"Too many tags ({tag_count}) found in content. Max is {MAX_TAG_COUNT}.")

    for tag in CONTEXT_TAGS + (PRIVATE_TAG,):
        pattern = re.compile(rf"<{tag}>[\s\S]*?</{tag}>", re.IGNORECASE)
        if pattern.search(text):
            stripped = True
        text = pattern.sub("", text)

    return text.strip(), stripped
