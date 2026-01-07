"""LLM-Agnostic Injection Strategies for ai-mem.

This module provides host-aware context injection strategies that work
with any LLM (Claude, Gemini, Cursor, etc.) without depending on
host-specific features.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class InjectionMethod(str, Enum):
    """Methods for injecting context into LLM prompts."""
    SYSTEM_PROMPT = "system_prompt"  # Inject into system prompt
    USER_PROMPT_PREFIX = "prompt_prefix"  # Prepend to user message
    MCP_TOOLS = "mcp_tools"  # Use MCP tools (on-demand retrieval)
    HYBRID = "hybrid"  # Combination of methods


class ContextPosition(str, Enum):
    """Where to position the injected context."""
    SYSTEM_PROMPT = "system_prompt"
    USER_PROMPT_START = "user_prompt_start"
    USER_PROMPT_END = "user_prompt_end"
    BOTH = "both"


@dataclass
class HostConfig:
    """Configuration for a specific LLM host."""
    name: str
    injection_method: InjectionMethod = InjectionMethod.USER_PROMPT_PREFIX
    supports_mcp: bool = False
    context_position: ContextPosition = ContextPosition.USER_PROMPT_START
    format: str = "markdown"  # "markdown", "xml_tags", "plain"
    max_context_tokens: Optional[int] = None
    disclosure_mode: str = "auto"  # Default disclosure mode for this host


# Default configurations for known hosts
DEFAULT_HOST_CONFIGS: Dict[str, HostConfig] = {
    "claude-code": HostConfig(
        name="claude-code",
        injection_method=InjectionMethod.MCP_TOOLS,
        supports_mcp=True,
        context_position=ContextPosition.SYSTEM_PROMPT,
        format="markdown",
        max_context_tokens=8000,
        disclosure_mode="compact",  # Prefer compact for MCP
    ),
    "claude-desktop": HostConfig(
        name="claude-desktop",
        injection_method=InjectionMethod.MCP_TOOLS,
        supports_mcp=True,
        context_position=ContextPosition.SYSTEM_PROMPT,
        format="markdown",
        max_context_tokens=8000,
        disclosure_mode="compact",
    ),
    "gemini-cli": HostConfig(
        name="gemini-cli",
        injection_method=InjectionMethod.USER_PROMPT_PREFIX,
        supports_mcp=False,
        context_position=ContextPosition.USER_PROMPT_START,
        format="markdown",
        max_context_tokens=4000,
        disclosure_mode="standard",
    ),
    "gemini-api": HostConfig(
        name="gemini-api",
        injection_method=InjectionMethod.USER_PROMPT_PREFIX,
        supports_mcp=False,
        context_position=ContextPosition.USER_PROMPT_START,
        format="markdown",
        max_context_tokens=8000,
        disclosure_mode="standard",
    ),
    "cursor": HostConfig(
        name="cursor",
        injection_method=InjectionMethod.HYBRID,
        supports_mcp=True,
        context_position=ContextPosition.BOTH,
        format="markdown",
        max_context_tokens=4000,
        disclosure_mode="compact",
    ),
    "vscode": HostConfig(
        name="vscode",
        injection_method=InjectionMethod.USER_PROMPT_PREFIX,
        supports_mcp=False,
        context_position=ContextPosition.USER_PROMPT_START,
        format="markdown",
        max_context_tokens=4000,
        disclosure_mode="standard",
    ),
    "generic": HostConfig(
        name="generic",
        injection_method=InjectionMethod.USER_PROMPT_PREFIX,
        supports_mcp=False,
        context_position=ContextPosition.USER_PROMPT_START,
        format="xml_tags",
        max_context_tokens=2000,
        disclosure_mode="compact",  # Safe default
    ),
}


# Model-specific token budgets (approximate available context)
MODEL_TOKEN_BUDGETS: Dict[str, int] = {
    # Claude models
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    "claude-3.5-sonnet": 200000,
    "claude-opus-4": 200000,
    # Gemini models
    "gemini-pro": 30000,
    "gemini-1.5-pro": 1000000,
    "gemini-1.5-flash": 1000000,
    "gemini-2.0-flash": 1000000,
    # GPT models
    "gpt-4": 8000,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    # Open source
    "llama-3": 8000,
    "llama-3.1": 128000,
    "mistral": 32000,
    "mixtral": 32000,
}


def get_host_config(host: str) -> HostConfig:
    """Get configuration for a specific host.

    Args:
        host: Host identifier (e.g., "claude-code", "gemini-cli")

    Returns:
        HostConfig for the host, or generic config if unknown
    """
    host_lower = host.lower().replace("_", "-")

    # Check for exact match
    if host_lower in DEFAULT_HOST_CONFIGS:
        return DEFAULT_HOST_CONFIGS[host_lower]

    # Check for partial matches
    for key, config in DEFAULT_HOST_CONFIGS.items():
        if key in host_lower or host_lower in key:
            return config

    # Return generic config
    return DEFAULT_HOST_CONFIGS["generic"]


def get_model_token_budget(model: str) -> int:
    """Get token budget for a specific model.

    Args:
        model: Model identifier

    Returns:
        Estimated available tokens (conservative estimate)
    """
    model_lower = model.lower()

    # Check for exact match
    if model_lower in MODEL_TOKEN_BUDGETS:
        return MODEL_TOKEN_BUDGETS[model_lower]

    # Check for partial matches
    for key, budget in MODEL_TOKEN_BUDGETS.items():
        if key in model_lower or model_lower in key:
            return budget

    # Default conservative estimate
    return 4000


def calculate_context_budget(
    model: Optional[str] = None,
    host: Optional[str] = None,
    available_tokens: Optional[int] = None,
    context_percentage: float = 0.15,  # Use max 15% for context
) -> int:
    """Calculate how many tokens to allocate for memory context.

    Args:
        model: Model identifier
        host: Host identifier
        available_tokens: Override token count
        context_percentage: Percentage of context to use for memory

    Returns:
        Number of tokens to allocate for context injection
    """
    if available_tokens is not None:
        base_budget = available_tokens
    elif model:
        base_budget = get_model_token_budget(model)
    elif host:
        config = get_host_config(host)
        base_budget = config.max_context_tokens or 4000
    else:
        base_budget = 4000  # Conservative default

    # Apply percentage and cap
    budget = int(base_budget * context_percentage)

    # Minimum budget
    return max(100, min(budget, 10000))


def determine_optimal_disclosure_mode(
    host: Optional[str] = None,
    model: Optional[str] = None,
    observation_count: int = 0,
    estimated_tokens: int = 0,
    max_budget: Optional[int] = None,
) -> str:
    """Determine the optimal disclosure mode based on host/model.

    Args:
        host: Host identifier
        model: Model identifier
        observation_count: Number of observations to include
        estimated_tokens: Estimated tokens if using full content
        max_budget: Maximum token budget

    Returns:
        Disclosure mode: "compact", "standard", or "full"
    """
    # Calculate budget
    budget = max_budget or calculate_context_budget(model=model, host=host)

    # Get host config
    host_config = get_host_config(host) if host else DEFAULT_HOST_CONFIGS["generic"]

    # If host supports MCP, prefer compact (on-demand retrieval)
    if host_config.supports_mcp:
        return "compact"

    # If estimated tokens exceed budget, use compact
    if estimated_tokens > budget:
        return "compact"

    # If under budget with room to spare, use standard
    if estimated_tokens < budget * 0.5:
        return "standard"

    # If tight on budget, use compact
    return "compact"


@dataclass
class InjectionResult:
    """Result of context injection."""
    context_text: str
    injection_method: InjectionMethod
    position: ContextPosition
    token_estimate: int
    disclosure_mode: str
    host: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def format_context_for_injection(
    context_text: str,
    host: Optional[str] = None,
    position: Optional[ContextPosition] = None,
) -> str:
    """Format context text for injection based on host preferences.

    Args:
        context_text: Raw context text
        host: Host identifier
        position: Override position

    Returns:
        Formatted context ready for injection
    """
    config = get_host_config(host) if host else DEFAULT_HOST_CONFIGS["generic"]
    pos = position or config.context_position

    # Already wrapped in ai-mem-context tags, just add position hint
    if pos == ContextPosition.USER_PROMPT_START:
        return f"[Memory Context]\n{context_text}\n\n[User Message]\n"
    elif pos == ContextPosition.USER_PROMPT_END:
        return f"\n\n[Relevant Memory]\n{context_text}"
    elif pos == ContextPosition.SYSTEM_PROMPT:
        return f"# Memory Context\n{context_text}"
    else:
        return context_text
