# Work Modes

Work Modes are predefined configurations that optimize ai-mem behavior for different workflows.

## Overview

Instead of manually tuning settings for each task, Work Modes provide instant configuration presets:

- **Code Mode**: Software development and coding
- **Email Mode**: Email investigation and analysis
- **Research Mode**: Deep exploration and research
- **Debug Mode**: Debugging and troubleshooting
- **Chill Mode**: Casual conversation

## Available Modes

### Code Mode

Optimized for software development tasks.

| Setting | Value |
|---------|-------|
| Context observations | 12 |
| Full observations | 4 |
| Priority types | bugfix, feature, decision, refactor, tool_output |
| Priority concepts | gotcha, pattern, trade-off, how-it-works |
| Auto-tags | code |
| Endless Mode | Enabled |
| Compression ratio | 4.0 |

```bash
ai-mem mode code
# or
export AI_MEM_MODE=code
```

### Email Investigation Mode

For analyzing communications and email threads.

| Setting | Value |
|---------|-------|
| Context observations | 20 |
| Full observations | 8 |
| Priority types | note, discovery, decision, interaction |
| Priority concepts | problem-solution, why-it-exists |
| Auto-tags | investigation, email |
| Endless Mode | Enabled |
| Compression ratio | 2.0 |

```bash
ai-mem mode email
```

### Research Mode

For deep research and exploration tasks.

| Setting | Value |
|---------|-------|
| Context observations | 25 |
| Full observations | 10 |
| Priority types | discovery, note, decision, summary |
| Priority concepts | how-it-works, why-it-exists, architecture |
| Auto-tags | research |
| Endless Mode | Enabled |
| Compression ratio | 3.0 |
| Disclosure mode | Full |

```bash
ai-mem mode research
```

### Debug Mode

For debugging and troubleshooting.

| Setting | Value |
|---------|-------|
| Context observations | 15 |
| Full observations | 6 |
| Priority types | bugfix, tool_output, discovery, note |
| Priority concepts | gotcha, problem-solution, workaround |
| Auto-tags | debug |
| Endless Mode | Enabled |
| Compression ratio | 2.0 |

```bash
ai-mem mode debug
```

### Chill Mode

Casual conversation with minimal memory overhead.

| Setting | Value |
|---------|-------|
| Context observations | 5 |
| Full observations | 2 |
| Priority types | note, preference |
| Priority concepts | (none) |
| Auto-tags | casual |
| Endless Mode | Disabled |
| Compression ratio | 8.0 |
| Disclosure mode | Compact |

```bash
ai-mem mode chill
```

## CLI Commands

```bash
# List all available modes
ai-mem mode --list

# Activate a mode
ai-mem mode code
ai-mem mode research

# Show current mode
ai-mem mode --show

# Clear mode (back to defaults)
ai-mem mode --clear
```

## Environment Variable

Set mode via environment:

```bash
export AI_MEM_MODE=code
```

This applies the mode to all ai-mem commands in that shell session.

## Programmatic Usage

```python
from ai_mem.modes import (
    ModeManager,
    get_mode_config,
    list_modes,
    apply_mode,
)

# Use ModeManager
manager = ModeManager()
manager.set_mode("code")

# Get mode config
config = get_mode_config("research")
print(f"Context: {config.context_total} observations")

# Apply to AppConfig
from ai_mem.config import load_config
app_config = load_config()
app_config = apply_mode(app_config, "debug")

# Get auto-tags
tags = manager.get_auto_tags()  # ["code"]

# Get priority types
types = manager.get_priority_types()  # ["bugfix", "feature", ...]
```

## How Modes Affect Behavior

### Context Generation

Modes adjust:
- How many observations appear in context
- Which observation types are prioritized
- Which concepts are emphasized
- Disclosure level (compact/standard/full)

### Search Weighting

Modes adjust search weights:
- FTS (full-text search) weight
- Vector similarity weight
- Recency weight

### Endless Mode

Each mode controls:
- Whether Endless Mode is active
- Compression ratio for that workflow

## Creating Custom Modes

Modes are defined in `modes.py`. To add a custom mode:

```python
MODES["custom"] = ModeConfig(
    name="Custom Mode",
    description="My custom workflow",
    context_total=15,
    context_full=5,
    priority_types=["note", "decision"],
    priority_concepts=["pattern"],
    auto_tags=["custom"],
    endless_enabled=True,
    compression_ratio=3.0,
    fts_weight=0.4,
    vector_weight=0.4,
    recency_weight=0.2,
    disclosure_mode="standard",
    context_description="Custom session",
)
```

## Mode Context Header

When a mode is active, context includes a header:

```
[Mode: Code Mode] Development session - prioritizing code decisions and patterns
```

This helps the LLM understand the current workflow context.

## Best Practices

1. **Start with a mode** before beginning work
2. **Use Research mode** for open-ended exploration
3. **Use Debug mode** when troubleshooting issues
4. **Use Chill mode** for casual conversations to save tokens
5. **Clear mode** when switching tasks to avoid confusion
