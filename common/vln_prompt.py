MEMORY_PROMPT = (
    "Global environment memory: "
    "<|memory_start|><|memory_pad|><|memory_end|>"
)


def resolve_memory_prompt(enabled: bool) -> str:
    """Return text safe to pass to the Qwen chat template."""
    if not isinstance(enabled, bool):
        raise TypeError(
            "data.vln.memory_prompt must be a boolean, "
            f"got {type(enabled).__name__}"
        )
    return MEMORY_PROMPT if enabled else ""
