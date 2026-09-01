MEMORY_PROMPT = (
    "Global environment memory: "
    "<|memory_start|><|memory_pad|><|memory_end|>"
)
NAVIGATION_PROMPT = (
    "You are a navigation robot to perform the instruction: {instruction} "
    "{act_hist_prompt}What is your next action?"
)
TRAJECTORY_SUMMARY_PROMPT = (
    "You are a navigation robot. Given the environment memory, infer and "
    "summarize the instruction."
)
HISTORICAL_OBSERVATION_LABEL = "Historical observation: "
CURRENT_OBSERVATION_LABEL = "Current observation: "


def resolve_memory_prompt(enabled: bool) -> str:
    """Return text safe to pass to the Qwen chat template."""
    if not isinstance(enabled, bool):
        raise TypeError(
            "data.vln.memory_prompt must be a boolean, "
            f"got {type(enabled).__name__}"
        )
    return MEMORY_PROMPT if enabled else ""


def _build_visual_content(observations):
    observations = list(observations)
    if not observations:
        raise ValueError("VLN prompt construction requires at least one observation")

    content = []
    if len(observations) > 1:
        content.extend(
            [
                {"type": "text", "text": HISTORICAL_OBSERVATION_LABEL},
                {"type": "video", "video": observations[:-1]},
            ]
        )
    content.extend(
        [
            {"type": "text", "text": CURRENT_OBSERVATION_LABEL},
            {"type": "image", "image": observations[-1]},
        ]
    )
    return content


def build_navigation_user_message(
    memory_prompt,
    observations,
    instruction,
    act_hist_prompt="",
):
    """Build the canonical MemVLN navigation message and its text query."""
    input_txt = NAVIGATION_PROMPT.format(
        instruction=instruction,
        act_hist_prompt=act_hist_prompt,
    )
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": memory_prompt},
            *_build_visual_content(observations),
            {"type": "text", "text": input_txt},
        ],
    }
    return message, input_txt


def build_trajectory_summary_user_message(memory_prompt, observations):
    """Build the canonical MemVLN trajectory-summary message and text query."""
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": memory_prompt},
            *_build_visual_content(observations),
            {"type": "text", "text": TRAJECTORY_SUMMARY_PROMPT},
        ],
    }
    return message, TRAJECTORY_SUMMARY_PROMPT
