"""Shared action parsing primitives for VLN-CE evaluators."""

import re


STOP_ACTION = 0
FORWARD_ACTION = 1
LEFT_ACTION = 2
RIGHT_ACTION = 3

_ACTION_INDEX = (
    ("stop", STOP_ACTION),
    ("forward", FORWARD_ACTION),
    ("left", LEFT_ACTION),
    ("right", RIGHT_ACTION),
)
_MAGNITUDE_PATTERN = re.compile(r"-?\d+")


def parse_navigation_action(action, *, require_magnitude=True):
    """Return ``(action_index, magnitude)`` or ``(None, None)``.

    Merged-action generation requires a distance/angle for directional
    actions. Non-merged generation only needs the action class and can set
    ``require_magnitude=False``.
    """
    if not isinstance(action, str):
        return None, None

    for token, action_index in _ACTION_INDEX:
        if token not in action:
            continue
        if action_index == STOP_ACTION:
            return STOP_ACTION, None

        match = _MAGNITUDE_PATTERN.search(action)
        if match is None:
            if require_magnitude:
                return None, None
            return action_index, None
        return action_index, float(match.group())

    return None, None
