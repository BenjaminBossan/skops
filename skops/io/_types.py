from dataclasses import dataclass
from typing import Any


@dataclass
class State:
    """Base type for all States returned by get_state functions

    The 'content' field should be overwritten to match the specific class.

    """

    cls: str
    module: str
    content: Any
