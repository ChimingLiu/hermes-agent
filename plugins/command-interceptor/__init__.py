"""command-interceptor plugin — configurable tool-call blocking.

Wires two behaviours:

1. ``pre_tool_call`` hook — matches tool calls against user-configured
   rules and blocks them with a custom message when a rule matches.

2. ``post_tool_call`` hook — logs blocked calls to an audit file and
   optionally dispatches follow-up tools configured via ``on_block``.

Rules live in ``~/.hermes/config.yaml`` under ``command_interceptor``.
See ``interceptor.py`` for config format and matching logic.
"""

from __future__ import annotations

import logging

from . import interceptor

logger = logging.getLogger(__name__)


def register(ctx) -> None:
    interceptor.set_plugin_context(ctx)
    ctx.register_hook("pre_tool_call", interceptor.on_pre_tool_call)
    ctx.register_hook("post_tool_call", interceptor.on_post_tool_call)
