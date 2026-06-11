"""Core logic for the command-interceptor plugin.

Rule matching, hook callbacks, config loading, and audit logging.

Config format (in ``~/.hermes/config.yaml``)::

    command_interceptor:
      enabled: true                       # defaults to true
      log_file: "~/.hermes/command-interceptor.log"
      rules:
        - name: "block-dangerous-rm"
          tool: "terminal"
          match:
            field: "command"
            pattern: "rm\\s+-rf\\s+/"
            mode: "regex"                 # regex | substring | exact
          action: block                   # block (only action for now)
          message: "Blocked: `{command}`. Use a safer alternative."
          redirect: "Use `rm -i` instead."

        - name: "block-env-write"
          tool: "write_file"
          match:
            field: "path"
            pattern: "\\.env$"
            mode: "regex"
          action: block
          message: "Cannot write to .env files directly."

Rule fields:
  - name:        unique rule identifier (used in logs)
  - tool:        tool name to match against (exact match)
  - match.field: which arg field to inspect (e.g. "command", "path")
  - match.pattern: string or regex to match
  - match.mode:  "regex" | "substring" | "exact" (default: "substring")
  - action:      "block" (only action for now)
  - message:     message returned to the LLM; supports ``{field_name}`` templates
  - redirect:    optional hint shown after the message (what to do instead)
  - on_block:    optional list of {tool, args} dicts to dispatch on block
                 (side-effect only; results are not fed back to the LLM)
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Held reference to PluginContext, set in register().
_ctx = None
_ctx_lock = threading.Lock()

# Cache for loaded rules — keyed by config file mtime + size.
_rules_cache: Dict[str, List[Dict[str, Any]]] = {}
_rules_lock = threading.Lock()

# Track the matched rule name per tool_call_id so post_tool_call can use it.
_pending_blocks: Dict[str, Dict[str, Any]] = {}
_pending_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config():
    """Load config.yaml, returning the raw dict (or {} on failure)."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        return cfg if isinstance(cfg, dict) else {}
    except Exception as exc:
        logger.debug("command-interceptor: could not load config: %s", exc)
        return {}


def _cfg_get(d, *keys, default=None):
    """Safe nested dict traversal, mirrors hermes_cli.config.cfg_get."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key)
        if d is None:
            return default
    return d


def _config_mtime_key() -> str:
    """Return a cache key based on config file mtime + size."""
    config_path = Path.home() / ".hermes" / "config.yaml"
    try:
        stat = config_path.stat()
        return f"{config_path}:{stat.st_mtime}:{stat.st_size}"
    except OSError:
        return str(config_path)


def _load_rules() -> List[Dict[str, Any]]:
    """Load and validate rules from config, with caching."""
    cache_key = _config_mtime_key()
    with _rules_lock:
        cached = _rules_cache.get(cache_key)
        if cached is not None:
            return cached

    cfg = _load_config()
    section = _cfg_get(cfg, "command_interceptor")
    if not isinstance(section, dict):
        rules = []
    else:
        raw_rules = section.get("rules")
        rules = raw_rules if isinstance(raw_rules, list) else []

    validated = _validate_rules(rules)
    with _rules_lock:
        _rules_cache[cache_key] = validated
    return validated


def _validate_rules(raw_rules: List[Any]) -> List[Dict[str, Any]]:
    """Validate and normalise rules, compiling regexes upfront."""
    valid: List[Dict[str, Any]] = []
    seen_names: set = set()
    for i, r in enumerate(raw_rules):
        if not isinstance(r, dict):
            continue

        name = r.get("name", f"rule-{i}")
        if name in seen_names:
            logger.warning(
                "command-interceptor: duplicate rule name %r, skipping", name
            )
            continue
        seen_names.add(name)

        tool = r.get("tool")
        if not isinstance(tool, str) or not tool:
            logger.warning(
                "command-interceptor: rule %r missing 'tool', skipping", name
            )
            continue

        match_block = r.get("match")
        if not isinstance(match_block, dict):
            logger.warning(
                "command-interceptor: rule %r missing 'match' block, skipping", name
            )
            continue

        field = match_block.get("field")
        pattern = match_block.get("pattern")
        if not isinstance(field, str) or not isinstance(pattern, str):
            logger.warning(
                "command-interceptor: rule %r has invalid match.field or match.pattern",
                name,
            )
            continue

        mode = match_block.get("mode", "substring")
        if mode not in ("regex", "substring", "exact"):
            logger.warning(
                "command-interceptor: rule %r unknown mode %r, defaulting to substring",
                name, mode,
            )
            mode = "substring"

        compiled_re = None
        if mode == "regex":
            try:
                compiled_re = re.compile(pattern)
            except re.error as exc:
                logger.warning(
                    "command-interceptor: rule %r invalid regex %r: %s",
                    name, pattern, exc,
                )
                continue

        action = r.get("action", "block")
        if action not in ("block",):
            logger.warning(
                "command-interceptor: rule %r unknown action %r, skipping",
                name, action,
            )
            continue

        message = r.get("message", f"Command blocked by rule: {name}")

        normalised: Dict[str, Any] = {
            "name": name,
            "tool": tool,
            "match_field": field,
            "match_pattern": pattern,
            "match_mode": mode,
            "compiled_re": compiled_re,
            "action": action,
            "message": message,
            "redirect": r.get("redirect", ""),
            "on_block": r.get("on_block", []),
        }
        valid.append(normalised)

    return valid


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _match_rule(rule: Dict[str, Any], tool_name: str, args: Dict[str, Any]) -> Optional[str]:
    """Try to match a rule against a tool call.

    Returns the formatted block message on match, or None.
    """
    if tool_name != rule["tool"]:
        return None

    field_value = args.get(rule["match_field"])
    if not isinstance(field_value, str) or not field_value:
        return None

    mode = rule["match_mode"]
    pattern = rule["match_pattern"]
    matched = False

    if mode == "regex":
        matched = bool(rule["compiled_re"].search(field_value))
    elif mode == "substring":
        matched = pattern in field_value
    elif mode == "exact":
        matched = pattern == field_value

    if not matched:
        return None

    # Format message with template variables from args.
    msg = rule["message"]
    try:
        msg = msg.format(**{k: v for k, v in args.items() if isinstance(v, (str, int, float, bool))})
    except (KeyError, ValueError):
        pass

    # Append redirect hint.
    redirect = rule.get("redirect", "")
    if isinstance(redirect, str) and redirect:
        msg += f"\n\n💡 {redirect}"

    return msg


def _find_match(tool_name: str, args: Dict[str, Any]) -> Optional[tuple]:
    """Find the first matching rule. Returns (rule, formatted_message) or None."""
    if not isinstance(args, dict):
        return None

    rules = _load_rules()
    for rule in rules:
        msg = _match_rule(rule, tool_name, args)
        if msg is not None:
            return (rule, msg)
    return None


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------

def _get_log_path() -> Optional[Path]:
    """Get the configured log file path, expanded."""
    cfg = _load_config()
    section = _cfg_get(cfg, "command_interceptor")
    if not isinstance(section, dict):
        return None
    log_file = section.get("log_file")
    if isinstance(log_file, str) and log_file.strip():
        return Path(log_file.strip()).expanduser()
    return None


def _write_audit_log(
    event: str,
    rule_name: str,
    tool_name: str,
    args: Dict[str, Any],
    session_id: str = "",
    tool_call_id: str = "",
) -> None:
    """Write a JSON-lines audit log entry."""
    log_path = _get_log_path()
    if log_path is None:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "event": event,
            "rule": rule_name,
            "tool": tool_name,
            "args": {k: v for k, v in args.items() if isinstance(v, (str, int, float, bool, type(None)))},
            "session_id": session_id,
            "tool_call_id": tool_call_id,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.debug("command-interceptor: audit log write failed: %s", exc)


# ---------------------------------------------------------------------------
# Plugin context
# ---------------------------------------------------------------------------

def set_plugin_context(ctx) -> None:
    global _ctx
    with _ctx_lock:
        _ctx = ctx


def _dispatch_on_block(on_block: List[Any]) -> None:
    """Dispatch follow-up tools configured in a rule's on_block list.

    These are fire-and-forget side effects — results are not fed back to
    the LLM. Each entry should be ``{tool: "...", args: {...}}``.
    """
    global _ctx
    with _ctx_lock:
        plugin_ctx = _ctx
    if plugin_ctx is None:
        return
    if not isinstance(on_block, list):
        return
    for entry in on_block:
        if not isinstance(entry, dict):
            continue
        tool_name = entry.get("tool")
        tool_args = entry.get("args")
        if not isinstance(tool_name, str) or not isinstance(tool_args, dict):
            continue
        try:
            plugin_ctx.dispatch_tool(tool_name, tool_args)
        except Exception as exc:
            logger.warning(
                "command-interceptor: on_block dispatch %r failed: %s",
                tool_name, exc,
            )


# ---------------------------------------------------------------------------
# Hook callbacks
# ---------------------------------------------------------------------------

def on_pre_tool_call(
    tool_name: str = "",
    args: Any = None,
    session_id: str = "",
    tool_call_id: str = "",
    **_: Any,
) -> Optional[Dict[str, str]]:
    """Pre-tool-call hook: block matched calls."""
    result = _find_match(tool_name, args if isinstance(args, dict) else {})
    if result is None:
        return None

    rule, message = result

    # Store for post_tool_call so it can log + dispatch on_block.
    if tool_call_id:
        with _pending_lock:
            _pending_blocks[tool_call_id] = {
                "rule_name": rule["name"],
                "tool_name": tool_name,
                "args": args if isinstance(args, dict) else {},
                "session_id": session_id,
            }

    _write_audit_log(
        "blocked", rule["name"], tool_name,
        args if isinstance(args, dict) else {},
        session_id=session_id,
        tool_call_id=tool_call_id,
    )

    return {"action": "block", "message": message}


def on_post_tool_call(
    tool_name: str = "",
    args: Any = None,
    session_id: str = "",
    tool_call_id: str = "",
    status: str = "",
    error_type: str = "",
    **_: Any,
) -> None:
    """Post-tool-call hook: dispatch on_block actions and log blocked calls."""
    if status != "blocked" or not tool_call_id:
        # Also clean up any stale pending entry for this tool_call_id.
        with _pending_lock:
            _pending_blocks.pop(tool_call_id, None)
        return

    with _pending_lock:
        pending = _pending_blocks.pop(tool_call_id, None)

    if pending is None:
        return

    # Dispatch on_block follow-up tools.
    rule_name = pending["rule_name"]
    rules = _load_rules()
    for rule in rules:
        if rule["name"] == rule_name:
            on_block = rule.get("on_block", [])
            if on_block:
                _dispatch_on_block(on_block)
            break
