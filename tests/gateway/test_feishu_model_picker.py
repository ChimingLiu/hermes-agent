"""Tests for Feishu interactive model-picker cards."""

import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


def _ensure_feishu_mocks():
    if importlib.util.find_spec("lark_oapi") is None and "lark_oapi" not in sys.modules:
        mod = MagicMock()
        for name in (
            "lark_oapi", "lark_oapi.api.im.v1",
            "lark_oapi.event", "lark_oapi.event.callback_type",
        ):
            sys.modules.setdefault(name, mod)
    if importlib.util.find_spec("aiohttp") is None and "aiohttp" not in sys.modules:
        aio = MagicMock()
        sys.modules.setdefault("aiohttp", aio)
        sys.modules.setdefault("aiohttp.web", aio.web)


_ensure_feishu_mocks()

from gateway.config import PlatformConfig
import gateway.platforms.feishu as feishu_module
from gateway.platforms.feishu import FeishuAdapter


class _FakeCallBackCard:
    def __init__(self):
        self.type = None
        self.data = None


class _FakeP2Response:
    def __init__(self):
        self.card = None


def _make_adapter() -> FeishuAdapter:
    config = PlatformConfig(enabled=True)
    adapter = FeishuAdapter(config)
    adapter._client = MagicMock()
    return adapter


def _make_providers(with_current: str = "openai") -> list:
    return [
        {
            "slug": "anthropic",
            "name": "Anthropic",
            "is_current": with_current == "anthropic",
            "models": ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"],
            "total_models": 3,
        },
        {
            "slug": "openai",
            "name": "OpenAI",
            "is_current": with_current == "openai",
            "models": ["gpt-4o", "gpt-4o-mini"],
            "total_models": 2,
        },
    ]


def _make_card_action_data(
    action_value: dict,
    chat_id: str = "oc_12345",
    open_id: str = "ou_user1",
    token: str = "tok_abc",
) -> SimpleNamespace:
    return SimpleNamespace(
        event=SimpleNamespace(
            token=token,
            context=SimpleNamespace(open_chat_id=chat_id),
            operator=SimpleNamespace(open_id=open_id),
            action=SimpleNamespace(tag="button", value=action_value),
        ),
    )


def _close_submitted_coro(coro, _loop):
    coro.close()
    return SimpleNamespace(add_done_callback=lambda *_a, **_k: None)


@pytest.fixture
def _patch_cb_types(monkeypatch):
    monkeypatch.setattr(feishu_module, "P2CardActionTriggerResponse", _FakeP2Response)
    monkeypatch.setattr(feishu_module, "CallBackCard", _FakeCallBackCard)


def _all_buttons(card: dict) -> list:
    return [b for e in card["elements"] if e.get("tag") == "action" for b in e["actions"]]


def _picker_kinds(card: dict) -> set:
    return {b["value"].get("hermes_picker") for b in _all_buttons(card)}


# ===========================================================================
# Card builders
# ===========================================================================

class TestBuildProviderCard:
    def test_renders_every_provider_with_sid(self):
        providers = _make_providers("openai")
        card = FeishuAdapter._build_provider_card(
            providers=providers,
            current_model="gpt-4o",
            current_provider="openai",
            sid="deadbeef1234",
        )
        assert "gpt-4o" in card["elements"][0]["content"]

        provider_btns = [b for b in _all_buttons(card) if b["value"].get("hermes_picker") == "provider"]
        assert {b["value"]["slug"] for b in provider_btns} == {p["slug"] for p in providers}
        for b in provider_btns:
            assert b["value"]["sid"] == "deadbeef1234"

        current = next(b for b in provider_btns if b["value"]["slug"] == "openai")
        assert current["type"] == "primary"
        assert current["text"]["content"].startswith("✓ ")

        kinds = _picker_kinds(card)
        assert "provider" in kinds and "cancel" in kinds and "back" not in kinds

    def test_buttons_split_into_rows(self):
        providers = [
            {"slug": f"p{i}", "name": f"P{i}", "models": ["m"], "total_models": 1}
            for i in range(5)
        ]
        card = FeishuAdapter._build_provider_card(
            providers=providers, current_model="", current_provider="", sid="sid1",
        )
        provider_rows = [
            e for e in card["elements"]
            if e.get("tag") == "action"
            and any(b["value"].get("hermes_picker") == "provider" for b in e["actions"])
        ]
        assert sum(len(r["actions"]) for r in provider_rows) == 5
        assert all(len(r["actions"]) <= 2 for r in provider_rows)


class TestBuildModelCard:
    def test_lists_all_models_with_nav(self):
        provider = {
            "slug": "anthropic",
            "name": "Anthropic",
            "models": ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"],
            "total_models": 3,
        }
        card = FeishuAdapter._build_model_card(provider=provider, sid="sid2")

        model_btns = [b for b in _all_buttons(card) if b["value"].get("hermes_picker") == "model"]
        assert {b["value"]["model"] for b in model_btns} == set(provider["models"])
        for b in model_btns:
            assert b["value"]["slug"] == "anthropic"
            assert b["value"]["sid"] == "sid2"

        kinds = _picker_kinds(card)
        assert kinds == {"model", "back", "cancel"}

    def test_truncates_when_over_page_size(self):
        provider = {
            "slug": "openrouter",
            "name": "OpenRouter",
            "models": [f"vendor/model-{i}" for i in range(20)],
            "total_models": 20,
        }
        card = FeishuAdapter._build_model_card(provider=provider, sid="sid3")
        model_btns = [b for b in _all_buttons(card) if b["value"].get("hermes_picker") == "model"]
        assert len(model_btns) == feishu_module._FEISHU_MODEL_PAGE_SIZE

        hint_md = card["elements"][0]["content"]
        assert "12" in hint_md and "more" in hint_md

        for b in model_btns:
            assert "/" not in b["text"]["content"]

    def test_empty_model_list_shows_nav_only(self):
        provider = {"slug": "empty", "name": "Empty", "models": [], "total_models": 0}
        card = FeishuAdapter._build_model_card(provider=provider, sid="sidE")
        assert _picker_kinds(card) == {"back", "cancel"}


# ===========================================================================
# send_model_picker
# ===========================================================================

class TestSendModelPicker:
    @pytest.mark.asyncio
    async def test_sends_interactive_card(self):
        adapter = _make_adapter()
        cb = AsyncMock(return_value="Model switched.")

        response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_picker_1"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=response,
        ) as mock_send:
            result = await adapter.send_model_picker(
                chat_id="oc_xyz",
                providers=_make_providers("openai"),
                current_model="gpt-4o",
                current_provider="openai",
                session_key="agent:main:feishu:group:oc_xyz",
                on_model_selected=cb,
                metadata={"thread_id": "t1"},
            )

        assert result.success is True
        assert result.message_id == "msg_picker_1"
        kwargs = mock_send.call_args[1]
        assert kwargs["msg_type"] == "interactive"
        assert kwargs["chat_id"] == "oc_xyz"

        card = json.loads(kwargs["payload"])
        for b in _all_buttons(card):
            assert "hermes_picker" in b["value"]
            assert b["value"].get("sid")

        sid = FeishuAdapter._make_picker_sid("agent:main:feishu:group:oc_xyz")
        assert sid in adapter._model_picker_state

    @pytest.mark.asyncio
    async def test_not_connected_returns_error(self):
        adapter = _make_adapter()
        adapter._client = None
        result = await adapter.send_model_picker(
            chat_id="oc_x", providers=_make_providers(), current_model="m",
            current_provider="openai", session_key="s",
            on_model_selected=AsyncMock(),
        )
        assert result.success is False


# ===========================================================================
# Card action dispatch
# ===========================================================================

def _seed_state(adapter: FeishuAdapter, sid: str, *, cb=None, providers=None) -> dict:
    state = {
        "providers": providers if providers is not None else _make_providers("openai"),
        "on_model_selected": cb or AsyncMock(return_value="ok"),
        "current_model": "gpt-4o",
        "current_provider": "openai",
        "chat_id": "oc_seed",
        "metadata": None,
        "created_at": time.time(),
    }
    adapter._model_picker_state[sid] = state
    return state


def _adapter_with_loop() -> FeishuAdapter:
    adapter = _make_adapter()
    adapter._loop = MagicMock()
    adapter._loop.is_closed = MagicMock(return_value=False)
    return adapter


class TestPickerCardActionRouting:
    def test_provider_click_returns_model_card(self, _patch_cb_types):
        adapter = _adapter_with_loop()
        sid = FeishuAdapter._make_picker_sid("sess-1")
        _seed_state(adapter, sid)

        data = _make_card_action_data({
            "hermes_picker": "provider", "slug": "anthropic", "sid": sid,
        })
        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro):
            response = adapter._on_card_action_trigger(data)

        assert response.card is not None
        card = response.card.data
        assert "Anthropic" in card["elements"][0]["content"]
        assert _picker_kinds(card) == {"model", "back", "cancel"}

    def test_back_click_returns_provider_card(self, _patch_cb_types):
        adapter = _adapter_with_loop()
        sid = FeishuAdapter._make_picker_sid("sess-1")
        _seed_state(adapter, sid)

        data = _make_card_action_data({"hermes_picker": "back", "sid": sid})
        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro):
            response = adapter._on_card_action_trigger(data)

        assert _picker_kinds(response.card.data) == {"provider", "cancel"}

    def test_cancel_click_returns_cancellation_card(self, _patch_cb_types):
        adapter = _adapter_with_loop()
        sid = FeishuAdapter._make_picker_sid("sess-1")
        _seed_state(adapter, sid)

        data = _make_card_action_data({"hermes_picker": "cancel", "sid": sid})
        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro):
            response = adapter._on_card_action_trigger(data)

        assert response.card is not None
        assert not _all_buttons(response.card.data)

    def test_model_click_invokes_callback(self, _patch_cb_types):
        adapter = _adapter_with_loop()
        sid = FeishuAdapter._make_picker_sid("sess-1")
        cb = AsyncMock(return_value="Model switched to `claude-sonnet-4-6`")
        _seed_state(adapter, sid, cb=cb)

        loop = asyncio.new_event_loop()

        def _run_submitted(coro, _loop):
            loop.run_until_complete(coro)
            return SimpleNamespace(add_done_callback=lambda *_a, **_k: None)

        data = _make_card_action_data({
            "hermes_picker": "model",
            "slug": "anthropic",
            "model": "claude-sonnet-4-6",
            "sid": sid,
        })
        try:
            with (
                patch("asyncio.run_coroutine_threadsafe", side_effect=_run_submitted),
                patch.object(
                    adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
                    return_value=SimpleNamespace(
                        success=lambda: True,
                        data=SimpleNamespace(message_id="msg_conf"),
                    ),
                ),
            ):
                response = adapter._on_card_action_trigger(data)
        finally:
            loop.close()

        assert response.card is not None
        cb.assert_awaited_once_with("oc_seed", "claude-sonnet-4-6", "anthropic")

    def test_expired_state_returns_error_card(self, _patch_cb_types):
        adapter = _adapter_with_loop()
        data = _make_card_action_data({
            "hermes_picker": "provider", "slug": "anthropic", "sid": "ghost-sid",
        })
        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro):
            response = adapter._on_card_action_trigger(data)

        assert response.card is not None
        assert not _all_buttons(response.card.data)

    def test_auto_selects_single_model_provider(self, _patch_cb_types):
        adapter = _adapter_with_loop()
        sid = FeishuAdapter._make_picker_sid("sess-1")
        cb = AsyncMock(return_value="Model switched to `only-model`")
        _seed_state(adapter, sid, cb=cb, providers=[
            {"slug": "tiny", "name": "Tiny", "models": ["only-model"], "total_models": 1},
        ])

        loop = asyncio.new_event_loop()

        def _run_submitted(coro, _loop):
            loop.run_until_complete(coro)
            return SimpleNamespace(add_done_callback=lambda *_a, **_k: None)

        data = _make_card_action_data({
            "hermes_picker": "provider", "slug": "tiny", "sid": sid,
        })
        try:
            with (
                patch("asyncio.run_coroutine_threadsafe", side_effect=_run_submitted),
                patch.object(
                    adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
                    return_value=SimpleNamespace(
                        success=lambda: True,
                        data=SimpleNamespace(message_id="msg_conf_auto"),
                    ),
                ),
            ):
                adapter._on_card_action_trigger(data)
        finally:
            loop.close()

        cb.assert_awaited_once_with("oc_seed", "only-model", "tiny")


class TestPickerStateLifecycle:
    def test_prune_drops_stale_entries(self):
        adapter = _make_adapter()
        adapter._model_picker_state["fresh"] = {"created_at": time.time()}
        adapter._model_picker_state["stale"] = {
            "created_at": time.time() - feishu_module._FEISHU_MODEL_PICKER_TTL_SECONDS - 10,
        }
        adapter._prune_picker_state()
        assert "fresh" in adapter._model_picker_state
        assert "stale" not in adapter._model_picker_state

    def test_sid_deterministic_and_short(self):
        a = FeishuAdapter._make_picker_sid("agent:main:feishu:group:oc_xxxxxxxxxxxxxxxxxxxx")
        b = FeishuAdapter._make_picker_sid("agent:main:feishu:group:oc_xxxxxxxxxxxxxxxxxxxx")
        c = FeishuAdapter._make_picker_sid("different-key")
        assert a == b
        assert a != c
        assert len(a) <= 16
