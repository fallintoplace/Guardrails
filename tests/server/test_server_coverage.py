# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import importlib
import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemoguardrails.exceptions import StreamingNotSupportedError
from nemoguardrails.server import api
from nemoguardrails.server.datastore import redis_store
from nemoguardrails.server.datastore.datastore import DataStore


@pytest.fixture
def reset_api_state():
    old_challenges = list(api.challenges)
    old_instances = dict(api.llm_rails_instances)
    old_history = dict(api.llm_rails_events_history_cache)
    old_datastore = api.datastore
    old_loggers = list(api.registered_loggers)
    old_attrs = {
        "rails_config_path": api.app.rails_config_path,
        "single_config_mode": api.app.single_config_mode,
        "single_config_id": api.app.single_config_id,
        "default_config_id": api.app.default_config_id,
        "auto_reload": api.app.auto_reload,
        "stop_signal": api.app.stop_signal,
    }
    yield
    api.challenges[:] = old_challenges
    api.llm_rails_instances.clear()
    api.llm_rails_instances.update(old_instances)
    api.llm_rails_events_history_cache.clear()
    api.llm_rails_events_history_cache.update(old_history)
    api.datastore = old_datastore
    api.registered_loggers[:] = old_loggers
    for name, value in old_attrs.items():
        setattr(api.app, name, value)


@pytest.mark.asyncio
async def test_lifespan_loads_challenges_and_single_config(tmp_path, reset_api_state):
    app = api.GuardrailsApp()
    app.rails_config_path = str(tmp_path)
    (tmp_path / "config.yml").write_text("models: []\n", encoding="utf-8")
    (tmp_path / "challenges.json").write_text('[{"name": "c", "content": "prompt"}]', encoding="utf-8")

    with patch("nemoguardrails.telemetry.set_deployment_type") as mock_set_deployment_type:
        async with api.lifespan(app):
            assert app.single_config_mode is True
            assert app.single_config_id == tmp_path.name
            assert api.challenges[-1] == {"name": "c", "content": "prompt"}

    mock_set_deployment_type.assert_called_once()


@pytest.mark.asyncio
async def test_lifespan_loads_config_py_init(tmp_path, reset_api_state):
    app = api.GuardrailsApp()
    app.rails_config_path = str(tmp_path)
    (tmp_path / "config.py").write_text("def init(app):\n    app.loaded_from_config_py = True\n", encoding="utf-8")

    async with api.lifespan(app):
        assert app.loaded_from_config_py is True


@pytest.mark.asyncio
async def test_lifespan_auto_reload_sets_and_cancels_task(tmp_path, reset_api_state):
    app = api.GuardrailsApp()
    app.rails_config_path = str(tmp_path)
    app.auto_reload = True
    task = MagicMock()
    loop = MagicMock()
    loop.run_in_executor.return_value = task

    with patch("asyncio.get_running_loop", return_value=loop):
        async with api.lifespan(app):
            assert app.loop == loop
            assert app.task == task

    assert app.stop_signal is True
    task.cancel.assert_called_once()


@pytest.mark.asyncio
async def test_get_rails_cache_and_invalid_single_config(reset_api_state):
    cached = SimpleNamespace(events_history_cache={})
    api.llm_rails_instances["cfg"] = cached
    assert await api._get_rails(["cfg"]) is cached

    api.app.single_config_mode = True
    api.app.single_config_id = "only"
    with pytest.raises(ValueError, match="Invalid configuration ids"):
        await api._get_rails(["other"])


@pytest.mark.asyncio
async def test_get_rails_rejects_bad_config_ids(tmp_path, reset_api_state):
    api.app.rails_config_path = str(tmp_path)
    for config_id in ["../outside", "nested/config"]:
        with pytest.raises(ValueError):
            await api._get_rails([config_id])


@pytest.mark.asyncio
async def test_get_rails_loads_config_updates_model_and_restores_history(tmp_path, reset_api_state):
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    api.app.rails_config_path = str(tmp_path)
    api.llm_rails_events_history_cache["cfg:override-model"] = {"events": [1]}
    config = MagicMock()
    config.models = []
    config.model_copy.side_effect = lambda update: SimpleNamespace(models=update["models"])
    rails = SimpleNamespace(events_history_cache={})

    with (
        patch.object(api.RailsConfig, "from_path", return_value=config) as mock_from_path,
        patch.object(api, "LLMRails", return_value=rails) as mock_llm_rails,
        patch.dict("os.environ", {"MAIN_MODEL_ENGINE": "openai", "MAIN_MODEL_BASE_URL": "http://model"}),
    ):
        result = await api._get_rails(["cfg"], model_name="override-model")

    assert result is rails
    expected_path = os.path.normpath(os.path.join(os.path.abspath(str(tmp_path)), "cfg"))
    mock_from_path.assert_called_once_with(expected_path)
    mock_llm_rails.assert_called_once()
    assert rails.events_history_cache == {"events": [1]}
    assert "cfg:override-model" in api.llm_rails_instances


@pytest.mark.asyncio
async def test_format_streaming_response_yields_error_and_done():
    async def stream():
        yield '{"error": {"message": "bad"}}'
        yield "ignored"

    chunks = [chunk async for chunk in api._format_streaming_response(stream(), model_name="model")]

    assert '"message": "bad"' in chunks[0]
    assert chunks[-1] == "data: [DONE]\n\n"


def test_process_chunk_handles_unexpected_validation_error(monkeypatch):
    def boom(value):
        raise RuntimeError("bad validator")

    monkeypatch.setattr(api.ChunkError, "model_validate_json", boom)

    assert api.process_chunk("plain") == "plain"


def test_registration_helpers(reset_api_state):
    api.register_challenges([{"name": "one"}])
    assert asyncio.run(api.get_challenges()) == [{"name": "one"}]

    store = object()
    logger = object()
    api.register_datastore(store)
    api.register_logger(logger)
    api.set_default_config_id("default")

    assert api.datastore is store
    assert logger in api.registered_loggers
    assert api.app.default_config_id == "default"
    assert isinstance(api.GuardrailsConfigurationError(), Exception)


def test_start_auto_reload_monitoring_clears_changed_config_cache(tmp_path, monkeypatch, reset_api_state):
    watchdog = types.ModuleType("watchdog")
    events = types.ModuleType("watchdog.events")
    observers = types.ModuleType("watchdog.observers")

    class FileSystemEventHandler:
        pass

    class Observer:
        def schedule(self, handler, path, recursive):
            self.handler = handler
            assert path == api.app.rails_config_path
            assert recursive is True

        def start(self):
            self.handler.on_any_event(SimpleNamespace(is_directory=True, event_type="modified", src_path="ignored"))
            self.handler.on_any_event(
                SimpleNamespace(is_directory=False, event_type="modified", src_path=str(tmp_path / "cfg" / ".hidden"))
            )
            self.handler.on_any_event(
                SimpleNamespace(
                    is_directory=False, event_type="modified", src_path=str(tmp_path / "cfg" / "config.yml")
                )
            )

        def stop(self):
            self.stopped = True

        def join(self):
            self.joined = True

    events.FileSystemEventHandler = FileSystemEventHandler
    observers.Observer = Observer
    monkeypatch.setitem(sys.modules, "watchdog", watchdog)
    monkeypatch.setitem(sys.modules, "watchdog.events", events)
    monkeypatch.setitem(sys.modules, "watchdog.observers", observers)
    (tmp_path / "cfg").mkdir()
    (tmp_path / "cfg" / "config.yml").write_text("models: []\n", encoding="utf-8")
    api.app.rails_config_path = str(tmp_path)
    api.app.stop_signal = False
    api.llm_rails_instances["cfg"] = SimpleNamespace(events_history_cache={"cached": True})

    def stop_loop(seconds):
        api.app.stop_signal = True

    monkeypatch.setattr(api.time, "sleep", stop_loop)

    api.start_auto_reload_monitoring()

    assert "cfg" not in api.llm_rails_instances
    assert api.llm_rails_events_history_cache["cfg"] == {"cached": True}


def test_start_auto_reload_monitoring_import_error(monkeypatch):
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("watchdog"):
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    monkeypatch.setattr(api.os, "_exit", MagicMock(side_effect=SystemExit(-1)))

    with pytest.raises(SystemExit):
        api.start_auto_reload_monitoring()


class FakeSession:
    def __init__(self):
        self.values = {}

    def set(self, key, value):
        self.values[key] = value

    def get(self, key):
        return self.values.get(key)


class FakeMessage:
    sent = []
    streamed = []
    updated = []

    def __init__(self, content=""):
        self.content = content

    async def send(self):
        self.sent.append(self)
        return self

    async def stream_token(self, token):
        self.streamed.append(token)

    async def update(self):
        self.updated.append(self.content)


class FakeChatSettings:
    def __init__(self, widgets):
        self.widgets = widgets

    async def send(self):
        return {"config_id": "cfg2"}


class FakeSelect:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeStarter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def server_app_module(monkeypatch):
    import nemoguardrails.server.app as server_app

    server_app = importlib.reload(server_app)
    FakeMessage.sent = []
    FakeMessage.streamed = []
    FakeMessage.updated = []
    session = FakeSession()
    fake_cl = SimpleNamespace(
        User=object,
        Starter=FakeStarter,
        Message=FakeMessage,
        ChatSettings=FakeChatSettings,
        input_widget=SimpleNamespace(Select=FakeSelect),
        user_session=session,
    )
    monkeypatch.setattr(server_app, "cl", fake_cl)
    old_challenges = list(server_app.challenges)
    old_path = server_app.app.rails_config_path
    old_single = server_app.app.single_config_mode
    old_single_id = server_app.app.single_config_id
    old_default = server_app.app.default_config_id
    yield server_app, session
    server_app.challenges[:] = old_challenges
    server_app.app.rails_config_path = old_path
    server_app.app.single_config_mode = old_single
    server_app.app.single_config_id = old_single_id
    server_app.app.default_config_id = old_default


@pytest.mark.asyncio
async def test_chat_app_starters_and_config_discovery(tmp_path, server_app_module):
    server_app, _ = server_app_module
    server_app.challenges[:] = []
    assert await server_app.set_starters() == []

    server_app.challenges[:] = [{"name": "n", "content": "content", "icon": "i"}]
    starters = await server_app.set_starters()
    assert starters[0].kwargs == {"label": "n", "message": "content", "icon": "i"}

    server_app.app.single_config_mode = True
    server_app.app.single_config_id = "single"
    assert server_app._discover_configs() == ["single"]

    server_app.app.single_config_mode = False
    server_app.app.rails_config_path = str(tmp_path)
    (tmp_path / "cfg").mkdir()
    (tmp_path / "cfg" / "config.yml").write_text("models: []\n", encoding="utf-8")
    (tmp_path / ".hidden").mkdir()
    assert server_app._discover_configs() == ["cfg"]

    server_app.app.rails_config_path = str(tmp_path / "missing")
    assert server_app._discover_configs() == []


@pytest.mark.asyncio
async def test_chat_app_start_settings_and_no_config(server_app_module):
    server_app, session = server_app_module
    with patch.object(server_app, "_discover_configs", return_value=[]):
        await server_app.on_chat_start()
    assert session.values["config_id"] is None
    assert "No guardrails" in FakeMessage.sent[-1].content

    with patch.object(server_app, "_discover_configs", return_value=["cfg1", "cfg2"]):
        server_app.app.default_config_id = "cfg1"
        await server_app.on_chat_start()
    assert session.values["config_id"] == "cfg2"

    await server_app.on_settings_update({"config_id": "cfg1"})
    assert session.values["config_id"] == "cfg1"
    assert session.values["messages"] == []


@pytest.mark.asyncio
async def test_chat_app_on_message_paths(server_app_module):
    server_app, session = server_app_module
    await server_app.on_message(SimpleNamespace(content="hello"))
    assert "No guardrails" in FakeMessage.sent[-1].content

    session.set("config_id", "cfg")
    session.set("messages", [])
    with patch.object(server_app, "_get_rails", AsyncMock(side_effect=RuntimeError("bad"))):
        await server_app.on_message(SimpleNamespace(content="hello"))
    assert session.get("messages") == []
    assert "Error loading" in FakeMessage.sent[-1].content

    async def stream_success(messages):
        yield "hi"
        yield " there"

    rails = SimpleNamespace(stream_async=stream_success)
    with patch.object(server_app, "_get_rails", AsyncMock(return_value=rails)):
        await server_app.on_message(SimpleNamespace(content="hello"))
    assert FakeMessage.streamed[-2:] == ["hi", " there"]
    assert session.get("messages")[-1] == {"role": "assistant", "content": "hi there"}

    async def stream_unsupported(messages):
        raise StreamingNotSupportedError("no stream")
        yield "never"

    rails = SimpleNamespace(
        stream_async=stream_unsupported, generate_async=AsyncMock(return_value={"content": "fallback"})
    )
    with patch.object(server_app, "_get_rails", AsyncMock(return_value=rails)):
        await server_app.on_message(SimpleNamespace(content="again"))
    assert FakeMessage.updated[-1] == "fallback"

    async def stream_error(messages):
        raise RuntimeError("boom")
        yield "never"

    rails = SimpleNamespace(stream_async=stream_error)
    with patch.object(server_app, "_get_rails", AsyncMock(return_value=rails)):
        await server_app.on_message(SimpleNamespace(content="bad"))
    assert "An error occurred" in FakeMessage.updated[-1]


@pytest.mark.asyncio
async def test_datastore_base_methods_raise():
    store = DataStore()
    with pytest.raises(NotImplementedError):
        await store.set("key", "value")
    with pytest.raises(NotImplementedError):
        await store.get("key")


@pytest.mark.asyncio
async def test_redis_store_import_error_and_client_calls(monkeypatch):
    monkeypatch.setattr(redis_store, "aioredis", None)
    with pytest.raises(ImportError, match="aioredis is required"):
        redis_store.RedisStore("redis://localhost")

    client = AsyncMock()
    fake_aioredis = SimpleNamespace(from_url=MagicMock(return_value=client))
    monkeypatch.setattr(redis_store, "aioredis", fake_aioredis)
    store = redis_store.RedisStore("redis://localhost", username="user", password="pass")
    await store.set("key", "value")
    client.get.return_value = "value"
    assert await store.get("key") == "value"
    fake_aioredis.from_url.assert_called_once_with(
        url="redis://localhost", username="user", password="pass", decode_responses=True
    )
    client.set.assert_awaited_once_with("key", "value")
    client.get.assert_awaited_once_with("key")
