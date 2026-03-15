"""Tests for transcription service."""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.bus.events import TranscribeRequest
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import Config, ProvidersConfig, TranscriptionConfig
from nanobot.providers.transcription import TranscriptionService, create_transcription_service


def test_create_transcription_service_no_config():
    """Test that service creation returns None when not configured."""
    bus = MessageBus()
    config = Config(
        transcription=TranscriptionConfig(provider="", model=""), providers=ProvidersConfig()
    )
    service = create_transcription_service(config, bus)
    assert service is None


def test_create_transcription_service_missing_provider():
    """Test that service creation fails when provider is missing."""
    bus = MessageBus()
    config = Config(
        transcription=TranscriptionConfig(provider="", model="whisper-large-v3"),
        providers=ProvidersConfig(),
    )
    service = create_transcription_service(config, bus)
    assert service is None


def test_create_transcription_service_missing_model():
    """Test that service creation fails when model is missing."""
    bus = MessageBus()
    config = Config(
        transcription=TranscriptionConfig(provider="groq", model=""), providers=ProvidersConfig()
    )
    service = create_transcription_service(config, bus)
    assert service is None


def test_create_transcription_service_unknown_provider():
    """Test that service creation fails for unknown provider."""
    bus = MessageBus()
    config = Config(
        transcription=TranscriptionConfig(provider="unknown", model="model"),
        providers=ProvidersConfig(),
    )
    service = create_transcription_service(config, bus)
    assert service is None


def test_create_transcription_service_missing_api_key():
    """Test that service creation fails when API key is missing."""
    bus = MessageBus()
    config = Config(
        transcription=TranscriptionConfig(provider="groq", model="whisper-large-v3"),
        providers=ProvidersConfig(),
    )
    config.providers.groq = MagicMock()
    config.providers.groq.api_key = ""
    service = create_transcription_service(config, bus)
    assert service is None


def test_create_transcription_service_success():
    """Test successful service creation with correct model prefixing."""
    bus = MessageBus()
    config = Config(
        transcription=TranscriptionConfig(provider="groq", model="whisper-large-v3"),
        providers=ProvidersConfig(),
    )
    config.providers.groq = MagicMock()
    config.providers.groq.api_key = "test_key"
    service = create_transcription_service(config, bus)
    assert service is not None
    assert isinstance(service, TranscriptionService)
    assert service._api_key == "test_key"
    # Groq has litellm_prefix="groq", so model should be prefixed
    assert service._model == "groq/whisper-large-v3"


@pytest.mark.asyncio
async def test_transcription_service_process_success():
    """Test TranscriptionService processes a request and publishes InboundMessage."""
    bus = MessageBus()
    service = TranscriptionService(bus=bus, model="groq/whisper-large-v3", api_key="test_key")

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp.write(b"fake audio data")
        tmp_path = tmp.name

    try:
        with patch(
            "nanobot.providers.transcription.litellm.atranscription", new_callable=AsyncMock
        ) as mock_litellm:
            mock_response = MagicMock()
            mock_response.text = "test transcription"
            mock_litellm.return_value = mock_response

            request = TranscribeRequest(
                file_path=tmp_path,
                channel="telegram",
                sender_id="user1",
                chat_id="chat1",
                media=[tmp_path],
            )
            await service._process(request)

            mock_litellm.assert_called_once()
            call_args = mock_litellm.call_args
            assert call_args[1]["model"] == "groq/whisper-large-v3"
            assert call_args[1]["api_key"] == "test_key"

            # Check that InboundMessage was published
            msg = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)
            assert msg.content == "[transcription: test transcription]"
            assert msg.channel == "telegram"
            assert msg.sender_id == "user1"
            assert msg.chat_id == "chat1"
            assert msg.media == [tmp_path]
    finally:
        os.unlink(tmp_path)


@pytest.mark.asyncio
async def test_transcription_service_file_not_found():
    """Test TranscriptionService publishes fallback for missing file."""
    bus = MessageBus()
    service = TranscriptionService(bus=bus, model="groq/whisper-large-v3", api_key="test_key")

    request = TranscribeRequest(
        file_path="/nonexistent/file.ogg",
        channel="telegram",
        sender_id="user1",
        chat_id="chat1",
    )
    await service._process(request)

    msg = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)
    assert msg.content == "[voice: /nonexistent/file.ogg]"
    assert msg.channel == "telegram"


@pytest.mark.asyncio
async def test_transcription_service_api_error():
    """Test TranscriptionService publishes fallback on API failure."""
    bus = MessageBus()
    service = TranscriptionService(bus=bus, model="groq/whisper-large-v3", api_key="test_key")

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp.write(b"fake audio data")
        tmp_path = tmp.name

    try:
        with patch(
            "nanobot.providers.transcription.litellm.atranscription", new_callable=AsyncMock
        ) as mock_litellm:
            mock_litellm.side_effect = Exception("API error")

            request = TranscribeRequest(
                file_path=tmp_path,
                channel="telegram",
                sender_id="user1",
                chat_id="chat1",
            )
            await service._process(request)

            msg = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)
            assert msg.content == f"[voice: {tmp_path}]"
    finally:
        os.unlink(tmp_path)


@pytest.mark.asyncio
async def test_transcribe_request_flows_through_bus():
    """Test that TranscribeRequest events flow correctly through the bus."""
    bus = MessageBus()

    request = TranscribeRequest(
        file_path="/path/to/audio.ogg",
        channel="telegram",
        sender_id="user1",
        chat_id="chat1",
        media=["/path/to/audio.ogg"],
        metadata={"message_id": 123},
        session_key_override="custom:key",
    )

    await bus.publish_transcription(request)
    consumed = await asyncio.wait_for(bus.consume_transcription(), timeout=1.0)

    assert consumed.file_path == "/path/to/audio.ogg"
    assert consumed.channel == "telegram"
    assert consumed.sender_id == "user1"
    assert consumed.chat_id == "chat1"
    assert consumed.media == ["/path/to/audio.ogg"]
    assert consumed.metadata == {"message_id": 123}
    assert consumed.session_key_override == "custom:key"


def test_bus_has_transcription_default():
    """Test that bus.has_transcription defaults to False."""
    bus = MessageBus()
    assert bus.has_transcription is False


def test_channel_init_without_transcription():
    """Test channels initialize without errors when transcription is not configured."""
    from nanobot.channels.base import BaseChannel

    mock_config = MagicMock()
    mock_config.allow_from = ["*"]
    bus = MessageBus()

    class TestChannel(BaseChannel):
        name = "test"
        display_name = "Test"

        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            pass

        async def send(self, msg):
            pass

    channel = TestChannel(mock_config, bus)
    assert channel.bus is bus
    assert channel.config is mock_config
