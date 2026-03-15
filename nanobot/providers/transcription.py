"""Transcription service — bus-consuming service using LiteLLM."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import litellm
from loguru import logger

from nanobot.bus.events import InboundMessage, TranscribeRequest
from nanobot.bus.queue import MessageBus
from nanobot.providers.registry import ProviderSpec, find_by_name

if TYPE_CHECKING:
    from nanobot.config.schema import Config


class TranscriptionService:
    """Independent bus consumer that transcribes audio and publishes results."""

    def __init__(
        self,
        *,
        bus: MessageBus,
        model: str,
        api_key: str,
        spec: ProviderSpec,
    ):
        self._bus = bus
        self._model = model
        self._api_key = api_key
        self._spec = spec

        # Setup env vars so LiteLLM can find credentials
        self._setup_env()

    def _setup_env(self) -> None:
        """Set environment variables based on provider spec."""
        if not self._spec.env_key:
            return

        os.environ.setdefault(self._spec.env_key, self._api_key)

        for env_name, env_val in self._spec.env_extras:
            resolved = env_val.replace("{api_key}", self._api_key)
            os.environ.setdefault(env_name, resolved)

    async def run(self) -> None:
        """Long-running consumer loop — reads TranscribeRequests, transcribes, publishes results."""
        logger.info("Transcription service started (model={})", self._model)
        while True:
            try:
                request = await self._bus.consume_transcription()
                await self._process(request)
            except Exception as e:
                logger.error("Transcription service error: {}", e)

    async def _process(self, request: TranscribeRequest) -> None:
        """Process a single transcription request."""
        path = Path(request.file_path)
        content: str

        if not path.exists():
            logger.error("Audio file not found: {}", request.file_path)
            content = f"[voice: {request.file_path}]"
        else:
            try:
                with open(path, "rb") as f:
                    response = await litellm.atranscription(
                        model=self._model,
                        file=(path.name, f),
                        api_key=self._api_key,
                    )
                content = f"[transcription: {response.text}]"
                logger.info("Transcribed {}: {}...", path.name, response.text[:50])
            except Exception as e:
                logger.error("Transcription failed for {}: {}", request.file_path, e)
                content = f"[voice: {request.file_path}]"

        # Publish result as InboundMessage
        msg = InboundMessage(
            channel=request.channel,
            sender_id=request.sender_id,
            chat_id=request.chat_id,
            content=content,
            media=request.media,
            metadata=request.metadata,
            session_key_override=request.session_key_override,
        )
        await self._bus.publish_inbound(msg)


def create_transcription_service(
    config: Config,
    bus: MessageBus,
) -> TranscriptionService | None:
    """Create a TranscriptionService from config. Returns None if not configured."""
    provider_name = config.transcription.provider
    model = config.transcription.model

    if not provider_name or not model:
        return None

    provider_cfg = getattr(config.providers, provider_name, None)
    if provider_cfg is None:
        logger.warning(
            "Transcription disabled: '{}' is not a known provider name",
            provider_name,
        )
        return None

    api_key = provider_cfg.api_key
    if not api_key:
        logger.warning(
            "Transcription disabled: providers.{}.api_key is not set",
            provider_name,
        )
        return None

    spec = find_by_name(provider_name)
    if not spec:
        logger.warning(
            "Transcription disabled: '{}' not found in provider registry",
            provider_name,
        )
        return None

    # Construct LiteLLM model string with provider's litellm_prefix
    if spec.litellm_prefix:
        litellm_model = f"{spec.litellm_prefix}/{model}"
    else:
        litellm_model = model

    return TranscriptionService(bus=bus, model=litellm_model, api_key=api_key, spec=spec)
