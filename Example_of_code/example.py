"""
FastAPI application for LLM service with OpenAI-compatible Chat Completions API.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import sys
import time
import uuid
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from sse_starlette.sse import EventSourceResponse

from app.config import get_settings
from app.model import LLMEngine
from app.schemas import (
    ChatChoice,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    StreamChoice,
    StreamChunk,
    StreamDelta,
    UsageInfo,
)
from app.security import basic_auth_configured, verify_basic_auth

settings = get_settings()

logger.remove()
logger.add(
    sys.stderr,
    level=settings.log_level.upper(),
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
)

llm_engine: LLMEngine | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan: load the model on startup and release resources on shutdown.

    Raises:
        RuntimeError: If HTTP Basic auth credentials are not configured in the environment.
    """
    global llm_engine

    logger.info("Starting LLM service...")
    if not basic_auth_configured():
        raise RuntimeError(
            "/chat requires HTTP Basic auth. "
            "Set LLM_BASIC_USER and LLM_BASIC_PASSWORD in settings/tokens.env."
        )
    logger.info("HTTP Basic auth enabled for /chat")
    llm_engine = LLMEngine(settings)
    await llm_engine.initialize()
    logger.info("LLM service ready")

    yield

    logger.info("Shutting down LLM service...")
    if llm_engine is not None:
        await llm_engine.shutdown()
    logger.info("LLM service stopped")


app = FastAPI(
    title="LLM Service",
    description="Ollama-backed microservice for text generation with OpenAI-compatible /chat API",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        Health status with model information
    """
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    return HealthResponse(
        status="healthy",
        model=settings.model_name,
        ready=llm_engine.is_ready(),
    )


@app.post(
    "/chat",
    response_model=None,
    dependencies=[Depends(verify_basic_auth)],
)
async def chat_completion(request: ChatRequest) -> ChatResponse | StreamingResponse:
    """
    OpenAI-compatible chat completion endpoint.

    Args:
        request: Chat completion request

    Returns:
        Chat completion response or streaming response

    Raises:
        HTTPException: If generation fails
    """
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        if request.stream:
            return EventSourceResponse(
                stream_generator(request),
                media_type="text/event-stream",
            )
        else:
            return await generate_response(request)

    except Exception as e:
        logger.exception("Chat completion failed")
        raise HTTPException(
            status_code=500,
            detail="Generation failed",
        ) from e


async def generate_response(request: ChatRequest) -> ChatResponse:
    """
    Generate non-streaming chat response.

    Args:
        request: Chat completion request

    Returns:
        Chat completion response
    """
    if llm_engine is None:
        raise RuntimeError("Engine not initialized")

    generated_text, prompt_tokens, completion_tokens = await llm_engine.generate(
        messages=request.messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
    )

    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    return ChatResponse(
        id=response_id,
        object="chat.completion",
        created=created,
        model=settings.model_name,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=generated_text),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


async def stream_generator(request: ChatRequest) -> AsyncIterator[str]:
    """
    Generate streaming chat response.

    Args:
        request: Chat completion request

    Yields:
        Server-Sent Events formatted chunks
    """
    if llm_engine is None:
        raise RuntimeError("Engine not initialized")

    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    first_chunk = True

    async for (
        delta,
        is_finished,
        prompt_tokens,
        completion_tokens,
    ) in llm_engine.generate_stream(
        messages=request.messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
    ):
        if first_chunk:
            stream_delta = StreamDelta(role="assistant", content=delta)
            first_chunk = False
        else:
            stream_delta = StreamDelta(content=delta)

        chunk = StreamChunk(
            id=response_id,
            object="chat.completion.chunk",
            created=created,
            model=settings.model_name,
            choices=[
                StreamChoice(
                    index=0,
                    delta=stream_delta,
                    finish_reason="stop" if is_finished else None,
                )
            ],
        )

        yield f"data: {chunk.model_dump_json()}\n\n"

    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
