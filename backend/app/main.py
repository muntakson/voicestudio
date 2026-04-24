"""FastAPI backend for the Qwen3-TTS voice cloning web application."""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import fastapi
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.asr_service import asr_service
from app.models import GenerateRequest, HealthResponse, RewriteRequest, VoiceInfo, VoiceListResponse
from app.tts_service import tts_service, OUTPUTS_DIR, UPLOADS_DIR, ALLOWED_AUDIO_EXTENSIONS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Qwen3-TTS backend ...")
    tts_service.load_model()
    logger.info("Model ready. Accepting requests.")
    yield
    logger.info("Shutting down Qwen3-TTS backend.")


app = FastAPI(title="Qwen3-TTS API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://voice.iotok.org", "http://voice.iotok.org",
                   "http://localhost:3000", "http://localhost:4729",
                   "http://127.0.0.1:3000", "http://127.0.0.1:4729"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    import torch
    return HealthResponse(
        status="ok" if tts_service.is_loaded else "loading",
        model_loaded=tts_service.is_loaded,
        gpu_available=torch.cuda.is_available(),
    )


@app.get("/api/voices", response_model=VoiceListResponse)
async def list_voices():
    raw = tts_service.list_voices()
    return VoiceListResponse(voices=[VoiceInfo(**v) for v in raw])


@app.post("/api/upload-voice")
async def upload_voice(
    file: UploadFile = File(...),
    name: str = fastapi.Form(...),
    ref_text: str = fastapi.Form(""),
    language: str = fastapi.Form("Auto"),
):
    """Upload a voice sample and register it with a speaker name."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Accepted: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}",
        )

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 50 MB)")

    result = tts_service.save_uploaded_voice(
        contents=contents,
        original_filename=file.filename,
        speaker_name=name.strip(),
        ref_text=ref_text.strip(),
        language=language.strip(),
    )

    logger.info("Registered voice '%s' from %s", result["name"], file.filename)
    return result


@app.post("/api/generate")
async def generate_audio(req: GenerateRequest):
    """Generate TTS audio via Server-Sent Events."""

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            if not tts_service.is_loaded:
                yield _sse({"status": "error", "message": "Model is not loaded yet"})
                return

            yield _sse({"status": "loading", "message": "Preparing voice clone..."})

            try:
                tts_service.resolve_voice(req.voice_id)
            except FileNotFoundError as exc:
                yield _sse({"status": "error", "message": str(exc)})
                return

            yield _sse({"status": "generating", "message": "Generating audio..."})

            import asyncio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: tts_service.generate(
                    text=req.text,
                    voice_id=req.voice_id,
                    language=req.language,
                    seed=req.seed,
                ),
            )

            yield _sse({
                "status": "complete",
                "audio_url": f"/api/outputs/{result['output_filename']}",
                "duration": result["duration"],
                "generation_time": result["generation_time"],
            })

        except Exception as exc:
            logger.error("Generation failed: %s", exc, exc_info=True)
            yield _sse({"status": "error", "message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.get("/api/outputs/{filename}")
async def get_output(filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    file_path = os.path.join(OUTPUTS_DIR, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="audio/wav", filename=filename)


# ---------------------------------------------------------------------------
# ASR (Speech-to-Text)
# ---------------------------------------------------------------------------

@app.post("/api/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    num_speakers: int = fastapi.Form(2),
):
    """Transcribe Korean audio with speaker diarization via SSE."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Accepted: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}",
        )

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(contents) > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 100 MB)")

    num_speakers = max(1, min(num_speakers, 5))

    tmp_dir = tempfile.mkdtemp()
    src_path = os.path.join(tmp_dir, f"input{ext}")
    wav_path = os.path.join(tmp_dir, "input.wav")

    with open(src_path, "wb") as f:
        f.write(contents)

    conv = subprocess.run(
        ["ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
        capture_output=True, timeout=120,
    )
    if conv.returncode != 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Failed to process audio file")

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            if not asr_service.is_loaded:
                yield _sse({"status": "loading", "message": "ASR 모델을 로딩 중입니다 (최초 1회, 잠시 기다려 주세요)..."})
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, asr_service.load_models)

            yield _sse({"status": "transcribing", "message": "음성을 인식하고 화자를 구분하는 중..."})

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: asr_service.transcribe(src_path, wav_path, num_speakers=num_speakers),
            )

            yield _sse({"status": "complete", **result})

        except Exception as exc:
            logger.error("Transcription failed: %s", exc, exc_info=True)
            yield _sse({"status": "error", "message": str(exc)})
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

REWRITE_SYSTEM_PROMPT = (
    "당신은 한국의 대표 작가 박완서의 문체로 글을 다시 쓰는 전문가입니다. "
    "박완서 특유의 문체를 살려주세요: "
    "일상적이고 솔직한 어조, 세밀한 관찰과 묘사, "
    "담담하면서도 깊은 감정, 사회 비판적 시선, "
    "구어체와 문어체를 자연스럽게 섞는 문장. "
    "원문의 내용과 의미는 유지하면서 박완서 스타일로 재구성하세요. "
    "추가 설명 없이 변환된 글만 출력하세요."
)

FIX_TYPOS_SYSTEM_PROMPT = (
    "당신은 한국어 맞춤법 및 오타 교정 전문가입니다. "
    "주어진 텍스트에서 오타, 맞춤법 오류, 띄어쓰기 오류를 수정하세요. "
    "원문의 내용, 의미, 문체, 화자 태그([화자 1] 등)는 그대로 유지하세요. "
    "추가 설명 없이 교정된 글만 출력하세요."
)


def _call_llm(model: str, system_prompt: str, user_text: str, temperature: float = 0.7) -> str:
    import requests as _requests

    if model.startswith("claude"):
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        resp = _requests.post(
            ANTHROPIC_API_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 4096,
                "temperature": temperature,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_text}],
            },
            timeout=120,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Claude API error ({resp.status_code}): {resp.text[:300]}")
        return resp.json()["content"][0]["text"]
    else:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        resp = _requests.post(
            GROQ_CHAT_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                "temperature": temperature,
                "max_tokens": 4096,
            },
            timeout=120,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Groq API error ({resp.status_code}): {resp.text[:300]}")
        return resp.json()["choices"][0]["message"]["content"]


@app.post("/api/fix-typos")
async def fix_typos(req: RewriteRequest):
    """Fix typos and spelling errors via LLM."""
    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            yield _sse({"status": "fixing", "message": "오타 수정 중..."})
            loop = asyncio.get_event_loop()
            fixed = await loop.run_in_executor(
                None, lambda: _call_llm(req.model, FIX_TYPOS_SYSTEM_PROMPT, req.text, 0.2),
            )
            yield _sse({"status": "complete", "fixed_text": fixed})
        except Exception as exc:
            logger.error("Fix typos failed: %s", exc, exc_info=True)
            yield _sse({"status": "error", "message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.post("/api/rewrite")
async def rewrite_text(req: RewriteRequest):
    """Rewrite text in Park Wan-seo's literary style via LLM."""
    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            yield _sse({"status": "rewriting", "message": "박완서 문체로 변환 중..."})
            loop = asyncio.get_event_loop()
            rewritten = await loop.run_in_executor(
                None, lambda: _call_llm(req.model, REWRITE_SYSTEM_PROMPT, req.text, 0.7),
            )
            yield _sse({"status": "complete", "rewritten_text": rewritten})
        except Exception as exc:
            logger.error("Rewrite failed: %s", exc, exc_info=True)
            yield _sse({"status": "error", "message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
