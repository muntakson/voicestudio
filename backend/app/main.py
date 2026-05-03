"""FastAPI backend for the Qwen3-TTS voice cloning web application."""

import asyncio
import json
import logging
import os
import queue
import shutil
import subprocess
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

import fastapi
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.asr_service import asr_service
from app.database import (
    init_db, list_projects, get_project, create_project,
    update_project, delete_project, add_project_audio,
    list_project_audio, upsert_project_artifact, list_project_artifacts,
    AUDIO_FILES_DIR, ARTIFACTS_DIR,
)
from app.models import (
    AudioDownloadRequest, GenerateRequest, HealthResponse, ProjectCreate,
    ProjectUpdate, RewriteRequest, VoiceInfo, VoiceListResponse,
)
from app.elevenlabs_service import elevenlabs_service
from app.tts_service import tts_service, OUTPUTS_DIR, UPLOADS_DIR, ALLOWED_AUDIO_EXTENSIONS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Qwen3-TTS backend ...")
    init_db()
    logger.info("Database initialized.")
    tts_service.load_model()
    logger.info("Model ready. Accepting requests.")
    yield
    logger.info("Shutting down Qwen3-TTS backend.")


app = FastAPI(title="Qwen3-TTS API", version="3.0.0", lifespan=lifespan)

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
# Health & Voice Routes
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
async def api_list_voices():
    raw = tts_service.list_voices()
    return VoiceListResponse(voices=[VoiceInfo(**v) for v in raw])


@app.get("/api/elevenlabs-voices")
async def api_elevenlabs_voices():
    raw = elevenlabs_service.list_voices()
    return {"voices": raw}


@app.post("/api/upload-voice")
async def upload_voice(
    file: UploadFile = File(...),
    name: str = fastapi.Form(...),
    ref_text: str = fastapi.Form(""),
    language: str = fastapi.Form("Auto"),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported format. Accepted: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}")
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 50 MB)")
    result = tts_service.save_uploaded_voice(
        contents=contents, original_filename=file.filename,
        speaker_name=name.strip(), ref_text=ref_text.strip(), language=language.strip(),
    )
    logger.info("Registered voice '%s' from %s", result["name"], file.filename)
    return result


# ---------------------------------------------------------------------------
# TTS Generation
# ---------------------------------------------------------------------------

@app.post("/api/generate")
async def generate_audio(req: GenerateRequest):
    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            if req.engine == "elevenlabs":
                voice_id = req.voice_id
                if not voice_id.startswith("el_"):
                    el_voices = elevenlabs_service.list_voices()
                    if not el_voices:
                        yield _sse({"status": "error", "message": "No ElevenLabs voices available"})
                        return
                    voice_id = el_voices[0]["id"]
                    logger.info("Voice '%s' not valid for ElevenLabs, falling back to '%s'", req.voice_id, voice_id)
                yield _sse({"status": "loading", "message": "ElevenLabs에서 음성 생성 중..."})
                yield _sse({"status": "generating", "message": "ElevenLabs 클라우드 TTS 생성 중..."})
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: elevenlabs_service.generate(text=req.text, voice_id=voice_id, output_name=req.output_name, voice_name=req.voice_name),
                )
                out_fname = result["output_filename"]
                if req.project_id:
                    out_path = os.path.join(OUTPUTS_DIR, out_fname)
                    fsize = os.path.getsize(out_path) if os.path.isfile(out_path) else 0
                    add_project_audio(req.project_id, out_fname, out_fname, fsize, "output")
                yield _sse({
                    "status": "complete",
                    "audio_url": f"/api/outputs/{out_fname}",
                    "duration": result["duration"],
                    "generation_time": result["generation_time"],
                    "engine": "elevenlabs",
                    "text_chars": result.get("text_chars", 0),
                })
            else:
                if not tts_service.is_loaded:
                    yield _sse({"status": "error", "message": "Model is not loaded yet"})
                    return
                import torch as _torch
                gpu_name = _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else "CPU"
                alloc_mb = round(_torch.cuda.memory_allocated() / 1024 / 1024, 1) if _torch.cuda.is_available() else 0
                yield _sse({"status": "loading", "message": f"Preparing voice clone... (GPU: {gpu_name}, VRAM: {alloc_mb}MB)"})
                try:
                    tts_service.resolve_voice(req.voice_id)
                except FileNotFoundError as exc:
                    yield _sse({"status": "error", "message": str(exc)})
                    return
                progress_q: queue.Queue = queue.Queue()

                def on_chunk_progress(idx, total, preview):
                    progress_q.put((idx, total, preview))

                yield _sse({"status": "generating", "message": "Generating audio..."})
                loop = asyncio.get_event_loop()
                gen_future = loop.run_in_executor(
                    None,
                    lambda: tts_service.generate(
                        text=req.text, voice_id=req.voice_id, language=req.language,
                        seed=req.seed, output_name=req.output_name, voice_name=req.voice_name,
                        on_progress=on_chunk_progress, postprocess=req.postprocess,
                    ),
                )
                elapsed = 0
                while not gen_future.done():
                    try:
                        await asyncio.wait_for(asyncio.shield(gen_future), timeout=15)
                    except asyncio.TimeoutError:
                        elapsed += 15
                        while not progress_q.empty():
                            idx, total, preview = progress_q.get_nowait()
                            yield _sse({"status": "generating", "message": f"Chunk {idx}/{total}: {preview}..."})
                        yield _sse({"status": "generating", "message": f"Generating audio... ({elapsed}s)"})
                result = gen_future.result()
                out_fname = result["output_filename"]
                if req.project_id:
                    out_path = os.path.join(OUTPUTS_DIR, out_fname)
                    fsize = os.path.getsize(out_path) if os.path.isfile(out_path) else 0
                    add_project_audio(req.project_id, out_fname, out_fname, fsize, "output")
                yield _sse({
                    "status": "complete",
                    "audio_url": f"/api/outputs/{out_fname}",
                    "duration": result["duration"],
                    "generation_time": result["generation_time"],
                    "engine": "qwen3",
                    "text_chars": len(req.text),
                    "num_chunks": result.get("num_chunks", 1),
                    "rtf": result.get("rtf"),
                    "gpu_peak_mb": result.get("gpu_peak_mb"),
                    "gpu_util_pct": result.get("gpu_util_pct"),
                })
        except Exception as exc:
            logger.error("Generation failed: %s", exc, exc_info=True)
            yield _sse({"status": "error", "message": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


@app.get("/api/outputs/{filename}")
async def get_output(filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    file_path = os.path.join(OUTPUTS_DIR, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    media_type = "audio/mpeg" if filename.endswith(".mp3") else "audio/wav"
    return FileResponse(file_path, media_type=media_type, filename=filename)


# ---------------------------------------------------------------------------
# ASR (standalone, backwards-compat)
# ---------------------------------------------------------------------------

@app.post("/api/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    num_speakers: int = fastapi.Form(2),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported format. Accepted: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}")
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
    conv = subprocess.run(["ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
                          capture_output=True, timeout=120)
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

            progress_q: queue.Queue = queue.Queue()
            def on_progress(msg: str):
                progress_q.put(msg)

            loop = asyncio.get_event_loop()
            transcribe_future = loop.run_in_executor(
                None, lambda: asr_service.transcribe(src_path, wav_path, num_speakers=num_speakers, on_progress=on_progress))

            while not transcribe_future.done():
                try:
                    await asyncio.wait_for(asyncio.shield(transcribe_future), timeout=3)
                except asyncio.TimeoutError:
                    while not progress_q.empty():
                        msg = progress_q.get_nowait()
                        yield _sse({"status": "transcribing", "message": msg})

            while not progress_q.empty():
                msg = progress_q.get_nowait()
                yield _sse({"status": "transcribing", "message": msg})

            result = transcribe_future.result()
            yield _sse({"status": "complete", **result})
        except Exception as exc:
            logger.error("Transcription failed: %s", exc, exc_info=True)
            yield _sse({"status": "error", "message": str(exc)})
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# Project CRUD
# ---------------------------------------------------------------------------

@app.get("/api/projects")
async def api_list_projects():
    return list_projects()


@app.post("/api/projects")
async def api_create_project(req: ProjectCreate):
    pid = str(uuid.uuid4())
    now = datetime.now().isoformat()
    proj = create_project(pid, req.name.strip(), now)
    logger.info("Created project '%s' (%s)", req.name, pid)
    return proj


@app.get("/api/projects/{project_id}")
async def api_get_project(project_id: str):
    proj = get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    return proj


def _artifact_base(proj: dict) -> str:
    src = proj.get("source_audio_original_name") or proj.get("source_audio_filename") or ""
    if src:
        return os.path.splitext(src)[0]
    return proj.get("name") or proj["id"]


@app.patch("/api/projects/{project_id}")
async def api_update_project(project_id: str, req: ProjectUpdate):
    proj = get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    fields = {k: v for k, v in req.model_dump().items() if v is not None}
    updated = update_project(project_id, **fields)
    base = _artifact_base(updated or proj)
    if "transcript_text" in fields:
        fname = f"{base}_groq_whisper.txt"
        fpath = os.path.join(ARTIFACTS_DIR, fname)
        content = fields["transcript_text"]
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        if content.strip():
            upsert_project_artifact(project_id, fname, "음성인식 원본 (ASR)", len(content.encode("utf-8")))
    if "edited_transcript" in fields:
        fname = f"{base}_edited.txt"
        fpath = os.path.join(ARTIFACTS_DIR, fname)
        content = fields["edited_transcript"]
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        if content.strip():
            upsert_project_artifact(project_id, fname, "편집된 녹취록", len(content.encode("utf-8")))
    if "rewritten_text" in fields:
        fname = f"{base}_rewritten.txt"
        fpath = os.path.join(ARTIFACTS_DIR, fname)
        content = fields["rewritten_text"]
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        if content.strip():
            upsert_project_artifact(project_id, fname, "LLM 변환 텍스트", len(content.encode("utf-8")))
    return updated


@app.delete("/api/projects/{project_id}")
async def api_delete_project(project_id: str):
    proj = get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    if proj.get("source_audio_filename"):
        audio_path = os.path.join(AUDIO_FILES_DIR, proj["source_audio_filename"])
        if os.path.isfile(audio_path):
            os.remove(audio_path)
    for suffix in ("_transcript.txt", "_edited.txt", "_rewritten.txt"):
        p = os.path.join(ARTIFACTS_DIR, f"{project_id}{suffix}")
        if os.path.isfile(p):
            os.remove(p)
    for art in list_project_artifacts(project_id):
        p = os.path.join(ARTIFACTS_DIR, art["filename"])
        if os.path.isfile(p):
            os.remove(p)
    delete_project(project_id)
    logger.info("Deleted project %s", project_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Project Audio Upload (from browser recorder)
# ---------------------------------------------------------------------------

@app.post("/api/projects/{project_id}/upload-audio")
async def api_project_upload_audio(
    project_id: str,
    file: UploadFile = File(...),
    filename: str = fastapi.Form(""),
):
    proj = get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(contents) > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 100 MB)")

    save_name = filename.strip() if filename.strip() else f"{uuid.uuid4().hex[:8]}_recording.webm"
    save_name = save_name.replace("/", "").replace("\\", "").replace("..", "")
    save_path = os.path.join(AUDIO_FILES_DIR, save_name)
    with open(save_path, "wb") as fout:
        fout.write(contents)

    update_project(
        project_id,
        source_audio_filename=save_name,
        source_audio_original_name=save_name,
        source_audio_size=len(contents),
        status="uploaded",
    )
    add_project_audio(project_id, save_name, save_name, len(contents))
    logger.info("Saved recording '%s' for project %s (%d bytes)", save_name, project_id, len(contents))
    return {"ok": True, "filename": save_name, "size": len(contents)}


# ---------------------------------------------------------------------------
# Project Audio Files
# ---------------------------------------------------------------------------

@app.get("/api/projects/{project_id}/audio-files")
async def api_project_audio_files(project_id: str):
    proj = get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    return list_project_audio(project_id)


@app.get("/api/projects/{project_id}/artifacts")
async def api_project_artifacts(project_id: str):
    proj = get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    return list_project_artifacts(project_id)


@app.get("/api/artifacts/{filename}")
async def api_get_artifact(filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = os.path.join(ARTIFACTS_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path, filename=filename)


# ---------------------------------------------------------------------------
# Audio Files (for ASR file picker)
# ---------------------------------------------------------------------------

@app.get("/api/audio-files")
async def api_list_audio_files():
    files = []
    for f in sorted(os.listdir(AUDIO_FILES_DIR)):
        path = os.path.join(AUDIO_FILES_DIR, f)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(f.lower())[1]
        if ext in ALLOWED_AUDIO_EXTENSIONS:
            stat = os.stat(path)
            files.append({
                "filename": f,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
    return files


def _resolve_audio_path(filename: str) -> str:
    """Find audio file in audio_files/ or outputs/ directory."""
    path = os.path.join(AUDIO_FILES_DIR, filename)
    if os.path.isfile(path):
        return path
    path = os.path.join(OUTPUTS_DIR, filename)
    if os.path.isfile(path):
        return path
    return ""


@app.get("/api/audio-files/{filename}")
async def api_get_audio_file(filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = _resolve_audio_path(filename)
    if not path:
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, filename=filename)


# ---------------------------------------------------------------------------
# Audio Waveform Peaks (server-side decoding for large files)
# ---------------------------------------------------------------------------

@app.get("/api/audio-peaks/{filename}")
async def api_audio_peaks(filename: str, num_peaks: int = 500):
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = _resolve_audio_path(filename)
    if not path:
        raise HTTPException(status_code=404, detail="File not found")

    num_peaks = max(100, min(num_peaks, 2000))

    def compute():
        import soundfile as sf
        import numpy as np
        info = sf.info(path)
        duration = info.duration
        sample_rate = info.samplerate
        total_frames = info.frames

        block_size = max(1, total_frames // num_peaks)
        peaks = []
        with sf.SoundFile(path) as f:
            for _ in range(num_peaks):
                data = f.read(block_size)
                if len(data) == 0:
                    break
                if data.ndim > 1:
                    data = data[:, 0]
                peaks.append(float(np.max(np.abs(data))))

        return {"peaks": peaks, "duration": duration, "sample_rate": sample_rate}

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, compute)
        return result
    except Exception as exc:
        # If soundfile can't read the format directly, convert via ffmpeg first
        logger.warning("soundfile failed for %s, trying ffmpeg: %s", filename, exc)
        wav_tmp = os.path.join(tempfile.mkdtemp(), "decoded.wav")
        try:
            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", path, "-ar", "22050", "-ac", "1", "-f", "wav", wav_tmp],
                capture_output=True, timeout=300,
            )
            if proc.returncode != 0:
                raise HTTPException(status_code=500, detail="Failed to decode audio")

            def compute_wav():
                import soundfile as sf
                import numpy as np
                info = sf.info(wav_tmp)
                block_size = max(1, info.frames // num_peaks)
                peaks = []
                with sf.SoundFile(wav_tmp) as f:
                    for _ in range(num_peaks):
                        data = f.read(block_size)
                        if len(data) == 0:
                            break
                        if data.ndim > 1:
                            data = data[:, 0]
                        peaks.append(float(np.max(np.abs(data))))
                return {"peaks": peaks, "duration": info.duration, "sample_rate": info.samplerate}

            result = await loop.run_in_executor(None, compute_wav)
            return result
        finally:
            shutil.rmtree(os.path.dirname(wav_tmp), ignore_errors=True)


# ---------------------------------------------------------------------------
# Audio Clip (server-side trim via ffmpeg)
# ---------------------------------------------------------------------------

@app.post("/api/audio-clip")
async def api_audio_clip(
    source: str = fastapi.Form(...),
    start: float = fastapi.Form(...),
    end: float = fastapi.Form(...),
    output_name: str = fastapi.Form(...),
    project_id: str = fastapi.Form(""),
):
    if "/" in source or "\\" in source or ".." in source:
        raise HTTPException(status_code=400, detail="Invalid source filename")
    src_path = os.path.join(AUDIO_FILES_DIR, source)
    if not os.path.isfile(src_path):
        raise HTTPException(status_code=404, detail="Source file not found")

    if end <= start or start < 0:
        raise HTTPException(status_code=400, detail="Invalid time range")

    safe_name = output_name.strip().replace("/", "").replace("\\", "").replace("..", "")
    if not safe_name:
        safe_name = "clip.wav"
    if not safe_name.endswith((".wav", ".mp3")):
        safe_name += ".wav"

    out_path = os.path.join(AUDIO_FILES_DIR, safe_name)
    counter = 1
    base, ext = os.path.splitext(safe_name)
    while os.path.exists(out_path):
        out_path = os.path.join(AUDIO_FILES_DIR, f"{base}_{counter}{ext}")
        counter += 1

    final_name = os.path.basename(out_path)
    duration = end - start

    def do_clip():
        cmd = [
            "ffmpeg", "-y",
            "-i", src_path,
            "-ss", str(start),
            "-t", str(duration),
            "-ar", "44100", "-ac", "1",
            out_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()[:200]}")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, do_clip)

    file_size = os.path.getsize(out_path)
    if project_id:
        add_project_audio(project_id, final_name, final_name, file_size)
    logger.info("Clipped %s [%.1f-%.1f] -> %s (%d bytes)", source, start, end, final_name, file_size)
    return {
        "ok": True,
        "filename": final_name,
        "size": file_size,
        "duration": duration,
        "audio_url": f"/api/audio-files/{final_name}",
    }


# ---------------------------------------------------------------------------
# Audio Download (yt-dlp)
# ---------------------------------------------------------------------------

@app.post("/api/download-audio")
async def api_download_audio(req: AudioDownloadRequest):
    import re
    import yt_dlp

    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    custom_name = req.filename.strip() if req.filename else None

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            yield _sse({"status": "starting", "message": f"URL 분석 중: {url}"})

            info_result: dict = {}

            def extract_info():
                with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "js_runtimes": {"node": {}}}) as ydl:
                    info = ydl.extract_info(url, download=False)
                    info_result["title"] = info.get("title", "audio")
                    info_result["duration"] = info.get("duration")
                    info_result["uploader"] = info.get("uploader", "")

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, extract_info)

            title = info_result.get("title", "audio")
            duration = info_result.get("duration")
            dur_str = fmtDuration(duration) if duration else "알 수 없음"
            yield _sse({
                "status": "info",
                "message": f"제목: {title} | 길이: {dur_str}",
                "title": title,
                "duration": duration,
            })

            safe_title = re.sub(r'[^\w\s가-힯ᄀ-ᇿ.-]', '', custom_name or title).strip()
            if not safe_title:
                safe_title = "download"
            safe_title = safe_title[:100]
            out_path = os.path.join(AUDIO_FILES_DIR, f"{safe_title}.mp3")

            counter = 1
            while os.path.exists(out_path):
                out_path = os.path.join(AUDIO_FILES_DIR, f"{safe_title}_{counter}.mp3")
                counter += 1

            progress_messages: queue.Queue = queue.Queue()

            def progress_hook(d):
                if d["status"] == "downloading":
                    pct = d.get("_percent_str", "?%").strip()
                    speed = d.get("_speed_str", "").strip()
                    eta = d.get("_eta_str", "").strip()
                    progress_messages.put(f"다운로드 중: {pct} (속도: {speed}, 남은시간: {eta})")
                elif d["status"] == "finished":
                    progress_messages.put("다운로드 완료, MP3 변환 중...")

            def do_download():
                ydl_opts = {
                    "format": "bestaudio/best",
                    "outtmpl": out_path.replace(".mp3", ".%(ext)s"),
                    "postprocessors": [{
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }],
                    "progress_hooks": [progress_hook],
                    "quiet": True,
                    "no_warnings": True,
                    "js_runtimes": {"node": {}},
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])

            yield _sse({"status": "downloading", "message": "다운로드 시작..."})

            dl_future = loop.run_in_executor(None, do_download)
            while not dl_future.done():
                try:
                    await asyncio.wait_for(asyncio.shield(dl_future), timeout=2)
                except asyncio.TimeoutError:
                    while not progress_messages.empty():
                        msg = progress_messages.get_nowait()
                        yield _sse({"status": "downloading", "message": msg})

            dl_future.result()

            while not progress_messages.empty():
                msg = progress_messages.get_nowait()
                yield _sse({"status": "downloading", "message": msg})

            final_path = out_path
            if not os.path.exists(final_path):
                for f in os.listdir(AUDIO_FILES_DIR):
                    base = os.path.basename(out_path).replace(".mp3", "")
                    if f.startswith(base) and f.endswith(".mp3"):
                        final_path = os.path.join(AUDIO_FILES_DIR, f)
                        break

            if not os.path.exists(final_path):
                yield _sse({"status": "error", "message": "다운로드된 파일을 찾을 수 없습니다"})
                return

            file_size = os.path.getsize(final_path)
            filename = os.path.basename(final_path)

            if req.project_id:
                add_project_audio(req.project_id, filename, title or filename, file_size)

            yield _sse({
                "status": "complete",
                "message": f"완료! {filename} ({file_size / 1024 / 1024:.1f} MB)",
                "filename": filename,
                "file_size": file_size,
                "audio_url": f"/api/audio-files/{filename}",
                "title": title,
                "duration": duration,
            })

        except Exception as exc:
            logger.error("Audio download failed: %s", exc, exc_info=True)
            yield _sse({"status": "error", "message": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


def fmtDuration(sec):
    if not sec:
        return "-"
    m = int(sec) // 60
    s = int(sec) % 60
    if m == 0:
        return f"{s}초"
    return f"{m}분 {s}초"


# ---------------------------------------------------------------------------
# Project-aware Transcription
# ---------------------------------------------------------------------------

@app.post("/api/projects/{project_id}/transcribe")
async def api_project_transcribe(
    project_id: str,
    file: UploadFile = File(None),
    existing_file: str = fastapi.Form(""),
    num_speakers: int = fastapi.Form(2),
):
    proj = get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    num_speakers = max(1, min(num_speakers, 5))

    if file and file.filename:
        ext = os.path.splitext(file.filename.lower())[1]
        if ext not in ALLOWED_AUDIO_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Unsupported audio format")
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        if len(contents) > 100 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 100 MB)")
        safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
        save_path = os.path.join(AUDIO_FILES_DIR, safe_name)
        with open(save_path, "wb") as fout:
            fout.write(contents)
        update_project(project_id,
                       source_audio_filename=safe_name,
                       source_audio_original_name=file.filename,
                       source_audio_size=len(contents),
                       num_speakers=num_speakers,
                       status="uploaded")
        add_project_audio(project_id, safe_name, file.filename, len(contents))
        src_path = save_path
    elif existing_file:
        if "/" in existing_file or "\\" in existing_file or ".." in existing_file:
            raise HTTPException(status_code=400, detail="Invalid filename")
        src_path = os.path.join(AUDIO_FILES_DIR, existing_file)
        if not os.path.isfile(src_path):
            raise HTTPException(status_code=404, detail="Audio file not found")
        fsize = os.path.getsize(src_path)
        update_project(project_id,
                       source_audio_filename=existing_file,
                       source_audio_original_name=existing_file,
                       source_audio_size=fsize,
                       num_speakers=num_speakers,
                       status="uploaded")
        add_project_audio(project_id, existing_file, existing_file, fsize)
    else:
        raise HTTPException(status_code=400, detail="No audio file provided")

    tmp_dir = tempfile.mkdtemp()
    wav_path = os.path.join(tmp_dir, "input.wav")
    conv = subprocess.run(["ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
                          capture_output=True, timeout=120)
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

            progress_q: queue.Queue = queue.Queue()
            def on_progress(msg: str):
                progress_q.put(msg)

            loop = asyncio.get_event_loop()
            transcribe_future = loop.run_in_executor(
                None, lambda: asr_service.transcribe(src_path, wav_path, num_speakers=num_speakers, on_progress=on_progress))

            while not transcribe_future.done():
                try:
                    await asyncio.wait_for(asyncio.shield(transcribe_future), timeout=3)
                except asyncio.TimeoutError:
                    while not progress_q.empty():
                        msg = progress_q.get_nowait()
                        yield _sse({"status": "transcribing", "message": msg})

            while not progress_q.empty():
                msg = progress_q.get_nowait()
                yield _sse({"status": "transcribing", "message": msg})

            result = transcribe_future.result()

            update_project(project_id,
                           transcript_json=json.dumps(result, ensure_ascii=False),
                           transcript_text=result.get("full_text", ""),
                           asr_model="whisper-large-v3",
                           asr_elapsed=result.get("processing_time", 0),
                           asr_audio_duration=result.get("duration", 0),
                           asr_cost=0.0,
                           status="transcribed")

            base = _artifact_base(get_project(project_id) or proj)
            artifact_name = f"{base}_groq_whisper.txt"
            artifact_path = os.path.join(ARTIFACTS_DIR, artifact_name)
            text_content = result.get("full_text", "")
            with open(artifact_path, "w", encoding="utf-8") as af:
                af.write(text_content)
            upsert_project_artifact(project_id, artifact_name, "음성인식 원본 (ASR)", len(text_content.encode("utf-8")))

            yield _sse({"status": "complete", **result})
        except Exception as exc:
            logger.error("Project transcription failed: %s", exc, exc_info=True)
            yield _sse({"status": "error", "message": str(exc)})
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# LLM Rewrite / Fix Typos
# ---------------------------------------------------------------------------

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


def _call_llm(model: str, system_prompt: str, user_text: str, temperature: float = 0.7) -> dict:
    """Returns {"text": str, "input_tokens": int, "output_tokens": int}."""
    import requests as _requests
    if model.startswith("claude"):
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        resp = _requests.post(ANTHROPIC_API_URL, headers={
            "x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json",
        }, json={
            "model": model, "max_tokens": 4096, "temperature": temperature,
            "system": system_prompt, "messages": [{"role": "user", "content": user_text}],
        }, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"Claude API error ({resp.status_code}): {resp.text[:300]}")
        body = resp.json()
        usage = body.get("usage", {})
        return {
            "text": body["content"][0]["text"],
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }
    else:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        resp = _requests.post(GROQ_CHAT_URL, headers={
            "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
        }, json={
            "model": model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}],
            "temperature": temperature, "max_tokens": 4096,
        }, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"Groq API error ({resp.status_code}): {resp.text[:300]}")
        body = resp.json()
        usage = body.get("usage", {})
        return {
            "text": body["choices"][0]["message"]["content"],
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }


@app.post("/api/fix-typos")
async def fix_typos(req: RewriteRequest):
    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            yield _sse({"status": "fixing", "message": "오타 수정 중..."})
            loop = asyncio.get_event_loop()
            t0 = time.time()
            result = await loop.run_in_executor(None, lambda: _call_llm(req.model, FIX_TYPOS_SYSTEM_PROMPT, req.text, 0.2))
            elapsed = round(time.time() - t0, 2)
            yield _sse({
                "status": "complete", "fixed_text": result["text"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "elapsed": elapsed,
            })
        except Exception as exc:
            logger.error("Fix typos failed: %s", exc, exc_info=True)
            yield _sse({"status": "error", "message": str(exc)})
    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


@app.post("/api/rewrite")
async def rewrite_text(req: RewriteRequest):
    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            yield _sse({"status": "rewriting", "message": "박완서 문체로 변환 중..."})
            loop = asyncio.get_event_loop()
            t0 = time.time()
            result = await loop.run_in_executor(None, lambda: _call_llm(req.model, REWRITE_SYSTEM_PROMPT, req.text, 0.7))
            elapsed = round(time.time() - t0, 2)
            yield _sse({
                "status": "complete", "rewritten_text": result["text"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "elapsed": elapsed,
            })
        except Exception as exc:
            logger.error("Rewrite failed: %s", exc, exc_info=True)
            yield _sse({"status": "error", "message": str(exc)})
    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# Infographic Generation (Gemini 2.5 Flash)
# ---------------------------------------------------------------------------

INFOGRAPHICS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "infographics")
os.makedirs(INFOGRAPHICS_DIR, exist_ok=True)


class InfographicRequest(fastapi.params.Depends):
    pass


@app.post("/api/generate-infographic")
async def generate_infographic(req: dict = fastapi.Body(...)):
    prompt = req.get("prompt", "").strip()
    project_id = req.get("project_id", "")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY is not set")

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            yield _sse({"status": "generating", "message": "Gemini 2.5 Flash로 인포그래픽 생성 중..."})
            loop = asyncio.get_event_loop()

            def _gen():
                from google import genai
                from google.genai import types as genai_types

                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash-image",
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"],
                    ),
                )
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        return part.inline_data.data
                return None

            t0 = time.time()
            image_bytes = await loop.run_in_executor(None, _gen)
            elapsed = round(time.time() - t0, 2)

            if not image_bytes:
                yield _sse({"status": "error", "message": "이미지 생성에 실패했습니다. 응답에 이미지가 없습니다."})
                return

            proj_name = ""
            if project_id:
                proj = get_project(project_id)
                if proj:
                    import re
                    proj_name = re.sub(r'[\\/:*?"<>|]', '', proj.get("name", "")).strip()
                    proj_name = re.sub(r'\s+', '_', proj_name)

            fname = f"{proj_name}_infographic.png" if proj_name else f"infographic_{uuid.uuid4().hex[:8]}.png"
            fpath = os.path.join(INFOGRAPHICS_DIR, fname)
            if os.path.exists(fpath):
                fname = f"{proj_name}_infographic_{uuid.uuid4().hex[:6]}.png" if proj_name else fname
                fpath = os.path.join(INFOGRAPHICS_DIR, fname)

            with open(fpath, "wb") as f:
                f.write(image_bytes)

            logger.info("Generated infographic '%s' in %.1fs (%d bytes)", fname, elapsed, len(image_bytes))
            yield _sse({
                "status": "complete",
                "image_url": f"/api/infographics/{fname}",
                "filename": fname,
                "elapsed": elapsed,
                "size": len(image_bytes),
            })
        except Exception as exc:
            logger.error("Infographic generation failed: %s", exc, exc_info=True)
            yield _sse({"status": "error", "message": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


@app.get("/api/infographics/{filename}")
async def api_get_infographic(filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = os.path.join(INFOGRAPHICS_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, filename=filename, media_type="image/png")


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
