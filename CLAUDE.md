# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Qwen3-TTS Voice Studio — a voice cloning, text-to-speech, and speech recognition web app. Two tabs: TTS (voice cloning via Qwen3-TTS 1.7B) and ASR (Korean interview transcription with speaker diarization via Groq Whisper API + local WavLM).

## Commands

### Frontend (Next.js 14 + React 18 + TypeScript + Tailwind CSS)

```bash
cd frontend
npm run dev          # Dev server on port 4729
npm run build        # Production build (standalone output)
npm start            # Production server on port 4729
```

### Backend (FastAPI + Python)

```bash
cd backend
source venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload   # Dev

# Production (GROQ_API_KEY required for ASR)
GROQ_API_KEY="..." gunicorn -w 1 -k uvicorn.workers.UvicornWorker --bind 127.0.0.1:4728 --timeout 900 app.main:app
```

No test suite, linter, or formatter is configured.

## Architecture

```
Browser (port 4729)  →  FastAPI backend (port 4728)  →  Qwen3-TTS model (CUDA:0)
     Next.js/React          REST + SSE               ├── vibevoice (TTS)
     Tabbed UI: TTS | ASR                            ├── Groq API / Whisper large-v3 (ASR)
                                                     └── WavLM (speaker embeddings, local GPU)
```

### Backend (`backend/app/`)

- **main.py** — FastAPI app with async lifespan. Loads TTS model on startup; ASR models lazy-load on first request. CORS allows `voice.iotok.org` and localhost origins. Routes:
  - `GET /api/health` — model/GPU status
  - `GET /api/voices` — combined preset + uploaded voice list
  - `POST /api/upload-voice` — multipart upload (max 50MB; wav/m4a/mp3/ogg/flac/webm)
  - `POST /api/generate` — TTS generation streamed via Server-Sent Events
  - `GET /api/outputs/{filename}` — serve generated WAV files
  - `POST /api/transcribe` — ASR: multipart audio upload (max 100MB, 15min), converts to 16kHz mono WAV via ffmpeg, returns SSE with speaker-diarized Korean transcript
- **tts_service.py** — TTS service. Wraps `Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")`. Voice upload converts to 24kHz mono WAV via ffmpeg. Generation is mutex-locked. Supports x-vector-only mode (no ref_text) or full voice cloning (with ref_text).
- **asr_service.py** — ASR service. Sends converted WAV to Groq API (Whisper large-v3, Korean) for transcription, then runs local WavLM (`microsoft/wavlm-base-plus-sv`) on GPU for speaker embeddings + sklearn AgglomerativeClustering for diarization (1–5 speakers). Mutex-locked. Requires `GROQ_API_KEY` env var.
- **models.py** — Pydantic schemas: `GenerateRequest`, `VoiceInfo`, `VoiceListResponse`, `HealthResponse`.

### Frontend (`frontend/app/`)

- **page.tsx** — Single component with tabbed UI (TTS | ASR). TTS tab: voice selection, text input, language/seed controls, upload modal, SSE generation. ASR tab: audio file upload (drag & drop), speaker count selector (default 2), SSE transcription progress, color-coded transcript viewer with copy/download. Uses `AbortController` for cancellation.
- **layout.tsx** — Root layout, dark mode enabled via `className="dark"`.
- **globals.css** — Tailwind directives plus custom component classes (`.card`, `.input-field`, `.btn-primary`, etc.) and `.progress-pulse` animation.

### Data directories (under `backend/`)

- `voices/` — Preset voice WAV files. Names encode metadata: `{lang}-{Name}_{gender}.wav`.
- `uploads/` — User-uploaded voices. Each voice is a `{uuid}.wav` + `{uuid}.json` metadata pair.
- `outputs/` — Generated speech WAV files (UUID-named).

### Key data flows

**TTS Generation**: Frontend POSTs JSON to `/api/generate` → backend resolves voice file, runs `model.generate_voice_clone()` with mutex lock, streams SSE events (`loading` → `generating` → `complete`/`error`) → frontend parses SSE stream and renders audio player with download link.

**Voice Upload**: Frontend sends FormData → backend validates format/size, converts via ffmpeg to 24kHz mono WAV, saves file + JSON metadata, returns voice_id → frontend refreshes voice list.

**ASR Transcription**: Frontend sends FormData (audio file + num_speakers) to `/api/transcribe` → backend converts to 16kHz mono WAV via ffmpeg → sends WAV to Groq Whisper API → gets timestamped segments → extracts WavLM speaker embeddings per segment on local GPU → clusters into N speakers via AgglomerativeClustering → streams SSE events (`loading` → `transcribing` → `complete`/`error`) → frontend renders color-coded speaker-labeled transcript with timestamps.

## Environment notes

- GPU: NVIDIA GB10 (CUDA capability 12.1). PyTorch works but CTranslate2 lacks CUDA support on this chip — do not use faster-whisper with `device="cuda"`.
- `torchaudio.load` / `torchaudio.info` require `torchcodec` in torchaudio 2.10+. Use `soundfile` for audio I/O instead.
- Audio files with misleading extensions (e.g., raw AAC with `.m4a` extension) must be converted via ffmpeg before sending to external APIs.
