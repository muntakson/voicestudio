# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Voice Studio — a multi-project audiobook biography (회고록) platform. Users record or upload interview audio, transcribe it with speaker diarization, edit and rewrite the text with LLMs, then generate audiobooks via cloud or local TTS. Hosted at `voice.iotok.org`.

## Commands

### Frontend (Next.js 14 + React 18 + TypeScript + Tailwind CSS)

```bash
cd frontend
npm run dev          # Dev server on port 4729
npm run build        # Production build — must rm -rf .next first for clean builds
npm start            # Production server on port 4729
```

Production is managed via PM2:
```bash
pm2 delete frontend && pm2 start "npm start" --name frontend --cwd /var/www/qwen3-tts/frontend
```

### Backend (FastAPI + Python)

```bash
cd backend
source venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload   # Dev
```

Production uses `backend/start.sh` which exports API keys (GROQ_API_KEY, ANTHROPIC_API_KEY, ELEVENLABS_API_KEY) and runs gunicorn on port 4728. There is also a systemd unit `qwen3-tts-backend.service` but it requires sudo to manage.

```bash
# Manual production start (if systemd is unavailable):
nohup /var/www/qwen3-tts/backend/start.sh > logs/stdout.log 2>&1 &
```

No test suite, linter, or formatter is configured.

## Architecture

```
Browser → Nginx (port 80) → /api/*  → FastAPI backend (port 4728)
          voice.iotok.org   /*      → Next.js frontend (port 4729)
          Cloudflare SSL
```

Nginx proxies with `proxy_buffering off` for SSE routes. Max upload 50MB. Timeouts set to 300s for long TTS/ASR operations.

### Pipeline (per project)

```
1. Record/Upload audio  →  2. ASR (Groq Whisper + WavLM diarization)
    ↓                          ↓
3. Edit transcript       →  4. LLM rewrite (fix typos / 박완서 style)
    ↓                          ↓
5. TTS generation (ElevenLabs cloud or Qwen3 local GPU)
```

Each step records model used, tokens consumed, elapsed time, and estimated cost to the database.

### Backend (`backend/app/`)

- **main.py** — FastAPI app. Routes:
  - Project CRUD: `GET/POST /api/projects`, `GET/PATCH/DELETE /api/projects/{id}`
  - Project audio: `POST /api/projects/{id}/upload-audio`, `POST /api/projects/{id}/transcribe`
  - TTS: `POST /api/generate` (supports `engine: "elevenlabs" | "qwen3"`)
  - Voices: `GET /api/voices` (Qwen3 presets), `GET /api/elevenlabs-voices`
  - Audio files: `GET /api/audio-files`, `GET /api/audio-files/{filename}`, `GET /api/outputs/{filename}`
  - LLM: `POST /api/fix-typos`, `POST /api/rewrite` (both use Groq or Anthropic depending on model)
  - Legacy: `POST /api/transcribe`, `POST /api/upload-voice`
- **elevenlabs_service.py** — ElevenLabs cloud TTS. Thread-locked. Uses `eleven_flash_v2_5` model. Output filenames: `{sanitized_project_name}_generated.mp3`.
- **tts_service.py** — Local Qwen3-TTS on GPU. Wraps `Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")`. Thread-locked. Output: `{name}_generated.wav`.
- **asr_service.py** — Groq Whisper large-v3 for transcription + local WavLM for speaker embeddings + AgglomerativeClustering for diarization. Thread-locked.
- **database.py** — SQLite3 with WAL mode. Thread-local connections. Auto-migration via `PRAGMA table_info`.
- **models.py** — Pydantic schemas: `GenerateRequest`, `ProjectCreate`, `ProjectUpdate`, `RewriteRequest`, `VoiceInfo`, etc.

### Database schema (key columns on `projects` table)

Text pipeline: `transcript_text` (raw ASR) → `edited_transcript` (user edits) → `rewritten_text` (LLM output)

Per-service cost tracking:
- ASR: `asr_model`, `asr_elapsed`, `asr_audio_duration`, `asr_cost`
- Fix typos: `fix_typos_model`, `fix_typos_input_tokens`, `fix_typos_output_tokens`, `fix_typos_elapsed`, `fix_typos_cost`
- Rewrite: `rewrite_model`, `rewrite_input_tokens`, `rewrite_output_tokens`, `rewrite_elapsed`, `rewrite_cost`
- TTS: `tts_engine`, `tts_model`, `tts_text_chars`, `tts_elapsed`, `tts_cost`
- `total_cost` — sum of all service costs

Status progression: `created` → `uploaded` → `transcribed` → `rewritten` → `generated`

### Frontend (`frontend/app/`)

Single-page app in **page.tsx** (~1900 lines). Six tabs in studio view:

1. **음성녹음 (Recorder)** — Browser MediaRecorder with waveform canvas, saves to project
2. **음성인식 (ASR)** — Upload or select audio, speaker count, SSE transcription
3. **글편집 (Editor)** — Edit transcript, remove speakers, fix typos / rewrite with LLM
4. **소스 (Source)** — View/edit all text versions (raw transcript, edited, rewritten) with word counts and manual save
5. **오디오북생성 (TTS)** — Dual engine (ElevenLabs/Qwen3), voice picker, text-to-speech with word count and cost estimate
6. **설정 (Settings)** — TTS engine and LLM model selection

Landing view shows project cards with AI service cost breakdown, text stats, collapsible transcript/rewrite previews, and audio player.

Key patterns:
- SSE streaming for all long-running operations (`readSSE` helper)
- Auto-save with 1.5s debounce for editor and rewritten text
- `patchProject()` sends PATCH to backend for incremental updates
- Cost calculation: LLM rates in `LLM_RATES` map, ElevenLabs at `$0.30/1K chars`
- `countWords()` handles Korean character counting separately from space-delimited words

### Data directories (under `backend/`)

- `voices/` — Preset Qwen3 voice WAVs: `{lang}-{Name}_{gender}.wav`
- `uploads/` — User-uploaded voices: `{uuid}.wav` + `{uuid}.json`
- `outputs/` — Generated audio: `{project_name}_generated.mp3` (ElevenLabs) or `.wav` (Qwen3)
- `audio_files/` — Source interview audio uploads
- `artifacts/` — Project artifacts
- `projects.db` — SQLite database

## Environment notes

- GPU: NVIDIA GB10 (CUDA capability 12.1). PyTorch works but CTranslate2 lacks CUDA support — do not use faster-whisper with `device="cuda"`.
- `torchaudio.load` / `torchaudio.info` require `torchcodec` in torchaudio 2.10+. Use `soundfile` for audio I/O instead.
- Audio files with misleading extensions (e.g., raw AAC with `.m4a`) must be converted via ffmpeg before sending to external APIs.
- Frontend build caching: Next.js can serve stale chunks after rebuild. Always `rm -rf .next` before `npm run build`, then fully restart PM2 (`pm2 delete frontend && pm2 start ...`).
- Three API keys required in production: `GROQ_API_KEY` (ASR + Groq LLMs), `ANTHROPIC_API_KEY` (Claude), `ELEVENLABS_API_KEY` (cloud TTS). All set in `backend/start.sh` and the systemd service file.
