# Voice Studio

A multi-project audiobook biography (회고록) platform. Record or upload interview audio, transcribe with speaker diarization, edit and rewrite text with LLMs, then generate audiobooks via cloud or local TTS.

**Live at**: [voice.iotok.org](https://voice.iotok.org)

## Features

- **Audio Recording & Upload** — Browser-based recorder with waveform visualization
- **Speech Recognition** — Groq Whisper large-v3 + WavLM speaker diarization
- **Text Editing** — Manual editing, LLM-powered typo fixing, and literary rewriting (박완서 style)
- **TTS Generation** — Dual engine support:
  - **ElevenLabs** — Cloud TTS, fast, multilingual (`eleven_flash_v2_5`)
  - **Qwen3-TTS** — Local GPU inference with voice cloning (`Qwen3-TTS-12Hz-1.7B-Base`)
- **Batch Narration** — Generate multiple audio files from `<narration>` tagged scripts
- **Voice Cloning** — Record or upload reference audio to clone voices for Qwen3-TTS
- **Audio Post-processing** — Despike, normalize loudness, compression for audiobook quality
- **Infographic Generation** — Life-journey infographics via Gemini 2.5 Flash
- **Cost Tracking** — Per-service cost breakdown (ASR, LLM, TTS) for each project

## Architecture

```
Browser → Nginx (port 80) → /api/*  → FastAPI backend (port 4728)
          voice.iotok.org   /*      → Next.js frontend (port 4729)
          Cloudflare SSL
```

### Pipeline

```
1. Record/Upload audio  →  2. ASR (Groq Whisper + WavLM diarization)
       ↓                          ↓
3. Edit transcript       →  4. LLM rewrite (fix typos / literary style)
       ↓                          ↓
5. TTS generation (ElevenLabs or Qwen3-TTS)
       ↓
6. Infographic generation (Gemini 2.5 Flash)
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 14, React 18, TypeScript 5, Tailwind CSS 3 |
| Backend | FastAPI, Python 3.12, SQLite3 (WAL mode) |
| ASR | Groq Whisper large-v3, WavLM speaker embeddings |
| TTS (Cloud) | ElevenLabs Flash v2.5 |
| TTS (Local) | Qwen3-TTS-12Hz-1.7B-Base on NVIDIA GPU |
| LLM | Claude (Anthropic), Groq (Llama) |
| Infographic | Gemini 2.5 Flash image generation |

## Setup

### Prerequisites

- Python 3.12+
- Node.js 18+
- NVIDIA GPU with CUDA 12.1+ (for local TTS)
- ffmpeg

### Environment Variables

Create `backend/start.sh` with:

```bash
#!/bin/bash
export GROQ_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export ELEVENLABS_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
cd /path/to/backend
exec gunicorn -w 1 -k uvicorn.workers.UvicornWorker --bind 127.0.0.1:4728 --timeout 900 app.main:app
```

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev    # Dev server on port 4729
```

### Production

```bash
# Frontend (PM2)
cd frontend && rm -rf .next && npm run build
pm2 start "npm start" --name frontend --cwd /path/to/frontend

# Backend
chmod +x backend/start.sh
./backend/start.sh
```

## UI Tabs

| Tab | Description |
|-----|-------------|
| 음성녹음 | Browser audio recorder with waveform |
| 음성인식 | Upload audio → transcription with speaker diarization |
| 글편집 | Edit transcript, fix typos, rewrite with LLM |
| 소스 | View/edit all text versions (raw, edited, rewritten) |
| 오디오북생성 | TTS generation with dual engine and batch mode |
| 인포그래픽 | Generate life-journey infographic from text |
| 설정 | TTS engine, voice clone, LLM model selection |

## License

MIT
