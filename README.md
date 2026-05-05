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

## Batch Narration (Multiple Poems Example)

Paste multiple poems or stories into the **오디오북생성** text box using `<narration>` tags. Each narration becomes a separate audio file, generated sequentially in one click.

### Format

```xml
<narration>
<title>제목</title>
<body>
본문 내용...
</body>
</narration>
```

### Example: Korean Poems Batch

Paste the following into the text box to generate 3 poem audio files at once:

```xml
<narration>
<title>진달래꽃</title>
<body>
나 보기가 역겨워
가실 때에는
말없이 고이 보내 드리오리다.

영변에 약산
진달래꽃
아름 따다 가실 길에 뿌리오리다.

가시는 걸음걸음
놓인 그 꽃을
사뿐히 즈려밟고 가시옵소서.

나 보기가 역겨워
가실 때에는
죽어도 아니 눈물 흘리오리다.
</body>
</narration>

<narration>
<title>서시</title>
<body>
죽는 날까지 하늘을 우러러
한 점 부끄럼이 없기를,
잎새에 이는 바람에도
나는 괴로워했다.

별을 노래하는 마음으로
모든 죽어 가는 것을 사랑해야지.

그리고 나한테 주어진 길을
걸어가야겠다.

오늘 밤에도 별이 바람에 스치운다.
</body>
</narration>

<narration>
<title>님의침묵</title>
<body>
님은 갔습니다. 아아, 사랑하는 나의 님은 갔습니다.
푸른 산빛을 깨치고 단풍나무 숲을 향하여
난 작은 길을 걸어서 차마 떨치고 갔습니다.

날카로운 첫 키스의 추억은
나의 운명의 지침을 돌려놓고
뒷걸음쳐서 사라졌습니다.
</body>
</narration>
```

### How it works

1. Paste the tagged text into the **Text to Speak** box
2. The UI detects narration tags and shows a **Batch: N편** badge
3. Click **Generate Speech** — each narration is generated one by one
4. Output files are named `{title}_{voice}.wav` (e.g., `진달래꽃_세월.wav`)
5. Progress and audio players appear in the **Batch Output** panel
6. Click **Batch 중단** to stop mid-batch if needed

### Tips

- Each `<title>` becomes the output filename
- `<body>` text is preprocessed: `[잠시멈춤]` tags become 1-second pauses, stage directions like `[부드러운 목소리로]` are stripped
- Blank lines between stanzas add natural pauses in the audio
- Works with any voice and language — select before generating

## License

MIT
