"""TTS service wrapping the Qwen3-TTS model for inference."""

import json
import logging
import os
import subprocess
import tempfile
import threading
import time
import uuid
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICES_DIR = os.path.join(BACKEND_DIR, "voices")
UPLOADS_DIR = os.path.join(BACKEND_DIR, "uploads")
OUTPUTS_DIR = os.path.join(BACKEND_DIR, "outputs")

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".m4a", ".mp3", ".ogg", ".flac", ".webm"}


def _convert_to_wav(src_path: str, dst_path: str) -> None:
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", src_path, "-ar", "24000", "-ac", "1", dst_path],
        capture_output=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr.decode()[:200]}")


class TTSService:
    def __init__(self):
        self.model = None
        self._lock = threading.Lock()
        self._loaded = False

    def load_model(self) -> None:
        from qwen_tts import Qwen3TTSModel

        model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        logger.info("Loading Qwen3-TTS model: %s", model_name)

        self.model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        self._loaded = True
        logger.info("Qwen3-TTS model loaded successfully")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Voice management
    # ------------------------------------------------------------------

    def save_uploaded_voice(
        self,
        contents: bytes,
        original_filename: str,
        speaker_name: str,
        ref_text: str = "",
        language: str = "Auto",
    ) -> dict:
        ext = os.path.splitext(original_filename.lower())[1]
        uid = uuid.uuid4().hex[:8]
        voice_id = f"upload-{uid}-{speaker_name}"
        wav_name = f"{voice_id}.wav"
        wav_path = os.path.join(UPLOADS_DIR, wav_name)

        if ext == ".wav":
            with open(wav_path, "wb") as f:
                f.write(contents)
        else:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
            try:
                _convert_to_wav(tmp_path, wav_path)
            finally:
                os.unlink(tmp_path)

        meta = {
            "name": speaker_name,
            "ref_text": ref_text,
            "language": language,
            "original_file": original_filename,
        }
        with open(os.path.join(UPLOADS_DIR, f"{voice_id}.json"), "w") as f:
            json.dump(meta, f)

        logger.info("Saved voice '%s' -> %s", speaker_name, wav_name)
        return {"id": voice_id, "name": speaker_name, "language": language, "filename": wav_name}

    def list_voices(self) -> list:
        voices = []

        # Preset voices from voices/ dir
        if os.path.isdir(VOICES_DIR):
            for fname in sorted(os.listdir(VOICES_DIR)):
                if not fname.lower().endswith(".wav"):
                    continue
                voice_id = os.path.splitext(fname)[0]
                meta_path = os.path.join(VOICES_DIR, f"{voice_id}.json")
                if os.path.isfile(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                    voices.append({
                        "id": voice_id,
                        "name": meta.get("name", voice_id),
                        "language": meta.get("language", "unknown"),
                        "ref_text": meta.get("ref_text", ""),
                        "source": "preset",
                    })
                else:
                    name = voice_id
                    lang = "unknown"
                    if "-" in voice_id:
                        parts = voice_id.split("-", 1)
                        lang = parts[0]
                        remainder = parts[1]
                        name = remainder.split("_")[0] if "_" in remainder else remainder
                    voices.append({
                        "id": voice_id,
                        "name": name,
                        "language": lang,
                        "ref_text": "",
                        "source": "preset",
                    })

        # Uploaded voices from uploads/ dir
        if os.path.isdir(UPLOADS_DIR):
            for fname in sorted(os.listdir(UPLOADS_DIR)):
                if not fname.endswith(".json"):
                    continue
                voice_id = fname[:-5]  # strip .json
                wav_path = os.path.join(UPLOADS_DIR, f"{voice_id}.wav")
                if not os.path.isfile(wav_path):
                    continue
                with open(os.path.join(UPLOADS_DIR, fname)) as f:
                    meta = json.load(f)
                voices.append({
                    "id": voice_id,
                    "name": meta.get("name", voice_id),
                    "language": meta.get("language", "custom"),
                    "ref_text": meta.get("ref_text", ""),
                    "source": "uploaded",
                })

        return voices

    def resolve_voice(self, voice_id: str) -> tuple[str, str]:
        """Return (wav_path, ref_text) for a voice_id."""
        # Check uploads first
        uploads_wav = os.path.join(UPLOADS_DIR, f"{voice_id}.wav")
        uploads_meta = os.path.join(UPLOADS_DIR, f"{voice_id}.json")
        if os.path.isfile(uploads_wav):
            ref_text = ""
            if os.path.isfile(uploads_meta):
                with open(uploads_meta) as f:
                    ref_text = json.load(f).get("ref_text", "")
            return uploads_wav, ref_text

        # Check presets
        preset_wav = os.path.join(VOICES_DIR, f"{voice_id}.wav")
        preset_meta = os.path.join(VOICES_DIR, f"{voice_id}.json")
        if os.path.isfile(preset_wav):
            ref_text = ""
            if os.path.isfile(preset_meta):
                with open(preset_meta) as f:
                    ref_text = json.load(f).get("ref_text", "")
            return preset_wav, ref_text

        raise FileNotFoundError(f"Voice '{voice_id}' not found")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        text: str,
        voice_id: str,
        language: str = "Auto",
        seed: Optional[int] = None,
    ) -> dict:
        if not self._loaded:
            raise RuntimeError("Model is not loaded yet")

        with self._lock:
            return self._generate_locked(text, voice_id, language, seed)

    def _generate_locked(
        self,
        text: str,
        voice_id: str,
        language: str,
        seed: Optional[int],
    ) -> dict:
        import soundfile as sf

        wav_path, ref_text = self.resolve_voice(voice_id)

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        start_time = time.time()

        use_xvector_only = not ref_text.strip()

        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=wav_path,
            ref_text=ref_text if not use_xvector_only else None,
            x_vector_only_mode=use_xvector_only,
        )

        generation_time = time.time() - start_time

        audio = wavs[0]
        duration = len(audio) / sr

        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        output_filename = f"{uuid.uuid4().hex}.wav"
        output_path = os.path.join(OUTPUTS_DIR, output_filename)
        sf.write(output_path, audio, sr)

        logger.info(
            "Generated %s: duration=%.2fs, gen_time=%.2fs, voice=%s",
            output_filename, duration, generation_time, voice_id,
        )

        return {
            "output_filename": output_filename,
            "duration": round(duration, 2),
            "generation_time": round(generation_time, 2),
        }


tts_service = TTSService()
