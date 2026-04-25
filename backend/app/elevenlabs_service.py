"""ElevenLabs TTS service for fast cloud-based text-to-speech."""

import logging
import os
import threading
import time
import uuid
from typing import Optional

import requests

logger = logging.getLogger(__name__)

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BACKEND_DIR, "outputs")

API_BASE = "https://api.elevenlabs.io/v1"
DEFAULT_MODEL = "eleven_flash_v2_5"

PRESET_VOICES = [
    {"id": "JBFqnCBsd6RMkjVDRZzb", "name": "George", "language": "English", "description": "warm, confident male"},
    {"id": "nPczCjzI2devNBz1zQrb", "name": "Brian", "language": "English", "description": "deep, narration male"},
    {"id": "cgSgspJ2msm6clMCkdW9", "name": "Jessica", "language": "English", "description": "expressive, young female"},
    {"id": "iP95p4xoKVk53GoZ742B", "name": "Chris", "language": "English", "description": "casual, young male"},
    {"id": "onwK4e9ZLuTAKqWW03F9", "name": "Daniel", "language": "English", "description": "authoritative, news male"},
    {"id": "N2lVS1w4EtoT3dr4eOWO", "name": "Callum", "language": "English", "description": "intense, transatlantic male"},
    {"id": "pFZP5JQG7iQjIQuC4Bku", "name": "Lily", "language": "English", "description": "warm, British female"},
    {"id": "TX3LPaxmHKxFdv7VOQHJ", "name": "Liam", "language": "English", "description": "articulate, young male"},
    {"id": "bIHbv24MWmeRgasZH58o", "name": "Will", "language": "English", "description": "friendly, young male"},
    {"id": "SAz9YHcvj6GT2YYXdXww", "name": "River", "language": "English", "description": "calm, nonbinary"},
    {"id": "CwhRBWXzGAHq8TQ4Fs17", "name": "Roger", "language": "English", "description": "confident, middle-aged male"},
    {"id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel", "language": "English", "description": "calm, young female"},
    {"id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella", "language": "English", "description": "soft, young female"},
    {"id": "ErXwobaYiN019PkySvjV", "name": "Antoni", "language": "English", "description": "well-rounded, young male"},
    {"id": "TxGEqnHWrfWFTfGW9XjX", "name": "Josh", "language": "English", "description": "deep, young male"},
    {"id": "VR6AewLTigWG4xSOukaG", "name": "Arnold", "language": "English", "description": "crisp, middle-aged male"},
    {"id": "pNInz6obpgDQGcFmaJgB", "name": "Adam", "language": "English", "description": "deep, middle-aged male"},
    {"id": "yoZ06aMxZJJ28mfd3POQ", "name": "Sam", "language": "English", "description": "raspy, young male"},
]


class ElevenLabsService:
    def __init__(self):
        self._lock = threading.Lock()

    def _get_api_key(self) -> str:
        key = os.environ.get("ELEVENLABS_API_KEY", "")
        if not key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set")
        return key

    def _headers(self) -> dict:
        return {"xi-api-key": self._get_api_key()}

    def list_voices(self) -> list[dict]:
        voices = []
        for v in PRESET_VOICES:
            voices.append({
                "id": f"el_{v['id']}",
                "name": v["name"],
                "language": v["language"],
                "ref_text": v.get("description", ""),
                "source": "elevenlabs",
            })

        try:
            resp = requests.get(
                f"{API_BASE}/voices",
                headers=self._headers(),
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                seen = {v["id"] for v in PRESET_VOICES}
                for v in data.get("voices", []):
                    if v["voice_id"] in seen:
                        continue
                    labels = v.get("labels", {})
                    lang = labels.get("language", "English")
                    voices.append({
                        "id": f"el_{v['voice_id']}",
                        "name": v["name"],
                        "language": lang,
                        "ref_text": labels.get("description", v.get("description", "")),
                        "source": "elevenlabs",
                    })
        except Exception as exc:
            logger.warning("Failed to fetch ElevenLabs voices: %s", exc)

        return voices

    def generate(self, text: str, voice_id: str, output_name: Optional[str] = None) -> dict:
        with self._lock:
            return self._generate_locked(text, voice_id, output_name)

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        import re
        name = re.sub(r'[\\/:*?"<>|]', '', name).strip()
        name = re.sub(r'\s+', '_', name)
        return name[:80] if name else ""

    def _generate_locked(self, text: str, voice_id: str, output_name: Optional[str] = None) -> dict:
        el_voice_id = voice_id.removeprefix("el_")

        start_time = time.time()

        resp = requests.post(
            f"{API_BASE}/text-to-speech/{el_voice_id}",
            headers={**self._headers(), "Content-Type": "application/json"},
            json={
                "text": text,
                "model_id": DEFAULT_MODEL,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                },
            },
            timeout=120,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"ElevenLabs API error ({resp.status_code}): {resp.text[:300]}")

        generation_time = time.time() - start_time

        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        prefix = self._sanitize_filename(output_name) if output_name else ""
        if prefix:
            output_filename = f"{prefix}_generated.mp3"
            output_path = os.path.join(OUTPUTS_DIR, output_filename)
            if os.path.exists(output_path):
                output_filename = f"{prefix}_generated_{uuid.uuid4().hex[:6]}.mp3"
                output_path = os.path.join(OUTPUTS_DIR, output_filename)
        else:
            output_filename = f"{uuid.uuid4().hex[:8]}.mp3"
            output_path = os.path.join(OUTPUTS_DIR, output_filename)
        with open(output_path, "wb") as f:
            f.write(resp.content)

        file_size = len(resp.content)
        duration_estimate = file_size / 16000

        logger.info(
            "ElevenLabs generated %s: ~%.1fs, gen_time=%.2fs, voice=%s",
            output_filename, duration_estimate, generation_time, el_voice_id,
        )

        return {
            "output_filename": output_filename,
            "duration": round(duration_estimate, 2),
            "generation_time": round(generation_time, 2),
            "text_chars": len(text),
        }


elevenlabs_service = ElevenLabsService()
