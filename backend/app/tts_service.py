"""TTS service wrapping the Qwen3-TTS model for inference."""

import json
import logging
import os
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import datetime
from typing import List, Optional

import re

import numpy as np
import torch

logger = logging.getLogger(__name__)

CHUNK_CHAR_LIMIT = 500


def _gpu_info() -> dict:
    """Return current GPU memory and utilization stats."""
    info = {"gpu_available": torch.cuda.is_available()}
    if not info["gpu_available"]:
        return info
    dev = torch.cuda.current_device()
    info["device_name"] = torch.cuda.get_device_name(dev)
    info["allocated_mb"] = round(torch.cuda.memory_allocated(dev) / 1024 / 1024, 1)
    info["reserved_mb"] = round(torch.cuda.memory_reserved(dev) / 1024 / 1024, 1)
    info["max_allocated_mb"] = round(torch.cuda.max_memory_allocated(dev) / 1024 / 1024, 1)
    try:
        free, total = torch.cuda.mem_get_info(dev)
        info["free_mb"] = round(free / 1024 / 1024, 1)
        info["total_mb"] = round(total / 1024 / 1024, 1)
        info["used_pct"] = round((1 - free / total) * 100, 1)
    except Exception:
        pass
    try:
        util = torch.cuda.utilization(dev)
        info["gpu_util_pct"] = util
    except Exception:
        pass
    return info

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICES_DIR = os.path.join(BACKEND_DIR, "voices")
UPLOADS_DIR = os.path.join(BACKEND_DIR, "uploads")
OUTPUTS_DIR = os.path.join(BACKEND_DIR, "outputs")

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".m4a", ".mp3", ".ogg", ".flac", ".webm"}


def _smooth_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    from app.audio_smoother import smooth_and_verify
    audio, sr, report = smooth_and_verify(audio, sr)
    logger.info("Audio smoother: %s", report.summary().replace("\n", " | "))
    return audio


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
        pre_gpu = _gpu_info()
        logger.info("Loading Qwen3-TTS model: %s | GPU before load: %s", model_name, pre_gpu)

        load_start = time.time()
        self.model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        load_time = time.time() - load_start
        self._loaded = True

        post_gpu = _gpu_info()
        logger.info(
            "Qwen3-TTS model loaded in %.1fs | GPU after load: alloc=%sMB reserved=%sMB free=%sMB used=%.1f%% util=%s%%",
            load_time,
            post_gpu.get("allocated_mb"),
            post_gpu.get("reserved_mb"),
            post_gpu.get("free_mb"),
            post_gpu.get("used_pct", 0),
            post_gpu.get("gpu_util_pct", "N/A"),
        )

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
        output_name: Optional[str] = None,
        voice_name: Optional[str] = None,
        on_progress: Optional[callable] = None,
        postprocess: bool = False,
    ) -> dict:
        if not self._loaded:
            raise RuntimeError("Model is not loaded yet")

        with self._lock:
            return self._generate_locked(text, voice_id, language, seed, output_name, voice_name, on_progress, postprocess)

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        import re
        name = re.sub(r'[\\/:*?"<>|]', '', name).strip()
        name = re.sub(r'\s+', '_', name)
        return name[:80] if name else ""

    @staticmethod
    def _split_text(text: str, limit: int = CHUNK_CHAR_LIMIT) -> list[tuple[str, str]]:
        """Split text into (chunk, break_type) tuples.

        break_type is 'paragraph' or 'sentence', indicating what boundary
        preceded this chunk. The first chunk always has 'paragraph'.
        """
        if len(text) <= limit:
            return [(text, "paragraph")]

        paragraphs = re.split(r'\n\s*\n', text)
        chunks: list[tuple[str, str]] = []
        buf = ""
        pending_break = "paragraph"

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if buf and len(buf) + len(para) + 2 > limit:
                chunks.append((buf.strip(), pending_break))
                buf = ""
                pending_break = "paragraph"
            if len(para) <= limit:
                if buf:
                    buf = buf + "\n\n" + para
                else:
                    buf = para
            else:
                if buf:
                    chunks.append((buf.strip(), pending_break))
                    buf = ""
                    pending_break = "paragraph"
                sentences = re.split(r'(?<=[.!?。！？\n])\s*', para)
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    if buf and len(buf) + len(sent) + 1 > limit:
                        chunks.append((buf.strip(), pending_break))
                        buf = ""
                        pending_break = "sentence"
                    buf = (buf + " " + sent) if buf else sent
            pending_break = "paragraph"
        if buf.strip():
            chunks.append((buf.strip(), pending_break))
        return chunks

    def _generate_locked(
        self,
        text: str,
        voice_id: str,
        language: str,
        seed: Optional[int],
        output_name: Optional[str] = None,
        voice_name: Optional[str] = None,
        on_progress: Optional[callable] = None,
        postprocess: bool = False,
    ) -> dict:
        import soundfile as sf

        wav_path, ref_text = self.resolve_voice(voice_id)

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        start_time = time.time()
        use_xvector_only = not ref_text.strip()

        chunk_pairs = self._split_text(text)
        total_chunks = len(chunk_pairs)

        pre_gpu = _gpu_info()
        logger.info(
            "=== TTS Generation Start === chunks=%d, total_chars=%d, voice=%s, lang=%s, xvector_only=%s",
            total_chunks, len(text), voice_id, language, use_xvector_only,
        )
        logger.info(
            "  GPU state: device=%s alloc=%sMB reserved=%sMB free=%sMB used=%.1f%% util=%s%%",
            pre_gpu.get("device_name", "N/A"),
            pre_gpu.get("allocated_mb"),
            pre_gpu.get("reserved_mb"),
            pre_gpu.get("free_mb"),
            pre_gpu.get("used_pct", 0),
            pre_gpu.get("gpu_util_pct", "N/A"),
        )
        logger.info("  ref_audio=%s, ref_text=%r", wav_path, ref_text[:80] if ref_text else "")

        torch.cuda.reset_peak_memory_stats()

        all_audio: list[np.ndarray] = []
        break_types: list[str] = []
        sr = None

        for i, (chunk, break_type) in enumerate(chunk_pairs):
            chunk_start = time.time()
            if on_progress and total_chunks > 1:
                on_progress(i + 1, total_chunks, chunk[:60])

            wavs, chunk_sr = self.model.generate_voice_clone(
                text=chunk,
                language=language,
                ref_audio=wav_path,
                ref_text=ref_text if not use_xvector_only else None,
                x_vector_only_mode=use_xvector_only,
            )
            sr = chunk_sr
            all_audio.append(wavs[0])
            break_types.append(break_type)
            chunk_time = time.time() - chunk_start
            chunk_dur = len(wavs[0]) / sr
            rtf = chunk_time / chunk_dur if chunk_dur > 0 else 0
            chunk_gpu = _gpu_info()
            logger.info(
                "  Chunk %d/%d: %d chars → %.1fs audio in %.1fs (RTF=%.2f) [%s] | GPU alloc=%sMB util=%s%%",
                i + 1, total_chunks, len(chunk), chunk_dur, chunk_time, rtf, break_type,
                chunk_gpu.get("allocated_mb"), chunk_gpu.get("gpu_util_pct", "N/A"),
            )

        # Fade chunk edges BEFORE compression to tame model edge artifacts
        fade_samples = int(sr * 0.3)
        fade_out = np.cos(np.linspace(0, np.pi / 2, fade_samples)) ** 2
        fade_in = np.cos(np.linspace(np.pi / 2, 0, fade_samples)) ** 2
        for i in range(len(all_audio)):
            seg = all_audio[i].astype(np.float64)
            n = min(fade_samples, len(seg) // 2)
            seg[:n] *= fade_in[:n]
            seg[-n:] *= fade_out[:n]
            all_audio[i] = seg

        PAUSE_PARAGRAPH = 1.5
        PAUSE_SENTENCE = 0.8
        if total_chunks > 1:
            parts = [all_audio[0]]
            for j in range(1, len(all_audio)):
                pause_sec = PAUSE_PARAGRAPH if break_types[j] == "paragraph" else PAUSE_SENTENCE
                parts.append(np.zeros(int(sr * pause_sec), dtype=np.float64))
                parts.append(all_audio[j])
            audio = np.concatenate(parts)
        else:
            audio = all_audio[0]

        if postprocess:
            audio = _smooth_audio(audio, sr)
        else:
            logger.info("Post-processing skipped (postprocess=False)")

        generation_time = time.time() - start_time
        duration = len(audio) / sr

        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        project_part = self._sanitize_filename(output_name) if output_name else ""
        voice_part = self._sanitize_filename(voice_name) if voice_name else ""
        parts = [p for p in [project_part, voice_part, "qwen", timestamp] if p]
        output_filename = "_".join(parts) + ".wav"
        output_path = os.path.join(OUTPUTS_DIR, output_filename)
        if os.path.exists(output_path):
            output_filename = "_".join(parts + [uuid.uuid4().hex[:4]]) + ".wav"
            output_path = os.path.join(OUTPUTS_DIR, output_filename)
        sf.write(output_path, audio, sr)

        rtf_total = generation_time / duration if duration > 0 else 0
        post_gpu = _gpu_info()
        peak_mb = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
        logger.info(
            "=== TTS Generation Done === file=%s, chunks=%d, duration=%.2fs, gen_time=%.2fs, RTF=%.2f",
            output_filename, total_chunks, duration, generation_time, rtf_total,
        )
        logger.info(
            "  GPU final: alloc=%sMB reserved=%sMB peak=%sMB free=%sMB used=%.1f%% util=%s%%",
            post_gpu.get("allocated_mb"),
            post_gpu.get("reserved_mb"),
            peak_mb,
            post_gpu.get("free_mb"),
            post_gpu.get("used_pct", 0),
            post_gpu.get("gpu_util_pct", "N/A"),
        )
        logger.info(
            "  Performance: %.1f chars/sec, %.1f audio_sec/wall_sec",
            len(text) / generation_time if generation_time > 0 else 0,
            duration / generation_time if generation_time > 0 else 0,
        )

        return {
            "output_filename": output_filename,
            "duration": round(duration, 2),
            "generation_time": round(generation_time, 2),
            "num_chunks": total_chunks,
            "rtf": round(rtf_total, 2),
            "gpu_peak_mb": peak_mb,
            "gpu_util_pct": post_gpu.get("gpu_util_pct"),
        }


tts_service = TTSService()
