"""ASR service: Korean speech transcription with speaker diarization.

Uses Groq API (Whisper large-v3) for fast transcription and local WavLM
for speaker embeddings. Speaker diarization via agglomerative clustering.
"""

import logging
import os
import threading
import time

import numpy as np
import requests
import soundfile as sf
import torch

logger = logging.getLogger(__name__)

MAX_DURATION_SEC = 900  # 15 minutes
GROQ_API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

MIME_TYPES = {
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
    ".ogg": "audio/ogg", ".flac": "audio/flac", ".webm": "audio/webm",
    ".mp4": "audio/mp4", ".mpeg": "audio/mpeg", ".mpga": "audio/mpeg",
    ".opus": "audio/opus",
}


class ASRService:
    def __init__(self):
        self._spk_extractor = None
        self._spk_model = None
        self._lock = threading.Lock()
        self._loaded = False

    def load_models(self) -> None:
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

        logger.info("Loading WavLM speaker embedding model ...")
        self._spk_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-base-plus-sv",
        )
        self._spk_model = (
            WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
            .to("cuda")
            .eval()
        )
        logger.info("Speaker embedding model loaded")
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _speaker_embedding(self, audio_np: np.ndarray) -> np.ndarray:
        inputs = self._spk_extractor(
            audio_np, sampling_rate=16000, return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(self._spk_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = self._spk_model(**inputs).embeddings
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.squeeze().cpu().numpy()

    def _transcribe_groq(self, audio_path: str) -> list[dict]:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")

        ext = os.path.splitext(audio_path)[1].lower()
        mime = MIME_TYPES.get(ext, "audio/wav")
        with open(audio_path, "rb") as f:
            resp = requests.post(
                GROQ_API_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (os.path.basename(audio_path), f, mime)},
                data={
                    "model": "whisper-large-v3",
                    "language": "ko",
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "segment",
                },
                timeout=300,
            )

        if resp.status_code != 200:
            raise RuntimeError(f"Groq API error ({resp.status_code}): {resp.text[:300]}")

        return resp.json().get("segments", [])

    def transcribe(self, original_path: str, wav_path: str, num_speakers: int = 2) -> dict:
        if not self._loaded:
            raise RuntimeError("ASR models not loaded")
        with self._lock:
            return self._transcribe_locked(wav_path, num_speakers)

    def _transcribe_locked(self, wav_path: str, num_speakers: int) -> dict:
        from sklearn.cluster import AgglomerativeClustering

        t0 = time.time()

        audio_np, sr = sf.read(wav_path, dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        duration_sec = len(audio_np) / sr

        if duration_sec > MAX_DURATION_SEC:
            raise ValueError(
                f"오디오가 너무 깁니다 ({duration_sec:.0f}초). "
                f"최대 {MAX_DURATION_SEC}초(15분)까지 지원합니다."
            )

        groq_segments = self._transcribe_groq(wav_path)

        if not groq_segments:
            return {
                "segments": [],
                "full_text": "",
                "duration": round(duration_sec, 2),
                "processing_time": round(time.time() - t0, 2),
            }

        embeddings: list[np.ndarray] = []
        valid_indices: list[int] = []
        for i, seg in enumerate(groq_segments):
            s = int(seg["start"] * sr)
            e = int(seg["end"] * sr)
            chunk = audio_np[s:e]
            if len(chunk) < sr // 2:
                continue
            if len(chunk) > sr * 10:
                chunk = chunk[: sr * 10]
            embeddings.append(self._speaker_embedding(chunk))
            valid_indices.append(i)

        label_map: dict[int, int] = {}
        if len(embeddings) >= num_speakers:
            X = np.stack(embeddings)
            labels = AgglomerativeClustering(
                n_clusters=min(num_speakers, len(embeddings)),
            ).fit_predict(X)
            for idx, lbl in zip(valid_indices, labels):
                label_map[idx] = int(lbl)
        else:
            for j, idx in enumerate(valid_indices):
                label_map[idx] = j

        result_segments = []
        for i, seg in enumerate(groq_segments):
            result_segments.append({
                "speaker": label_map.get(i, 0) + 1,
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"].strip(),
            })

        lines: list[str] = []
        prev_speaker = None
        for seg in result_segments:
            if seg["speaker"] != prev_speaker:
                if lines:
                    lines.append("")
                lines.append(f"[화자 {seg['speaker']}]")
                prev_speaker = seg["speaker"]
            lines.append(seg["text"])

        return {
            "segments": result_segments,
            "full_text": "\n".join(lines),
            "duration": round(duration_sec, 2),
            "processing_time": round(time.time() - t0, 2),
        }


asr_service = ASRService()
