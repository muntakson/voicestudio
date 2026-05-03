"""Voice audio smoother with spectral QC.

Pipeline: clip sample spikes → compress → gate silence → normalize → verify.

Usage:
    from app.audio_smoother import smooth_and_verify
    audio, sr, report = smooth_and_verify(audio, sr)

CLI:
    python -m app.audio_smoother input.wav [output.wav]
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from scipy.ndimage import uniform_filter1d

logger = logging.getLogger(__name__)


@dataclass
class QCReport:
    passed: bool = False
    spike_count: int = 0
    rms_range_ratio: float = 0.0
    pause_noise_db: float = 0.0
    crest_factor: float = 0.0
    iterations: int = 0

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] iter={self.iterations} "
            f"spikes={self.spike_count} "
            f"rms_range={self.rms_range_ratio:.2f}x "
            f"pause={self.pause_noise_db:.0f}dB "
            f"crest={self.crest_factor:.1f}"
        )


MAX_SPIKES_PER_SEC = 0.15
MAX_RMS_RANGE = 2.0
MAX_PAUSE_NOISE_DB = -40
MAX_CREST_FACTOR = 8.0


def _local_rms(audio: np.ndarray, sr: int, window_sec: float = 0.2) -> np.ndarray:
    win = max(int(sr * window_sec), 1)
    return np.sqrt(np.maximum(uniform_filter1d(audio ** 2, size=win, mode='constant'), 0))


def analyze(audio: np.ndarray, sr: int) -> QCReport:
    report = QCReport()
    audio = audio.astype(np.float64)
    duration = len(audio) / sr

    # 1. Spike count: samples where |sample| > 3x local RMS
    local_rms = _local_rms(audio, sr, 0.2)
    spike_threshold = local_rms * 3.0
    spike_threshold = np.maximum(spike_threshold, 0.01)
    spike_mask = np.abs(audio) > spike_threshold

    # Count distinct spike events (group within 50ms)
    min_gap = int(sr * 0.05)
    spike_indices = np.where(spike_mask)[0]
    spike_count = 0
    last_spike = -min_gap
    for idx in spike_indices:
        if idx - last_spike >= min_gap:
            spike_count += 1
            last_spike = idx
    report.spike_count = spike_count

    # 2. Per-second RMS range
    n_secs = len(audio) // sr
    if n_secs > 0:
        rms_secs = np.array([np.sqrt(np.mean(audio[i * sr:(i + 1) * sr] ** 2)) for i in range(n_secs)])
        speech_rms = rms_secs[rms_secs > 0.008]
        if len(speech_rms) >= 2:
            report.rms_range_ratio = float(np.max(speech_rms) / np.min(speech_rms))

    # 3. Pause noise
    env_100ms = max(int(sr * 0.1), 1)
    n_100 = len(audio) // env_100ms
    pause_vals = []
    for i in range(n_100):
        rms = np.sqrt(np.mean(audio[i * env_100ms:(i + 1) * env_100ms] ** 2))
        if rms < 0.008:
            pause_vals.append(rms)
    if pause_vals:
        report.pause_noise_db = float(20 * np.log10(np.mean(pause_vals) + 1e-10))
    else:
        report.pause_noise_db = -60.0

    # 4. Crest factor (speech only)
    speech_mask = local_rms > 0.008
    if np.sum(speech_mask) > sr:
        s = audio[speech_mask]
        speech_rms_val = np.sqrt(np.mean(s ** 2))
        if speech_rms_val > 1e-6:
            report.crest_factor = float(np.max(np.abs(s)) / speech_rms_val)

    max_spikes = max(3, int(duration * MAX_SPIKES_PER_SEC))
    report.passed = (
        report.spike_count <= max_spikes
        and report.rms_range_ratio <= MAX_RMS_RANGE
        and report.pause_noise_db <= MAX_PAUSE_NOISE_DB
        and report.crest_factor <= MAX_CREST_FACTOR
    )
    return report


def _clip_spikes(audio: np.ndarray, sr: int) -> np.ndarray:
    """Clip individual sample spikes to 2.5x local RMS.
    This is the core fix — the model produces single-sample glitches
    that envelope-based methods cannot catch."""
    local_rms = _local_rms(audio, sr, 0.2)
    ceiling = local_rms * 2.5
    ceiling = np.maximum(ceiling, 0.01)
    audio = np.clip(audio, -ceiling, ceiling)
    return audio


def _compress(audio: np.ndarray, sr: int, window_sec: float = 1.0) -> np.ndarray:
    """Single-pass RMS compressor with noise gate."""
    window_samples = max(int(sr * window_sec), 1)
    rms_env = np.sqrt(np.maximum(uniform_filter1d(audio ** 2, size=window_samples, mode='constant'), 0))

    gain = np.ones_like(rms_env)
    silent = rms_env <= 0.005
    gain[silent] = 0.0
    loud = (~silent) & (rms_env > 0.06)
    gain[loud] = (0.06 / rms_env[loud]) ** (1 - 1 / 30)
    quiet = (~silent) & (~loud)
    gain[quiet] = 0.06 / rms_env[quiet]
    gain[~silent] = np.clip(gain[~silent], 0.25, 15.0)

    smooth_samples = max(int(sr * 0.2), 1)
    gain = uniform_filter1d(gain, size=smooth_samples, mode='nearest')
    return audio * gain


def _gate_silence(audio: np.ndarray, sr: int) -> np.ndarray:
    """Zero out silence regions."""
    env_win = max(int(sr * 0.03), 1)
    fade_n = max(int(sr * 0.005), 1)
    n = len(audio) // env_win
    for i in range(n):
        s = i * env_win
        e = min(s + env_win, len(audio))
        if np.sqrt(np.mean(audio[s:e] ** 2)) < 0.005:
            cf = min(s + fade_n, e)
            if s > 0 and cf > s:
                audio[s:cf] *= np.linspace(1, 0, cf - s)
            audio[cf:e] = 0.0
    return audio


def _normalize(audio: np.ndarray) -> np.ndarray:
    target = 10 ** (-1 / 20)  # -1 dB
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio * (target / peak)
    return audio


def smooth_and_verify(audio: np.ndarray, sr: int, max_iterations: int = 8) -> tuple:
    """Iteratively: clip spikes → compress → gate → normalize → check.

    Returns (audio, sr, QCReport).
    """
    audio = audio.astype(np.float64)

    initial = analyze(audio, sr)
    logger.info("Smoother initial: %s", initial.summary())

    for iteration in range(1, max_iterations + 1):
        # Step 1: Clip sample-level spikes (the key fix)
        audio = _clip_spikes(audio, sr)

        # Step 2: Multi-pass compression for even loudness
        # Short window first to tame big swings, then long window for smoothness
        for window_sec in [0.3, 0.5, 1.0, 1.0]:
            audio = _compress(audio, sr, window_sec=window_sec)
            audio = _normalize(audio)

        # Step 3: Clip again after compression (compression can create new peaks)
        audio = _clip_spikes(audio, sr)

        # Step 4: Gate silence + normalize
        audio = _gate_silence(audio, sr)
        audio = _normalize(audio)

        report = analyze(audio, sr)
        report.iterations = iteration
        logger.info("Smoother iter %d: %s", iteration, report.summary())

        if report.passed:
            logger.info("QC PASSED after %d iterations", iteration)
            return audio, sr, report

    logger.info("Smoother done (max iter): %s", report.summary())
    return audio, sr, report


if __name__ == "__main__":
    import sys
    import soundfile as sf

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if len(sys.argv) < 2:
        print(f"Usage: python -m app.audio_smoother input.wav [output.wav]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) >= 3 else input_path.rsplit(".", 1)[0] + "_smooth.wav"

    audio, sr = sf.read(input_path)
    print(f"Input: {input_path} ({len(audio)/sr:.1f}s, sr={sr})\n")

    print("--- Initial ---")
    print(analyze(audio, sr).summary())

    print("\n--- Smoothing ---")
    audio, sr, report = smooth_and_verify(audio, sr)

    print(f"\n--- Final ---")
    print(report.summary())

    sf.write(output_path, audio, sr)
    print(f"\nSaved: {output_path}")
