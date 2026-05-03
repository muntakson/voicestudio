"""Post-process a TTS wav file: dynamic range compression + loudness normalization."""
import sys
import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d


def compress_and_normalize(audio: np.ndarray, sr: int) -> np.ndarray:
    audio = audio.astype(np.float64)

    for _ in range(4):
        window_samples = max(int(sr * 1.0), 1)
        sq = audio ** 2
        rms_env = np.sqrt(uniform_filter1d(sq, size=window_samples, mode='constant'))

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

        audio = audio * gain

        target_peak = 10 ** (-1 / 20)
        peak = np.max(np.abs(audio))
        if peak > 1e-6:
            audio = audio * (target_peak / peak)

    return audio


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} input.wav [output.wav]")
        sys.exit(1)

    input_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        base = input_path.rsplit('.', 1)[0]
        output_path = f"{base}_normalized.wav"

    audio, sr = sf.read(input_path)
    print(f"Input:  {input_path}")
    print(f"  Sample rate: {sr}, Duration: {len(audio)/sr:.1f}s, Peak: {np.max(np.abs(audio)):.4f}, RMS: {np.sqrt(np.mean(audio**2)):.4f}")

    audio = compress_and_normalize(audio, sr)

    sf.write(output_path, audio, sr)
    print(f"Output: {output_path}")
    print(f"  Peak: {np.max(np.abs(audio)):.4f}, RMS: {np.sqrt(np.mean(audio**2)):.4f}")


if __name__ == "__main__":
    main()
