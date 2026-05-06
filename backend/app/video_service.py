import logging
import os
import subprocess
import tempfile
import time

logger = logging.getLogger(__name__)

VIDEOS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)

FONT_NAME = "Noto Serif CJK KR"

WIDTH = 1080
HEIGHT = 1920


def _get_audio_duration(audio_path: str) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        text=True,
    )
    return float(out.strip())


def _format_ass_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _escape_ass(text: str) -> str:
    text = text.replace("\\", "\\\\")
    text = text.replace("{", "\\{")
    text = text.replace("}", "\\}")
    return text


def _build_ass(
    poem_title: str,
    poem_author: str,
    poem_body: str,
    duration: float,
) -> str:
    fade_in_ms = int(min(1.5, duration * 0.05) * 1000)
    scroll_start = 2.0
    scroll_end = max(duration - 1.0, scroll_start + 1)

    title_escaped = _escape_ass(poem_title)
    author_escaped = _escape_ass(poem_author)

    body_lines = poem_body.split("\n")
    body_text = "\\N".join(_escape_ass(line) for line in body_lines)

    line_height = 100
    total_body_height = len(body_lines) * line_height
    body_start_y = HEIGHT - 80
    body_end_y = 350 - total_body_height

    t_start = _format_ass_time(scroll_start)
    t_end = _format_ass_time(scroll_end)
    t_dur = _format_ass_time(duration)
    t_fade_start = _format_ass_time(min(1.5, duration * 0.05))

    ass = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {WIDTH}
PlayResY: {HEIGHT}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Title,{FONT_NAME},80,&H00FFFFFF,&H000000FF,&HB0000000,&H00000000,-1,0,0,0,100,100,0,0,1,4,0,8,80,80,140,1
Style: Author,{FONT_NAME},52,&HD9FFFFFF,&H000000FF,&H99000000,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,8,80,80,250,1
Style: Body,{FONT_NAME},64,&H00FFFFFF,&H000000FF,&HB0000000,&H00000000,-1,0,0,0,100,100,0,0,1,4,0,7,80,80,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 1,{t_fade_start},{t_dur},Title,,0,0,0,,{{\\fad({fade_in_ms},0)}}{title_escaped}
Dialogue: 1,{t_fade_start},{t_dur},Author,,0,0,0,,{{\\fad({fade_in_ms},0)}}{author_escaped}
Dialogue: 0,{t_start},{t_end},Body,,0,0,0,,{{\\move(80,{body_start_y},80,{body_end_y})}}{body_text}
"""
    return ass


def compose_poem_video(
    image_path: str,
    audio_path: str,
    poem_title: str,
    poem_author: str,
    poem_body: str,
    output_name: str,
) -> dict:
    t0 = time.time()
    duration = _get_audio_duration(audio_path)

    import uuid as _uuid
    out_filename = f"poem_short_{_uuid.uuid4().hex[:8]}.mp4"
    out_path = os.path.join(VIDEOS_DIR, out_filename)

    ass_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".ass", delete=False, encoding="utf-8"
    )
    try:
        ass_content = _build_ass(poem_title, poem_author, poem_body, duration)
        ass_file.write(ass_content)
        ass_file.close()

        filter_complex = (
            f"[0:v]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={WIDTH}:{HEIGHT},format=yuv420p,"
            f"ass={ass_file.name}"
        )

        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", image_path,
            "-i", audio_path,
            "-filter_complex", filter_complex,
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            out_path,
        ]

        logger.info("Running ffmpeg for poem video: %s", out_filename)
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            logger.error("ffmpeg stderr: %s", result.stderr[-1000:])
            raise RuntimeError(f"ffmpeg failed (code {result.returncode}): {result.stderr[-300:]}")

    finally:
        os.unlink(ass_file.name)

    file_size = os.path.getsize(out_path)
    elapsed = round(time.time() - t0, 2)
    logger.info("Poem video '%s' created: %.1fs duration, %d bytes, %.1fs elapsed",
                out_filename, duration, file_size, elapsed)

    return {
        "output_filename": out_filename,
        "duration": round(duration, 2),
        "elapsed": elapsed,
        "file_size": file_size,
    }
