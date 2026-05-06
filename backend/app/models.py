"""Pydantic models for the Qwen3-TTS API."""

from typing import Optional
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    voice_id: str = Field(..., description="Voice ID to clone")
    language: str = Field(default="Auto", description="Language: Auto, Chinese, English, Korean, Japanese, etc.")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    engine: str = Field(default="elevenlabs", description="TTS engine: 'qwen3' or 'elevenlabs'")
    output_name: Optional[str] = Field(default=None, description="Human-readable prefix for output filename")
    voice_name: Optional[str] = Field(default=None, description="Human-readable voice name for output filename")
    project_id: Optional[str] = Field(default=None, description="Associate generated audio with a project")
    postprocess: bool = Field(default=False, description="Apply audio post-processing (compression, normalization) for Qwen3")
    custom_filename: Optional[str] = Field(default=None, description="Exact output filename (without extension). If set, overrides output_name/voice_name naming.")
    poem_mode: bool = Field(default=False, description="Poem mode: longer pauses between lines and slower pace")


class VoiceInfo(BaseModel):
    id: str
    name: str
    language: str
    ref_text: str = ""
    source: str = "preset"


class VoiceListResponse(BaseModel):
    voices: list[VoiceInfo]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool


class RewriteRequest(BaseModel):
    text: str = Field(..., description="Text to rewrite")
    model: str = Field(default="claude-sonnet-4-6", description="LLM model ID")


class AudioDownloadRequest(BaseModel):
    url: str = Field(..., description="URL to download audio from (YouTube, etc.)")
    filename: Optional[str] = Field(default=None, description="Custom output filename (without extension)")
    project_id: Optional[str] = Field(default=None, description="Associate download with a project")


class ProjectCreate(BaseModel):
    name: str = Field(..., description="Project name")


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    transcript_text: Optional[str] = None
    num_speakers: Optional[int] = None
    llm_model: Optional[str] = None
    edited_transcript: Optional[str] = None
    rewritten_text: Optional[str] = None
    generated_audio_filename: Optional[str] = None
    generated_audio_size: Optional[int] = None
    generated_audio_duration: Optional[float] = None
    status: Optional[str] = None
    asr_model: Optional[str] = None
    asr_elapsed: Optional[float] = None
    asr_audio_duration: Optional[float] = None
    asr_cost: Optional[float] = None
    fix_typos_model: Optional[str] = None
    fix_typos_input_tokens: Optional[int] = None
    fix_typos_output_tokens: Optional[int] = None
    fix_typos_elapsed: Optional[float] = None
    fix_typos_cost: Optional[float] = None
    rewrite_model: Optional[str] = None
    rewrite_input_tokens: Optional[int] = None
    rewrite_output_tokens: Optional[int] = None
    rewrite_elapsed: Optional[float] = None
    rewrite_cost: Optional[float] = None
    tts_text: Optional[str] = None
    tts_engine: Optional[str] = None
    tts_model: Optional[str] = None
    tts_text_chars: Optional[int] = None
    tts_elapsed: Optional[float] = None
    tts_cost: Optional[float] = None
    total_cost: Optional[float] = None
    poem_text: Optional[str] = None
    poem_audio_filename: Optional[str] = None
    poem_audio_duration: Optional[float] = None
    poem_image_prompt: Optional[str] = None
    poem_image_filename: Optional[str] = None
    poem_video_prompt: Optional[str] = None
    poem_video_filename: Optional[str] = None
    poem_gen_elapsed: Optional[float] = None
    poem_gen_summary: Optional[str] = None
    category_id: Optional[int] = None
