"""Pydantic models for the Qwen3-TTS API."""

from typing import Optional
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    voice_id: str = Field(..., description="Voice ID to clone")
    language: str = Field(default="Auto", description="Language: Auto, Chinese, English, Korean, Japanese, etc.")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


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


class ProjectCreate(BaseModel):
    name: str = Field(..., description="Project name")


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    transcript_text: Optional[str] = None
    num_speakers: Optional[int] = None
    llm_model: Optional[str] = None
    rewritten_text: Optional[str] = None
    generated_audio_filename: Optional[str] = None
    generated_audio_size: Optional[int] = None
    status: Optional[str] = None
