"use client";

import { useState, useEffect, useRef, useCallback } from "react";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface Voice { id: string; name: string; language: string; ref_text: string; source: string; }
interface GenerationStatus { status: "idle" | "loading" | "generating" | "complete" | "error"; message: string; audioUrl: string | null; duration: number | null; }
interface TranscriptSegment { speaker: number; start: number; end: number; text: string; }
interface TranscriptResult { segments: TranscriptSegment[]; full_text: string; duration: number; processing_time: number; }

interface AuthUser { username: string; role: string; token: string; }

interface Project {
  id: string; name: string; created_at: string; owner: string | null;
  source_audio_filename: string | null; source_audio_original_name: string | null; source_audio_size: number;
  transcript_json: string | null; transcript_text: string | null; num_speakers: number;
  llm_model: string | null; edited_transcript: string | null; rewritten_text: string | null;
  generated_audio_filename: string | null; generated_audio_size: number; generated_audio_duration: number;
  status: string;
  asr_model: string | null; asr_elapsed: number; asr_audio_duration: number; asr_cost: number;
  fix_typos_model: string | null; fix_typos_input_tokens: number; fix_typos_output_tokens: number; fix_typos_elapsed: number; fix_typos_cost: number;
  rewrite_model: string | null; rewrite_input_tokens: number; rewrite_output_tokens: number; rewrite_elapsed: number; rewrite_cost: number;
  tts_text: string | null; tts_engine: string | null; tts_model: string | null; tts_text_chars: number; tts_elapsed: number; tts_cost: number;
  total_cost: number;
  poem_text: string | null; poem_audio_filename: string | null; poem_audio_duration: number;
  poem_image_prompt: string | null; poem_image_filename: string | null;
  poem_video_prompt: string | null; poem_video_filename: string | null; poem_gen_elapsed: number;
  poem_gen_summary: string | null;
}

interface AudioFileInfo { filename: string; size: number; modified?: string; original_name?: string; file_type?: string; created_at?: string; }

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

const LANGUAGES = ["Auto", "Korean", "English", "Chinese", "Japanese", "French", "German", "Spanish", "Russian", "Portuguese", "Italian"];
const EXAMPLES: { label: string; text: string; lang: string }[] = [
  { label: "English", text: "The sun dipped below the horizon, painting the sky in shades of amber and violet. A cool breeze swept through the valley, carrying with it the scent of pine and distant rain.", lang: "English" },
  { label: "Korean", text: "인공지능 기술이 빠르게 발전하면서 우리 생활의 많은 부분이 변화하고 있습니다. 음성 합성 기술도 그 중 하나로, 이제는 사람의 목소리와 거의 구별할 수 없는 수준에 이르렀습니다.", lang: "Korean" },
  { label: "Chinese", text: "人工智能技术正在以前所未有的速度改变我们的世界。从语音合成到自然语言处理，每一项突破都让我们离未来更近一步。", lang: "Chinese" },
  { label: "Japanese", text: "人工知能の技術は日々進化を続けています。音声合成の分野でも、人間の声と区別がつかないほどの品質が実現されています。", lang: "Japanese" },
];
const LLM_MODELS = [
  { id: "claude-sonnet-4-6", label: "Claude Sonnet 4.6", provider: "Anthropic" },
  { id: "qwen/qwen3-32b", label: "Qwen 3 32B", provider: "Groq" },
  { id: "llama-3.3-70b-versatile", label: "Llama 3.3 70B", provider: "Groq" },
  { id: "llama-3.1-8b-instant", label: "Llama 3.1 8B", provider: "Groq" },
];
const SPEAKER_COLORS = [
  { bg: "bg-blue-500/15", border: "border-blue-500/30", label: "bg-blue-500/25 text-blue-300" },
  { bg: "bg-emerald-500/15", border: "border-emerald-500/30", label: "bg-emerald-500/25 text-emerald-300" },
  { bg: "bg-amber-500/15", border: "border-amber-500/30", label: "bg-amber-500/25 text-amber-300" },
  { bg: "bg-pink-500/15", border: "border-pink-500/30", label: "bg-pink-500/25 text-pink-300" },
  { bg: "bg-cyan-500/15", border: "border-cyan-500/30", label: "bg-cyan-500/25 text-cyan-300" },
];

function stripThinkTags(t: string) { return t.replace(/<think>[\s\S]*?<\/think>/g, "").trim(); }
function fmtTime(sec: number) { return `${Math.floor(sec / 60)}:${Math.floor(sec % 60).toString().padStart(2, "0")}`; }
function fmtSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
function fmtDate(iso: string) {
  try { const d = new Date(iso); return `${d.getFullYear()}-${(d.getMonth()+1).toString().padStart(2,"0")}-${d.getDate().toString().padStart(2,"0")}`; }
  catch { return iso; }
}

function fmtDuration(sec: number) {
  if (!sec) return "-";
  if (sec < 60) return `${sec.toFixed(1)}초`;
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return `${m}분 ${s}초`;
}

const LLM_RATES: Record<string, [number, number]> = {
  "claude-sonnet-4-6": [3.0, 15.0],
  "qwen/qwen3-32b": [0.29, 0.39],
  "llama-3.3-70b-versatile": [0.59, 0.79],
  "llama-3.1-8b-instant": [0.05, 0.08],
};
const EL_COST_PER_CHAR = 0.00030;
const QWEN3_CHARS_PER_SEC = 7;
const EL_CHARS_PER_SEC = 80;

function estimateTtsTime(chars: number, engine: string): string {
  const cps = engine === "elevenlabs" ? EL_CHARS_PER_SEC : QWEN3_CHARS_PER_SEC;
  const secs = Math.ceil(chars / cps);
  if (secs < 60) return `~${secs}초`;
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  return s > 0 ? `~${m}분 ${s}초` : `~${m}분`;
}

const INFOGRAPHIC_PROMPT_TEMPLATE = `Create a professional, high-resolution infographic summarizing the life journey of [INSERT PERSON NAME]. Use a visual style similar to the attached example: a clean, aesthetic, and warm-toned illustration that combines metaphors with clear text.

Core Visual Concept: Use a 'growth' metaphor (such as a tree or flower) to represent the person's life stages.

Root/Base: Represent early life/background with symbolic icons (e.g., [insert symbol, like a seed or home]). Add a short text description: '[Insert early life text]'.

Main Stem/Growth: Depict milestones as leaves or branches. For each milestone, include:
A minimalist icon (e.g., graduation cap for education, heart for family, book for achievements).

A bold, concise title.

2 lines of descriptive text.


Timeline/Process: At the bottom or side, include a clear linear progression (e.g., 'Past' → 'Present' → 'Future') showing key achievements as stages.

Design Specifications:

Color Palette: Use soft, warm, and professional colors (creams, muted greens, soft pinks/oranges).

Layout: Well-balanced, structured, and easy to read. Ensure text is clearly separated from background elements.

Typography: Use a clean, legible, modern sans-serif font.

Style: Flat, modern vector illustration with subtle shadows for depth. Avoid busy or cluttered designs.

Language: All text must be in Korean.`;

function countWords(text: string): number {
  if (!text || !text.trim()) return 0;
  const korean = text.match(/[가-힯ᄀ-ᇿ㄰-㆏ꥠ-꥿ힰ-퟿]+/g);
  const other = text.replace(/[가-힯ᄀ-ᇿ㄰-㆏ꥠ-꥿ힰ-퟿]+/g, " ").trim().split(/\s+/).filter(Boolean);
  return (korean ? korean.join("").length : 0) + other.length;
}

function calcLLMCost(model: string, inTok: number, outTok: number): number {
  const [i, o] = LLM_RATES[model] || [0, 0];
  return (inTok * i + outTok * o) / 1_000_000;
}

function estimateCost(model: string, inputTokens: number, outputTokens: number): string {
  const cost = calcLLMCost(model, inputTokens, outputTokens);
  if (cost === 0) return "-";
  return cost < 0.01 ? `$${cost.toFixed(4)}` : `$${cost.toFixed(3)}`;
}

function buildProjectSummary(p: Project): string {
  const lines: string[] = [];
  lines.push(`## ${p.name}`);
  lines.push(`- **생성일**: ${fmtDate(p.created_at)}`);
  lines.push("");

  if (p.source_audio_original_name) {
    lines.push("### 🎵 소스 오디오");
    lines.push(`- **파일명**: ${p.source_audio_original_name}`);
    lines.push(`- **파일 크기**: ${fmtSize(p.source_audio_size)}`);
    if (p.asr_audio_duration) lines.push(`- **오디오 길이**: ${fmtDuration(p.asr_audio_duration)}`);
    lines.push("");
  }

  if (p.asr_elapsed) {
    lines.push("### 🎙 음성인식 (ASR)");
    lines.push(`- **엔진**: Groq Whisper large-v3 + WavLM 화자분리`);
    lines.push(`- **화자 수**: ${p.num_speakers || 2}명`);
    lines.push(`- **처리 시간**: ${fmtDuration(p.asr_elapsed)}`);
    lines.push(`- **예상 비용**: ${p.asr_cost ? `$${p.asr_cost.toFixed(4)}` : "무료 (Groq 무료 티어)"}`);
    lines.push("");
  }

  if (p.fix_typos_model) {
    const model = p.fix_typos_model;
    const totalTokens = p.fix_typos_input_tokens + p.fix_typos_output_tokens;
    lines.push("### ✏️ 오타수정");
    lines.push(`- **모델**: ${model}`);
    lines.push(`- **입력 토큰**: ${p.fix_typos_input_tokens.toLocaleString()} / **출력 토큰**: ${p.fix_typos_output_tokens.toLocaleString()} / **합계**: ${totalTokens.toLocaleString()}`);
    lines.push(`- **처리 시간**: ${fmtDuration(p.fix_typos_elapsed)}`);
    lines.push(`- **예상 비용**: ${p.fix_typos_cost ? (p.fix_typos_cost < 0.01 ? `$${p.fix_typos_cost.toFixed(4)}` : `$${p.fix_typos_cost.toFixed(3)}`) : estimateCost(model, p.fix_typos_input_tokens, p.fix_typos_output_tokens)}`);
    lines.push("");
  }

  if (p.rewrite_model) {
    const model = p.rewrite_model;
    const totalTokens = p.rewrite_input_tokens + p.rewrite_output_tokens;
    lines.push("### 🖊 박완서 문체 변환");
    lines.push(`- **모델**: ${model}`);
    lines.push(`- **입력 토큰**: ${p.rewrite_input_tokens.toLocaleString()} / **출력 토큰**: ${p.rewrite_output_tokens.toLocaleString()} / **합계**: ${totalTokens.toLocaleString()}`);
    lines.push(`- **처리 시간**: ${fmtDuration(p.rewrite_elapsed)}`);
    lines.push(`- **예상 비용**: ${p.rewrite_cost ? (p.rewrite_cost < 0.01 ? `$${p.rewrite_cost.toFixed(4)}` : `$${p.rewrite_cost.toFixed(3)}`) : estimateCost(model, p.rewrite_input_tokens, p.rewrite_output_tokens)}`);
    lines.push("");
  }

  const ttsSourceText = p.rewritten_text || p.edited_transcript || p.transcript_text || "";
  if (ttsSourceText) {
    const wc = countWords(ttsSourceText);
    lines.push("### 📝 텍스트");
    lines.push(`- **글자 수**: ${ttsSourceText.length.toLocaleString()}자 / **단어 수**: ${wc.toLocaleString()}단어`);
    if (p.transcript_text) lines.push(`- **원본 전사**: ${p.transcript_text.length.toLocaleString()}자`);
    if (p.edited_transcript) lines.push(`- **편집 녹취록**: ${p.edited_transcript.length.toLocaleString()}자`);
    if (p.rewritten_text) lines.push(`- **LLM 변환**: ${p.rewritten_text.length.toLocaleString()}자`);
    lines.push("");
  }

  if (p.generated_audio_filename) {
    lines.push("### 🔊 생성된 오디오북");
    lines.push(`- **엔진**: ${p.tts_model || (p.tts_engine === "elevenlabs" ? "eleven_flash_v2_5" : "Qwen3-TTS-1.7B")}`);
    lines.push(`- **파일명**: ${p.generated_audio_filename}`);
    if (p.generated_audio_size) lines.push(`- **파일 크기**: ${fmtSize(p.generated_audio_size)}`);
    if (p.generated_audio_duration) lines.push(`- **오디오 길이**: ${fmtDuration(p.generated_audio_duration)}`);
    if (p.tts_elapsed) lines.push(`- **생성 시간**: ${fmtDuration(p.tts_elapsed)}`);
    if (p.tts_text_chars > 0) lines.push(`- **문자 수**: ${p.tts_text_chars.toLocaleString()}자`);
    const tc = p.tts_cost || 0;
    lines.push(`- **예상 비용**: ${tc > 0 ? (tc < 0.01 ? `$${tc.toFixed(4)}` : `$${tc.toFixed(3)}`) : "무료 (로컬 GPU)"}`)
    lines.push("");
  }

  const sumCost = (p.asr_cost || 0) + (p.fix_typos_cost || 0) + (p.rewrite_cost || 0) + (p.tts_cost || 0);
  if (sumCost > 0) {
    lines.push("---");
    lines.push(`**총 예상 비용**: ${sumCost < 0.01 ? `$${sumCost.toFixed(4)}` : `$${sumCost.toFixed(3)}`}`);
  }

  return lines.join("\n");
}

function renderMarkdown(md: string) {
  const lines = md.split("\n");
  const elements: React.ReactNode[] = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.startsWith("## ")) {
      elements.push(<h2 key={i} className="text-sm font-bold text-white mb-1">{line.slice(3)}</h2>);
    } else if (line.startsWith("### ")) {
      elements.push(<h3 key={i} className="text-xs font-semibold text-accent-400 mt-2 mb-0.5">{line.slice(4)}</h3>);
    } else if (line.startsWith("- ")) {
      const content = line.slice(2).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
      elements.push(<p key={i} className="text-sm text-gray-300 pl-2" dangerouslySetInnerHTML={{ __html: `• ${content}` }} />);
    } else if (line.startsWith("---")) {
      elements.push(<hr key={i} className="border-[#364153] my-1.5" />);
    } else if (line.startsWith("**")) {
      const content = line.replace(/\*\*(.+?)\*\*/g, '<strong class="text-white">$1</strong>');
      elements.push(<p key={i} className="text-sm text-gray-300" dangerouslySetInnerHTML={{ __html: content }} />);
    } else if (line.trim() === "") {
      continue;
    } else {
      elements.push(<p key={i} className="text-sm text-gray-300">{line}</p>);
    }
  }
  return <>{elements}</>;
}

function audioBufferToWav(buf: AudioBuffer): Blob {
  const nCh = buf.numberOfChannels, sr = buf.sampleRate, len = buf.length;
  const dataSize = len * nCh * 2, ab = new ArrayBuffer(44 + dataSize);
  const v = new DataView(ab, 0, 44), s = new Int16Array(ab, 44);
  const w = (o: number, t: string) => { for (let i = 0; i < t.length; i++) v.setUint8(o + i, t.charCodeAt(i)); };
  w(0, "RIFF"); v.setUint32(4, 36 + dataSize, true); w(8, "WAVE"); w(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, nCh, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * nCh * 2, true);
  v.setUint16(32, nCh * 2, true); v.setUint16(34, 16, true); w(36, "data"); v.setUint32(40, dataSize, true);
  for (let i = 0; i < len; i++) for (let c = 0; c < nCh; c++) {
    const val = Math.max(-1, Math.min(1, buf.getChannelData(c)[i]));
    s[i * nCh + c] = val < 0 ? val * 0x8000 : val * 0x7FFF;
  }
  return new Blob([ab], { type: "audio/wav" });
}

function fmtAudTime(sec: number) {
  const m = Math.floor(sec / 60), s = sec % 60;
  return `${m.toString().padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}`;
}

/* ------------------------------------------------------------------ */
/*  SSE helper                                                         */
/* ------------------------------------------------------------------ */

async function readSSE(res: Response, onEvent: (evt: Record<string, unknown>) => void) {
  const reader = res.body?.getReader();
  if (!reader) return;
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || !trimmed.startsWith("data:")) continue;
      try { onEvent(JSON.parse(trimmed.slice(5).trim())); } catch { /* skip */ }
    }
  }
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function Home() {
  /* ---- Auth state ---- */
  const [authUser, setAuthUser] = useState<AuthUser | null>(null);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [authMode, setAuthMode] = useState<"signin" | "signup">("signin");
  const [authUsername, setAuthUsername] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authError, setAuthError] = useState("");
  const [authLoading, setAuthLoading] = useState(false);

  useEffect(() => {
    const saved = localStorage.getItem("voicestudio_auth");
    if (saved) {
      try {
        const parsed = JSON.parse(saved) as AuthUser;
        setAuthUser(parsed);
      } catch { /* ignore */ }
    }
  }, []);

  const authHeaders = (): Record<string, string> => {
    if (!authUser?.token) return { "Content-Type": "application/json" };
    return { "Content-Type": "application/json", Authorization: `Bearer ${authUser.token}` };
  };

  const handleAuth = async () => {
    if (!authUsername.trim() || !authPassword) return;
    setAuthLoading(true);
    setAuthError("");
    try {
      const endpoint = authMode === "signup" ? "/api/auth/signup" : "/api/auth/signin";
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: authUsername.trim(), password: authPassword }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Error" }));
        setAuthError(err.detail || "Failed");
        return;
      }
      const data: AuthUser = await res.json();
      setAuthUser(data);
      localStorage.setItem("voicestudio_auth", JSON.stringify(data));
      setShowAuthModal(false);
      setAuthUsername("");
      setAuthPassword("");
    } catch { setAuthError("Network error"); }
    finally { setAuthLoading(false); }
  };

  const handleSignout = () => {
    setAuthUser(null);
    localStorage.removeItem("voicestudio_auth");
  };

  /* ---- View state ---- */
  const [view, setView] = useState<"landing" | "studio">("landing");
  const [currentProjectId, setCurrentProjectId] = useState<string | null>(null);
  const [currentProjectName, setCurrentProjectName] = useState<string>("");

  /* ---- Landing state ---- */
  const [projects, setProjects] = useState<Project[]>([]);
  const [showNewModal, setShowNewModal] = useState(false);
  const [newProjectName, setNewProjectName] = useState("");
  const [expandedTranscripts, setExpandedTranscripts] = useState<Set<string>>(new Set());
  const [expandedRewrites, setExpandedRewrites] = useState<Set<string>>(new Set());
  const [expandedSummary, setExpandedSummary] = useState<Set<string>>(new Set());

  /* ---- Studio state ---- */
  const [activeTab, setActiveTab] = useState<"recorder" | "download" | "source" | "tts" | "infographic" | "poem-shorts" | "asr" | "editor" | "settings">("asr");

  /* TTS */
  const [ttsEngine, setTtsEngine] = useState<"elevenlabs" | "qwen3">("qwen3");
  const [text, setText] = useState("");
  const [language, setLanguage] = useState("Korean");
  const [voices, setVoices] = useState<Voice[]>([]);
  const [elVoices, setElVoices] = useState<Voice[]>([]);
  const [selectedVoice, setSelectedVoice] = useState("upload-be0916e0-세월");
  const [seed, setSeed] = useState("100");
  const [postprocess, setPostprocess] = useState(false);
  const [gen, setGen] = useState<GenerationStatus>({ status: "idle", message: "", audioUrl: null, duration: null });
  const [genElapsed, setGenElapsed] = useState(0);
  const genStartRef = useRef<number>(0);
  const genTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadName, setUploadName] = useState("");
  const [uploadRefText, setUploadRefText] = useState("");
  const [uploadLanguage, setUploadLanguage] = useState("Auto");
  const [uploading, setUploading] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  /* Batch narration */
  interface NarationItem { title: string; body: string; }
  const [batchStatus, setBatchStatus] = useState<{ running: boolean; current: number; total: number; results: { title: string; status: "pending" | "generating" | "complete" | "error"; audioUrl?: string; message?: string }[] }>({ running: false, current: 0, total: 0, results: [] });
  const batchAbortRef = useRef(false);

  /* Poems file picker */
  const [poemFiles, setPoemFiles] = useState<{ filename: string; name: string }[]>([]);
  const [showPoemPicker, setShowPoemPicker] = useState(false);
  const [poemLoading, setPoemLoading] = useState(false);

  /* ASR */
  const [asrFile, setAsrFile] = useState<File | null>(null);
  const [numSpeakers, setNumSpeakers] = useState(2);
  const [asrStatus, setAsrStatus] = useState<{ status: string; message: string }>({ status: "idle", message: "" });
  const [transcript, setTranscript] = useState<TranscriptResult | null>(null);
  const [copied, setCopied] = useState(false);
  const asrAbortRef = useRef<AbortController | null>(null);
  const asrFileRef = useRef<HTMLInputElement | null>(null);
  const [showAudioList, setShowAudioList] = useState(false);
  const [audioFiles, setAudioFiles] = useState<AudioFileInfo[]>([]);
  const [selectedExistingFile, setSelectedExistingFile] = useState<string | null>(null);

  /* Recorder */
  const [isRecording, setIsRecording] = useState(false);
  const [recorderElapsed, setRecorderElapsed] = useState(0);
  const [recordingFilename, setRecordingFilename] = useState("");
  const [recordingSaving, setRecordingSaving] = useState(false);
  const [recordingSaved, setRecordingSaved] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const recorderCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const recorderTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const animFrameRef = useRef<number>(0);

  /* Download tab */
  const [dlUrl, setDlUrl] = useState("");
  const [dlFilename, setDlFilename] = useState("");
  const [dlStatus, setDlStatus] = useState<{ status: string; message: string }>({ status: "idle", message: "" });
  const [dlLogs, setDlLogs] = useState<string[]>([]);
  const dlLogRef = useRef<HTMLDivElement | null>(null);
  const dlAbortRef = useRef<AbortController | null>(null);
  const [dlAudioUrl, setDlAudioUrl] = useState<string | null>(null);
  const [dlAudioFilename, setDlAudioFilename] = useState<string | null>(null);
  /* Download audio editor (server-side peaks + HTML5 audio) */
  const [dlAudReady, setDlAudReady] = useState(false);
  const [dlAudDuration, setDlAudDuration] = useState(0);
  const [dlAudPlaying, setDlAudPlaying] = useState(false);
  const [dlAudTime, setDlAudTime] = useState(0);
  const [dlAudSelection, setDlAudSelection] = useState<[number, number] | null>(null);
  const [dlAudSaving, setDlAudSaving] = useState(false);
  const [dlAudSaveMsg, setDlAudSaveMsg] = useState("");
  const dlAudCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const dlAudRef = useRef<HTMLAudioElement | null>(null);
  const dlAudAnimRef = useRef<number>(0);
  const dlAudPeaksRef = useRef<number[]>([]);
  const dlAudDragRef = useRef(false);
  const dlAudSelRef = useRef<[number, number] | null>(null);
  const dlAudUpdateRef = useRef(0);

  /* Audio Editor */
  const [audBuffer, setAudBuffer] = useState<AudioBuffer | null>(null);
  const [audUrl, setAudUrl] = useState<string | null>(null);
  const [audPlaying, setAudPlaying] = useState(false);
  const [audTime, setAudTime] = useState(0);
  const [audSelection, setAudSelection] = useState<[number, number] | null>(null);
  const [audSaving, setAudSaving] = useState(false);
  const [audSaveMsg, setAudSaveMsg] = useState("");
  const audCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const audRef = useRef<HTMLAudioElement | null>(null);
  const audAnimRef = useRef<number>(0);
  const audPeaksRef = useRef<number[]>([]);
  const audBlobRef = useRef<string | null>(null);
  const audDragRef = useRef(false);
  const audSelRef = useRef<[number, number] | null>(null);
  const audUpdateRef = useRef(0);

  /* Editor */
  const [editorText, setEditorText] = useState("");
  const [rewrittenText, setRewrittenText] = useState("");
  const [rewriteStatus, setRewriteStatus] = useState<{ status: string; message: string }>({ status: "idle", message: "" });
  const rewriteAbortRef = useRef<AbortController | null>(null);
  const editorFileRef = useRef<HTMLInputElement | null>(null);
  const editorSaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [editorSaved, setEditorSaved] = useState(false);
  const rewrittenSaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [rewrittenSaved, setRewrittenSaved] = useState(false);
  const ttsSaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [ttsSaved, setTtsSaved] = useState(false);

  /* Source tab */
  const [sourceProject, setSourceProject] = useState<Project | null>(null);
  const [srcTranscript, setSrcTranscript] = useState("");
  const [srcAudioFiles, setSrcAudioFiles] = useState<AudioFileInfo[]>([]);
  const [srcArtifacts, setSrcArtifacts] = useState<{ filename: string; label: string; artifact_type: string; file_size: number; created_at: string }[]>([]);
  const [expandedArtifacts, setExpandedArtifacts] = useState<Set<string>>(new Set());
  const [artifactContents, setArtifactContents] = useState<Record<string, string>>({});

  /* Voice clip editor modal */
  const [clipModal, setClipModal] = useState<{ filename: string; audioUrl: string } | null>(null);
  const [clipReady, setClipReady] = useState(false);
  const [clipDuration, setClipDuration] = useState(0);
  const [clipPlaying, setClipPlaying] = useState(false);
  const [clipTime, setClipTime] = useState(0);
  const [clipSelection, setClipSelection] = useState<[number, number] | null>(null);
  const [clipSaving, setClipSaving] = useState(false);
  const [clipSaveMsg, setClipSaveMsg] = useState("");
  const [clipCropName, setClipCropName] = useState("");
  const [clipVoiceName, setClipVoiceName] = useState("");
  const [clipRefText, setClipRefText] = useState("");
  const [clipRegistering, setClipRegistering] = useState(false);
  const [clipHistory, setClipHistory] = useState<{ from: string; to: string; start: number; end: number }[]>([]);
  const [clipLogs, setClipLogs] = useState<string[]>([]);
  const clipLogRef = useRef<HTMLDivElement | null>(null);
  const clipCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const clipAudioRef = useRef<HTMLAudioElement | null>(null);
  const clipAnimRef = useRef<number>(0);
  const clipPeaksRef = useRef<number[]>([]);
  const clipDragRef = useRef(false);
  const clipSelRef = useRef<[number, number] | null>(null);
  const clipUpdateRef = useRef(0);
  const [srcEdited, setSrcEdited] = useState("");
  const [srcRewritten, setSrcRewritten] = useState("");
  const [srcSaving, setSrcSaving] = useState<string | null>(null);

  /* Debug console */
  const [debugLogs, setDebugLogs] = useState<string[]>([]);
  const debugRef = useRef<HTMLDivElement | null>(null);
  const addDebug = useCallback((msg: string) => {
    const ts = new Date().toLocaleTimeString("en-GB", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit", fractionalSecondDigits: 3 });
    setDebugLogs((prev) => [...prev, `[${ts}] ${msg}`]);
  }, []);

  /* Infographic */
  const [infoPrompt, setInfoPrompt] = useState("");
  const [infoStatus, setInfoStatus] = useState<{ status: string; message: string }>({ status: "idle", message: "" });
  const [infoImageUrl, setInfoImageUrl] = useState<string | null>(null);
  const infoAbortRef = useRef<AbortController | null>(null);

  /* Poem Shorts */
  const [psPoem, setPsPoem] = useState("");
  const [psAudioUrl, setPsAudioUrl] = useState<string | null>(null);
  const [psAudioDuration, setPsAudioDuration] = useState<number | null>(null);
  const [psAudioStatus, setPsAudioStatus] = useState<{ status: string; message: string }>({ status: "idle", message: "" });
  const [psImagePrompt, setPsImagePrompt] = useState("");
  const [psImagePromptStatus, setPsImagePromptStatus] = useState<{ status: string; message: string }>({ status: "idle", message: "" });
  const [psImageUrl, setPsImageUrl] = useState<string | null>(null);
  const [psImageStatus, setPsImageStatus] = useState<{ status: string; message: string }>({ status: "idle", message: "" });
  const [psVideoPrompt, setPsVideoPrompt] = useState("");
  const [psVideoPromptStatus, setPsVideoPromptStatus] = useState<{ status: string; message: string }>({ status: "idle", message: "" });
  const [psVideoUrl, setPsVideoUrl] = useState<string | null>(null);
  const [psVideoStatus, setPsVideoStatus] = useState<{ status: string; message: string }>({ status: "idle", message: "" });
  const psAbortRef = useRef<AbortController | null>(null);
  const psPoemSaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [psSummaryOpen, setPsSummaryOpen] = useState(false);

  /* Settings */
  const [selectedModel, setSelectedModel] = useState("claude-sonnet-4-6");

  /* Voice Clone */
  const [vcMode, setVcMode] = useState<"idle" | "recording" | "recorded" | "uploading" | "done">("idle");
  const [vcName, setVcName] = useState("");
  const [vcLang, setVcLang] = useState("Korean");
  const [vcRefText, setVcRefText] = useState("");
  const [vcElapsed, setVcElapsed] = useState(0);
  const [vcAudioBlob, setVcAudioBlob] = useState<Blob | null>(null);
  const [vcAudioUrl, setVcAudioUrl] = useState<string | null>(null);
  const [vcError, setVcError] = useState("");
  const [vcSavedVoice, setVcSavedVoice] = useState<{ id: string; name: string } | null>(null);
  const vcRecorderRef = useRef<MediaRecorder | null>(null);
  const vcChunksRef = useRef<Blob[]>([]);
  const vcCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const vcAnalyserRef = useRef<AnalyserNode | null>(null);
  const vcAudioCtxRef = useRef<AudioContext | null>(null);
  const vcTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const vcAnimRef = useRef<number>(0);
  const vcFileRef = useRef<HTMLInputElement | null>(null);

  /* ================================================================ */
  /*  Project / Landing logic                                          */
  /* ================================================================ */

  const fetchProjects = useCallback(async () => {
    try {
      const headers: Record<string, string> = {};
      if (authUser?.token) headers["Authorization"] = `Bearer ${authUser.token}`;
      const res = await fetch("/api/projects", { headers });
      if (res.ok) setProjects(await res.json());
    } catch { /* silent */ }
  }, [authUser]);

  useEffect(() => { fetchProjects(); }, [fetchProjects]);
  useEffect(() => { debugRef.current?.scrollTo(0, debugRef.current.scrollHeight); }, [debugLogs]);

  const createProject = async () => {
    if (!newProjectName.trim()) return;
    if (!authUser) { setShowAuthModal(true); return; }
    try {
      const res = await fetch("/api/projects", {
        method: "POST", headers: authHeaders(),
        body: JSON.stringify({ name: newProjectName.trim() }),
      });
      if (!res.ok) return;
      const proj: Project = await res.json();
      setShowNewModal(false);
      setNewProjectName("");
      openProject(proj.id);
    } catch { /* silent */ }
  };

  const deleteProject = async (id: string) => {
    if (!confirm("이 프로젝트를 삭제하시겠습니까?")) return;
    await fetch(`/api/projects/${id}`, { method: "DELETE" });
    fetchProjects();
  };

  const resetStudioState = () => {
    setActiveTab("asr");
    setAsrFile(null); setSelectedExistingFile(null); setAsrStatus({ status: "idle", message: "" }); setTranscript(null); setCopied(false); setAsrSaved(false);
    setEditorText(""); setRewrittenText(""); setRewriteStatus({ status: "idle", message: "" });
    setText(""); setGen({ status: "idle", message: "", audioUrl: null, duration: null });
    setIsRecording(false); setRecorderElapsed(0); setRecordingFilename(""); setRecordingSaving(false); setRecordingSaved(false);
    if (audRef.current) audRef.current.pause();
    cancelAnimationFrame(audAnimRef.current);
    if (audBlobRef.current) { URL.revokeObjectURL(audBlobRef.current); audBlobRef.current = null; }
    setAudBuffer(null); setAudUrl(null); setAudPlaying(false); setAudTime(0);
    setAudSelection(null); audSelRef.current = null; setAudSaving(false); setAudSaveMsg("");
    audPeaksRef.current = [];
    // Download tab
    setDlUrl(""); setDlFilename(""); setDlStatus({ status: "idle", message: "" }); setDlLogs([]);
    setDlAudioUrl(null); setDlAudioFilename(null);
    if (dlAudRef.current) dlAudRef.current.pause();
    cancelAnimationFrame(dlAudAnimRef.current);
    setDlAudReady(false); setDlAudDuration(0); setDlAudPlaying(false); setDlAudTime(0);
    setDlAudSelection(null); dlAudSelRef.current = null; setDlAudSaving(false); setDlAudSaveMsg("");
    dlAudPeaksRef.current = [];
    // Poem Shorts
    setPsPoem(""); setPsAudioUrl(null); setPsAudioDuration(null);
    setPsAudioStatus({ status: "idle", message: "" });
    setPsImagePrompt(""); setPsImagePromptStatus({ status: "idle", message: "" });
    setPsImageUrl(null); setPsImageStatus({ status: "idle", message: "" });
    setPsVideoPrompt(""); setPsVideoPromptStatus({ status: "idle", message: "" });
    setPsVideoUrl(null); setPsVideoStatus({ status: "idle", message: "" });
  };

  const openProject = async (id: string) => {
    resetStudioState();
    setCurrentProjectId(id);
    setCurrentProjectName("");
    try {
      const res = await fetch(`/api/projects/${id}`);
      if (!res.ok) return;
      const proj: Project = await res.json();
      setCurrentProjectName(proj.name);
      if (proj.transcript_json) {
        try {
          const parsed = JSON.parse(proj.transcript_json);
          setTranscript(parsed as TranscriptResult);
          setAsrStatus({ status: "complete", message: "완료!" });
        } catch { /* skip */ }
      }
      if (proj.edited_transcript) {
        setEditorText(proj.edited_transcript);
      } else if (proj.transcript_text) {
        const lines = proj.transcript_text.split("\n").filter((l: string) => l.trim());
        setEditorText(lines.join("\n"));
      }
      if (proj.rewritten_text) {
        setRewrittenText(proj.rewritten_text);
        setRewriteStatus({ status: "complete", message: "변환 완료!" });
      }
      if (proj.llm_model) setSelectedModel(proj.llm_model);
      if (proj.num_speakers) setNumSpeakers(proj.num_speakers);
      if (proj.tts_text) {
        setText(proj.tts_text);
      }
      if (proj.tts_engine) {
        setTtsEngine(proj.tts_engine as "elevenlabs" | "qwen3");
      }
      if (proj.generated_audio_filename) {
        setGen({ status: "complete", message: "Done!", audioUrl: `/api/outputs/${proj.generated_audio_filename}`, duration: null });
      }
      // Poem Shorts
      if (proj.poem_text) setPsPoem(proj.poem_text);
      if (proj.poem_audio_filename) { setPsAudioUrl(`/api/outputs/${proj.poem_audio_filename}`); setPsAudioDuration(proj.poem_audio_duration || null); }
      if (proj.poem_image_prompt) setPsImagePrompt(proj.poem_image_prompt);
      if (proj.poem_image_filename) setPsImageUrl(`/api/infographics/${proj.poem_image_filename}`);
      if (proj.poem_video_prompt) setPsVideoPrompt(proj.poem_video_prompt);
      if (proj.poem_video_filename) setPsVideoUrl(`/api/videos/${proj.poem_video_filename}`);
      setActiveTab(proj.transcript_json ? "editor" : "asr");
    } catch { /* silent */ }
    setView("studio");
  };

  const goToLanding = () => {
    setView("landing");
    setCurrentProjectId(null);
    fetchProjects();
  };

  const patchProject = async (fields: Record<string, unknown>) => {
    if (!currentProjectId) return;
    try {
      await fetch(`/api/projects/${currentProjectId}`, {
        method: "PATCH", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(fields),
      });
    } catch { /* silent */ }
  };

  // Auto-save edited text to project with 1.5s debounce
  useEffect(() => {
    if (!currentProjectId || !editorText) return;
    setEditorSaved(false);
    if (editorSaveTimer.current) clearTimeout(editorSaveTimer.current);
    editorSaveTimer.current = setTimeout(async () => {
      await patchProject({ edited_transcript: editorText });
      setEditorSaved(true);
      setTimeout(() => setEditorSaved(false), 2000);
    }, 1500);
    return () => { if (editorSaveTimer.current) clearTimeout(editorSaveTimer.current); };
  }, [editorText, currentProjectId]);

  // Auto-save rewritten text to project with 1.5s debounce
  useEffect(() => {
    if (!currentProjectId || !rewrittenText) return;
    setRewrittenSaved(false);
    if (rewrittenSaveTimer.current) clearTimeout(rewrittenSaveTimer.current);
    rewrittenSaveTimer.current = setTimeout(async () => {
      await patchProject({ rewritten_text: rewrittenText });
      setRewrittenSaved(true);
      setTimeout(() => setRewrittenSaved(false), 2000);
    }, 1500);
    return () => { if (rewrittenSaveTimer.current) clearTimeout(rewrittenSaveTimer.current); };
  }, [rewrittenText, currentProjectId]);

  // Auto-save TTS text to project with 1.5s debounce
  useEffect(() => {
    if (!currentProjectId || !text) return;
    setTtsSaved(false);
    if (ttsSaveTimer.current) clearTimeout(ttsSaveTimer.current);
    ttsSaveTimer.current = setTimeout(async () => {
      await patchProject({ tts_text: text });
      setTtsSaved(true);
      setTimeout(() => setTtsSaved(false), 2000);
    }, 1500);
    return () => { if (ttsSaveTimer.current) clearTimeout(ttsSaveTimer.current); };
  }, [text, currentProjectId]);

  // Auto-save poem text to project with 1.5s debounce
  useEffect(() => {
    if (!currentProjectId || !psPoem) return;
    if (psPoemSaveTimer.current) clearTimeout(psPoemSaveTimer.current);
    psPoemSaveTimer.current = setTimeout(async () => {
      await patchProject({ poem_text: psPoem });
    }, 1500);
    return () => { if (psPoemSaveTimer.current) clearTimeout(psPoemSaveTimer.current); };
  }, [psPoem, currentProjectId]);

  /* ---- Poem Shorts handlers ---- */
  const handlePsGenerateAudio = async () => {
    if (!psPoem.trim()) return;
    psAbortRef.current?.abort();
    const controller = new AbortController();
    psAbortRef.current = controller;
    setPsAudioStatus({ status: "generating", message: "Qwen3-TTS로 시 낭독 생성 중..." });
    setPsAudioUrl(null); setPsAudioDuration(null);
    try {
      const res = await fetch("/api/poem-shorts/generate-audio", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ poem_text: psPoem, project_id: currentProjectId || "", voice_id: "upload-68582e2a-성우" }),
        signal: controller.signal,
      });
      if (!res.ok) { setPsAudioStatus({ status: "error", message: await res.text() }); return; }
      await readSSE(res, (evt) => {
        if (evt.status === "complete") {
          setPsAudioStatus({ status: "complete", message: `생성 완료 (${evt.generation_time}초, ${(evt.duration as number)?.toFixed(1)}초 오디오)` });
          setPsAudioUrl(evt.audio_url as string);
          setPsAudioDuration(evt.duration as number);
        } else if (evt.status === "error") {
          setPsAudioStatus({ status: "error", message: evt.message as string });
        } else if (evt.message) {
          setPsAudioStatus({ status: "generating", message: evt.message as string });
        }
      });
    } catch (err) {
      if ((err as Error).name !== "AbortError")
        setPsAudioStatus({ status: "error", message: (err as Error).message });
    }
  };

  const handlePsGenerateImagePrompt = async () => {
    if (!psPoem.trim()) return;
    setPsImagePromptStatus({ status: "generating", message: "이미지 프롬프트 생성 중..." });
    try {
      const res = await fetch("/api/poem-shorts/generate-image-prompt", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ poem_text: psPoem, project_id: currentProjectId || "", model: selectedModel }),
      });
      if (!res.ok) { setPsImagePromptStatus({ status: "error", message: await res.text() }); return; }
      await readSSE(res, (evt) => {
        if (evt.status === "complete") {
          setPsImagePrompt(evt.prompt as string);
          setPsImagePromptStatus({ status: "complete", message: `완료 (${evt.elapsed}초)` });
        } else if (evt.status === "error") {
          setPsImagePromptStatus({ status: "error", message: evt.message as string });
        } else if (evt.message) {
          setPsImagePromptStatus({ status: "generating", message: evt.message as string });
        }
      });
    } catch (err) {
      setPsImagePromptStatus({ status: "error", message: (err as Error).message });
    }
  };

  const handlePsGenerateImage = async () => {
    if (!psImagePrompt.trim()) return;
    setPsImageStatus({ status: "generating", message: "Gemini 2.5 Flash로 배경 이미지 생성 중..." });
    setPsImageUrl(null);
    try {
      const res = await fetch("/api/poem-shorts/generate-image", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: psImagePrompt, project_id: currentProjectId || "" }),
      });
      if (!res.ok) { setPsImageStatus({ status: "error", message: await res.text() }); return; }
      await readSSE(res, (evt) => {
        if (evt.status === "complete") {
          setPsImageUrl(evt.image_url as string);
          setPsImageStatus({ status: "complete", message: `완료 (${evt.model || "unknown"}, ${evt.elapsed}초, ${((evt.size as number) / 1024).toFixed(0)}KB)` });
        } else if (evt.status === "error") {
          setPsImageStatus({ status: "error", message: evt.message as string });
        } else if (evt.message) {
          setPsImageStatus({ status: "generating", message: evt.message as string });
        }
      });
    } catch (err) {
      setPsImageStatus({ status: "error", message: (err as Error).message });
    }
  };

  const handlePsGenerateVideoPrompt = async () => {
    if (!psPoem.trim()) return;
    setPsVideoPromptStatus({ status: "generating", message: "영상 프롬프트 생성 중..." });
    try {
      const res = await fetch("/api/poem-shorts/generate-video-prompt", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ poem_text: psPoem, image_prompt: psImagePrompt, project_id: currentProjectId || "", model: selectedModel }),
      });
      if (!res.ok) { setPsVideoPromptStatus({ status: "error", message: await res.text() }); return; }
      await readSSE(res, (evt) => {
        if (evt.status === "complete") {
          setPsVideoPrompt(evt.prompt as string);
          setPsVideoPromptStatus({ status: "complete", message: `완료 (${evt.elapsed}초)` });
        } else if (evt.status === "error") {
          setPsVideoPromptStatus({ status: "error", message: evt.message as string });
        } else if (evt.message) {
          setPsVideoPromptStatus({ status: "generating", message: evt.message as string });
        }
      });
    } catch (err) {
      setPsVideoPromptStatus({ status: "error", message: (err as Error).message });
    }
  };

  const handlePsGenerateVideo = async () => {
    if (!currentProjectId) return;
    setPsVideoStatus({ status: "generating", message: "영상 합성 중 (ffmpeg)..." });
    setPsVideoUrl(null);
    try {
      const res = await fetch("/api/poem-shorts/generate-video", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project_id: currentProjectId }),
      });
      if (!res.ok) { setPsVideoStatus({ status: "error", message: await res.text() }); return; }
      await readSSE(res, (evt) => {
        if (evt.status === "complete") {
          setPsVideoUrl(evt.video_url as string);
          setPsVideoStatus({ status: "complete", message: `완료 (${evt.elapsed}초, ${((evt.file_size as number) / 1024 / 1024).toFixed(1)}MB)` });
        } else if (evt.status === "error") {
          setPsVideoStatus({ status: "error", message: evt.message as string });
        } else if (evt.message) {
          setPsVideoStatus({ status: "generating", message: evt.message as string });
        }
      });
    } catch (err) {
      setPsVideoStatus({ status: "error", message: (err as Error).message });
    }
  };

  // Load source data when switching to source tab
  useEffect(() => {
    if (activeTab !== "source" || !currentProjectId) return;
    (async () => {
      try {
        const res = await fetch(`/api/projects/${currentProjectId}`);
        if (!res.ok) return;
        const proj: Project = await res.json();
        setSourceProject(proj);
        setSrcTranscript(proj.transcript_text || "");
        setSrcEdited(proj.edited_transcript || "");
        setSrcRewritten(proj.rewritten_text || "");
      } catch { /* silent */ }
    })();
    (async () => {
      try {
        const res = await fetch(`/api/projects/${currentProjectId}/audio-files`);
        if (res.ok) setSrcAudioFiles(await res.json());
      } catch { /* silent */ }
    })();
    (async () => {
      try {
        const res = await fetch(`/api/projects/${currentProjectId}/artifacts`);
        if (res.ok) setSrcArtifacts(await res.json());
      } catch { /* silent */ }
    })();
  }, [activeTab, currentProjectId]);

  const saveSourceField = async (field: string, value: string) => {
    if (!currentProjectId) return;
    setSrcSaving(field);
    await patchProject({ [field]: value });
    setSrcSaving(null);
  };

  const toggleExpanded = (set: Set<string>, id: string, setter: (s: Set<string>) => void) => {
    const next = new Set(set);
    if (next.has(id)) next.delete(id); else next.add(id);
    setter(next);
  };

  /* ================================================================ */
  /*  Audio file list                                                   */
  /* ================================================================ */

  const fetchAudioFiles = async () => {
    try {
      if (currentProjectId) {
        const res = await fetch(`/api/projects/${currentProjectId}/audio-files`);
        if (res.ok) {
          const data = await res.json();
          setAudioFiles(data.map((f: any) => ({ ...f, size: f.file_size ?? f.size, modified: f.created_at ?? f.modified })));
        }
      } else {
        const res = await fetch("/api/audio-files");
        if (res.ok) setAudioFiles(await res.json());
      }
    } catch { /* silent */ }
    setShowAudioList(true);
  };

  const selectExistingAudio = (filename: string) => {
    setShowAudioList(false);
    setSelectedExistingFile(filename);
    setAsrFile(null);
    setAsrStatus({ status: "idle", message: "" });
    setTranscript(null); setCopied(false); setAsrSaved(false);
  };

  /* ================================================================ */
  /*  Recorder logic                                                   */
  /* ================================================================ */

  useEffect(() => {
    if (activeTab === "recorder" && !recordingFilename && currentProjectId) {
      const proj = projects.find((p) => p.id === currentProjectId);
      const now = new Date();
      const ts = `${now.getFullYear()}${(now.getMonth() + 1).toString().padStart(2, "0")}${now.getDate().toString().padStart(2, "0")}_${now.getHours().toString().padStart(2, "0")}${now.getMinutes().toString().padStart(2, "0")}`;
      setRecordingFilename(`${proj?.name || "recording"}_${ts}.webm`);
    }
  }, [activeTab, currentProjectId]);

  useEffect(() => {
    const canvas = recorderCanvasRef.current;
    if (!canvas || activeTab !== "recorder" || isRecording) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "#1e2939";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#364153";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
  }, [activeTab, isRecording]);

  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stream.getTracks().forEach((t) => t.stop());
        mediaRecorderRef.current.stop();
      }
      if (recorderTimerRef.current) clearInterval(recorderTimerRef.current);
      cancelAnimationFrame(animFrameRef.current);
      audioCtxRef.current?.close();
    };
  }, []);

  const startRecording = async () => {
    resetAudEditor();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      const audioCtx = new AudioContext();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      audioCtxRef.current = audioCtx;
      analyserRef.current = analyser;

      let opts: MediaRecorderOptions = {};
      if (typeof MediaRecorder.isTypeSupported === "function") {
        if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) opts = { mimeType: "audio/webm;codecs=opus" };
        else if (MediaRecorder.isTypeSupported("audio/webm")) opts = { mimeType: "audio/webm" };
      }
      const recorder = new MediaRecorder(stream, opts);
      audioChunksRef.current = [];
      recorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunksRef.current.push(e.data); };
      mediaRecorderRef.current = recorder;
      recorder.start(100);

      setIsRecording(true);
      setRecorderElapsed(0);
      setRecordingSaved(false);

      const proj = projects.find((p) => p.id === currentProjectId);
      const now = new Date();
      const ts = `${now.getFullYear()}${(now.getMonth() + 1).toString().padStart(2, "0")}${now.getDate().toString().padStart(2, "0")}_${now.getHours().toString().padStart(2, "0")}${now.getMinutes().toString().padStart(2, "0")}`;
      setRecordingFilename(`${proj?.name || "recording"}_${ts}.webm`);

      const startTime = Date.now();
      recorderTimerRef.current = setInterval(() => {
        setRecorderElapsed(Math.floor((Date.now() - startTime) / 1000));
      }, 200);

      const canvas = recorderCanvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        const bufLen = analyser.frequencyBinCount;
        const dataArr = new Uint8Array(bufLen);
        const draw = () => {
          animFrameRef.current = requestAnimationFrame(draw);
          analyser.getByteTimeDomainData(dataArr);
          const w = canvas.width, h = canvas.height;
          if (ctx) {
            ctx.fillStyle = "#1e2939";
            ctx.fillRect(0, 0, w, h);
            ctx.strokeStyle = "#364153";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, h / 2);
            ctx.lineTo(w, h / 2);
            ctx.stroke();
            ctx.lineWidth = 2;
            ctx.strokeStyle = "#8b5cf6";
            ctx.beginPath();
            const sliceW = w / bufLen;
            let x = 0;
            for (let i = 0; i < bufLen; i++) {
              const v = dataArr[i] / 128.0;
              const y = (v * h) / 2;
              if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
              x += sliceW;
            }
            ctx.lineTo(w, h / 2);
            ctx.stroke();
          }
        };
        draw();
      }
    } catch {
      alert("마이크 접근 권한이 필요합니다.");
    }
  };

  const stopRecording = () => {
    const recorder = mediaRecorderRef.current;
    if (!recorder || recorder.state === "inactive") return;

    const filename = recordingFilename || "recording.webm";
    const projectId = currentProjectId;

    recorder.onstop = async () => {
      if (recorderTimerRef.current) { clearInterval(recorderTimerRef.current); recorderTimerRef.current = null; }
      cancelAnimationFrame(animFrameRef.current);
      recorder.stream.getTracks().forEach((t) => t.stop());
      audioCtxRef.current?.close();
      audioCtxRef.current = null;
      analyserRef.current = null;
      setIsRecording(false);

      const blob = new Blob(audioChunksRef.current, { type: recorder.mimeType || "audio/webm" });
      if (blob.size > 0 && projectId) {
        setRecordingSaving(true);
        try {
          const form = new FormData();
          form.append("file", blob, filename);
          form.append("filename", filename);
          const res = await fetch(`/api/projects/${projectId}/upload-audio`, { method: "POST", body: form });
          if (res.ok) setRecordingSaved(true);
        } catch { /* silent */ }
        finally { setRecordingSaving(false); }

        try {
          const ab = await blob.arrayBuffer();
          const actx = new AudioContext();
          const buf = await actx.decodeAudioData(ab);
          actx.close();
          loadAud(buf, blob);
        } catch { /* silent */ }
      }
    };
    recorder.stop();
  };

  /* ================================================================ */
  /*  Audio Editor logic                                               */
  /* ================================================================ */

  const loadAud = (buffer: AudioBuffer, sourceBlob?: Blob) => {
    if (audRef.current) audRef.current.pause();
    cancelAnimationFrame(audAnimRef.current);

    const data = buffer.getChannelData(0);
    const numPeaks = 500;
    const blockSize = Math.max(1, Math.floor(data.length / numPeaks));
    const peaks: number[] = [];
    for (let i = 0; i < numPeaks && i * blockSize < data.length; i++) {
      let max = 0;
      for (let j = 0; j < blockSize && i * blockSize + j < data.length; j++) {
        const abs = Math.abs(data[i * blockSize + j]);
        if (abs > max) max = abs;
      }
      peaks.push(max);
    }
    audPeaksRef.current = peaks;
    setAudBuffer(buffer);
    setAudTime(0);
    setAudPlaying(false);
    setAudSelection(null);
    audSelRef.current = null;
    setAudSaveMsg("");

    const blob = sourceBlob || audioBufferToWav(buffer);
    if (audBlobRef.current) URL.revokeObjectURL(audBlobRef.current);
    const url = URL.createObjectURL(blob);
    audBlobRef.current = url;
    setAudUrl(url);
  };

  const drawAudCanvas = () => {
    const canvas = audCanvasRef.current;
    const audio = audRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const w = canvas.width, h = canvas.height;
    const peaks = audPeaksRef.current;
    const dur = audio?.duration || 0;
    const curTime = audio ? audio.currentTime : 0;
    const sel = audSelRef.current;

    ctx.fillStyle = "#1e2939";
    ctx.fillRect(0, 0, w, h);
    if (peaks.length === 0 || dur === 0) return;

    if (sel) {
      const s = Math.min(sel[0], sel[1]), e = Math.max(sel[0], sel[1]);
      const x1 = (s / dur) * w, x2 = (e / dur) * w;
      ctx.fillStyle = "rgba(139, 92, 246, 0.2)";
      ctx.fillRect(x1, 0, x2 - x1, h);
      ctx.strokeStyle = "#8b5cf6";
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x1, 0); ctx.lineTo(x1, h); ctx.moveTo(x2, 0); ctx.lineTo(x2, h); ctx.stroke();
    }

    const barW = w / peaks.length;
    for (let i = 0; i < peaks.length; i++) {
      const barH = Math.max(1, peaks[i] * h * 0.85);
      const peakT = (i / peaks.length) * dur;
      ctx.fillStyle = peakT <= curTime ? "#8b5cf6" : "#364153";
      ctx.fillRect(i * barW, (h - barH) / 2, Math.max(1, barW - 0.5), barH);
    }

    ctx.strokeStyle = "#364153"; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();

    const cx = (curTime / dur) * w;
    ctx.strokeStyle = "#ef4444"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, h); ctx.stroke();
    ctx.fillStyle = "#ef4444"; ctx.beginPath(); ctx.arc(cx, 4, 4, 0, Math.PI * 2); ctx.fill();
  };

  useEffect(() => { if (audBuffer) drawAudCanvas(); }, [audBuffer, audTime, audSelection]);

  useEffect(() => {
    const handleMove = (e: MouseEvent) => {
      if (!audDragRef.current) return;
      const canvas = audCanvasRef.current;
      const audio = audRef.current;
      if (!canvas || !audio || !audio.duration) return;
      const rect = canvas.getBoundingClientRect();
      const x = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
      const t = (x / rect.width) * audio.duration;
      audSelRef.current = audSelRef.current ? [audSelRef.current[0], t] : null;
      setAudSelection(audSelRef.current);
      drawAudCanvas();
    };
    const handleUp = () => {
      if (!audDragRef.current) return;
      audDragRef.current = false;
      const sel = audSelRef.current;
      if (sel && Math.abs(sel[1] - sel[0]) < 0.05) {
        const audio = audRef.current;
        if (audio) { audio.currentTime = sel[0]; setAudTime(sel[0]); }
        audSelRef.current = null;
        setAudSelection(null);
      }
      drawAudCanvas();
    };
    document.addEventListener("mousemove", handleMove);
    document.addEventListener("mouseup", handleUp);
    return () => { document.removeEventListener("mousemove", handleMove); document.removeEventListener("mouseup", handleUp); };
  }, []);

  const onAudMouseDown = (e: React.MouseEvent) => {
    const canvas = audCanvasRef.current;
    const audio = audRef.current;
    if (!canvas || !audio || !audio.duration) return;
    const rect = canvas.getBoundingClientRect();
    const t = ((e.clientX - rect.left) / rect.width) * audio.duration;
    audDragRef.current = true;
    audSelRef.current = [t, t];
    setAudSelection([t, t]);
  };

  const audTick = () => {
    const audio = audRef.current;
    if (!audio || audio.paused || audio.ended) {
      if (audio?.ended) { setAudPlaying(false); setAudTime(0); audio.currentTime = 0; }
      return;
    }
    const now = performance.now();
    if (now - audUpdateRef.current > 80) { setAudTime(audio.currentTime); audUpdateRef.current = now; }
    drawAudCanvas();
    audAnimRef.current = requestAnimationFrame(audTick);
  };

  const audPlay = () => {
    const audio = audRef.current;
    if (!audio || !audBuffer) return;
    audio.play();
    setAudPlaying(true);
    audAnimRef.current = requestAnimationFrame(audTick);
  };

  const audPause = () => {
    audRef.current?.pause();
    cancelAnimationFrame(audAnimRef.current);
    setAudPlaying(false);
    if (audRef.current) setAudTime(audRef.current.currentTime);
    drawAudCanvas();
  };

  const audStop = () => {
    const audio = audRef.current;
    if (audio) { audio.pause(); audio.currentTime = 0; }
    cancelAnimationFrame(audAnimRef.current);
    setAudPlaying(false);
    setAudTime(0);
    drawAudCanvas();
  };

  const audSeek = (delta: number) => {
    const audio = audRef.current;
    if (!audio) return;
    audio.currentTime = Math.max(0, Math.min(audio.duration || 0, audio.currentTime + delta));
    setAudTime(audio.currentTime);
    drawAudCanvas();
  };

  const audCut = () => {
    if (!audBuffer || !audSelection) return;
    const [a, b] = audSelection;
    const s = Math.min(a, b), e = Math.max(a, b);
    const sr = audBuffer.sampleRate, nCh = audBuffer.numberOfChannels;
    const ss = Math.floor(s * sr), es = Math.floor(e * sr);
    const newLen = audBuffer.length - (es - ss);
    if (newLen <= 0) return;
    const actx = new OfflineAudioContext(nCh, newLen, sr);
    const nb = actx.createBuffer(nCh, newLen, sr);
    for (let c = 0; c < nCh; c++) {
      const od = audBuffer.getChannelData(c), nd = nb.getChannelData(c);
      nd.set(od.subarray(0, ss));
      nd.set(od.subarray(es), ss);
    }
    loadAud(nb);
    if (audRef.current) audRef.current.currentTime = Math.min(s, nb.duration);
    setAudTime(Math.min(s, nb.duration));
  };

  const audTrim = () => {
    if (!audBuffer || !audSelection) return;
    const [a, b] = audSelection;
    const s = Math.min(a, b), e = Math.max(a, b);
    const sr = audBuffer.sampleRate, nCh = audBuffer.numberOfChannels;
    const ss = Math.floor(s * sr), es = Math.floor(e * sr);
    const newLen = es - ss;
    if (newLen <= 0) return;
    const actx = new OfflineAudioContext(nCh, newLen, sr);
    const nb = actx.createBuffer(nCh, newLen, sr);
    for (let c = 0; c < nCh; c++) {
      nb.getChannelData(c).set(audBuffer.getChannelData(c).subarray(ss, es));
    }
    loadAud(nb);
    setAudTime(0);
  };

  const audSave = async () => {
    if (!audBuffer || !currentProjectId) return;
    setAudSaving(true);
    try {
      const wav = audioBufferToWav(audBuffer);
      const fname = recordingFilename.replace(/\.\w+$/, ".wav");
      const form = new FormData();
      form.append("file", wav, fname);
      form.append("filename", fname);
      const res = await fetch(`/api/projects/${currentProjectId}/upload-audio`, { method: "POST", body: form });
      if (res.ok) setAudSaveMsg("저장 완료!");
    } catch { setAudSaveMsg("저장 실패"); }
    finally { setAudSaving(false); setTimeout(() => setAudSaveMsg(""), 3000); }
  };

  const audSaveAs = async () => {
    if (!audBuffer || !currentProjectId) return;
    const defName = recordingFilename.replace(/\.\w+$/, "_edited.wav");
    const newName = prompt("파일명을 입력하세요:", defName);
    if (!newName) return;
    setAudSaving(true);
    try {
      const wav = audioBufferToWav(audBuffer);
      const form = new FormData();
      form.append("file", wav, newName);
      form.append("filename", newName);
      const res = await fetch(`/api/projects/${currentProjectId}/upload-audio`, { method: "POST", body: form });
      if (res.ok) { setRecordingFilename(newName); setAudSaveMsg(`"${newName}" 저장 완료!`); }
    } catch { setAudSaveMsg("저장 실패"); }
    finally { setAudSaving(false); setTimeout(() => setAudSaveMsg(""), 3000); }
  };

  const resetAudEditor = () => {
    if (audRef.current) audRef.current.pause();
    cancelAnimationFrame(audAnimRef.current);
    if (audBlobRef.current) { URL.revokeObjectURL(audBlobRef.current); audBlobRef.current = null; }
    setAudBuffer(null); setAudUrl(null); setAudPlaying(false); setAudTime(0);
    setAudSelection(null); audSelRef.current = null; setAudSaveMsg("");
    audPeaksRef.current = [];
  };

  /* ================================================================ */
  /*  Download Audio Editor logic                                      */
  /* ================================================================ */

  useEffect(() => { dlLogRef.current?.scrollTo(0, dlLogRef.current.scrollHeight); }, [dlLogs]);

  const addDlLog = useCallback((msg: string) => {
    const ts = new Date().toLocaleTimeString("en-GB", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit", fractionalSecondDigits: 3 });
    setDlLogs((prev) => [...prev, `[${ts}] ${msg}`]);
  }, []);

  const dlLoadPeaks = async (filename: string) => {
    if (dlAudRef.current) dlAudRef.current.pause();
    cancelAnimationFrame(dlAudAnimRef.current);
    setDlAudTime(0);
    setDlAudPlaying(false);
    setDlAudSelection(null);
    dlAudSelRef.current = null;
    setDlAudSaveMsg("");
    dlAudPeaksRef.current = [];
    setDlAudReady(false);

    addDlLog("파형 데이터 로딩 중...");
    try {
      const res = await fetch(`/api/audio-peaks/${encodeURIComponent(filename)}?num_peaks=800`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      dlAudPeaksRef.current = data.peaks;
      setDlAudDuration(data.duration);
      setDlAudReady(true);
      addDlLog(`파형 로딩 완료 (${Math.floor(data.duration / 60)}분 ${Math.floor(data.duration % 60)}초)`);
    } catch (e) {
      addDlLog(`파형 로딩 실패: ${e}`);
    }
  };

  const drawDlAudCanvas = useCallback(() => {
    const canvas = dlAudCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const w = canvas.width, h = canvas.height;
    const peaks = dlAudPeaksRef.current;
    const dur = dlAudDuration || (dlAudRef.current?.duration || 0);
    const curTime = dlAudRef.current ? dlAudRef.current.currentTime : 0;
    const sel = dlAudSelRef.current;
    ctx.fillStyle = "#1e2939";
    ctx.fillRect(0, 0, w, h);
    if (peaks.length === 0 || dur === 0) return;
    if (sel) {
      const s = Math.min(sel[0], sel[1]), e = Math.max(sel[0], sel[1]);
      const x1 = (s / dur) * w, x2 = (e / dur) * w;
      ctx.fillStyle = "rgba(139, 92, 246, 0.2)";
      ctx.fillRect(x1, 0, x2 - x1, h);
      ctx.strokeStyle = "#8b5cf6";
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x1, 0); ctx.lineTo(x1, h); ctx.moveTo(x2, 0); ctx.lineTo(x2, h); ctx.stroke();
    }
    const barW = w / peaks.length;
    for (let i = 0; i < peaks.length; i++) {
      const barH = Math.max(1, peaks[i] * h * 0.85);
      const peakT = (i / peaks.length) * dur;
      ctx.fillStyle = peakT <= curTime ? "#8b5cf6" : "#364153";
      ctx.fillRect(i * barW, (h - barH) / 2, Math.max(1, barW - 0.5), barH);
    }
    ctx.strokeStyle = "#364153"; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
    const cx = (curTime / dur) * w;
    ctx.strokeStyle = "#ef4444"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, h); ctx.stroke();
    ctx.fillStyle = "#ef4444"; ctx.beginPath(); ctx.arc(cx, 4, 4, 0, Math.PI * 2); ctx.fill();
  }, [dlAudDuration]);

  useEffect(() => { if (dlAudReady) drawDlAudCanvas(); }, [dlAudReady, dlAudTime, dlAudSelection, drawDlAudCanvas]);

  useEffect(() => {
    const handleMove = (e: MouseEvent) => {
      if (!dlAudDragRef.current) return;
      const canvas = dlAudCanvasRef.current;
      if (!canvas) return;
      const dur = dlAudDuration || (dlAudRef.current?.duration || 0);
      if (!dur) return;
      const rect = canvas.getBoundingClientRect();
      const x = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
      const t = (x / rect.width) * dur;
      dlAudSelRef.current = dlAudSelRef.current ? [dlAudSelRef.current[0], t] : null;
      setDlAudSelection(dlAudSelRef.current);
      drawDlAudCanvas();
    };
    const handleUp = () => {
      if (!dlAudDragRef.current) return;
      dlAudDragRef.current = false;
      const sel = dlAudSelRef.current;
      const dur = dlAudDuration || (dlAudRef.current?.duration || 0);
      if (sel && Math.abs(sel[1] - sel[0]) < 0.05 * (dur / 100 || 1)) {
        const audio = dlAudRef.current;
        if (audio) { audio.currentTime = sel[0]; setDlAudTime(sel[0]); }
        dlAudSelRef.current = null;
        setDlAudSelection(null);
      }
      drawDlAudCanvas();
    };
    document.addEventListener("mousemove", handleMove);
    document.addEventListener("mouseup", handleUp);
    return () => { document.removeEventListener("mousemove", handleMove); document.removeEventListener("mouseup", handleUp); };
  }, [dlAudDuration, drawDlAudCanvas]);

  const onDlAudMouseDown = (e: React.MouseEvent) => {
    const canvas = dlAudCanvasRef.current;
    if (!canvas) return;
    const dur = dlAudDuration || (dlAudRef.current?.duration || 0);
    if (!dur) return;
    const rect = canvas.getBoundingClientRect();
    const t = ((e.clientX - rect.left) / rect.width) * dur;
    dlAudDragRef.current = true;
    dlAudSelRef.current = [t, t];
    setDlAudSelection([t, t]);
    const audio = dlAudRef.current;
    if (audio) { audio.currentTime = t; setDlAudTime(t); }
  };

  const dlAudTick = () => {
    const audio = dlAudRef.current;
    if (!audio || audio.paused || audio.ended) {
      if (audio?.ended) { setDlAudPlaying(false); setDlAudTime(0); audio.currentTime = 0; }
      return;
    }
    const now = performance.now();
    if (now - dlAudUpdateRef.current > 80) { setDlAudTime(audio.currentTime); dlAudUpdateRef.current = now; }
    drawDlAudCanvas();
    dlAudAnimRef.current = requestAnimationFrame(dlAudTick);
  };

  const dlAudPlay = () => {
    const audio = dlAudRef.current;
    if (!audio || !dlAudReady) return;
    audio.play();
    setDlAudPlaying(true);
    dlAudAnimRef.current = requestAnimationFrame(dlAudTick);
  };

  const dlAudPause = () => {
    dlAudRef.current?.pause();
    cancelAnimationFrame(dlAudAnimRef.current);
    setDlAudPlaying(false);
    if (dlAudRef.current) setDlAudTime(dlAudRef.current.currentTime);
    drawDlAudCanvas();
  };

  const dlAudStop = () => {
    const audio = dlAudRef.current;
    if (audio) { audio.pause(); audio.currentTime = 0; }
    cancelAnimationFrame(dlAudAnimRef.current);
    setDlAudPlaying(false);
    setDlAudTime(0);
    drawDlAudCanvas();
  };

  const dlAudSeek = (delta: number) => {
    const audio = dlAudRef.current;
    if (!audio) return;
    audio.currentTime = Math.max(0, Math.min(audio.duration || 0, audio.currentTime + delta));
    setDlAudTime(audio.currentTime);
    drawDlAudCanvas();
  };

  const dlAudSaveClip = async () => {
    if (!dlAudSelection || !dlAudioFilename) return;
    const [a, b] = dlAudSelection;
    const s = Math.min(a, b), e = Math.max(a, b);
    if (e - s < 0.1) return;
    const defName = dlAudioFilename.replace(/\.\w+$/, `_${fmtAudTime(s).replace(":", "m")}s-${fmtAudTime(e).replace(":", "m")}s.wav`);
    const newName = prompt("저장할 파일명을 입력하세요:", defName);
    if (!newName) return;
    setDlAudSaving(true);
    addDlLog(`클립 추출 중: ${fmtAudTime(s)} ~ ${fmtAudTime(e)} → ${newName}`);
    try {
      const form = new FormData();
      form.append("source", dlAudioFilename);
      form.append("start", s.toString());
      form.append("end", e.toString());
      form.append("output_name", newName);
      const res = await fetch("/api/audio-clip", { method: "POST", body: form });
      if (res.ok) {
        const data = await res.json();
        setDlAudSaveMsg(`"${data.filename}" 저장 완료!`);
        addDlLog(`클립 저장 완료: ${data.filename} (${(data.size / 1024 / 1024).toFixed(1)} MB)`);
      } else {
        const err = await res.text();
        setDlAudSaveMsg("저장 실패");
        addDlLog(`클립 저장 실패: ${err}`);
      }
    } catch { setDlAudSaveMsg("저장 실패"); }
    finally { setDlAudSaving(false); setTimeout(() => setDlAudSaveMsg(""), 5000); }
  };

  const resetDlAudEditor = () => {
    if (dlAudRef.current) dlAudRef.current.pause();
    cancelAnimationFrame(dlAudAnimRef.current);
    setDlAudReady(false); setDlAudDuration(0); setDlAudPlaying(false); setDlAudTime(0);
    setDlAudSelection(null); dlAudSelRef.current = null; setDlAudSaveMsg("");
    dlAudPeaksRef.current = [];
  };

  const startDownload = async () => {
    if (!dlUrl.trim()) return;
    setDlStatus({ status: "downloading", message: "시작 중..." });
    setDlLogs([]);
    setDlAudioUrl(null);
    setDlAudioFilename(null);
    resetDlAudEditor();
    addDlLog(`다운로드 시작: ${dlUrl}`);

    try {
      const ctrl = new AbortController();
      dlAbortRef.current = ctrl;
      const res = await fetch("/api/download-audio", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: dlUrl, filename: dlFilename || null, project_id: currentProjectId || null }),
        signal: ctrl.signal,
      });
      if (!res.ok) {
        const err = await res.text();
        setDlStatus({ status: "error", message: err });
        addDlLog(`오류: ${err}`);
        return;
      }
      await readSSE(res, (evt) => {
        const status = evt.status as string;
        const message = evt.message as string;
        addDlLog(message);
        setDlStatus({ status, message });
        if (status === "complete") {
          const audioUrl = evt.audio_url as string;
          const filename = evt.filename as string;
          setDlAudioUrl(audioUrl);
          setDlAudioFilename(filename);
          addDlLog(`파일 저장됨: ${filename}`);
          dlLoadPeaks(filename);
        }
      });
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        setDlStatus({ status: "error", message: String(e) });
        addDlLog(`오류: ${e}`);
      }
    }
  };

  /* ================================================================ */
  /*  Voice Clip Editor Modal logic                                     */
  /* ================================================================ */

  useEffect(() => { clipLogRef.current?.scrollTo(0, clipLogRef.current.scrollHeight); }, [clipLogs]);

  const addClipLog = useCallback((msg: string) => {
    const ts = new Date().toLocaleTimeString("en-GB", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit", fractionalSecondDigits: 3 });
    setClipLogs((prev) => [...prev, `[${ts}] ${msg}`]);
  }, []);

  const loadClipFile = async (filename: string) => {
    const audioUrl = `/api/audio-files/${encodeURIComponent(filename)}`;
    setClipModal({ filename, audioUrl });
    setClipReady(false); setClipDuration(0); setClipTime(0); setClipPlaying(false);
    setClipSelection(null); clipSelRef.current = null;
    setClipSaveMsg(""); setClipSaving(false); setClipCropName("");
    clipPeaksRef.current = [];
    if (clipAudioRef.current) { clipAudioRef.current.pause(); clipAudioRef.current.src = audioUrl; }

    addClipLog(`파일 로딩: ${filename}`);
    try {
      const res = await fetch(`/api/audio-peaks/${encodeURIComponent(filename)}?num_peaks=800`);
      if (!res.ok) { addClipLog(`파형 로딩 실패 (HTTP ${res.status})`); return; }
      const data = await res.json();
      clipPeaksRef.current = data.peaks;
      setClipDuration(data.duration);
      setClipReady(true);
      const m = Math.floor(data.duration / 60), s = Math.floor(data.duration % 60);
      addClipLog(`파형 로딩 완료: ${m}분 ${s}초, ${data.peaks.length}개 피크`);
    } catch (e) { addClipLog(`파형 로딩 오류: ${e}`); }
  };

  const openClipModal = async (file: AudioFileInfo) => {
    setClipVoiceName(""); setClipRefText(""); setClipRegistering(false);
    setClipHistory([]);
    setClipLogs([]);
    await loadClipFile(file.filename);
  };

  const closeClipModal = () => {
    if (clipAudioRef.current) clipAudioRef.current.pause();
    cancelAnimationFrame(clipAnimRef.current);
    setClipModal(null);
    setClipReady(false);
  };

  const drawClipCanvas = useCallback(() => {
    const canvas = clipCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const w = canvas.width, h = canvas.height;
    const peaks = clipPeaksRef.current;
    const dur = clipDuration || (clipAudioRef.current?.duration || 0);
    const curTime = clipAudioRef.current ? clipAudioRef.current.currentTime : 0;
    const sel = clipSelRef.current;
    ctx.fillStyle = "#1e2939";
    ctx.fillRect(0, 0, w, h);
    if (peaks.length === 0 || dur === 0) return;
    if (sel) {
      const s = Math.min(sel[0], sel[1]), e = Math.max(sel[0], sel[1]);
      const x1 = (s / dur) * w, x2 = (e / dur) * w;
      ctx.fillStyle = "rgba(139, 92, 246, 0.2)";
      ctx.fillRect(x1, 0, x2 - x1, h);
      ctx.strokeStyle = "#8b5cf6"; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x1, 0); ctx.lineTo(x1, h); ctx.moveTo(x2, 0); ctx.lineTo(x2, h); ctx.stroke();
    }
    const barW = w / peaks.length;
    for (let i = 0; i < peaks.length; i++) {
      const barH = Math.max(1, peaks[i] * h * 0.85);
      const peakT = (i / peaks.length) * dur;
      ctx.fillStyle = peakT <= curTime ? "#8b5cf6" : "#364153";
      ctx.fillRect(i * barW, (h - barH) / 2, Math.max(1, barW - 0.5), barH);
    }
    ctx.strokeStyle = "#364153"; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
    const cx = (curTime / dur) * w;
    ctx.strokeStyle = "#ef4444"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, h); ctx.stroke();
    ctx.fillStyle = "#ef4444"; ctx.beginPath(); ctx.arc(cx, 4, 4, 0, Math.PI * 2); ctx.fill();
  }, [clipDuration]);

  useEffect(() => { if (clipReady) drawClipCanvas(); }, [clipReady, clipTime, clipSelection, drawClipCanvas]);

  useEffect(() => {
    const handleMove = (e: MouseEvent) => {
      if (!clipDragRef.current) return;
      const canvas = clipCanvasRef.current;
      if (!canvas) return;
      const dur = clipDuration || (clipAudioRef.current?.duration || 0);
      if (!dur) return;
      const rect = canvas.getBoundingClientRect();
      const x = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
      const t = (x / rect.width) * dur;
      clipSelRef.current = clipSelRef.current ? [clipSelRef.current[0], t] : null;
      setClipSelection(clipSelRef.current);
      drawClipCanvas();
    };
    const handleUp = () => {
      if (!clipDragRef.current) return;
      clipDragRef.current = false;
      const sel = clipSelRef.current;
      const dur = clipDuration || (clipAudioRef.current?.duration || 0);
      if (sel && Math.abs(sel[1] - sel[0]) < 0.05 * (dur / 100 || 1)) {
        const audio = clipAudioRef.current;
        if (audio) { audio.currentTime = sel[0]; setClipTime(sel[0]); }
        clipSelRef.current = null;
        setClipSelection(null);
      } else if (sel && Math.abs(sel[1] - sel[0]) >= 0.3) {
        const s = Math.min(sel[0], sel[1]), e = Math.max(sel[0], sel[1]);
        const base = (clipModal?.filename || "clip").replace(/\.\w+$/, "");
        setClipCropName(`${base}_${fmtAudTime(s).replace(":", "m")}s-${fmtAudTime(e).replace(":", "m")}s`);
      }
      drawClipCanvas();
    };
    document.addEventListener("mousemove", handleMove);
    document.addEventListener("mouseup", handleUp);
    return () => { document.removeEventListener("mousemove", handleMove); document.removeEventListener("mouseup", handleUp); };
  }, [clipDuration, drawClipCanvas]);

  const onClipMouseDown = (e: React.MouseEvent) => {
    const canvas = clipCanvasRef.current;
    if (!canvas) return;
    const dur = clipDuration || (clipAudioRef.current?.duration || 0);
    if (!dur) return;
    const rect = canvas.getBoundingClientRect();
    const t = ((e.clientX - rect.left) / rect.width) * dur;
    clipDragRef.current = true;
    clipSelRef.current = [t, t];
    setClipSelection([t, t]);
    const audio = clipAudioRef.current;
    if (audio) { audio.currentTime = t; setClipTime(t); }
  };

  const clipTick = () => {
    const audio = clipAudioRef.current;
    if (!audio || audio.paused || audio.ended) {
      if (audio?.ended) { setClipPlaying(false); setClipTime(0); audio.currentTime = 0; }
      return;
    }
    const now = performance.now();
    if (now - clipUpdateRef.current > 80) { setClipTime(audio.currentTime); clipUpdateRef.current = now; }
    drawClipCanvas();
    clipAnimRef.current = requestAnimationFrame(clipTick);
  };

  const clipPlay = () => {
    const audio = clipAudioRef.current;
    if (!audio || !clipReady) return;
    audio.play(); setClipPlaying(true);
    clipAnimRef.current = requestAnimationFrame(clipTick);
  };
  const clipPause = () => {
    clipAudioRef.current?.pause();
    cancelAnimationFrame(clipAnimRef.current);
    setClipPlaying(false);
    if (clipAudioRef.current) setClipTime(clipAudioRef.current.currentTime);
    drawClipCanvas();
  };
  const clipStop = () => {
    const audio = clipAudioRef.current;
    if (audio) { audio.pause(); audio.currentTime = 0; }
    cancelAnimationFrame(clipAnimRef.current);
    setClipPlaying(false); setClipTime(0);
    drawClipCanvas();
  };
  const clipSeek = (delta: number) => {
    const audio = clipAudioRef.current;
    if (!audio) return;
    audio.currentTime = Math.max(0, Math.min(audio.duration || 0, audio.currentTime + delta));
    setClipTime(audio.currentTime);
    drawClipCanvas();
  };

  const clipSaveCrop = async () => {
    if (!clipModal) return;
    if (!clipSelection) { addClipLog("구간을 선택하세요 — 파형 위에서 드래그하세요"); return; }
    const [a, b] = clipSelection;
    const s = Math.min(a, b), e = Math.max(a, b);
    if (e - s < 0.3) { addClipLog("선택 구간이 너무 짧습니다 (최소 0.3초)"); return; }
    if (!clipCropName.trim()) { addClipLog("파일이름을 넣으세요"); return; }

    setClipSaving(true);
    setClipSaveMsg("클립 추출 중...");
    addClipLog(`클립 추출: ${clipModal.filename} [${fmtAudTime(s)} ~ ${fmtAudTime(e)}] (${(e - s).toFixed(1)}초)`);

    try {
      let saveName = clipCropName.trim();
      if (!saveName.endsWith(".wav") && !saveName.endsWith(".mp3")) saveName += ".wav";
      addClipLog(`저장 파일명: ${saveName}`);
      const form = new FormData();
      form.append("source", clipModal.filename);
      form.append("start", s.toString());
      form.append("end", e.toString());
      form.append("output_name", saveName);
      if (currentProjectId) form.append("project_id", currentProjectId);
      const res = await fetch("/api/audio-clip", { method: "POST", body: form });
      if (!res.ok) {
        const errText = await res.text().catch(() => "알 수 없는 오류");
        addClipLog(`클립 추출 실패: ${errText}`);
        setClipSaveMsg("클립 추출 실패"); setClipSaving(false); return;
      }
      const data = await res.json();
      setClipHistory((prev) => [...prev, { from: clipModal.filename, to: data.filename, start: s, end: e }]);
      addClipLog(`저장 완료: ${data.filename} (${(data.size / 1024).toFixed(1)} KB, ${(e - s).toFixed(1)}초)`);
      setClipSaveMsg(`"${data.filename}" 저장 완료 (${(e - s).toFixed(1)}초)`);

      addClipLog(`잘린 파일 다시 로딩 중...`);
      setTimeout(() => loadClipFile(data.filename), 500);
    } catch (err) { addClipLog(`오류: ${err}`); setClipSaveMsg(`오류: ${err}`); }
    finally { setClipSaving(false); }
  };

  const clipRegisterVoice = async () => {
    if (!clipModal) return;
    if (!clipVoiceName.trim()) { addClipLog("화자 이름을 넣으세요"); return; }

    setClipRegistering(true);
    setClipSaveMsg("음성 등록 중...");
    addClipLog(`음성 등록 시작: 화자="${clipVoiceName.trim()}", 파일=${clipModal.filename}`);
    if (clipRefText.trim()) addClipLog(`참조 텍스트: "${clipRefText.trim()}"`);

    try {
      addClipLog("오디오 파일 다운로드 중...");
      const fileRes = await fetch(clipModal.audioUrl);
      const blob = await fileRes.blob();
      addClipLog(`파일 크기: ${(blob.size / 1024).toFixed(1)} KB`);
      const regForm = new FormData();
      regForm.append("file", blob, clipModal.filename);
      regForm.append("name", clipVoiceName.trim());
      regForm.append("ref_text", clipRefText.trim());
      regForm.append("language", "Auto");
      addClipLog("Qwen3-TTS 음성 등록 요청 중...");
      const regRes = await fetch("/api/upload-voice", { method: "POST", body: regForm });
      if (regRes.ok) {
        const result = await regRes.json();
        await fetchVoices();
        setSelectedVoice(result.id);
        setClipSaveMsg(`"${clipVoiceName.trim()}" 음성 등록 완료!`);
        addClipLog(`음성 등록 완료! ID: ${result.id}, 이름: ${result.name}`);
        addClipLog("오디오북생성 탭에서 이 음성을 선택할 수 있습니다.");
      } else {
        const err = await regRes.json().catch(() => ({ detail: "등록 실패" }));
        setClipSaveMsg(`등록 실패: ${err.detail || "오류"}`);
        addClipLog(`등록 실패: ${err.detail || "알 수 없는 오류"}`);
      }
    } catch (err) { setClipSaveMsg(`오류: ${err}`); addClipLog(`오류: ${err}`); }
    finally { setClipRegistering(false); }
  };

  /* ================================================================ */
  /*  Voice Clone logic                                                */
  /* ================================================================ */

  const vcStartRecording = async () => {
    setVcError("");
    setVcSavedVoice(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioCtx = new AudioContext();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      vcAudioCtxRef.current = audioCtx;
      vcAnalyserRef.current = analyser;

      let opts: MediaRecorderOptions = {};
      if (typeof MediaRecorder.isTypeSupported === "function") {
        if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) opts = { mimeType: "audio/webm;codecs=opus" };
        else if (MediaRecorder.isTypeSupported("audio/webm")) opts = { mimeType: "audio/webm" };
      }
      const recorder = new MediaRecorder(stream, opts);
      vcChunksRef.current = [];
      recorder.ondataavailable = (e) => { if (e.data.size > 0) vcChunksRef.current.push(e.data); };
      vcRecorderRef.current = recorder;
      recorder.start(100);
      setVcMode("recording");
      setVcElapsed(0);
      if (vcAudioUrl) { URL.revokeObjectURL(vcAudioUrl); setVcAudioUrl(null); }
      setVcAudioBlob(null);

      const startTime = Date.now();
      vcTimerRef.current = setInterval(() => setVcElapsed(Math.floor((Date.now() - startTime) / 1000)), 200);

      const canvas = vcCanvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        const bufLen = analyser.frequencyBinCount;
        const dataArr = new Uint8Array(bufLen);
        const draw = () => {
          vcAnimRef.current = requestAnimationFrame(draw);
          analyser.getByteTimeDomainData(dataArr);
          const w = canvas.width, h = canvas.height;
          if (ctx) {
            ctx.fillStyle = "#1e2939";
            ctx.fillRect(0, 0, w, h);
            ctx.strokeStyle = "#364153";
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
            ctx.lineWidth = 2;
            ctx.strokeStyle = "#8b5cf6";
            ctx.beginPath();
            const sliceW = w / bufLen;
            let x = 0;
            for (let i = 0; i < bufLen; i++) {
              const v = dataArr[i] / 128.0;
              const y = (v * h) / 2;
              if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
              x += sliceW;
            }
            ctx.lineTo(w, h / 2);
            ctx.stroke();
          }
        };
        draw();
      }
    } catch { setVcError("마이크 접근 권한이 필요합���다."); }
  };

  const vcStopRecording = () => {
    const recorder = vcRecorderRef.current;
    if (!recorder || recorder.state === "inactive") return;
    recorder.onstop = () => {
      if (vcTimerRef.current) { clearInterval(vcTimerRef.current); vcTimerRef.current = null; }
      cancelAnimationFrame(vcAnimRef.current);
      recorder.stream.getTracks().forEach((t) => t.stop());
      vcAudioCtxRef.current?.close();
      vcAudioCtxRef.current = null;
      vcAnalyserRef.current = null;
      const blob = new Blob(vcChunksRef.current, { type: recorder.mimeType || "audio/webm" });
      setVcAudioBlob(blob);
      setVcAudioUrl(URL.createObjectURL(blob));
      setVcMode("recorded");
    };
    recorder.stop();
  };

  const vcHandleFile = (file: File) => {
    setVcError("");
    setVcSavedVoice(null);
    if (vcAudioUrl) URL.revokeObjectURL(vcAudioUrl);
    setVcAudioBlob(file);
    setVcAudioUrl(URL.createObjectURL(file));
    if (!vcName) setVcName(file.name.replace(/\.[^.]+$/, ""));
    setVcMode("recorded");
  };

  const vcSave = async () => {
    if (!vcAudioBlob || !vcName.trim()) return;
    setVcMode("uploading");
    setVcError("");
    try {
      const form = new FormData();
      const ext = vcAudioBlob instanceof File ? vcAudioBlob.name : "recording.webm";
      form.append("file", vcAudioBlob, ext);
      form.append("name", vcName.trim());
      form.append("ref_text", vcRefText.trim());
      form.append("language", vcLang);
      const res = await fetch("/api/upload-voice", { method: "POST", body: form });
      if (!res.ok) { const err = await res.json().catch(() => ({ detail: "Upload failed" })); throw new Error(err.detail || "Upload failed"); }
      const result = await res.json();
      setVcSavedVoice({ id: result.id, name: vcName.trim() });
      setVcMode("done");
      await fetchVoices();
      setSelectedVoice(result.id);
    } catch (err) {
      setVcError(err instanceof Error ? err.message : "등록 실패");
      setVcMode("recorded");
    }
  };

  const vcReset = () => {
    if (vcAudioUrl) URL.revokeObjectURL(vcAudioUrl);
    setVcMode("idle");
    setVcName("");
    setVcRefText("");
    setVcLang("Korean");
    setVcElapsed(0);
    setVcAudioBlob(null);
    setVcAudioUrl(null);
    setVcError("");
    setVcSavedVoice(null);
  };

  /* ================================================================ */
  /*  TTS logic                                                        */
  /* ================================================================ */

  const fetchVoices = useCallback(async () => {
    try {
      const res = await fetch("/api/voices");
      if (!res.ok) return;
      const data = await res.json();
      const list: Voice[] = (data.voices ?? []).sort((a: Voice, b: Voice) => {
        if (a.source === "uploaded" && b.source !== "uploaded") return -1;
        if (a.source !== "uploaded" && b.source === "uploaded") return 1;
        return 0;
      });
      setVoices(list);
      if (list.length > 0) {
        const preferred = list.find((v) => v.id === "upload-be0916e0-세월");
        setSelectedVoice((prev) => prev ? prev : (preferred?.id || list[0].id));
      }
    } catch { /* silent */ }
  }, []);

  const fetchElVoices = useCallback(async () => {
    try {
      const res = await fetch("/api/elevenlabs-voices");
      if (!res.ok) return;
      const data = await res.json();
      const list: Voice[] = data.voices ?? [];
      setElVoices(list);
      if (list.length > 0) setSelectedVoice((prev) => (!prev || prev.startsWith("el_")) ? list[0].id : prev);
    } catch { /* silent */ }
  }, []);

  useEffect(() => { fetchVoices(); fetchElVoices(); }, []);

  const fetchPoemFiles = useCallback(async () => {
    try {
      const res = await fetch("/api/poems");
      if (!res.ok) return;
      const data = await res.json();
      setPoemFiles(data.poems ?? []);
    } catch { /* silent */ }
  }, []);

  const loadPoemFile = async (filename: string) => {
    setPoemLoading(true);
    try {
      const res = await fetch(`/api/poems/${encodeURIComponent(filename)}`);
      if (!res.ok) throw new Error("Failed to load");
      const data = await res.json();
      setText(data.content);
      setShowPoemPicker(false);
    } catch { /* silent */ }
    finally { setPoemLoading(false); }
  };

  const onFileSelected = (file: File) => { setUploadFile(file); setUploadName(file.name.replace(/\.[^.]+$/, "")); setUploadRefText(""); setUploadLanguage("Auto"); setShowUploadModal(true); };

  const submitUpload = async () => {
    if (!uploadFile || !uploadName.trim()) return;
    setUploading(true);
    try {
      const form = new FormData();
      form.append("file", uploadFile); form.append("name", uploadName.trim()); form.append("ref_text", uploadRefText.trim()); form.append("language", uploadLanguage);
      const res = await fetch("/api/upload-voice", { method: "POST", body: form });
      if (!res.ok) { const err = await res.json().catch(() => ({ detail: "Upload failed" })); throw new Error(err.detail || "Upload failed"); }
      const result = await res.json();
      await fetchVoices(); setSelectedVoice(result.id); setShowUploadModal(false); setUploadFile(null);
    } catch (err: unknown) { alert(err instanceof Error ? err.message : "Upload failed"); }
    finally { setUploading(false); }
  };

  const stripChineseParens = (t: string): string => {
    return t.replace(/\([^)]*[一-鿿㐀-䶿][^)]*\)/g, "").replace(/\([^)]*[一-鿿㐀-䶿][^)]*\)/g, "");
  };

  const poemPause = (t: string): string => {
    return t.replace(/\n(?!\n)/g, "\n\n");
  };

  const parseNarations = (input: string): NarationItem[] => {
    const items: NarationItem[] = [];
    const regex = /<narration>\s*<title>([\s\S]*?)<\/title>\s*<body>([\s\S]*?)<\/body>\s*<\/narration>/gi;
    let match;
    while ((match = regex.exec(input)) !== null) {
      const title = match[1].trim();
      const body = poemPause(stripChineseParens(match[2].trim()));
      if (title && body) items.push({ title, body });
    }
    return items;
  };

  const generateBatch = async () => {
    if (!text.trim() || !selectedVoice) return;
    const narations = parseNarations(text);
    if (narations.length === 0) { generate(); return; }

    let voiceToUse = selectedVoice;
    if (ttsEngine === "elevenlabs" && !voiceToUse.startsWith("el_")) {
      const fallback = elVoices[0]?.id;
      if (!fallback) { setGen({ status: "error", message: "ElevenLabs 음성을 먼저 선택하세요", audioUrl: null, duration: null }); return; }
      voiceToUse = fallback; setSelectedVoice(fallback);
    }
    if (ttsEngine === "qwen3" && voiceToUse.startsWith("el_")) {
      const fallback = voices[0]?.id;
      if (!fallback) { setGen({ status: "error", message: "Qwen3 음성을 먼저 선택하세요", audioUrl: null, duration: null }); return; }
      voiceToUse = fallback; setSelectedVoice(fallback);
    }

    const voiceObj = (ttsEngine === "elevenlabs" ? elVoices : voices).find((v) => v.id === voiceToUse);
    const voiceName = voiceObj?.name || "";
    batchAbortRef.current = false;
    const initResults = narations.map((n) => ({ title: n.title, status: "pending" as const }));
    setBatchStatus({ running: true, current: 0, total: narations.length, results: initResults });
    setDebugLogs([]);
    addDebug(`Batch mode: ${narations.length} narrations detected`);

    for (let i = 0; i < narations.length; i++) {
      if (batchAbortRef.current) {
        addDebug(`Batch aborted at ${i + 1}/${narations.length}`);
        setBatchStatus((prev) => ({ ...prev, running: false }));
        return;
      }
      const item = narations[i];
      const customFilename = `${item.title}_${voiceName}`;
      setBatchStatus((prev) => {
        const results = [...prev.results];
        results[i] = { ...results[i], status: "generating" };
        return { ...prev, current: i + 1, results };
      });
      addDebug(`[${i + 1}/${narations.length}] Generating: "${item.title}" (${item.body.length} chars)`);

      const body: Record<string, unknown> = { text: item.body, voice_id: voiceToUse, language, engine: ttsEngine, custom_filename: customFilename, poem_mode: true };
      if (voiceName) body.voice_name = voiceName;
      if (seed.trim() && ttsEngine === "qwen3") body.seed = parseInt(seed, 10);
      if (ttsEngine === "qwen3") body.postprocess = postprocess;
      if (currentProjectId) body.project_id = currentProjectId;

      try {
        const res = await fetch("/api/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
        if (!res.ok) {
          const errText = await res.text().catch(() => "Request failed");
          setBatchStatus((prev) => {
            const results = [...prev.results];
            results[i] = { ...results[i], status: "error", message: errText };
            return { ...prev, results };
          });
          addDebug(`[${i + 1}] ERROR: ${errText.slice(0, 200)}`);
          continue;
        }
        let completed = false;
        await readSSE(res, (event) => {
          if (event.status === "complete") {
            completed = true;
            const audioUrl = event.audio_url as string;
            setBatchStatus((prev) => {
              const results = [...prev.results];
              results[i] = { ...results[i], status: "complete", audioUrl };
              return { ...prev, results };
            });
            addDebug(`[${i + 1}] Complete: ${audioUrl}`);
          } else if (event.status === "error") {
            setBatchStatus((prev) => {
              const results = [...prev.results];
              results[i] = { ...results[i], status: "error", message: event.message as string };
              return { ...prev, results };
            });
            addDebug(`[${i + 1}] Error: ${event.message}`);
          } else if (event.status === "generating" || event.status === "loading") {
            const msg = (event.message as string) || "";
            setBatchStatus((prev) => {
              const results = [...prev.results];
              results[i] = { ...results[i], status: "generating", message: msg };
              return { ...prev, results };
            });
            addDebug(`[${i + 1}] ${msg}`);
          }
        });
        if (!completed) {
          setBatchStatus((prev) => {
            const results = [...prev.results];
            if (results[i].status === "generating") results[i] = { ...results[i], status: "error", message: "Stream ended without completion" };
            return { ...prev, results };
          });
        }
      } catch (err) {
        setBatchStatus((prev) => {
          const results = [...prev.results];
          results[i] = { ...results[i], status: "error", message: err instanceof Error ? err.message : "Unknown error" };
          return { ...prev, results };
        });
        addDebug(`[${i + 1}] FETCH ERROR: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
    setBatchStatus((prev) => ({ ...prev, running: false }));
    addDebug(`Batch complete: ${narations.length} items processed`);
  };

  const generate = async () => {
    if (!text.trim() || !selectedVoice) return;
    setDebugLogs([]);
    let voiceToUse = selectedVoice;
    addDebug(`Engine: ${ttsEngine}, Voice: ${voiceToUse}, Text length: ${text.trim().length} chars`);
    if (ttsEngine === "elevenlabs" && !voiceToUse.startsWith("el_")) {
      const fallback = elVoices[0]?.id;
      if (!fallback) { addDebug("ERROR: No ElevenLabs voices available"); setGen({ status: "error", message: "ElevenLabs 음성을 먼저 선택하세요", audioUrl: null, duration: null }); return; }
      addDebug(`Voice fallback: ${voiceToUse} → ${fallback}`);
      voiceToUse = fallback;
      setSelectedVoice(fallback);
    }
    if (ttsEngine === "qwen3" && voiceToUse.startsWith("el_")) {
      const fallback = voices[0]?.id;
      if (!fallback) { addDebug("ERROR: No Qwen3 voices available"); setGen({ status: "error", message: "Qwen3 음성을 먼저 선택하세요", audioUrl: null, duration: null }); return; }
      addDebug(`Voice fallback: ${voiceToUse} → ${fallback}`);
      voiceToUse = fallback;
      setSelectedVoice(fallback);
    }
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    genStartRef.current = Date.now();
    setGenElapsed(0);
    if (genTimerRef.current) clearInterval(genTimerRef.current);
    genTimerRef.current = setInterval(() => setGenElapsed(Math.floor((Date.now() - genStartRef.current) / 1000)), 1000);
    setGen({ status: "loading", message: ttsEngine === "elevenlabs" ? "ElevenLabs 준비 중..." : "Preparing...", audioUrl: null, duration: null });
    const voiceObj = (ttsEngine === "elevenlabs" ? elVoices : voices).find((v) => v.id === voiceToUse);
    const body: Record<string, unknown> = { text: text.trim(), voice_id: voiceToUse, language, engine: ttsEngine };
    if (voiceObj?.name) body.voice_name = voiceObj.name;
    if (seed.trim() && ttsEngine === "qwen3") body.seed = parseInt(seed, 10);
    if (ttsEngine === "qwen3") body.postprocess = postprocess;
    if (currentProjectId && currentProjectName) {
      body.output_name = currentProjectName;
      body.project_id = currentProjectId;
    }
    addDebug(`POST /api/generate → ${JSON.stringify(body).slice(0, 300)}`);
    const t0 = Date.now();
    try {
      const res = await fetch("/api/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: controller.signal });
      addDebug(`Response: ${res.status} ${res.statusText} (${Date.now() - t0}ms)`);
      if (!res.ok) {
        const errText = await res.text().catch(() => "Request failed");
        addDebug(`ERROR body: ${errText.slice(0, 500)}`);
        setGen({ status: "error", message: errText, audioUrl: null, duration: null }); return;
      }
      addDebug("SSE stream started, reading events...");
      let eventCount = 0;
      await readSSE(res, (event) => {
        eventCount++;
        addDebug(`SSE #${eventCount}: ${JSON.stringify(event).slice(0, 300)}`);
        if (event.status === "complete") {
          if (genTimerRef.current) { clearInterval(genTimerRef.current); genTimerRef.current = null; }
          addDebug(`Generation complete in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
          setGen({ status: "complete", message: "Done!", audioUrl: (event.audio_url as string) ?? null, duration: (event.duration as number) ?? null });
          if (currentProjectId && event.audio_url) {
            const fname = (event.audio_url as string).split("/").pop() || "";
            const eng = (event.engine as string) || ttsEngine;
            const chars = (event.text_chars as number) || 0;
            const ttsCost = eng === "elevenlabs" ? chars * EL_COST_PER_CHAR : 0;
            patchProject({
              generated_audio_filename: fname, status: "generated",
              tts_engine: eng,
              tts_model: eng === "elevenlabs" ? "eleven_flash_v2_5" : "Qwen3-TTS-1.7B",
              tts_text_chars: chars,
              tts_elapsed: (event.generation_time as number) || 0,
              tts_cost: ttsCost,
            });
          }
        } else if (event.status === "error") {
          if (genTimerRef.current) { clearInterval(genTimerRef.current); genTimerRef.current = null; }
          addDebug(`SSE error: ${event.message}`);
          setGen({ status: "error", message: (event.message as string) ?? "Failed", audioUrl: null, duration: null });
        } else {
          setGen((p) => ({ ...p, status: event.status as GenerationStatus["status"], message: (event.message as string) ?? p.message }));
        }
      });
      addDebug(`SSE stream ended. Total events: ${eventCount}, elapsed: ${((Date.now() - t0) / 1000).toFixed(1)}s`);
      if (eventCount === 0) {
        addDebug("WARNING: Stream ended with 0 events — possible timeout or empty response");
      }
    } catch (err: unknown) {
      if (genTimerRef.current) { clearInterval(genTimerRef.current); genTimerRef.current = null; }
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
      if ((err as Error).name === "AbortError") { addDebug(`Aborted by user after ${elapsed}s`); return; }
      addDebug(`FETCH ERROR after ${elapsed}s: ${err instanceof Error ? `${err.name}: ${err.message}` : String(err)}`);
      setGen({ status: "error", message: err instanceof Error ? err.message : "Unknown error", audioUrl: null, duration: null });
    }
  };

  const isGenerating = gen.status === "loading" || gen.status === "generating" || batchStatus.running;
  const activeVoices = ttsEngine === "elevenlabs" ? elVoices : voices;
  const currentVoice = activeVoices.find((v) => v.id === selectedVoice);

  /* ================================================================ */
  /*  ASR logic                                                        */
  /* ================================================================ */

  const transcribe = async () => {
    if (!asrFile && !selectedExistingFile) return;
    asrAbortRef.current?.abort();
    const controller = new AbortController();
    asrAbortRef.current = controller;
    setAsrStatus({ status: "loading", message: "오디오 파일 처리 중..." });
    setTranscript(null); setCopied(false); setAsrSaved(false);

    const form = new FormData();
    if (asrFile) {
      form.append("file", asrFile);
    } else if (selectedExistingFile) {
      form.append("existing_file", selectedExistingFile);
    }
    form.append("num_speakers", numSpeakers.toString());

    const url = currentProjectId ? `/api/projects/${currentProjectId}/transcribe` : "/api/transcribe";

    try {
      const res = await fetch(url, { method: "POST", body: form, signal: controller.signal });
      if (!res.ok) { const err = await res.json().catch(() => ({ detail: "요청 실패" })); setAsrStatus({ status: "error", message: err.detail || "요청 실패" }); return; }
      await readSSE(res, (event) => {
        if (event.status === "complete") {
          setTranscript({ segments: event.segments as TranscriptSegment[], full_text: event.full_text as string, duration: event.duration as number, processing_time: event.processing_time as number });
          setAsrStatus({ status: "complete", message: "완료!" });
        } else if (event.status === "error") { setAsrStatus({ status: "error", message: (event.message as string) || "실패" }); }
        else { setAsrStatus({ status: event.status as string, message: (event.message as string) || "" }); }
      });
    } catch (err: unknown) {
      if ((err as Error).name === "AbortError") return;
      setAsrStatus({ status: "error", message: err instanceof Error ? err.message : "알 수 없는 오류" });
    }
  };

  const hasAsrInput = !!(asrFile || selectedExistingFile);
  const isTranscribing = asrStatus.status === "loading" || asrStatus.status === "transcribing";
  const [asrSaved, setAsrSaved] = useState(false);
  const [asrSaving, setAsrSaving] = useState(false);
  const copyTranscript = async () => { if (!transcript?.full_text) return; await navigator.clipboard.writeText(transcript.full_text); setCopied(true); setTimeout(() => setCopied(false), 2000); };
  const downloadTranscript = () => { if (!transcript?.full_text) return; const b = new Blob([transcript.full_text], { type: "text/plain;charset=utf-8" }); const u = URL.createObjectURL(b); const a = document.createElement("a"); a.href = u; a.download = "transcript.txt"; a.click(); URL.revokeObjectURL(u); };
  const saveTranscript = async () => {
    if (!transcript?.full_text || !currentProjectId) return;
    setAsrSaving(true);
    try {
      await patchProject({ transcript_text: transcript.full_text });
      setAsrSaved(true);
      setTimeout(() => setAsrSaved(false), 2000);
    } catch { /* silent */ }
    setAsrSaving(false);
  };

  /* ================================================================ */
  /*  Editor logic                                                     */
  /* ================================================================ */

  const loadTranscriptToEditor = useCallback(() => {
    if (!transcript) return;
    setEditorText(transcript.segments.map((s) => `[화자 ${s.speaker}] ${s.text}`).join("\n"));
    setRewrittenText(""); setRewriteStatus({ status: "idle", message: "" });
  }, [transcript]);

  useEffect(() => { if (activeTab === "editor" && transcript && !editorText) loadTranscriptToEditor(); }, [activeTab, transcript, editorText, loadTranscriptToEditor]);

  useEffect(() => {
    if (activeTab === "infographic" && !infoPrompt && srcRewritten) {
      setInfoPrompt(INFOGRAPHIC_PROMPT_TEMPLATE + "\n\n---\n\n아래는 인포그래픽에 사용할 LLM 변환텍스트입니다:\n\n" + srcRewritten);
    }
  }, [activeTab, infoPrompt, srcRewritten]);

  const removeSpeaker = (n: number) => { setEditorText(editorText.split("\n").filter((l) => !l.startsWith(`[화자 ${n}]`)).join("\n")); };
  const speakersInText = (): number[] => {
    const s = new Set<number>();
    for (const l of editorText.split("\n")) { const m = l.match(/^\[화자 (\d+)\]/); if (m) s.add(parseInt(m[1], 10)); }
    return Array.from(s).sort((a, b) => a - b);
  };

  const rewriteText = async () => {
    if (!editorText.trim()) return;
    rewriteAbortRef.current?.abort();
    const controller = new AbortController(); rewriteAbortRef.current = controller;
    setRewriteStatus({ status: "rewriting", message: "박완서 문체로 변환 중..." }); setRewrittenText("");
    try {
      const res = await fetch("/api/rewrite", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: editorText.trim(), model: selectedModel }), signal: controller.signal });
      if (!res.ok) { const err = await res.json().catch(() => ({ detail: "요청 실패" })); setRewriteStatus({ status: "error", message: err.detail || "요청 실패" }); return; }
      await readSSE(res, (event) => {
        if (event.status === "complete") {
          const rw = stripThinkTags(event.rewritten_text as string);
          setRewrittenText(rw); setRewriteStatus({ status: "complete", message: "변환 완료!" });
          const rwIn = (event.input_tokens as number) || 0;
          const rwOut = (event.output_tokens as number) || 0;
          patchProject({
            rewritten_text: rw, llm_model: selectedModel, status: "rewritten",
            rewrite_model: selectedModel,
            rewrite_input_tokens: rwIn,
            rewrite_output_tokens: rwOut,
            rewrite_elapsed: event.elapsed || 0,
            rewrite_cost: calcLLMCost(selectedModel, rwIn, rwOut),
          });
        } else if (event.status === "error") { setRewriteStatus({ status: "error", message: (event.message as string) || "실패" }); }
        else { setRewriteStatus({ status: event.status as string, message: (event.message as string) || "" }); }
      });
    } catch (err: unknown) {
      if ((err as Error).name === "AbortError") return;
      setRewriteStatus({ status: "error", message: err instanceof Error ? err.message : "알 수 없는 오류" });
    }
  };

  const fixTypos = async () => {
    if (!editorText.trim()) return;
    rewriteAbortRef.current?.abort();
    const controller = new AbortController(); rewriteAbortRef.current = controller;
    setRewriteStatus({ status: "rewriting", message: "오타 수정 중..." }); setRewrittenText("");
    try {
      const res = await fetch("/api/fix-typos", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: editorText.trim(), model: selectedModel }), signal: controller.signal });
      if (!res.ok) { const err = await res.json().catch(() => ({ detail: "요청 실패" })); setRewriteStatus({ status: "error", message: err.detail || "요청 실패" }); return; }
      await readSSE(res, (event) => {
        if (event.status === "complete") {
          const fixed = stripThinkTags(event.fixed_text as string);
          setRewrittenText(fixed); setRewriteStatus({ status: "complete", message: "오타 수정 완료!" });
          const ftIn = (event.input_tokens as number) || 0;
          const ftOut = (event.output_tokens as number) || 0;
          patchProject({
            fix_typos_model: selectedModel,
            fix_typos_input_tokens: ftIn,
            fix_typos_output_tokens: ftOut,
            fix_typos_elapsed: event.elapsed || 0,
            fix_typos_cost: calcLLMCost(selectedModel, ftIn, ftOut),
          });
        } else if (event.status === "error") { setRewriteStatus({ status: "error", message: (event.message as string) || "실패" }); }
        else { setRewriteStatus({ status: event.status as string, message: (event.message as string) || "" }); }
      });
    } catch (err: unknown) {
      if ((err as Error).name === "AbortError") return;
      setRewriteStatus({ status: "error", message: err instanceof Error ? err.message : "알 수 없는 오류" });
    }
  };

  const isRewriting = rewriteStatus.status === "rewriting";

  /* ================================================================ */
  /*  Render: Landing Page                                             */
  /* ================================================================ */

  if (view === "landing") {
    return (
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        {/* Auth button - top right */}
        <div className="flex justify-end mb-2">
          {authUser ? (
            <div className="flex items-center gap-3">
              <span className="text-sm text-gray-300">
                <svg className="inline w-4 h-4 mr-1 -mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" /></svg>
                <span className="font-medium text-white">{authUser.username}</span>
                {authUser.role === "admin" && <span className="ml-1 text-xs text-accent-400">(admin)</span>}
              </span>
              <button onClick={handleSignout} className="text-sm text-gray-400 hover:text-red-400 transition-colors">로그아웃</button>
            </div>
          ) : (
            <button onClick={() => { setAuthMode("signin"); setAuthError(""); setShowAuthModal(true); }}
              className="flex items-center gap-1.5 text-sm text-gray-300 hover:text-white transition-colors">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" /></svg>
              로그인
            </button>
          )}
        </div>

        <header className="mb-10 text-center">
          <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
            <span className="bg-gradient-to-r from-accent-400 to-purple-400 bg-clip-text text-transparent">
              Voice Studio
            </span>
          </h1>
          <p className="mt-2 text-lg text-gray-300">오픈소스 오디오북 생성기</p>
        </header>

        <div className="mb-8 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <a href="/" className="inline-flex items-center gap-1.5 rounded-lg border border-accent-500/50 bg-accent-600/20 px-3 py-1.5 text-sm font-medium text-accent-300 hover:bg-accent-600/40 hover:text-white transition-colors">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" /></svg>
              Home
            </a>
            <h2 className="text-xl font-semibold text-white">프로젝트</h2>
          </div>
          {authUser ? (
            <button className="btn-primary text-sm" onClick={() => { setNewProjectName(""); setShowNewModal(true); }}>
              <PlusIcon /> 새 프로젝트
            </button>
          ) : (
            <button className="btn-primary text-sm" onClick={() => { setAuthMode("signin"); setAuthError(""); setShowAuthModal(true); }}>
              <PlusIcon /> 로그인하여 프로젝트 만들기
            </button>
          )}
        </div>

        {projects.length === 0 ? (
          <div className="card text-center py-16">
            <BookIcon />
            <p className="mt-4 text-gray-300">아직 프로젝트가 없습니다</p>
            <p className="mt-1 text-sm text-gray-400">{authUser ? "새 프로젝트를 만들어 회고록 작업을 시작하세요" : "로그인하여 프로젝트를 만드세요"}</p>
            {authUser ? (
              <button className="btn-primary mt-6" onClick={() => { setNewProjectName(""); setShowNewModal(true); }}>
                <PlusIcon /> 새 프로젝트 만들기
              </button>
            ) : (
              <button className="btn-primary mt-6" onClick={() => { setAuthMode("signin"); setAuthError(""); setShowAuthModal(true); }}>
                로그인
              </button>
            )}
          </div>
        ) : (
          <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
            {projects.map((p) => (
              <div key={p.id} className="flex flex-col rounded-2xl border border-[#364153] bg-[#1e2939] p-6 shadow-xl hover:border-[#4a5565] transition-colors">
                {/* Header */}
                <div className="mb-4 flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <h3 className="text-xl font-bold text-white leading-tight truncate">{p.name}</h3>
                    <p className="mt-1 text-sm text-gray-300">{fmtDate(p.created_at)}</p>
                  </div>
                  <span className={`shrink-0 rounded-full px-3 py-1 text-xs font-semibold tracking-wide uppercase ${
                    p.status === "created" ? "bg-slate-500/25 text-slate-200 ring-1 ring-slate-500/40" :
                    p.status === "uploaded" ? "bg-amber-500/25 text-amber-200 ring-1 ring-amber-500/40" :
                    p.status === "transcribed" ? "bg-sky-500/25 text-sky-200 ring-1 ring-sky-500/40" :
                    p.status === "rewritten" ? "bg-violet-500/25 text-violet-200 ring-1 ring-violet-500/40" :
                    p.status === "generated" ? "bg-emerald-500/25 text-emerald-200 ring-1 ring-emerald-500/40" :
                    "bg-slate-500/25 text-slate-200 ring-1 ring-slate-500/40"
                  }`}>
                    {p.status === "created" ? "생성됨" : p.status === "uploaded" ? "오디오준비됨" : p.status === "transcribed" ? "녹취록완료" : p.status === "rewritten" ? "편집완료" : p.status === "generated" ? "오디오북완료" : p.status}
                  </span>
                </div>

                {/* Audio source */}
                {p.source_audio_original_name && (
                  <div className="mb-3 flex items-center gap-2 rounded-lg bg-[#1e2939] px-3 py-2">
                    <MicIcon />
                    <span className="truncate text-sm font-medium text-[#e5e7eb]">{p.source_audio_original_name}</span>
                    <span className="ml-auto shrink-0 text-sm text-gray-300">{fmtSize(p.source_audio_size)}</span>
                  </div>
                )}

                {/* AI Services usage */}
                {(p.transcript_text || p.fix_typos_model || p.rewrite_model || p.generated_audio_filename) && (() => {
                  const totalCost = (p.asr_cost || 0) + (p.fix_typos_cost || 0) + (p.rewrite_cost || 0) + (p.tts_cost || 0);
                  const fmtCost = (c: number) => c === 0 ? "무료" : c < 0.01 ? `$${c.toFixed(4)}` : `$${c.toFixed(3)}`;

                  return (
                    <div className="mb-3 rounded-lg border border-[#364153] bg-[#101828] overflow-hidden">
                      <div className="px-3 py-2 border-b border-[#364153] flex items-center justify-between">
                        <span className="text-xs font-semibold uppercase tracking-wider text-gray-300">AI Services</span>
                        {totalCost > 0 && <span className="text-xs font-bold text-amber-300">총 {fmtCost(totalCost)}</span>}
                      </div>
                      <div className="divide-y divide-[#364153]">
                        {/* ASR */}
                        {p.transcript_text && (
                          <div className="px-3 py-2">
                            <div className="flex items-center justify-between">
                              <span className="text-xs font-semibold text-sky-300">🎙 ASR</span>
                              <span className="text-xs font-medium text-emerald-400">무료</span>
                            </div>
                            <div className="mt-0.5 text-sm text-gray-300">
                              Groq Whisper large-v3 + WavLM ({p.num_speakers || 2}명)
                              {p.asr_audio_duration > 0 && <span className="ml-1 text-gray-400">· {fmtDuration(p.asr_audio_duration)}</span>}
                              {p.asr_elapsed > 0 && <span className="ml-1 text-gray-400">· {fmtDuration(p.asr_elapsed)}</span>}
                            </div>
                          </div>
                        )}
                        {/* Fix Typos */}
                        {p.fix_typos_model && (
                          <div className="px-3 py-2">
                            <div className="flex items-center justify-between">
                              <span className="text-xs font-semibold text-orange-300">✏️ 오타수정</span>
                              <span className="text-xs font-medium text-amber-300">{fmtCost(p.fix_typos_cost || 0)}</span>
                            </div>
                            <div className="mt-0.5 text-sm text-gray-300">
                              {LLM_MODELS.find((m) => m.id === p.fix_typos_model)?.label ?? p.fix_typos_model}
                              <span className="ml-1.5 text-gray-400">{(p.fix_typos_input_tokens + p.fix_typos_output_tokens).toLocaleString()} tok</span>
                              <span className="ml-1 text-[#6a7282]">({p.fix_typos_input_tokens.toLocaleString()}↓ {p.fix_typos_output_tokens.toLocaleString()}↑)</span>
                            </div>
                          </div>
                        )}
                        {/* Rewrite */}
                        {p.rewrite_model && (
                          <div className="px-3 py-2">
                            <div className="flex items-center justify-between">
                              <span className="text-xs font-semibold text-violet-300">🖊 문체변환</span>
                              <span className="text-xs font-medium text-amber-300">{fmtCost(p.rewrite_cost || 0)}</span>
                            </div>
                            <div className="mt-0.5 text-sm text-gray-300">
                              {LLM_MODELS.find((m) => m.id === p.rewrite_model)?.label ?? p.rewrite_model}
                              <span className="ml-1.5 text-gray-400">{(p.rewrite_input_tokens + p.rewrite_output_tokens).toLocaleString()} tok</span>
                              <span className="ml-1 text-[#6a7282]">({p.rewrite_input_tokens.toLocaleString()}↓ {p.rewrite_output_tokens.toLocaleString()}↑)</span>
                            </div>
                          </div>
                        )}
                        {/* TTS */}
                        {p.generated_audio_filename && (
                          <div className="px-3 py-2">
                            <div className="flex items-center justify-between">
                              <span className="text-xs font-semibold text-emerald-300">🔊 TTS</span>
                              <span className={`text-xs font-medium ${p.tts_engine === "elevenlabs" ? "text-amber-300" : "text-emerald-400"}`}>
                                {p.tts_engine === "elevenlabs" ? fmtCost(p.tts_cost || 0) : "로컬 GPU"}
                              </span>
                            </div>
                            <div className="mt-0.5 text-sm text-gray-300">
                              {p.tts_model || (p.tts_engine === "elevenlabs" ? "eleven_flash_v2_5" : "Qwen3-TTS-1.7B")}
                              {p.tts_engine === "elevenlabs" && p.tts_text_chars > 0 && (
                                <span className="ml-1.5 text-gray-400">{p.tts_text_chars.toLocaleString()} chars</span>
                              )}
                              {p.generated_audio_duration > 0 && <span className="ml-1 text-gray-400">· {fmtDuration(p.generated_audio_duration)}</span>}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })()}

                {/* Text stats */}
                {(() => {
                  const bestText = p.rewritten_text || p.edited_transcript || p.transcript_text || "";
                  if (!bestText) return null;
                  const wc = countWords(bestText);
                  const label = p.rewritten_text ? "LLM 변환" : p.edited_transcript ? "편집 녹취록" : "원본 전사";
                  return (
                    <div className="mb-3 flex items-center gap-2 rounded-lg bg-[#1e2939] px-3 py-2">
                      <span className="text-sm text-gray-300">📝 {label}: {bestText.length.toLocaleString()}자 · {wc.toLocaleString()}단어</span>
                    </div>
                  );
                })()}

                {/* Transcript (collapsible) */}
                {p.transcript_text && (
                  <div className="mb-3">
                    <button className="flex w-full items-center gap-1.5 rounded-md px-2 py-1.5 text-sm font-semibold text-sky-300 hover:bg-sky-500/10 hover:text-sky-200 transition-colors"
                      onClick={() => toggleExpanded(expandedTranscripts, p.id, setExpandedTranscripts)}>
                      <ChevronIcon open={expandedTranscripts.has(p.id)} /> 원본 전사
                    </button>
                    {expandedTranscripts.has(p.id) && (
                      <div className="mt-1.5 max-h-[200px] overflow-y-auto rounded-lg border border-sky-500/20 bg-[#101828] px-4 py-3 text-base leading-relaxed text-[#e5e7eb] whitespace-pre-wrap">
                        {p.transcript_text}
                      </div>
                    )}
                  </div>
                )}

                {/* Rewritten (collapsible) */}
                {p.rewritten_text && (
                  <div className="mb-3">
                    <button className="flex w-full items-center gap-1.5 rounded-md px-2 py-1.5 text-sm font-semibold text-violet-300 hover:bg-violet-500/10 hover:text-violet-200 transition-colors"
                      onClick={() => toggleExpanded(expandedRewrites, p.id, setExpandedRewrites)}>
                      <ChevronIcon open={expandedRewrites.has(p.id)} /> 박완서 문체
                    </button>
                    {expandedRewrites.has(p.id) && (
                      <div className="mt-1.5 max-h-[200px] overflow-y-auto rounded-lg border border-violet-500/20 bg-violet-500/5 px-4 py-3 text-base leading-relaxed text-[#e5e7eb] whitespace-pre-wrap">
                        {p.rewritten_text}
                      </div>
                    )}
                  </div>
                )}

                {/* Generated audio */}
                {p.generated_audio_filename && (
                  <div className="mb-3">
                    <AudioPlayer src={`/api/outputs/${p.generated_audio_filename}`} />
                  </div>
                )}

                {/* Summary (collapsible) */}
                <div className="mb-3">
                  <button className="flex w-full items-center gap-1.5 rounded-md px-2 py-1.5 text-sm font-semibold text-emerald-300 hover:bg-emerald-500/10 hover:text-emerald-200 transition-colors"
                    onClick={() => toggleExpanded(expandedSummary, p.id, setExpandedSummary)}>
                    <ChevronIcon open={expandedSummary.has(p.id)} /> 프로젝트 요약
                  </button>
                  {expandedSummary.has(p.id) && (
                    <div className="mt-1.5 max-h-[300px] overflow-y-auto rounded-lg border border-emerald-500/20 bg-emerald-500/5 px-4 py-3 text-base leading-relaxed text-[#e5e7eb]">
                      {renderMarkdown(buildProjectSummary(p))}
                    </div>
                  )}
                </div>

                <div className="mt-auto flex gap-3 pt-4 border-t border-[#364153]">
                  <button className="btn-primary flex-1 text-base py-2.5" onClick={() => openProject(p.id)}>열기</button>
                  <button className="inline-flex items-center justify-center rounded-lg border border-[#364153] bg-[#1e2939] px-4 py-2.5 text-gray-300 hover:bg-red-500/15 hover:text-red-300 hover:border-red-500/30 transition-colors" onClick={() => deleteProject(p.id)}><TrashIcon /></button>
                </div>
              </div>
            ))}
          </div>
        )}

        <footer className="mt-12 border-t border-[#364153] pt-6 text-center text-sm text-gray-400 space-y-1">
          <p>오픈소스 오디오 자서전, 시낭독 영상 제작 소프트웨어</p>
          <p>2026년 5월, Sonny. 소스코드 : <a href="https://github.com/muntakson/voicestudio" target="_blank" rel="noopener noreferrer" className="underline hover:text-[#a78bfa]">github.com/muntakson/voicestudio</a></p>
          <p>제작목적 : AI 코딩 교육 &middot; mtshon@gmail.com</p>
        </footer>

        {/* New Project Modal */}
        {showNewModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
            <div className="card mx-4 w-full max-w-md space-y-4">
              <h2 className="text-lg font-semibold">새 프로젝트</h2>
              <div>
                <label className="label">프로젝트 이름</label>
                <input type="text" className="input-field" placeholder="예: 아버지의 이야기, 어머니의 회고록..."
                  value={newProjectName} onChange={(e) => setNewProjectName(e.target.value)}
                  autoFocus onKeyDown={(e) => e.key === "Enter" && createProject()} />
              </div>
              <div className="flex gap-3 pt-2">
                <button className="btn-secondary flex-1" onClick={() => setShowNewModal(false)}>취소</button>
                <button className="btn-primary flex-1" disabled={!newProjectName.trim()} onClick={createProject}>만들기</button>
              </div>
            </div>
          </div>
        )}

        {/* Auth Modal */}
        {showAuthModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={() => setShowAuthModal(false)}>
            <div className="card mx-4 w-full max-w-sm space-y-4" onClick={(e) => e.stopPropagation()}>
              <h2 className="text-lg font-semibold text-center">{authMode === "signin" ? "로그인" : "회원가입"}</h2>
              {authError && <p className="text-sm text-red-400 text-center">{authError}</p>}
              <div>
                <label className="label">사용자 이름</label>
                <input type="text" className="input-field" placeholder="username"
                  value={authUsername} onChange={(e) => setAuthUsername(e.target.value)}
                  autoFocus onKeyDown={(e) => e.key === "Enter" && handleAuth()} />
              </div>
              <div>
                <label className="label">비밀번호</label>
                <input type="password" className="input-field" placeholder="password"
                  value={authPassword} onChange={(e) => setAuthPassword(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleAuth()} />
              </div>
              <div className="flex gap-3 pt-2">
                <button className="btn-secondary flex-1" onClick={() => setShowAuthModal(false)}>취소</button>
                <button className="btn-primary flex-1" disabled={authLoading || !authUsername.trim() || !authPassword} onClick={handleAuth}>
                  {authLoading ? "..." : authMode === "signin" ? "로그인" : "가입"}
                </button>
              </div>
              <p className="text-center text-sm text-gray-400">
                {authMode === "signin" ? (
                  <>계정이 없으신가요? <button className="text-accent-400 hover:underline" onClick={() => { setAuthMode("signup"); setAuthError(""); }}>회원가입</button></>
                ) : (
                  <>이미 계정이 있으신가요? <button className="text-accent-400 hover:underline" onClick={() => { setAuthMode("signin"); setAuthError(""); }}>로그인</button></>
                )}
              </p>
            </div>
          </div>
        )}
      </div>
    );
  }

  /* ================================================================ */
  /*  Render: Studio View                                              */
  /* ================================================================ */

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      <header className="mb-10 text-center">
        <div className="mb-4 flex items-center justify-center gap-4">
          <a href="/" className="inline-flex items-center gap-1.5 rounded-lg border border-accent-500/50 bg-accent-600/20 px-3 py-1.5 text-sm font-medium text-accent-300 hover:bg-accent-600/40 hover:text-white transition-colors">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" /></svg>
            Home
          </a>
          <button onClick={goToLanding} className="inline-flex items-center gap-1 text-sm text-gray-300 hover:text-white transition-colors">
            <BackIcon /> 프로젝트 목록
          </button>
        </div>
        <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
          <span className="bg-gradient-to-r from-accent-400 to-purple-400 bg-clip-text text-transparent">Voice Studio</span>
        </h1>
        {currentProjectName && (
          <div className="mt-2 flex items-center justify-center gap-2">
            <span className="text-lg font-medium text-white">{currentProjectName}</span>
            {currentProjectId && <span className="rounded bg-[#364153] px-2 py-0.5 text-xs font-mono text-gray-400">{currentProjectId.slice(0, 8)}</span>}
          </div>
        )}
        <p className="mt-1 text-gray-300">오디오회고록 제작 서비스 - 음성인식, 녹취록 생성, AI편집, AI 오디오북 생성</p>
        <div className="mt-6 inline-flex rounded-lg border border-[#364153] bg-[#101828] p-1">
          {(["recorder", "download", "asr", "editor", "source", "tts", "infographic", "poem-shorts", "settings"] as const).map((tab) => (
            <button key={tab} onClick={() => setActiveTab(tab)}
              className={`rounded-md px-6 py-2 text-sm font-medium transition-colors ${activeTab === tab ? "bg-accent-600 text-white" : "text-gray-300 hover:text-white"}`}>
              {tab === "recorder" ? "음성녹음" : tab === "download" ? "오디오다운로드" : tab === "source" ? "소스" : tab === "tts" ? "오디오북생성" : tab === "infographic" ? "인포그래픽" : tab === "poem-shorts" ? "시 숏폼" : tab === "asr" ? "음성인식" : tab === "editor" ? "글편집" : "설정"}
            </button>
          ))}
        </div>
      </header>

      {/* ============ TTS Tab ============ */}
      {activeTab === "tts" && (
        <div className="grid gap-6 lg:grid-cols-[1fr_340px]">
          <div className="space-y-6">
            <section className="card">
              <div className="mb-3 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <h2 className="text-lg font-semibold">Text to Speak</h2>
                  <button className="p-1 rounded hover:bg-[#1e2939] text-gray-300 hover:text-white transition-colors" title="시 모음 불러오기"
                    onClick={() => { fetchPoemFiles(); setShowPoemPicker(true); }}>
                    <FileIcon />
                  </button>
                  {ttsSaved && <span className="text-xs text-green-400">저장됨</span>}
                  {text.trim() && parseNarations(text).length > 0 && (
                    <span className="text-xs px-2 py-0.5 rounded bg-purple-500/20 text-purple-300 border border-purple-500/30">
                      Batch: {parseNarations(text).length}편
                    </span>
                  )}
                </div>
                {text.trim() && <span className="text-sm text-gray-300">{text.length.toLocaleString()}자 · {countWords(text).toLocaleString()}단어 · {estimateTtsTime(text.length, ttsEngine)}{ttsEngine === "elevenlabs" && <span className="ml-1 text-amber-300">(≈${(text.length * EL_COST_PER_CHAR).toFixed(3)})</span>}</span>}
              </div>
              <textarea className="input-field min-h-[200px] resize-y text-base leading-relaxed" placeholder="Type or paste the text you want to convert to speech...

Batch mode: paste multiple stories with tags:
<narration><title>제목</title><body>본문...</body></narration>"
                value={text} onChange={(e) => setText(e.target.value)} />
              <div className="mt-3 flex flex-wrap gap-2">
                {EXAMPLES.map((ex) => (<button key={ex.label} className="btn-secondary text-xs" onClick={() => { setText(ex.text); setLanguage(ex.lang); }}>{ex.label}</button>))}
              </div>
            </section>
            <section className="card">
              <h2 className="mb-4 text-lg font-semibold">Settings</h2>
              <div className="mb-4">
                <label className="label">TTS 엔진</label>
                <div className="flex gap-2">
                  <button disabled
                    className="flex-1 rounded-lg px-3 py-2 text-sm font-medium bg-[#1e2939] border border-transparent text-gray-400 cursor-not-allowed opacity-50">
                    ElevenLabs <span className="text-[10px]">Cloud (비활성)</span>
                  </button>
                  <button onClick={() => { setTtsEngine("qwen3"); setLanguage("Korean"); const pref = voices.find((v) => v.id === "upload-be0916e0-세월"); setSelectedVoice(pref?.id || voices[0]?.id || ""); }}
                    className={`flex-1 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${ttsEngine === "qwen3" ? "bg-accent-600/30 border border-accent-500/50 text-white" : "bg-[#1e2939] border border-transparent hover:bg-[#1e2939] text-[#d1d5dc]"}`}>
                    Qwen3-TTS <span className="text-[10px] text-accent-400/70">Local</span>
                  </button>
                </div>
              </div>
              {ttsEngine === "qwen3" && (
                <div className="space-y-3">
                  <div className="grid gap-4 sm:grid-cols-2">
                    <div><label className="label">Language</label><select className="input-field cursor-pointer" value={language} onChange={(e) => setLanguage(e.target.value)}>{LANGUAGES.map((l) => <option key={l} value={l}>{l}</option>)}</select></div>
                    <div><label className="label">Seed (optional)</label><input type="number" className="input-field" placeholder="Random" value={seed} onChange={(e) => setSeed(e.target.value)} /></div>
                  </div>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input type="checkbox" checked={postprocess} onChange={(e) => setPostprocess(e.target.checked)} className="h-4 w-4 rounded border-[#364153] bg-[#1e2939] text-accent-500 focus:ring-accent-500/50" />
                    <span className="text-sm text-[#d1d5dc]">Audio post-processing</span>
                    <span className="text-[10px] text-gray-400">(compression, normalization)</span>
                  </label>
                </div>
              )}
              <button className="btn-primary mt-6 w-full text-lg" disabled={!text.trim() || !selectedVoice || isGenerating} onClick={generateBatch}>
                {batchStatus.running ? <><Spinner /> Batch {batchStatus.current}/{batchStatus.total}...</> : isGenerating ? <><Spinner /> Generating...{genElapsed > 0 && <span className="ml-2 font-mono text-sm opacity-80">{Math.floor(genElapsed / 60)}:{(genElapsed % 60).toString().padStart(2, "0")}</span>}</> : <><PlayIcon /> Generate Speech</>}
              </button>
              {batchStatus.running && (
                <button className="btn-secondary mt-2 w-full text-sm text-red-400" onClick={() => { batchAbortRef.current = true; }}>
                  Batch 중단
                </button>
              )}
            </section>
            {gen.status !== "idle" && (
              <section className="card">
                <h2 className="mb-4 text-lg font-semibold">Output</h2>
                {(gen.status === "loading" || gen.status === "generating") && (
                  <div className="space-y-3"><div className="flex items-center gap-3"><Spinner /><span className="text-sm text-gray-300">{gen.message}</span><span className="text-sm font-mono text-accent-400">{genElapsed > 0 && `${Math.floor(genElapsed / 60)}:${(genElapsed % 60).toString().padStart(2, "0")}`}</span></div>
                    <div className="h-2 overflow-hidden rounded-full bg-[#364153]"><div className="progress-pulse h-full rounded-full bg-gradient-to-r from-accent-600 to-purple-500" style={{ width: gen.status === "loading" ? "40%" : "75%", transition: "width 0.5s ease" }} /></div></div>)}
                {gen.status === "error" && <div className="rounded-lg border border-red-800/50 bg-red-900/20 px-4 py-3 text-sm text-red-300">{gen.message}</div>}
                {gen.status === "complete" && gen.audioUrl && (
                  <div className="space-y-4"><div className="flex items-center gap-2 text-sm text-green-400"><CheckIcon /><span>Done!{genElapsed > 0 && <span className="ml-2 text-gray-400">(소요시간: {genElapsed < 60 ? `${genElapsed}초` : `${Math.floor(genElapsed / 60)}분 ${genElapsed % 60}초`})</span>}{gen.duration != null && <span className="ml-1 text-gray-400">· 오디오 {gen.duration.toFixed(1)}s</span>}</span></div>
                    <AudioPlayer src={gen.audioUrl} /><a href={gen.audioUrl} download className="btn-secondary inline-flex mt-2"><DownloadIcon /> Download</a></div>)}
              </section>
            )}
            {batchStatus.total > 0 && (
              <section className="card">
                <div className="mb-3 flex items-center justify-between">
                  <h2 className="text-lg font-semibold">Batch Output ({batchStatus.results.filter((r) => r.status === "complete").length}/{batchStatus.total})</h2>
                  {!batchStatus.running && <button className="btn-secondary text-xs" onClick={() => setBatchStatus({ running: false, current: 0, total: 0, results: [] })}>Clear</button>}
                </div>
                {batchStatus.running && (
                  <div className="mb-3 h-2 overflow-hidden rounded-full bg-[#364153]">
                    <div className="h-full rounded-full bg-gradient-to-r from-accent-600 to-purple-500 transition-all duration-500" style={{ width: `${(batchStatus.current / batchStatus.total) * 100}%` }} />
                  </div>
                )}
                <div className="space-y-2 max-h-[400px] overflow-y-auto">
                  {batchStatus.results.map((r, i) => (
                    <div key={i} className={`rounded-lg px-4 py-3 text-sm ${r.status === "complete" ? "bg-green-900/20 border border-green-800/40" : r.status === "error" ? "bg-red-900/20 border border-red-800/40" : r.status === "generating" ? "bg-accent-900/20 border border-accent-500/40" : "bg-[#1e2939] border border-transparent"}`}>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2 min-w-0">
                          <span className="font-medium text-white truncate">{i + 1}. {r.title}</span>
                          {r.status === "generating" && <Spinner small />}
                        </div>
                        <div className="flex items-center gap-2 shrink-0">
                          {r.status === "complete" && <span className="text-xs text-green-400">완료</span>}
                          {r.status === "error" && <span className="text-xs text-red-400">실패</span>}
                          {r.status === "pending" && <span className="text-sm text-gray-400">대기</span>}
                          {r.status === "generating" && <span className="text-xs text-accent-400">생성 중</span>}
                        </div>
                      </div>
                      {r.status === "complete" && r.audioUrl && (
                        <div className="mt-2">
                          <AudioPlayer src={r.audioUrl} />
                          <a href={r.audioUrl} download className="btn-secondary inline-flex mt-1 text-xs"><DownloadIcon /> Download</a>
                        </div>
                      )}
                      {r.status === "generating" && r.message && (() => {
                        const chunkMatch = r.message.match(/Chunk (\d+)\/(\d+)/);
                        const chunkCur = chunkMatch ? parseInt(chunkMatch[1]) : 0;
                        const chunkTotal = chunkMatch ? parseInt(chunkMatch[2]) : 0;
                        const pct = chunkTotal > 0 ? Math.round((chunkCur / chunkTotal) * 100) : 0;
                        return (
                          <div className="mt-2 space-y-1">
                            {chunkTotal > 0 && (
                              <div className="flex items-center gap-2">
                                <div className="flex-1 h-1.5 rounded-full bg-[#364153] overflow-hidden">
                                  <div className="h-full rounded-full bg-accent-500 transition-all duration-300" style={{ width: `${pct}%` }} />
                                </div>
                                <span className="text-xs font-mono text-accent-400 shrink-0">{chunkCur}/{chunkTotal} ({pct}%)</span>
                              </div>
                            )}
                            <p className="text-xs text-accent-300/70 truncate">{r.message}</p>
                          </div>
                        );
                      })()}
                      {r.status === "error" && r.message && <p className="mt-1 text-xs text-red-300">{r.message}</p>}
                    </div>
                  ))}
                </div>
              </section>
            )}
            {debugLogs.length > 0 && (
              <section className="card">
                <div className="mb-3 flex items-center justify-between">
                  <h2 className="text-lg font-semibold">Debug Console</h2>
                  <button className="btn-secondary text-xs" onClick={() => setDebugLogs([])}>Clear</button>
                </div>
                <div ref={debugRef} className="max-h-[300px] overflow-y-auto rounded-lg bg-[#101828] border border-[#364153] p-3 font-mono text-xs leading-relaxed">
                  {debugLogs.map((log, i) => (
                    <div key={i} className={`${log.includes("ERROR") || log.includes("FETCH ERROR") ? "text-red-400" : log.includes("WARNING") ? "text-amber-400" : log.includes("complete") ? "text-green-400" : "text-gray-300"}`}>{log}</div>
                  ))}
                </div>
              </section>
            )}
          </div>
          <aside className="space-y-6">
            <section className="card">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-lg font-semibold">Voice</h2>
                {ttsEngine === "qwen3" && (
                  <>
                    <button className="btn-secondary text-xs" onClick={() => fileInputRef.current?.click()}><UploadIcon /> Add Voice</button>
                    <input ref={fileInputRef} type="file" accept=".wav,.m4a,.mp3,.ogg,.flac,.webm,audio/*" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) onFileSelected(f); e.target.value = ""; }} />
                  </>
                )}
              </div>
              {activeVoices.length === 0 ? <p className="text-sm text-gray-400">No voices yet.</p> : (
                <div className="space-y-1.5 max-h-[400px] overflow-y-auto pr-1">
                  {activeVoices.map((v) => (
                    <button key={v.id} onClick={() => setSelectedVoice(v.id)}
                      className={`w-full rounded-lg px-3 py-2.5 text-left text-sm transition-colors ${selectedVoice === v.id ? "bg-accent-600/30 border border-accent-500/50 text-white" : "bg-[#1e2939] border border-transparent hover:bg-[#1e2939] text-[#d1d5dc]"}`}>
                      <div className="flex items-center justify-between"><span className="font-medium">{v.name}</span><span className="text-sm text-gray-400">{v.language}</span></div>
                      {v.source === "uploaded" && <span className="mt-0.5 inline-block text-[10px] text-accent-400/70">uploaded</span>}
                      {v.source === "elevenlabs" && <span className="mt-0.5 inline-block text-[10px] text-emerald-400/70">{v.ref_text}</span>}
                    </button>
                  ))}
                </div>
              )}
              {ttsEngine === "qwen3" && currentVoice?.ref_text && <div className="mt-3 rounded-lg bg-[#1e2939] px-3 py-2"><p className="text-[10px] uppercase tracking-wider text-gray-400 mb-1">Reference transcript</p><p className="text-sm text-gray-300 line-clamp-3">{currentVoice.ref_text}</p></div>}
            </section>
          </aside>
        </div>
      )}

      {/* ============ Infographic Tab ============ */}
      {activeTab === "infographic" && (
        <div className="mx-auto max-w-4xl space-y-6">
          <section className="card">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-lg font-semibold">Prompt</h2>
              <button
                className="btn-primary text-sm"
                disabled={infoStatus.status === "generating" || !infoPrompt.trim()}
                onClick={async () => {
                  infoAbortRef.current?.abort();
                  const controller = new AbortController();
                  infoAbortRef.current = controller;
                  setInfoStatus({ status: "generating", message: "Gemini 2.5 Flash로 인포그래픽 생성 중..." });
                  setInfoImageUrl(null);
                  try {
                    const res = await fetch("/api/generate-infographic", {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({ prompt: infoPrompt, project_id: currentProjectId || "" }),
                      signal: controller.signal,
                    });
                    if (!res.ok) {
                      setInfoStatus({ status: "error", message: await res.text().catch(() => "Request failed") });
                      return;
                    }
                    await readSSE(res, (event) => {
                      if (event.status === "complete") {
                        setInfoStatus({ status: "complete", message: `생성 완료 (${event.elapsed}초)` });
                        setInfoImageUrl(event.image_url as string);
                      } else if (event.status === "error") {
                        setInfoStatus({ status: "error", message: event.message as string });
                      } else if (event.message) {
                        setInfoStatus({ status: "generating", message: event.message as string });
                      }
                    });
                  } catch (err) {
                    if ((err as Error).name !== "AbortError") {
                      setInfoStatus({ status: "error", message: (err as Error).message });
                    }
                  }
                }}
              >
                {infoStatus.status === "generating" ? <><Spinner small /> 생성 중...</> : "인포그래픽 생성"}
              </button>
            </div>
            <textarea
              className="input-field min-h-[300px] resize-y text-base leading-relaxed"
              value={infoPrompt}
              onChange={(e) => setInfoPrompt(e.target.value)}
              placeholder="인포그래픽 프롬프트를 입력하세요..."
            />
            {!infoPrompt && srcRewritten && (
              <button
                className="mt-2 btn-secondary text-xs"
                onClick={() => setInfoPrompt(INFOGRAPHIC_PROMPT_TEMPLATE + "\n\n---\n\n아래는 인포그래픽에 사용할 LLM 변환텍스트입니다:\n\n" + srcRewritten)}
              >
                LLM 변환텍스트로 프롬프트 채우기
              </button>
            )}
            {!infoPrompt && !srcRewritten && (
              <p className="mt-2 text-sm text-gray-400">소스 탭의 LLM 변환텍스트가 있으면 자동으로 프롬프트에 포함할 수 있습니다.</p>
            )}
            {infoStatus.status === "generating" && (
              <div className="mt-3 flex items-center gap-2 text-sm text-accent-400"><Spinner small /> {infoStatus.message}</div>
            )}
            {infoStatus.status === "error" && (
              <div className="mt-3 rounded-lg bg-red-500/10 border border-red-500/30 px-4 py-3 text-sm text-red-300">{infoStatus.message}</div>
            )}
          </section>

          {infoImageUrl && (
            <section className="card">
              <div className="mb-3 flex items-center justify-between">
                <h2 className="text-lg font-semibold">생성된 인포그래픽</h2>
                <a
                  href={infoImageUrl}
                  download
                  className="btn-secondary text-sm flex items-center gap-1.5"
                >
                  <DownloadIcon /> 다운로드
                </a>
              </div>
              <div className="rounded-lg overflow-hidden border border-[#364153]">
                <img src={infoImageUrl} alt="Generated infographic" className="w-full h-auto" />
              </div>
              {infoStatus.status === "complete" && infoStatus.message && (
                <p className="mt-2 text-sm text-gray-400">{infoStatus.message}</p>
              )}
            </section>
          )}
        </div>
      )}

      {/* ============ Poem Shorts Tab ============ */}
      {activeTab === "poem-shorts" && (
        <div className="mx-auto max-w-3xl space-y-6">
          {/* Step 1: Poem Input */}
          <section className="card">
            <h2 className="mb-2 text-lg font-semibold">1. 시 입력</h2>
            <p className="mb-3 text-sm text-gray-400">첫째 줄: 제목 &nbsp;|&nbsp; 둘째 줄: 작가 &nbsp;|&nbsp; 나머지: 시 본문</p>
            <textarea className="w-full rounded-lg border border-[#364153] bg-[#101828] p-4 text-base leading-relaxed text-white placeholder-[#6a7282] focus:border-accent-500 focus:outline-none resize-y min-h-[220px]"
              value={psPoem} onChange={(e) => setPsPoem(e.target.value)}
              placeholder={"꽃\n김춘수\n\n내가 그의 이름을 불러주기 전에는\n그는 다만\n하나의 몸짓에 지나지 않았다.\n\n내가 그의 이름을 불러주었을 때\n그는 나에게로 와서\n꽃이 되었다."} />
            {psPoem && <p className="mt-1 text-sm text-gray-400">{psPoem.length}자 · {psPoem.split("\n").filter(l => l.trim()).length}줄</p>}
          </section>

          {/* Step 2: Generate Audio */}
          <section className="card">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-lg font-semibold">2. 음성 생성 (낭독)</h2>
              <button className="rounded-lg bg-accent-600 px-5 py-2 text-sm font-medium text-white hover:bg-accent-700 transition-colors disabled:opacity-50"
                disabled={!psPoem.trim() || psAudioStatus.status === "generating"}
                onClick={handlePsGenerateAudio}>
                {psAudioStatus.status === "generating" ? "생성 중..." : "음성 생성"}
              </button>
            </div>
            {psAudioStatus.status === "generating" && (
              <div className="flex items-center gap-2 text-sm text-accent-400">
                <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>
                {psAudioStatus.message}
              </div>
            )}
            {psAudioStatus.status === "error" && <div className="rounded-lg bg-red-500/10 border border-red-500/30 px-4 py-3 text-sm text-red-300">{psAudioStatus.message}</div>}
            {psAudioStatus.status === "complete" && <p className="text-xs text-green-400">{psAudioStatus.message}</p>}
            {psAudioUrl && (
              <div className="mt-3">
                <audio src={psAudioUrl} controls className="w-full" />
              </div>
            )}
          </section>

          {/* Step 3: Image Prompt */}
          <section className="card">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-lg font-semibold">3. 이미지 프롬프트</h2>
              <button className="rounded-lg border border-[#364153] bg-[#101828] px-5 py-2 text-sm font-medium text-gray-300 hover:text-white hover:border-accent-500 transition-colors disabled:opacity-50"
                disabled={!psPoem.trim() || psImagePromptStatus.status === "generating"}
                onClick={handlePsGenerateImagePrompt}>
                {psImagePromptStatus.status === "generating" ? "생성 중..." : "프롬프트 생성"}
              </button>
            </div>
            {psImagePromptStatus.status === "generating" && (
              <div className="flex items-center gap-2 text-sm text-accent-400">
                <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>
                {psImagePromptStatus.message}
              </div>
            )}
            {psImagePromptStatus.status === "error" && <div className="rounded-lg bg-red-500/10 border border-red-500/30 px-4 py-3 text-sm text-red-300">{psImagePromptStatus.message}</div>}
            {psImagePromptStatus.status === "complete" && <p className="text-xs text-green-400 mb-2">{psImagePromptStatus.message}</p>}
            <textarea className="w-full rounded-lg border border-[#364153] bg-[#101828] p-4 text-sm text-white placeholder-[#6a7282] focus:border-accent-500 focus:outline-none resize-y min-h-[100px]"
              value={psImagePrompt} onChange={(e) => setPsImagePrompt(e.target.value)}
              placeholder="이미지 생성 프롬프트가 여기에 표시됩니다..." />
          </section>

          {/* Step 4: Generate Image */}
          <section className="card">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-lg font-semibold">4. 배경 이미지 생성</h2>
              <button className="rounded-lg bg-accent-600 px-5 py-2 text-sm font-medium text-white hover:bg-accent-700 transition-colors disabled:opacity-50"
                disabled={!psImagePrompt.trim() || psImageStatus.status === "generating"}
                onClick={handlePsGenerateImage}>
                {psImageStatus.status === "generating" ? "생성 중..." : "이미지 생성"}
              </button>
            </div>
            {psImageStatus.status === "generating" && (
              <div className="flex items-center gap-2 text-sm text-accent-400">
                <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>
                {psImageStatus.message}
              </div>
            )}
            {psImageStatus.status === "error" && <div className="rounded-lg bg-red-500/10 border border-red-500/30 px-4 py-3 text-sm text-red-300">{psImageStatus.message}</div>}
            {psImageStatus.status === "complete" && <p className="text-xs text-green-400 mb-2">{psImageStatus.message}</p>}
            {psImageUrl && (
              <div className="rounded-lg overflow-hidden border border-[#364153]">
                <img src={psImageUrl} alt="Poem background" className="w-full h-auto max-h-[500px] object-contain bg-black" />
              </div>
            )}
          </section>

          {/* Step 5: Video Prompt */}
          <section className="card">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-lg font-semibold">5. 영상 프롬프트</h2>
              <button className="rounded-lg border border-[#364153] bg-[#101828] px-5 py-2 text-sm font-medium text-gray-300 hover:text-white hover:border-accent-500 transition-colors disabled:opacity-50"
                disabled={!psPoem.trim() || psVideoPromptStatus.status === "generating"}
                onClick={handlePsGenerateVideoPrompt}>
                {psVideoPromptStatus.status === "generating" ? "생성 중..." : "프롬프트 생성"}
              </button>
            </div>
            {psVideoPromptStatus.status === "generating" && (
              <div className="flex items-center gap-2 text-sm text-accent-400">
                <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>
                {psVideoPromptStatus.message}
              </div>
            )}
            {psVideoPromptStatus.status === "error" && <div className="rounded-lg bg-red-500/10 border border-red-500/30 px-4 py-3 text-sm text-red-300">{psVideoPromptStatus.message}</div>}
            {psVideoPromptStatus.status === "complete" && <p className="text-xs text-green-400 mb-2">{psVideoPromptStatus.message}</p>}
            <textarea className="w-full rounded-lg border border-[#364153] bg-[#101828] p-4 text-sm text-white placeholder-[#6a7282] focus:border-accent-500 focus:outline-none resize-y min-h-[100px]"
              value={psVideoPrompt} onChange={(e) => setPsVideoPrompt(e.target.value)}
              placeholder="영상 구성 프롬프트가 여기에 표시됩니다..." />
          </section>

          {/* Step 6: Generate Video */}
          <section className="card">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-lg font-semibold">6. 영상 생성</h2>
              <button className="rounded-lg bg-accent-600 px-5 py-2 text-sm font-medium text-white hover:bg-accent-700 transition-colors disabled:opacity-50"
                disabled={!psAudioUrl || !psImageUrl || !currentProjectId || psVideoStatus.status === "generating"}
                onClick={handlePsGenerateVideo}>
                {psVideoStatus.status === "generating" ? "생성 중..." : "영상 생성"}
              </button>
            </div>
            {!psAudioUrl && <p className="text-sm text-gray-400">음성과 배경 이미지를 먼저 생성해주세요</p>}
            {psVideoStatus.status === "generating" && (
              <div className="flex items-center gap-2 text-sm text-accent-400">
                <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>
                {psVideoStatus.message}
              </div>
            )}
            {psVideoStatus.status === "error" && <div className="rounded-lg bg-red-500/10 border border-red-500/30 px-4 py-3 text-sm text-red-300">{psVideoStatus.message}</div>}
            {psVideoStatus.status === "complete" && <p className="text-xs text-green-400 mb-2">{psVideoStatus.message}</p>}
            {psVideoUrl && (
              <div className="space-y-3">
                <div className="rounded-lg overflow-hidden border border-[#364153]">
                  <video src={psVideoUrl} controls className="w-full max-h-[600px]" />
                </div>
                <a href={psVideoUrl} download className="inline-flex items-center gap-2 rounded-lg border border-[#364153] bg-[#101828] px-4 py-2 text-sm text-gray-300 hover:text-white hover:border-accent-500 transition-colors">
                  <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
                  다운로드
                </a>
              </div>
            )}
          </section>

          {/* Results overview */}
          {(psAudioUrl || psImageUrl || psVideoUrl) && (
            <section className="card">
              <h2 className="mb-3 text-lg font-semibold">생성 결과</h2>
              <div className="space-y-2 text-sm">
                <div className="flex items-center gap-2">
                  <span className={psAudioUrl ? "text-green-400" : "text-[#6a7282]"}>{psAudioUrl ? "✓" : "○"}</span>
                  <span className={psAudioUrl ? "text-white" : "text-[#6a7282]"}>음성 낭독</span>
                  {psAudioDuration && <span className="text-gray-400">({psAudioDuration.toFixed(1)}초)</span>}
                </div>
                <div className="flex items-center gap-2">
                  <span className={psImageUrl ? "text-green-400" : "text-[#6a7282]"}>{psImageUrl ? "✓" : "○"}</span>
                  <span className={psImageUrl ? "text-white" : "text-[#6a7282]"}>배경 이미지</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className={psVideoUrl ? "text-green-400" : "text-[#6a7282]"}>{psVideoUrl ? "✓" : "○"}</span>
                  <span className={psVideoUrl ? "text-white" : "text-[#6a7282]"}>숏폼 영상</span>
                </div>
              </div>
            </section>
          )}
        </div>
      )}

      {/* ============ Recorder Tab ============ */}
      {activeTab === "recorder" && (
        <div className="grid gap-6 lg:grid-cols-[1fr_340px]">
          <div className="space-y-6">
            <section className="card">
              <h2 className="mb-3 text-lg font-semibold">음성 녹음</h2>
              <div className="relative rounded-lg border border-[#364153] overflow-hidden">
                <canvas ref={recorderCanvasRef} width={800} height={160} className="w-full h-40 block" />
                <div className="absolute top-3 right-3 rounded-lg bg-black/60 px-3 py-1.5">
                  <span className="text-2xl font-mono text-white tabular-nums">
                    {Math.floor(recorderElapsed / 60).toString().padStart(2, "0")}:{(recorderElapsed % 60).toString().padStart(2, "0")}
                  </span>
                </div>
                {!isRecording && recorderElapsed === 0 && !recordingSaved && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <p className="text-[#99a1af] text-sm">녹음 버튼을 눌러 시작하세요</p>
                  </div>
                )}
                {isRecording && (
                  <div className="absolute top-3 left-3 flex items-center gap-2">
                    <span className="h-3 w-3 rounded-full bg-red-500 animate-pulse" />
                    <span className="text-xs font-medium text-red-400">녹음 중</span>
                  </div>
                )}
              </div>
            </section>

            <section className="card">
              <h2 className="mb-4 text-lg font-semibold">설정</h2>
              <div className="mb-4">
                <label className="label">파일명</label>
                <input type="text" className="input-field" value={recordingFilename}
                  onChange={(e) => setRecordingFilename(e.target.value)} disabled={isRecording} />
              </div>
              {!isRecording ? (
                <button className="btn-primary w-full text-lg" onClick={startRecording} disabled={recordingSaving}>
                  <RecordIcon /> 녹음 시작
                </button>
              ) : (
                <button className="w-full rounded-lg bg-red-600 px-4 py-3 text-lg font-medium text-white transition-colors hover:bg-red-700 flex items-center justify-center gap-2" onClick={stopRecording}>
                  <StopIcon /> 정지 및 저장
                </button>
              )}
              {recordingSaving && (
                <div className="mt-3 flex items-center gap-2 text-sm text-[#d1d5dc]">
                  <Spinner /> 저장 중...
                </div>
              )}
              {recordingSaved && (
                <div className="mt-3 space-y-2">
                  <div className="flex items-center gap-2 text-sm text-green-400">
                    <CheckIcon /> 저장 완료! ({recordingFilename})
                  </div>
                  <button className="btn-secondary text-xs" onClick={() => { setRecordingSaved(false); setRecorderElapsed(0); setRecordingFilename(""); resetAudEditor(); }}>
                    새로 녹음하기
                  </button>
                </div>
              )}
            </section>

            {/* Audio Editor */}
            {audBuffer && (
              <section className="card">
                <h2 className="mb-3 text-lg font-semibold">오디오 편집기</h2>
                {/* eslint-disable-next-line jsx-a11y/no-static-element-interactions */}
                <div className="relative rounded-lg border border-[#364153] overflow-hidden cursor-crosshair"
                  onMouseDown={onAudMouseDown}>
                  <canvas ref={audCanvasRef} width={900} height={140} className="w-full h-36 block" />
                </div>

                {/* Time + Selection info */}
                <div className="mt-2 flex items-center justify-between text-xs">
                  <span className="font-mono text-[#d1d5dc] tabular-nums">
                    {fmtAudTime(audTime)} / {fmtAudTime(audBuffer.duration)}
                  </span>
                  {audSelection && Math.abs(audSelection[1] - audSelection[0]) >= 0.05 && (
                    <span className="font-mono text-purple-400 tabular-nums">
                      선택: {fmtAudTime(Math.min(audSelection[0], audSelection[1]))} &ndash; {fmtAudTime(Math.max(audSelection[0], audSelection[1]))}
                      {" "}({fmtAudTime(Math.abs(audSelection[1] - audSelection[0]))})
                    </span>
                  )}
                </div>

                {/* Playback controls */}
                <div className="mt-3 flex items-center gap-2">
                  <button className="btn-secondary px-3 py-2 text-sm" onClick={() => audSeek(-5)} title="-5초"><BwdIcon /></button>
                  {!audPlaying ? (
                    <button className="btn-primary px-4 py-2 text-sm" onClick={audPlay}><PlayIcon /> 재생</button>
                  ) : (
                    <button className="btn-secondary px-4 py-2 text-sm" onClick={audPause}><PauseIcon /> 일시정지</button>
                  )}
                  <button className="btn-secondary px-3 py-2 text-sm" onClick={audStop} title="정지"><AudStopIcon /></button>
                  <button className="btn-secondary px-3 py-2 text-sm" onClick={() => audSeek(5)} title="+5초"><FwdIcon /></button>
                </div>

                {/* Edit tools */}
                <div className="mt-3 flex items-center gap-2 border-t border-[#364153] pt-3">
                  <span className="text-xs text-[#99a1af] mr-1">편집:</span>
                  <button className="btn-secondary px-3 py-1.5 text-xs" onClick={audCut}
                    disabled={!audSelection || Math.abs(audSelection[1] - audSelection[0]) < 0.05}>
                    <ScissorsIcon /> 잘라내기
                  </button>
                  <button className="btn-secondary px-3 py-1.5 text-xs" onClick={audTrim}
                    disabled={!audSelection || Math.abs(audSelection[1] - audSelection[0]) < 0.05}>
                    <TrimIcon /> 선택 영역만
                  </button>
                  <button className="btn-secondary px-3 py-1.5 text-xs" onClick={() => { setAudSelection(null); audSelRef.current = null; drawAudCanvas(); }}
                    disabled={!audSelection}>
                    선택 해제
                  </button>
                </div>

                {/* Save */}
                <div className="mt-3 flex items-center gap-2 border-t border-[#364153] pt-3">
                  <button className="btn-primary px-4 py-2 text-sm" onClick={audSave} disabled={audSaving}>
                    <SaveIcon /> 저장
                  </button>
                  <button className="btn-secondary px-4 py-2 text-sm" onClick={audSaveAs} disabled={audSaving}>
                    <SaveAsIcon /> 다른 이름 저장
                  </button>
                  {audSaving && <Spinner />}
                  {audSaveMsg && <span className={`text-xs ${audSaveMsg.includes("실패") ? "text-red-400" : "text-green-400"}`}>{audSaveMsg}</span>}
                </div>

                <audio ref={audRef} src={audUrl || undefined} preload="auto" style={{ display: "none" }}
                  onEnded={() => { cancelAnimationFrame(audAnimRef.current); setAudPlaying(false); setAudTime(0); drawAudCanvas(); }} />
              </section>
            )}
          </div>
          <aside className="space-y-6">
            <section className="card">
              <h2 className="mb-3 text-lg font-semibold">사용 방법</h2>
              <ul className="space-y-2 text-sm text-[#d1d5dc]">
                <li className="flex gap-2"><span className="text-accent-400">1.</span>녹음 시작 버튼을 클릭하세요.</li>
                <li className="flex gap-2"><span className="text-accent-400">2.</span>마이크 접근 권한을 허용해 주세요.</li>
                <li className="flex gap-2"><span className="text-accent-400">3.</span>녹음이 시작되면 파형이 표시됩니다.</li>
                <li className="flex gap-2"><span className="text-accent-400">4.</span>정지 버튼을 누르면 자동으로 저장됩니다.</li>
                <li className="flex gap-2"><span className="text-accent-400">5.</span>파형을 클릭하여 탐색, 드래그하여 영역을 선택하세요.</li>
                <li className="flex gap-2"><span className="text-accent-400">6.</span>잘라내기/선택 영역만 도구로 오디오를 편집하세요.</li>
              </ul>
            </section>
          </aside>
        </div>
      )}

      {/* ============ Download Tab ============ */}
      {activeTab === "download" && (
        <div className="grid gap-6 lg:grid-cols-[1fr_340px]">
          <div className="space-y-6">
            <section className="card">
              <h2 className="mb-3 text-lg font-semibold">오디오 다운로드</h2>
              <div className="space-y-3">
                <div>
                  <label className="label">URL</label>
                  <input type="text" className="input-field" placeholder="YouTube, SoundCloud 등 URL을 입력하세요"
                    value={dlUrl} onChange={(e) => setDlUrl(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter" && dlStatus.status !== "downloading") startDownload(); }}
                    disabled={dlStatus.status === "downloading"} />
                </div>
                <div>
                  <label className="label">파일명 (선택)</label>
                  <input type="text" className="input-field" placeholder="저장할 파일명 (비어있으면 원본 제목 사용)"
                    value={dlFilename} onChange={(e) => setDlFilename(e.target.value)}
                    disabled={dlStatus.status === "downloading"} />
                </div>
                <div className="flex items-center gap-3">
                  <button className="btn-primary px-6 py-2.5" onClick={startDownload}
                    disabled={!dlUrl.trim() || dlStatus.status === "downloading"}>
                    {dlStatus.status === "downloading" ? <><Spinner small /> 다운로드 중...</> : <><DownloadIcon /> 다운로드</>}
                  </button>
                  {dlStatus.status === "downloading" && (
                    <button className="btn-secondary px-4 py-2.5 text-sm" onClick={() => dlAbortRef.current?.abort()}>취소</button>
                  )}
                </div>
                {dlStatus.status === "error" && (
                  <p className="text-sm text-red-400">{dlStatus.message}</p>
                )}
                {dlStatus.status === "complete" && (
                  <p className="text-sm text-green-400">{dlStatus.message}</p>
                )}
              </div>
            </section>

            {/* Debug Console */}
            <section className="card">
              <div className="mb-2 flex items-center justify-between">
                <h2 className="text-sm font-semibold text-gray-300">Debug Console</h2>
                <button className="text-sm text-gray-400 hover:text-white" onClick={() => setDlLogs([])}>Clear</button>
              </div>
              <div ref={dlLogRef} className="h-40 overflow-y-auto rounded-lg bg-[#101828] border border-[#364153] p-3 font-mono text-sm text-gray-400 space-y-0.5">
                {dlLogs.length === 0 ? (
                  <p className="text-[#4a5565]">다운로드를 시작하면 로그가 표시됩니다...</p>
                ) : dlLogs.map((log, i) => (
                  <p key={i} className={log.includes("오류") || log.includes("실패") ? "text-red-400" : log.includes("완료") || log.includes("저장") ? "text-green-400" : ""}>{log}</p>
                ))}
              </div>
            </section>

            {/* Audio Editor (server-side peaks) */}
            {dlAudReady && dlAudioUrl && (
              <section className="card">
                <h2 className="mb-3 text-lg font-semibold">오디오 편집기</h2>
                <p className="mb-2 text-sm text-gray-400">{dlAudioFilename}</p>
                {/* eslint-disable-next-line jsx-a11y/no-static-element-interactions */}
                <div className="relative rounded-lg border border-[#364153] overflow-hidden cursor-crosshair"
                  onMouseDown={onDlAudMouseDown}>
                  <canvas ref={dlAudCanvasRef} width={900} height={140} className="w-full h-36 block" />
                </div>

                <div className="mt-2 flex items-center justify-between text-xs">
                  <span className="font-mono text-[#d1d5dc] tabular-nums">
                    {fmtAudTime(dlAudTime)} / {fmtAudTime(dlAudDuration)}
                  </span>
                  {dlAudSelection && Math.abs(dlAudSelection[1] - dlAudSelection[0]) >= 0.5 && (
                    <span className="font-mono text-purple-400 tabular-nums">
                      선택: {fmtAudTime(Math.min(dlAudSelection[0], dlAudSelection[1]))} &ndash; {fmtAudTime(Math.max(dlAudSelection[0], dlAudSelection[1]))}
                      {" "}({fmtAudTime(Math.abs(dlAudSelection[1] - dlAudSelection[0]))})
                    </span>
                  )}
                </div>

                <div className="mt-3 flex items-center gap-2">
                  <button className="btn-secondary px-3 py-2 text-sm" onClick={() => dlAudSeek(-5)} title="-5초"><BwdIcon /></button>
                  {!dlAudPlaying ? (
                    <button className="btn-primary px-4 py-2 text-sm" onClick={dlAudPlay}><PlayIcon /> 재생</button>
                  ) : (
                    <button className="btn-secondary px-4 py-2 text-sm" onClick={dlAudPause}><PauseIcon /> 일시정지</button>
                  )}
                  <button className="btn-secondary px-3 py-2 text-sm" onClick={dlAudStop} title="정지"><AudStopIcon /></button>
                  <button className="btn-secondary px-3 py-2 text-sm" onClick={() => dlAudSeek(5)} title="+5초"><FwdIcon /></button>
                  <button className="btn-secondary px-3 py-2 text-sm" onClick={() => dlAudSeek(30)} title="+30초"><FwdIcon /> 30</button>
                  <button className="btn-secondary px-3 py-2 text-sm" onClick={() => dlAudSeek(-30)} title="-30초">30 <BwdIcon /></button>
                </div>

                <div className="mt-3 flex items-center gap-2 border-t border-[#364153] pt-3">
                  <span className="text-xs text-[#99a1af] mr-1">선택 영역:</span>
                  <button className="btn-secondary px-3 py-1.5 text-xs" onClick={() => { setDlAudSelection(null); dlAudSelRef.current = null; drawDlAudCanvas(); }}
                    disabled={!dlAudSelection}>
                    선택 해제
                  </button>
                  <button className="btn-primary px-4 py-1.5 text-xs" onClick={dlAudSaveClip}
                    disabled={dlAudSaving || !dlAudSelection || Math.abs(dlAudSelection[1] - dlAudSelection[0]) < 0.5}>
                    <ScissorsIcon /> 선택 구간 저장
                  </button>
                  {dlAudSaving && <Spinner small />}
                  {dlAudSaveMsg && <span className={`text-xs ${dlAudSaveMsg.includes("실패") ? "text-red-400" : "text-green-400"}`}>{dlAudSaveMsg}</span>}
                </div>

                <audio ref={dlAudRef} src={dlAudioUrl} preload="metadata"
                  onLoadedMetadata={() => { if (dlAudRef.current && dlAudDuration === 0) setDlAudDuration(dlAudRef.current.duration); }}
                  onEnded={() => { cancelAnimationFrame(dlAudAnimRef.current); setDlAudPlaying(false); setDlAudTime(0); drawDlAudCanvas(); }}
                  style={{ display: "none" }} />
              </section>
            )}
          </div>
          <aside className="space-y-6">
            <section className="card">
              <h2 className="mb-3 text-lg font-semibold">사용 방법</h2>
              <ul className="space-y-2 text-sm text-[#d1d5dc]">
                <li className="flex gap-2"><span className="text-accent-400">1.</span>YouTube, SoundCloud 등의 URL을 입력하세요.</li>
                <li className="flex gap-2"><span className="text-accent-400">2.</span>파일명을 지정하거나 비워두면 원본 제목을 사용합니다.</li>
                <li className="flex gap-2"><span className="text-accent-400">3.</span>다운로드 버튼을 클릭하면 MP3로 변환됩니다.</li>
                <li className="flex gap-2"><span className="text-accent-400">4.</span>다운로드 완료 후 파형 에디터가 나타납니다.</li>
                <li className="flex gap-2"><span className="text-accent-400">5.</span>파형을 드래그하여 원하는 구간을 선택하세요.</li>
                <li className="flex gap-2"><span className="text-accent-400">6.</span>&ldquo;선택 영역만&rdquo;으로 원하는 부분만 추출하세요.</li>
                <li className="flex gap-2"><span className="text-accent-400">7.</span>&ldquo;다른 이름으로 저장&rdquo;으로 클립을 저장하세요.</li>
              </ul>
            </section>
          </aside>
        </div>
      )}

      {/* ============ ASR Tab ============ */}
      {activeTab === "asr" && (
        <div className="grid gap-6 lg:grid-cols-[1fr_340px]">
          <div className="space-y-6">
            <section className="card">
              <div className="mb-3 flex items-center justify-between">
                <h2 className="text-lg font-semibold">음성 파일 업로드</h2>
                <button className="btn-secondary text-xs" onClick={fetchAudioFiles} title="저장된 음성 파일 목록">
                  <ListIcon /> 파일 목록
                </button>
              </div>
              <div className="relative flex min-h-[160px] cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-[#364153] bg-[#364153]/50 p-6 transition-colors hover:border-accent-500/50 hover:bg-[#364153]"
                onClick={() => asrFileRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
                onDrop={(e) => { e.preventDefault(); e.stopPropagation(); const f = e.dataTransfer.files?.[0]; if (f) { setAsrFile(f); setSelectedExistingFile(null); setAsrStatus({ status: "idle", message: "" }); setTranscript(null); } }}>
                {asrFile ? (<><MicIcon /><p className="mt-3 text-sm font-medium text-white">{asrFile.name}</p><p className="mt-1 text-sm text-gray-400">{fmtSize(asrFile.size)}</p><p className="mt-2 text-xs text-accent-400">클릭하여 다른 파일 선택</p></>)
                  : selectedExistingFile ? (<><MicIcon /><p className="mt-3 text-sm font-medium text-white">{selectedExistingFile}</p><p className="mt-2 text-xs text-accent-400">클릭하여 다른 파일 선택</p></>)
                  : (<><UploadIcon /><p className="mt-3 text-sm text-gray-300">클릭하거나 파일을 드래그하세요</p><p className="mt-1 text-sm text-gray-400">WAV, MP3, M4A, OGG, FLAC, WebM</p></>)}
                <input ref={asrFileRef} type="file" accept=".wav,.m4a,.mp3,.ogg,.flac,.webm,audio/*" className="hidden"
                  onChange={(e) => { const f = e.target.files?.[0]; if (f) { setAsrFile(f); setSelectedExistingFile(null); setAsrStatus({ status: "idle", message: "" }); setTranscript(null); } e.target.value = ""; }} />
              </div>
            </section>
            <section className="card">
              <h2 className="mb-4 text-lg font-semibold">설정</h2>
              <div><label className="label">화자 수</label><select className="input-field cursor-pointer" value={numSpeakers} onChange={(e) => setNumSpeakers(Number(e.target.value))}>{[1,2,3,4,5].map((n) => <option key={n} value={n}>{n}명</option>)}</select></div>
              <button className="btn-primary mt-6 w-full text-lg" disabled={!hasAsrInput || isTranscribing} onClick={transcribe}>
                {isTranscribing ? <><Spinner /> 처리 중...</> : <><MicIcon /> 음성 인식 시작</>}
              </button>
            </section>
            {(asrStatus.status === "loading" || asrStatus.status === "transcribing") && (
              <section className="card"><div className="space-y-3"><div className="flex items-center gap-3"><Spinner /><span className="text-sm text-gray-300">{asrStatus.message}</span></div>
                <div className="h-2 overflow-hidden rounded-full bg-[#364153]"><div className="progress-pulse h-full rounded-full bg-gradient-to-r from-accent-600 to-purple-500" style={{ width: asrStatus.status === "loading" ? "30%" : "65%", transition: "width 0.5s ease" }} /></div></div></section>)}
            {asrStatus.status === "error" && <section className="card"><div className="rounded-lg border border-red-800/50 bg-red-900/20 px-4 py-3 text-sm text-red-300">{asrStatus.message}</div></section>}
            {transcript && asrStatus.status === "complete" && (
              <section className="card">
                <div className="mb-4 flex items-center justify-between"><h2 className="text-lg font-semibold">인식 결과</h2>
                  <div className="flex gap-2">
                    <button className="btn-primary text-xs" disabled={asrSaving} onClick={saveTranscript}>{asrSaved ? <><CheckIcon /> 저장됨</> : asrSaving ? "저장 중..." : "저장"}</button>
                    <button className="btn-secondary text-xs" onClick={copyTranscript}>{copied ? <><CheckIcon /> 복사됨</> : <><CopyIcon /> 복사</>}</button>
                    <button className="btn-secondary text-xs" onClick={downloadTranscript}><DownloadIcon /> 다운로드</button></div></div>
                <div className="mb-4 flex gap-4 text-sm text-gray-400"><span>오디오 길이: {fmtTime(transcript.duration)}</span><span>처리 시간: {transcript.processing_time.toFixed(1)}초</span></div>
                <div className="space-y-2 max-h-[500px] overflow-y-auto pr-1">
                  {transcript.segments.map((seg, i) => { const color = SPEAKER_COLORS[(seg.speaker - 1) % SPEAKER_COLORS.length]; return (
                    <div key={i} className={`rounded-lg border ${color.border} ${color.bg} px-3 py-2`}>
                      <div className="mb-1 flex items-center gap-2"><span className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${color.label}`}>화자 {seg.speaker}</span><span className="text-[10px] text-gray-400">{fmtTime(seg.start)} &ndash; {fmtTime(seg.end)}</span></div>
                      <p className="text-sm text-white">{seg.text}</p></div>); })}
                </div>
              </section>
            )}
          </div>
          <aside className="space-y-6">
            <section className="card"><h2 className="mb-3 text-lg font-semibold">사용 방법</h2><ul className="space-y-2 text-sm text-gray-300">
              <li className="flex gap-2"><span className="text-accent-400">1.</span>음성 파일을 업로드하거나 파일 목록에서 선택하세요.</li>
              <li className="flex gap-2"><span className="text-accent-400">2.</span>화자 수를 선택하세요 (기본: 2명).</li>
              <li className="flex gap-2"><span className="text-accent-400">3.</span>&ldquo;음성 인식 시작&rdquo; 버튼을 클릭하면 자동 전사됩니다.</li>
              <li className="flex gap-2"><span className="text-accent-400">4.</span>완료 후 &ldquo;글편집&rdquo; 탭에서 편집할 수 있습니다.</li>
            </ul></section>
          </aside>
        </div>
      )}

      {/* ============ Editor Tab ============ */}
      {activeTab === "editor" && (
        <div className="grid gap-6 lg:grid-cols-[1fr_340px]">
          <div className="space-y-6">
            <section className="card">
              <div className="mb-3 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <h2 className="text-lg font-semibold">원본 텍스트</h2>
                  {editorSaved && currentProjectId && <span className="text-xs text-green-400 transition-opacity">✓ 저장됨</span>}
                </div>
                <div className="flex gap-2">
                  <button className="btn-secondary text-xs" onClick={() => editorFileRef.current?.click()} title="텍스트 파일 불러오기"><FileIcon /> 파일 열기</button>
                  <button className="btn-secondary text-xs" onClick={loadTranscriptToEditor} disabled={!transcript}><RefreshIcon /> 음성인식 결과</button>
                </div>
                <input ref={editorFileRef} type="file" accept=".txt,.srt,.vtt,.csv,.tsv,.md,.json" className="hidden"
                  onChange={(e) => { const f = e.target.files?.[0]; if (f) { const r = new FileReader(); r.onload = (ev) => { if (typeof ev.target?.result === "string") { setEditorText(ev.target.result); setRewrittenText(""); setRewriteStatus({ status: "idle", message: "" }); } }; r.readAsText(f); } e.target.value = ""; }} />
              </div>
              <textarea className="input-field min-h-[250px] resize-y text-base leading-relaxed" value={editorText} onChange={(e) => setEditorText(e.target.value)} placeholder="텍스트를 붙여넣거나, 파일을 열거나, 음성인식 결과를 불러오세요..." />
            </section>
            {editorText && (
              <section className="card">
                <h2 className="mb-4 text-lg font-semibold">편집 도구</h2>
                {speakersInText().length > 0 && (<div className="mb-4"><label className="label">화자 삭제</label><div className="flex flex-wrap gap-2">
                  {speakersInText().map((spk) => { const color = SPEAKER_COLORS[(spk - 1) % SPEAKER_COLORS.length]; return (
                    <button key={spk} className={`rounded-lg border ${color.border} ${color.bg} px-3 py-1.5 text-sm font-medium transition-colors hover:opacity-80`} onClick={() => removeSpeaker(spk)}><TrashIcon /> 화자 {spk} 삭제</button>); })}
                </div></div>)}
                <div className="flex gap-3">
                  <button className="btn-secondary flex-1 text-base py-2.5" disabled={!editorText.trim() || isRewriting} onClick={fixTypos}>
                    {isRewriting && rewriteStatus.message.includes("오타") ? <><Spinner /> 수정 중...</> : <><SpellCheckIcon /> 오타수정</>}</button>
                  <button className="btn-primary flex-1 text-base py-2.5" disabled={!editorText.trim() || isRewriting} onClick={rewriteText}>
                    {isRewriting && !rewriteStatus.message.includes("오타") ? <><Spinner /> 변환 중...</> : <><PenIcon /> 박완서 문체</>}</button>
                </div>
                <p className="mt-2 text-sm text-gray-400">모델: {LLM_MODELS.find((m) => m.id === selectedModel)?.label ?? selectedModel} · 설정 탭에서 변경 가능</p>
              </section>
            )}
            {rewriteStatus.status === "rewriting" && (
              <section className="card"><div className="space-y-3"><div className="flex items-center gap-3"><Spinner /><span className="text-sm text-gray-300">{rewriteStatus.message}</span></div>
                <div className="h-2 overflow-hidden rounded-full bg-[#364153]"><div className="progress-pulse h-full rounded-full bg-gradient-to-r from-accent-600 to-purple-500" style={{ width: "60%", transition: "width 0.5s ease" }} /></div></div></section>)}
            {rewriteStatus.status === "error" && <section className="card"><div className="rounded-lg border border-red-800/50 bg-red-900/20 px-4 py-3 text-sm text-red-300">{rewriteStatus.message}</div></section>}
            {rewrittenText && rewriteStatus.status === "complete" && (
              <section className="card">
                <div className="mb-3 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <h2 className="text-lg font-semibold">변환 결과</h2>
                    {rewrittenSaved && currentProjectId && <span className="text-xs text-green-400 transition-opacity">✓ 저장됨</span>}
                  </div>
                  <div className="flex gap-2">
                    <button className="btn-secondary text-xs" onClick={async () => { await navigator.clipboard.writeText(rewrittenText); }}><CopyIcon /> 복사</button>
                    <button className="btn-secondary text-xs" onClick={() => { const b = new Blob([rewrittenText], { type: "text/plain;charset=utf-8" }); const u = URL.createObjectURL(b); const a = document.createElement("a"); a.href = u; a.download = "rewritten.txt"; a.click(); URL.revokeObjectURL(u); }}><DownloadIcon /> 다운로드</button>
                  </div></div>
                <textarea className="input-field min-h-[250px] resize-y text-base leading-relaxed border-purple-500/30 bg-purple-500/10 text-white" value={rewrittenText} onChange={(e) => setRewrittenText(e.target.value)} />
              </section>
            )}
          </div>
          <aside className="space-y-6">
            <section className="card"><h2 className="mb-3 text-lg font-semibold">사용 방법</h2><ul className="space-y-2 text-sm text-gray-300">
              <li className="flex gap-2"><span className="text-accent-400">1.</span>음성인식 탭에서 인식을 완료한 후 이 탭으로 이동하세요.</li>
              <li className="flex gap-2"><span className="text-accent-400">2.</span>화자 삭제 버튼으로 특정 화자의 발화를 제거할 수 있습니다.</li>
              <li className="flex gap-2"><span className="text-accent-400">3.</span>&ldquo;박완서 문체&rdquo; 버튼을 클릭하면 AI가 문체를 변환합니다.</li>
              <li className="flex gap-2"><span className="text-accent-400">4.</span>설정 탭에서 사용할 LLM 모델을 변경할 수 있습니다.</li>
            </ul></section>
            <section className="card"><h2 className="mb-3 text-lg font-semibold">박완서 문체란?</h2><p className="text-sm text-gray-300 leading-relaxed">박완서(1931–2011)는 한국 문학의 거장으로, 일상의 세밀한 관찰, 솔직하고 담담한 어조, 깊은 감정 묘사, 사회 비판적 시선이 특징입니다.</p></section>
          </aside>
        </div>
      )}

      {/* ============ Source Tab ============ */}
      {activeTab === "source" && (
        <div className="mx-auto max-w-4xl space-y-6">
          {/* Source Audio */}
          <section className="card">
            <h2 className="mb-3 text-lg font-semibold">🎵 소스 오디오</h2>
            {sourceProject?.source_audio_original_name ? (
              <div className="flex items-center gap-3 rounded-lg bg-[#1e2939] px-4 py-3">
                <MicIcon />
                <div className="min-w-0 flex-1">
                  <p className="font-medium text-[#e5e7eb] truncate">{sourceProject.source_audio_original_name}</p>
                  <p className="text-sm text-gray-400">{fmtSize(sourceProject.source_audio_size)}{sourceProject.asr_audio_duration > 0 && ` · ${fmtDuration(sourceProject.asr_audio_duration)}`}</p>
                </div>
              </div>
            ) : (
              <p className="text-sm text-gray-400">소스 오디오가 없습니다. 음성녹음 또는 음성인식 탭에서 오디오를 추가하세요.</p>
            )}
          </section>

          {/* Downloaded Audio Files */}
          {srcAudioFiles.length > 0 && (
            <section className="card">
              <div className="mb-3 flex items-center justify-between">
                <h2 className="text-lg font-semibold">🎧 오디오 파일 목록</h2>
                <span className="text-sm text-gray-400">{srcAudioFiles.length}개 파일 · 클릭하여 음성 클립 추출</span>
              </div>
              <div className="max-h-[300px] overflow-y-auto space-y-1.5">
                {srcAudioFiles.map((af) => (
                  <div key={af.filename} className="flex items-center rounded-lg bg-[#1e2939] border border-transparent hover:bg-[#1e2939] hover:border-accent-500/30 transition-colors">
                    <button onClick={() => openClipModal(af)} className="flex-1 min-w-0 px-4 py-3 text-left">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2 min-w-0">
                          <MicIcon />
                          <span className="text-sm text-white truncate">{af.original_name || af.filename}</span>
                          {af.file_type === "output" && <span className="text-[10px] px-1.5 py-0.5 rounded bg-green-900/40 text-green-400 shrink-0">TTS</span>}
                        </div>
                        <div className="flex items-center gap-3 shrink-0 ml-2">
                          <span className="text-sm text-gray-400">{fmtSize(af.size)}</span>
                          <span className="text-sm text-gray-400">{fmtDate(af.created_at || af.modified || "")}</span>
                          <span className="text-[10px] text-accent-400">편집</span>
                        </div>
                      </div>
                    </button>
                    <a href={`/api/audio-files/${encodeURIComponent(af.filename)}`} download
                      className="shrink-0 p-3 text-gray-400 hover:text-accent-400 transition-colors" title="다운로드">
                      <DownloadIcon />
                    </a>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Artifact Files */}
          {srcArtifacts.length > 0 && (
            <section className="card">
              <div className="mb-3 flex items-center justify-between">
                <h2 className="text-lg font-semibold">📄 텍스트 파일</h2>
                <span className="text-sm text-gray-400">{srcArtifacts.length}개 파일</span>
              </div>
              <div className="space-y-1.5">
                {srcArtifacts.map((a) => {
                  const isOpen = expandedArtifacts.has(a.filename);
                  return (
                  <div key={a.filename}>
                    <button onClick={async () => {
                      const next = new Set(expandedArtifacts);
                      if (next.has(a.filename)) { next.delete(a.filename); } else {
                        next.add(a.filename);
                        if (!artifactContents[a.filename]) {
                          try {
                            const res = await fetch(`/api/artifacts/${encodeURIComponent(a.filename)}`);
                            if (res.ok) setArtifactContents((prev) => ({ ...prev, [a.filename]: "" }));
                            const text = await res.text();
                            setArtifactContents((prev) => ({ ...prev, [a.filename]: text }));
                          } catch { /* silent */ }
                        }
                      }
                      setExpandedArtifacts(next);
                    }}
                      className="flex items-center justify-between w-full rounded-lg bg-[#1e2939] border border-transparent hover:bg-[#1e2939] hover:border-accent-500/30 px-4 py-3 text-left transition-colors">
                      <div className="flex items-center gap-2 min-w-0">
                        <span className={`text-xs transition-transform ${isOpen ? "rotate-90" : ""}`}>&#9654;</span>
                        <div className="min-w-0">
                          <span className="text-sm text-white truncate block">{a.filename}</span>
                          <span className="text-[10px] text-gray-400">{a.label}</span>
                        </div>
                      </div>
                      <div className="flex items-center gap-3 shrink-0 ml-2">
                        <span className="text-sm text-gray-400">{fmtSize(a.file_size)}</span>
                        <span className="text-sm text-gray-400">{fmtDate(a.created_at)}</span>
                      </div>
                    </button>
                    {isOpen && (
                      <div className="mt-1 rounded-lg bg-[#101828] border border-[#364153] px-4 py-3">
                        {artifactContents[a.filename] ? (
                          <pre className="text-sm text-[#d1d5dc] whitespace-pre-wrap max-h-[400px] overflow-y-auto leading-relaxed">{artifactContents[a.filename]}</pre>
                        ) : (
                          <p className="text-sm text-gray-400">로딩 중...</p>
                        )}
                      </div>
                    )}
                  </div>
                  );
                })}
              </div>
            </section>
          )}

          {/* Raw Transcript */}
          <section className="card">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-lg font-semibold">🎙 음성인식 원본 (ASR)</h2>
              <div className="flex items-center gap-3">
                {srcTranscript && <span className="text-sm text-gray-300">{srcTranscript.length.toLocaleString()}자 · {countWords(srcTranscript).toLocaleString()}단어</span>}
                <button className="btn-secondary text-xs" disabled={srcSaving === "transcript_text"}
                  onClick={() => saveSourceField("transcript_text", srcTranscript)}>
                  {srcSaving === "transcript_text" ? "저장 중..." : "저장"}
                </button>
              </div>
            </div>
            <textarea className="input-field min-h-[180px] resize-y text-base leading-relaxed"
              value={srcTranscript} onChange={(e) => setSrcTranscript(e.target.value)}
              placeholder="음성인식 결과가 여기에 표시됩니다. 비어있으면 직접 입력할 수 있습니다." />
          </section>

          {/* Edited Transcript */}
          <section className="card">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-lg font-semibold">✏️ 편집된 녹취록</h2>
              <div className="flex items-center gap-3">
                {srcEdited && <span className="text-sm text-gray-300">{srcEdited.length.toLocaleString()}자 · {countWords(srcEdited).toLocaleString()}단어</span>}
                <button className="btn-secondary text-xs" disabled={srcSaving === "edited_transcript"}
                  onClick={() => saveSourceField("edited_transcript", srcEdited)}>
                  {srcSaving === "edited_transcript" ? "저장 중..." : "저장"}
                </button>
              </div>
            </div>
            <textarea className="input-field min-h-[180px] resize-y text-base leading-relaxed border-orange-500/20 bg-orange-500/5"
              value={srcEdited} onChange={(e) => setSrcEdited(e.target.value)}
              placeholder="글편집 탭에서 오타수정/화자삭제 등의 편집 결과가 저장됩니다. 직접 입력할 수도 있습니다." />
          </section>

          {/* Rewritten Text */}
          <section className="card">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-lg font-semibold">🖊 LLM 변환 텍스트</h2>
              <div className="flex items-center gap-3">
                {srcRewritten && <span className="text-sm text-gray-300">{srcRewritten.length.toLocaleString()}자 · {countWords(srcRewritten).toLocaleString()}단어</span>}
                <button className="btn-secondary text-xs" disabled={srcSaving === "rewritten_text"}
                  onClick={() => saveSourceField("rewritten_text", srcRewritten)}>
                  {srcSaving === "rewritten_text" ? "저장 중..." : "저장"}
                </button>
              </div>
            </div>
            <textarea className="input-field min-h-[180px] resize-y text-base leading-relaxed border-violet-500/20 bg-violet-500/5"
              value={srcRewritten} onChange={(e) => setSrcRewritten(e.target.value)}
              placeholder="박완서 문체 변환 결과가 여기에 저장됩니다. 직접 입력할 수도 있습니다." />
            {srcRewritten && (
              <p className="mt-2 text-sm text-gray-400">
                이 텍스트가 오디오북생성 탭의 &quot;Text to Speak&quot;에 사용됩니다.
              </p>
            )}
          </section>

          {/* Short Form Artifacts */}
          {sourceProject && (sourceProject.poem_text || sourceProject.poem_audio_filename || sourceProject.poem_image_prompt || sourceProject.poem_image_filename || sourceProject.poem_video_prompt || sourceProject.poem_video_filename) && (
            <section className="card">
              <h2 className="mb-4 text-lg font-semibold">🎬 시 숏폼 (Short Form)</h2>

              {/* Collapsible Summary */}
              {(() => {
                const summary = sourceProject?.poem_gen_summary ? (() => { try { return JSON.parse(sourceProject.poem_gen_summary); } catch { return null; } })() : null;
                if (!summary || Object.keys(summary).length === 0) return null;
                const stepLabels: Record<string, string> = { audio: "낭독 음성", image_prompt: "이미지 프롬프트", image: "배경 이미지", video_prompt: "영상 프롬프트", video: "숏폼 영상" };
                const totalElapsed = Object.values(summary).reduce((s: number, v: any) => s + (v?.elapsed || 0), 0);
                const KRW_PER_USD = 1380;
                const IMAGE_COST_USD = 0.02;
                const stepCost = (d: any, step: string): number => {
                  if (!d) return 0;
                  if (step === "image") return IMAGE_COST_USD;
                  if (d.input_tokens != null && d.output_tokens != null && d.model) {
                    const rates = LLM_RATES[d.model];
                    if (rates) return (d.input_tokens * rates[0] + d.output_tokens * rates[1]) / 1_000_000;
                  }
                  return 0;
                };
                const totalCostUsd = (["audio", "image_prompt", "image", "video_prompt", "video"] as const).reduce((s, step) => s + stepCost(summary[step], step), 0);
                const totalCostKrw = Math.round(totalCostUsd * KRW_PER_USD);
                const totalTokensIn = (["image_prompt", "video_prompt"] as const).reduce((s, step) => s + (summary[step]?.input_tokens || 0), 0);
                const totalTokensOut = (["image_prompt", "video_prompt"] as const).reduce((s, step) => s + (summary[step]?.output_tokens || 0), 0);
                return (
                  <div className="mb-4">
                    <button onClick={() => setPsSummaryOpen(!psSummaryOpen)}
                      className="flex items-center gap-2 w-full rounded-lg bg-[#1e2939] border border-accent-500/20 px-4 py-2.5 text-left hover:bg-[#1e2939] transition-colors">
                      <span className={`text-xs transition-transform ${psSummaryOpen ? "rotate-90" : ""}`}>&#9654;</span>
                      <span className="text-sm font-medium text-accent-300">Summary</span>
                      <span className="text-sm text-gray-400 ml-auto">{Object.keys(summary).length}단계 · 총 {totalElapsed.toFixed(1)}초 · 토큰 {totalTokensIn.toLocaleString()}→{totalTokensOut.toLocaleString()} · ₩{totalCostKrw.toLocaleString()} (${totalCostUsd.toFixed(4)})</span>
                    </button>
                    {psSummaryOpen && (
                      <div className="mt-1 rounded-lg bg-[#101828] border border-[#364153] px-4 py-3 space-y-3">
                        {(["audio", "image_prompt", "image", "video_prompt", "video"] as const).map((step) => {
                          const d = summary[step];
                          if (!d) return null;
                          const cost = stepCost(d, step);
                          const costKrw = Math.round(cost * KRW_PER_USD);
                          return (
                            <div key={step} className="text-sm leading-relaxed">
                              <div className="flex items-center gap-2 mb-1">
                                <span className="font-medium text-white">{stepLabels[step]}</span>
                                <span className="text-gray-400">—</span>
                                <span className="text-accent-400 font-mono">{d.model}</span>
                              </div>
                              <div className="flex flex-wrap gap-x-4 gap-y-0.5 text-gray-400 pl-2">
                                {d.api_key && <span>API Key: <span className="font-mono text-gray-300">{d.api_key}</span></span>}
                                {d.elapsed != null && <span>생성시간: <span className="text-gray-300">{d.elapsed}초</span></span>}
                                {d.size != null && <span>크기: <span className="text-gray-300">{d.size >= 1048576 ? (d.size / 1048576).toFixed(1) + "MB" : (d.size / 1024).toFixed(0) + "KB"}</span></span>}
                                {d.duration != null && <span>길이: <span className="text-gray-300">{d.duration.toFixed(1)}초</span></span>}
                                {d.input_tokens != null && <span>토큰: <span className="text-gray-300">{d.input_tokens}→{d.output_tokens}</span></span>}
                                {d.voice && <span>음성: <span className="text-gray-300">{d.voice}</span></span>}
                                {cost > 0 && <span>비용: <span className="text-amber-300 font-medium">₩{costKrw.toLocaleString()}</span> <span className="text-gray-400">(${cost.toFixed(4)})</span></span>}
                              </div>
                            </div>
                          );
                        })}
                        <div className="border-t border-[#364153] pt-2 mt-2 flex items-center justify-between text-sm">
                          <span className="text-gray-400">총 토큰: <span className="text-white font-medium">{totalTokensIn.toLocaleString()} → {totalTokensOut.toLocaleString()}</span> ({(totalTokensIn + totalTokensOut).toLocaleString()})</span>
                          <span className="text-gray-400">총 비용: <span className="text-amber-300 font-semibold">₩{totalCostKrw.toLocaleString()}</span> <span className="text-gray-400">(${totalCostUsd.toFixed(4)})</span></span>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })()}

              <div className="space-y-3">
                {/* 1. Audio */}
                <div className="rounded-lg bg-[#1e2939] border border-transparent px-4 py-3">
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono px-1.5 py-0.5 rounded bg-blue-900/40 text-blue-300">1</span>
                      <span className="text-sm font-medium text-white">낭독 음성</span>
                    </div>
                    {sourceProject.poem_audio_filename ? (
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-green-400">{sourceProject.poem_audio_filename}</span>
                        {sourceProject.poem_audio_duration > 0 && <span className="text-sm text-gray-400">{fmtDuration(sourceProject.poem_audio_duration)}</span>}
                        <a href={`/api/outputs/${sourceProject.poem_audio_filename}`} download className="text-gray-400 hover:text-accent-400 transition-colors" title="다운로드"><DownloadIcon /></a>
                      </div>
                    ) : <span className="text-sm text-gray-400">미생성</span>}
                  </div>
                  {sourceProject.poem_audio_filename && (
                    <audio src={`/api/outputs/${sourceProject.poem_audio_filename}`} controls className="w-full h-8 mt-2" />
                  )}
                </div>

                {/* 2. Image Prompt */}
                <div className="rounded-lg bg-[#1e2939] border border-transparent px-4 py-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono px-1.5 py-0.5 rounded bg-purple-900/40 text-purple-300">2</span>
                      <span className="text-sm font-medium text-white">이미지 프롬프트</span>
                    </div>
                    {sourceProject.poem_image_prompt ? <span className="text-xs text-green-400">생성됨</span> : <span className="text-sm text-gray-400">미생성</span>}
                  </div>
                  {sourceProject.poem_image_prompt && (
                    <p className="mt-2 text-sm text-gray-300 leading-relaxed whitespace-pre-wrap max-h-[120px] overflow-y-auto">{sourceProject.poem_image_prompt}</p>
                  )}
                </div>

                {/* 3. Background Image */}
                <div className="rounded-lg bg-[#1e2939] border border-transparent px-4 py-3">
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono px-1.5 py-0.5 rounded bg-emerald-900/40 text-emerald-300">3</span>
                      <span className="text-sm font-medium text-white">배경 이미지</span>
                    </div>
                    {sourceProject.poem_image_filename ? (
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-green-400">{sourceProject.poem_image_filename}</span>
                        <a href={`/api/infographics/${sourceProject.poem_image_filename}`} download className="text-gray-400 hover:text-accent-400 transition-colors" title="다운로드"><DownloadIcon /></a>
                      </div>
                    ) : <span className="text-sm text-gray-400">미생성</span>}
                  </div>
                  {sourceProject.poem_image_filename && (
                    <img src={`/api/infographics/${sourceProject.poem_image_filename}`} alt="배경" className="mt-2 rounded-lg max-h-[200px] object-contain" />
                  )}
                </div>

                {/* 4. Video Prompt */}
                <div className="rounded-lg bg-[#1e2939] border border-transparent px-4 py-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono px-1.5 py-0.5 rounded bg-amber-900/40 text-amber-300">4</span>
                      <span className="text-sm font-medium text-white">영상 프롬프트</span>
                    </div>
                    {sourceProject.poem_video_prompt ? <span className="text-xs text-green-400">생성됨</span> : <span className="text-sm text-gray-400">미생성</span>}
                  </div>
                  {sourceProject.poem_video_prompt && (
                    <p className="mt-2 text-sm text-gray-300 leading-relaxed whitespace-pre-wrap max-h-[120px] overflow-y-auto">{sourceProject.poem_video_prompt}</p>
                  )}
                </div>

                {/* 5. Video */}
                <div className="rounded-lg bg-[#1e2939] border border-transparent px-4 py-3">
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono px-1.5 py-0.5 rounded bg-red-900/40 text-red-300">5</span>
                      <span className="text-sm font-medium text-white">숏폼 영상</span>
                    </div>
                    {sourceProject.poem_video_filename ? (
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-green-400">{sourceProject.poem_video_filename}</span>
                        {sourceProject.poem_gen_elapsed > 0 && <span className="text-sm text-gray-400">{sourceProject.poem_gen_elapsed}초</span>}
                        <a href={`/api/videos/${sourceProject.poem_video_filename}`} download className="text-gray-400 hover:text-accent-400 transition-colors" title="다운로드"><DownloadIcon /></a>
                      </div>
                    ) : <span className="text-sm text-gray-400">미생성</span>}
                  </div>
                  {sourceProject.poem_video_filename && (
                    <video src={`/api/videos/${sourceProject.poem_video_filename}`} controls className="mt-2 rounded-lg w-full max-h-[300px]" />
                  )}
                </div>
              </div>
            </section>
          )}
        </div>
      )}

      {/* ============ Settings Tab ============ */}
      {activeTab === "settings" && (
        <div className="mx-auto max-w-2xl space-y-6">
          <section className="card">
            <h2 className="mb-4 text-lg font-semibold">TTS 엔진 설정</h2>
            <p className="mb-4 text-sm text-gray-300">오디오북 생성에 사용할 TTS 엔진을 선택하세요.</p>
            <div className="space-y-2">
              <button disabled
                className="w-full rounded-lg px-4 py-3 text-left text-sm bg-[#1e2939] border border-transparent text-gray-400 cursor-not-allowed opacity-50">
                <div className="flex items-center justify-between">
                  <div><span className="font-medium">ElevenLabs</span><span className="ml-2 rounded px-1.5 py-0.5 text-[10px] bg-gray-500/20 text-gray-400">비활성</span></div>
                </div>
                <p className="mt-1 text-sm text-gray-400">API 키 갱신 필요</p>
              </button>
              <button onClick={() => { setTtsEngine("qwen3"); setLanguage("Korean"); const pref = voices.find((v) => v.id === "upload-be0916e0-세월"); if (voices.length > 0) setSelectedVoice(pref?.id || voices[0].id); }}
                className={`w-full rounded-lg px-4 py-3 text-left text-sm transition-colors ${ttsEngine === "qwen3" ? "bg-accent-600/30 border border-accent-500/50 text-white" : "bg-[#1e2939] border border-transparent hover:bg-[#1e2939] text-[#d1d5dc]"}`}>
                <div className="flex items-center justify-between">
                  <div><span className="font-medium">Qwen3-TTS 1.7B</span><span className="ml-2 rounded px-1.5 py-0.5 text-[10px] bg-blue-500/20 text-blue-300">Local GPU</span></div>
                  {ttsEngine === "qwen3" && <span className="text-xs text-accent-400">선택됨</span>}
                </div>
                <p className="mt-1 text-sm text-gray-400">로컬 GPU 음성 클론, 무료, 느림 (GB10)</p>
              </button>
            </div>
          </section>
          <section className="card">
            <h2 className="mb-4 text-lg font-semibold">LLM 모델 설정</h2>
            <p className="mb-4 text-sm text-gray-300">글편집 탭의 오타수정 및 문체 변환에 사용할 LLM 모델을 선택하세요.</p>
            <div className="space-y-2">
              {LLM_MODELS.map((m) => (
                <button key={m.id} onClick={() => setSelectedModel(m.id)}
                  className={`w-full rounded-lg px-4 py-3 text-left text-sm transition-colors ${selectedModel === m.id ? "bg-accent-600/30 border border-accent-500/50 text-white" : "bg-[#1e2939] border border-transparent hover:bg-[#1e2939] text-[#d1d5dc]"}`}>
                  <div className="flex items-center justify-between"><div><span className="font-medium">{m.label}</span><span className="ml-2 rounded px-1.5 py-0.5 text-[10px] bg-[#364153] text-gray-300">{m.provider}</span></div>
                    {selectedModel === m.id && <span className="text-xs text-accent-400">선택됨</span>}</div>
                </button>
              ))}
            </div>
          </section>

          {/* Voice Clone */}
          <section className="card">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-semibold">음성 클론 (Voice Clone)</h2>
              {vcMode !== "idle" && vcMode !== "uploading" && (
                <button className="btn-secondary text-xs" onClick={vcReset}>초기화</button>
              )}
            </div>
            <p className="mb-4 text-sm text-gray-300">Qwen3-TTS로 음성을 복제하려면 3초 이상의 음성 샘플과 해당 스크립트를 등록하세요.</p>

            {/* Record / Upload area */}
            {vcMode === "idle" && (
              <div className="space-y-3">
                <div className="flex gap-3">
                  <button className="btn-primary flex-1 flex items-center justify-center gap-2" onClick={vcStartRecording}>
                    <MicIcon /> 녹음하기
                  </button>
                  <button className="btn-secondary flex-1 flex items-center justify-center gap-2" onClick={() => vcFileRef.current?.click()}>
                    <UploadIcon /> 파일 업로드
                  </button>
                  <input ref={vcFileRef} type="file" accept=".wav,.m4a,.mp3,.ogg,.flac,.webm,audio/*" className="hidden"
                    onChange={(e) => { const f = e.target.files?.[0]; if (f) vcHandleFile(f); e.target.value = ""; }} />
                </div>
                <p className="text-sm text-gray-400">WAV, MP3, M4A, WebM 등 지원 (최대 50MB, 권장 3~30초)</p>
              </div>
            )}

            {/* Recording */}
            {vcMode === "recording" && (
              <div className="space-y-3">
                <div className="relative">
                  <canvas ref={vcCanvasRef} width={600} height={80} className="w-full rounded-lg bg-[#1e2939]" />
                  <div className="absolute right-3 top-2 flex items-center gap-2">
                    <span className="h-2.5 w-2.5 rounded-full bg-red-500 animate-pulse" />
                    <span className="font-mono text-sm text-red-400">{Math.floor(vcElapsed / 60)}:{(vcElapsed % 60).toString().padStart(2, "0")}</span>
                  </div>
                </div>
                <button className="btn-primary w-full flex items-center justify-center gap-2" onClick={vcStopRecording}>
                  <StopIcon /> 녹음 중지
                </button>
              </div>
            )}

            {/* Recorded / Done — show form */}
            {(vcMode === "recorded" || vcMode === "done" || vcMode === "uploading") && (
              <div className="space-y-4">
                {/* Audio preview */}
                {vcAudioUrl && (
                  <div className="rounded-lg bg-[#1e2939] p-3">
                    <div className="mb-1 flex items-center justify-between">
                      <span className="text-sm text-gray-400">음성 미리듣기</span>
                      {vcElapsed > 0 && <span className="text-sm text-gray-400">녹음: {vcElapsed}초</span>}
                    </div>
                    <audio src={vcAudioUrl} controls className="w-full h-8" style={{ filter: "invert(0.85) hue-rotate(180deg)" }} />
                  </div>
                )}

                {/* Voice Name */}
                <div>
                  <label className="label">음성 이름 *</label>
                  <input type="text" className="input-field" placeholder="예: 엄마, 아버지, 나레이터" value={vcName}
                    onChange={(e) => setVcName(e.target.value)} disabled={vcMode === "uploading" || vcMode === "done"} />
                </div>

                {/* Language */}
                <div>
                  <label className="label">언어</label>
                  <select className="input-field cursor-pointer" value={vcLang} onChange={(e) => setVcLang(e.target.value)}
                    disabled={vcMode === "uploading" || vcMode === "done"}>
                    {LANGUAGES.map((l) => <option key={l} value={l}>{l}</option>)}
                  </select>
                </div>

                {/* Reference Script */}
                <div>
                  <label className="label">음성 스크립트 (Reference Text)</label>
                  <textarea className="input-field min-h-[80px] resize-y text-base leading-relaxed"
                    placeholder="녹음/업로드한 음성에서 말하는 내용을 정확히 입력하세요. 음성 복제 품질이 크게 향상됩니다."
                    value={vcRefText} onChange={(e) => setVcRefText(e.target.value)}
                    disabled={vcMode === "uploading" || vcMode === "done"} />
                  <p className="mt-1 text-sm text-gray-400">
                    {vcRefText.trim() ? `${vcRefText.trim().length}자` : "스크립트 없이도 음성 복제가 가능하지만, 입력하면 품질이 훨씬 좋��집니다."}
                  </p>
                </div>

                {/* Error */}
                {vcError && <div className="rounded-lg border border-red-800/50 bg-red-900/20 px-4 py-2 text-sm text-red-300">{vcError}</div>}

                {/* Save / Done */}
                {vcMode === "done" && vcSavedVoice ? (
                  <div className="flex items-center gap-2 rounded-lg border border-green-800/50 bg-green-900/20 px-4 py-3 text-sm text-green-300">
                    <CheckIcon />
                    <span>&quot;{vcSavedVoice.name}&quot; 음성이 등록되었습니다. 오디오북생성 탭에서 사용할 수 있습니다.</span>
                  </div>
                ) : (
                  <div className="flex gap-3">
                    <button className="btn-secondary flex-1" onClick={vcReset} disabled={vcMode === "uploading"}>취소</button>
                    <button className="btn-primary flex-1" disabled={!vcName.trim() || vcMode === "uploading"} onClick={vcSave}>
                      {vcMode === "uploading" ? <><Spinner small /> 등록 중...</> : "음성 등록"}
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* Registered voices list */}
            {voices.filter((v) => v.source === "uploaded").length > 0 && (
              <div className="mt-6 border-t border-[#364153] pt-4">
                <h3 className="mb-3 text-sm font-medium text-gray-300">등록된 음성 ({voices.filter((v) => v.source === "uploaded").length})</h3>
                <div className="space-y-1.5 max-h-[200px] overflow-y-auto pr-1">
                  {voices.filter((v) => v.source === "uploaded").map((v) => (
                    <div key={v.id} className="flex items-center justify-between rounded-lg bg-[#1e2939] px-3 py-2 text-sm">
                      <div>
                        <span className="font-medium text-white">{v.name}</span>
                        <span className="ml-2 text-sm text-gray-400">{v.language}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        {v.ref_text && <span className="text-[10px] text-accent-400/70">스크립트 있음</span>}
                        {!v.ref_text && <span className="text-[10px] text-amber-400/70">��크립트 없음</span>}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>
        </div>
      )}

      <footer className="mt-12 border-t border-[#364153] pt-6 text-center text-sm text-gray-400 space-y-1">
        <p>오픈소스 오디오 자서전, 시낭독 영상 제작 소프트웨어</p>
        <p>2026년 5월, Sonny. 소스코드 : <a href="https://github.com/muntakson/voicestudio" target="_blank" rel="noopener noreferrer" className="underline hover:text-[#a78bfa]">github.com/muntakson/voicestudio</a></p>
        <p>제작목적 : AI 코딩 교육 &middot; mtshon@gmail.com</p>
      </footer>

      {/* ---- Poem Picker Modal ---- */}
      {showPoemPicker && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={() => setShowPoemPicker(false)}>
          <div className="card mx-4 w-full max-w-sm space-y-3" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">시 모음 선택</h2>
              <button className="p-1 rounded hover:bg-[#1e2939] text-gray-300" onClick={() => setShowPoemPicker(false)}><CloseIcon /></button>
            </div>
            {poemFiles.length === 0 ? (
              <p className="text-sm text-gray-400 py-4 text-center">파일이 없습니다</p>
            ) : (
              <div className="space-y-1.5 max-h-[400px] overflow-y-auto pr-1">
                {poemFiles.map((p) => (
                  <button key={p.filename} disabled={poemLoading}
                    className="w-full rounded-lg px-4 py-3 text-left text-sm transition-colors bg-[#1e2939] border border-transparent hover:bg-[#1e2939] hover:border-accent-500/30 text-[#d1d5dc]"
                    onClick={() => loadPoemFile(p.filename)}>
                    <span className="font-medium text-white">{p.name}</span>
                  </button>
                ))}
              </div>
            )}
            {poemLoading && <div className="flex items-center justify-center gap-2 py-2"><Spinner small /><span className="text-sm text-gray-300">불러오는 중...</span></div>}
          </div>
        </div>
      )}

      {/* ---- Upload Modal (TTS) ---- */}
      {showUploadModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="card mx-4 w-full max-w-md space-y-4">
            <h2 className="text-lg font-semibold">Register Voice</h2>
            <div className="rounded-lg bg-[#1e2939] px-3 py-2 text-sm text-gray-300">File: {uploadFile?.name}</div>
            <div><label className="label">Speaker Name *</label><input type="text" className="input-field" placeholder="e.g. John, Narrator" value={uploadName} onChange={(e) => setUploadName(e.target.value)} autoFocus /></div>
            <div><label className="label">Reference Transcript (optional)</label><textarea className="input-field min-h-[80px] resize-y text-sm" placeholder="Type what is spoken in the audio..." value={uploadRefText} onChange={(e) => setUploadRefText(e.target.value)} /></div>
            <div><label className="label">Language</label><select className="input-field cursor-pointer" value={uploadLanguage} onChange={(e) => setUploadLanguage(e.target.value)}>{LANGUAGES.map((l) => <option key={l} value={l}>{l}</option>)}</select></div>
            <div className="flex gap-3 pt-2">
              <button className="btn-secondary flex-1" onClick={() => { setShowUploadModal(false); setUploadFile(null); }} disabled={uploading}>Cancel</button>
              <button className="btn-primary flex-1" disabled={!uploadName.trim() || uploading} onClick={submitUpload}>{uploading ? <><Spinner small /> Registering...</> : "Register Voice"}</button>
            </div>
          </div>
        </div>
      )}

      {/* ---- Audio File List Modal ---- */}
      {showAudioList && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="card mx-4 w-full max-w-lg space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">저장된 음성 파일</h2>
              <button className="text-gray-400 hover:text-white" onClick={() => setShowAudioList(false)}><CloseIcon /></button>
            </div>
            {audioFiles.length === 0 ? (
              <p className="text-sm text-gray-400 py-4 text-center">저장된 음성 파일이 없습니다</p>
            ) : (
              <div className="max-h-[400px] overflow-y-auto space-y-1.5">
                {audioFiles.map((af) => (
                  <button key={af.filename} onClick={() => selectExistingAudio(af.filename)}
                    className="w-full rounded-lg bg-[#1e2939] border border-transparent hover:bg-[#1e2939] hover:border-accent-500/30 px-4 py-3 text-left transition-colors">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-white truncate">{af.filename}</span>
                      <span className="text-sm text-gray-400 ml-2 shrink-0">{fmtSize(af.size)}</span>
                    </div>
                    <p className="text-[10px] text-gray-400 mt-0.5">{fmtDate(af.modified || af.created_at || "")}</p>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ---- Voice Clip Editor Modal ---- */}
      {clipModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="card w-full max-w-3xl max-h-[90vh] overflow-y-auto space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">음성 클립 추출</h2>
              <button className="text-gray-400 hover:text-white" onClick={closeClipModal}><CloseIcon /></button>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-300 truncate flex-1">{clipModal.filename}</span>
              {clipDuration > 0 && <span className="text-sm text-gray-400 shrink-0">{fmtAudTime(clipDuration)}</span>}
            </div>

            {/* Clip history */}
            {clipHistory.length > 0 && (
              <div className="rounded-lg bg-[#101828] border border-[#364153] px-3 py-2 space-y-1">
                <p className="text-[10px] text-gray-400 font-medium">편집 이력</p>
                {clipHistory.map((h, i) => (
                  <p key={i} className="text-sm text-gray-400 font-mono">
                    {i + 1}. {h.from} → <span className="text-accent-400">{h.to}</span> ({fmtAudTime(h.start)}~{fmtAudTime(h.end)})
                  </p>
                ))}
              </div>
            )}

            {!clipReady ? (
              <div className="flex items-center gap-3 py-8 justify-center">
                <Spinner /> <span className="text-sm text-gray-300">파형 로딩 중...</span>
              </div>
            ) : (
              <>
                {/* Waveform */}
                {/* eslint-disable-next-line jsx-a11y/no-static-element-interactions */}
                <div className="relative rounded-lg border border-[#364153] overflow-hidden cursor-crosshair"
                  onMouseDown={onClipMouseDown}>
                  <canvas ref={clipCanvasRef} width={900} height={140} className="w-full h-36 block" />
                </div>

                {/* Time display */}
                <div className="flex items-center justify-between text-xs">
                  <span className="font-mono text-[#d1d5dc] tabular-nums">
                    {fmtAudTime(clipTime)} / {fmtAudTime(clipDuration)}
                  </span>
                  {clipSelection && Math.abs(clipSelection[1] - clipSelection[0]) >= 0.3 && (
                    <span className="font-mono text-purple-400 tabular-nums">
                      선택: {fmtAudTime(Math.min(clipSelection[0], clipSelection[1]))} &ndash; {fmtAudTime(Math.max(clipSelection[0], clipSelection[1]))}
                      {" "}({fmtAudTime(Math.abs(clipSelection[1] - clipSelection[0]))})
                    </span>
                  )}
                </div>

                {/* Playback controls */}
                <div className="flex items-center gap-2 flex-wrap">
                  <button className="btn-secondary px-3 py-2 text-sm" onClick={() => clipSeek(-5)} title="-5초"><BwdIcon /></button>
                  {!clipPlaying ? (
                    <button className="btn-primary px-4 py-2 text-sm" onClick={clipPlay}><PlayIcon /> 재생</button>
                  ) : (
                    <button className="btn-secondary px-4 py-2 text-sm" onClick={clipPause}><PauseIcon /> 일시정지</button>
                  )}
                  <button className="btn-secondary px-3 py-2 text-sm" onClick={clipStop} title="정지"><AudStopIcon /></button>
                  <button className="btn-secondary px-3 py-2 text-sm" onClick={() => clipSeek(5)} title="+5초"><FwdIcon /></button>
                  <button className="btn-secondary px-3 py-2 text-sm" onClick={() => clipSeek(30)} title="+30초"><FwdIcon /> 30</button>
                  <button className="btn-secondary px-3 py-2 text-sm" onClick={() => clipSeek(-30)} title="-30초">30 <BwdIcon /></button>
                  <button className="btn-secondary px-3 py-1.5 text-xs ml-auto" onClick={() => { setClipSelection(null); clipSelRef.current = null; drawClipCanvas(); }}
                    disabled={!clipSelection}>
                    선택 해제
                  </button>
                </div>

                {/* Debug Console */}
                <div className="mt-1">
                  <div className="mb-1 flex items-center justify-between">
                    <span className="text-xs font-medium text-gray-400">Debug Console</span>
                    <button className="text-[10px] text-gray-400 hover:text-white" onClick={() => setClipLogs([])}>Clear</button>
                  </div>
                  <div ref={clipLogRef} className="h-28 overflow-y-auto rounded-lg bg-[#101828] border border-[#364153] px-3 py-2 font-mono text-sm text-gray-400 space-y-0.5">
                    {clipLogs.length === 0 ? (
                      <p className="text-[#4a5565]">작업 로그가 여기에 표시됩니다...</p>
                    ) : clipLogs.map((log, i) => (
                      <p key={i} className={log.includes("오류") || log.includes("실패") ? "text-red-400" : log.includes("완료") || log.includes("등록 완료") ? "text-green-400" : log.includes("넣으세요") || log.includes("선택하세요") || log.includes("짧습니다") ? "text-amber-400" : ""}>{log}</p>
                    ))}
                  </div>
                </div>

                {/* Step 1: Crop & Save */}
                <div className="border-t border-[#364153] pt-4 space-y-3">
                  <div className="flex items-center gap-2">
                    <span className="flex h-6 w-6 items-center justify-center rounded-full bg-accent-600 text-xs font-bold text-white">1</span>
                    <h3 className="text-sm font-semibold text-white">구간 선택 &amp; 저장</h3>
                  </div>
                  <p className="text-sm text-gray-400">파형에서 원하는 구간을 드래그로 선택하고, 파일명을 입력한 후 저장하세요. 저장 후 잘린 파일이 다시 로드되어 반복 편집할 수 있습니다.</p>
                  <div className="flex items-center gap-3">
                    <input type="text" className="input-field flex-1" placeholder="저장할 파일명 (예: 아버지_목소리_01)"
                      value={clipCropName} onChange={(e) => setClipCropName(e.target.value)}
                      onKeyDown={(e) => { if (e.key === "Enter" && !clipSaving) clipSaveCrop(); }}
                      disabled={clipSaving} />
                    <button className="btn-primary px-5 py-2.5 shrink-0" onClick={clipSaveCrop}
                      disabled={clipSaving}>
                      {clipSaving ? <><Spinner small /> 저장 중...</> : <><ScissorsIcon /> 클립 저장</>}
                    </button>
                  </div>
                </div>

                {/* Step 2: Register Voice */}
                <div className="border-t border-[#364153] pt-4 space-y-3">
                  <div className="flex items-center gap-2">
                    <span className="flex h-6 w-6 items-center justify-center rounded-full bg-emerald-600 text-xs font-bold text-white">2</span>
                    <h3 className="text-sm font-semibold text-white">음성 등록 (Qwen3-TTS)</h3>
                  </div>
                  <p className="text-sm text-gray-400">현재 로드된 오디오가 원하는 음성 클립이면, 화자 이름을 입력하고 등록하세요. 등록된 음성은 오디오북생성 탭에서 사용할 수 있습니다.</p>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <div>
                      <label className="label">화자 이름 *</label>
                      <input type="text" className="input-field" placeholder="예: 아버지, 어머니, 나레이터"
                        value={clipVoiceName} onChange={(e) => setClipVoiceName(e.target.value)} />
                    </div>
                    <div>
                      <label className="label">참조 텍스트 (선택)</label>
                      <input type="text" className="input-field" placeholder="이 오디오에서 말하는 내용"
                        value={clipRefText} onChange={(e) => setClipRefText(e.target.value)} />
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <button className="btn-primary bg-emerald-600 hover:bg-emerald-500 px-6 py-2.5" onClick={clipRegisterVoice}
                      disabled={clipRegistering}>
                      {clipRegistering ? <><Spinner small /> 등록 중...</> : <><UploadIcon /> 음성 등록</>}
                    </button>
                    {clipSaveMsg && (
                      <span className={`text-xs ${clipSaveMsg.includes("실패") || clipSaveMsg.includes("오류") ? "text-red-400" : clipSaveMsg.includes("완료") ? "text-green-400" : "text-gray-300"}`}>
                        {clipSaveMsg}
                      </span>
                    )}
                  </div>
                </div>

                <audio ref={clipAudioRef} src={clipModal.audioUrl} preload="metadata"
                  onLoadedMetadata={() => { if (clipAudioRef.current && clipDuration === 0) setClipDuration(clipAudioRef.current.duration); }}
                  onEnded={() => { cancelAnimationFrame(clipAnimRef.current); setClipPlaying(false); setClipTime(0); drawClipCanvas(); }}
                  style={{ display: "none" }} />
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Icons                                                              */
/* ------------------------------------------------------------------ */

function AudioPlayer({ src }: { src: string }) {
  const ref = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [cur, setCur] = useState(0);
  const [dur, setDur] = useState(0);

  useEffect(() => {
    const a = ref.current;
    if (!a) return;
    const onTime = () => setCur(a.currentTime);
    const onMeta = () => setDur(a.duration || 0);
    const onEnd = () => setPlaying(false);
    a.addEventListener("timeupdate", onTime);
    a.addEventListener("loadedmetadata", onMeta);
    a.addEventListener("ended", onEnd);
    return () => { a.removeEventListener("timeupdate", onTime); a.removeEventListener("loadedmetadata", onMeta); a.removeEventListener("ended", onEnd); };
  }, []);

  const toggle = () => {
    const a = ref.current;
    if (!a) return;
    if (playing) { a.pause(); setPlaying(false); } else { a.play(); setPlaying(true); }
  };
  const seek = (e: React.MouseEvent<HTMLDivElement>) => {
    const a = ref.current;
    if (!a || !dur) return;
    const rect = e.currentTarget.getBoundingClientRect();
    a.currentTime = ((e.clientX - rect.left) / rect.width) * dur;
  };
  const fmt = (s: number) => { const m = Math.floor(s / 60); const sec = Math.floor(s % 60); return `${m}:${sec.toString().padStart(2, "0")}`; };
  const pct = dur > 0 ? (cur / dur) * 100 : 0;

  return (
    <div className="flex items-center gap-3 rounded-xl bg-[#101828] border border-[#364153] px-4 py-3">
      <button onClick={toggle} className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-accent-500 text-white hover:bg-accent-400 transition-colors">
        {playing
          ? <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20"><rect x="5" y="3" width="4" height="14" rx="1" /><rect x="11" y="3" width="4" height="14" rx="1" /></svg>
          : <svg className="h-4 w-4 ml-0.5" fill="currentColor" viewBox="0 0 20 20"><path d="M6.3 2.841A1.5 1.5 0 004 4.11v11.78a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" /></svg>}
      </button>
      <span className="w-11 shrink-0 text-xs font-mono text-[#e5e7eb]">{fmt(cur)}</span>
      <div className="relative flex-1 cursor-pointer py-1" onClick={seek}>
        <div className="h-2 rounded-full bg-[#364153]">
          <div className="h-full rounded-full bg-accent-400 transition-[width] duration-100" style={{ width: `${pct}%` }} />
        </div>
      </div>
      <span className="w-11 shrink-0 text-xs font-mono text-gray-300">{fmt(dur)}</span>
      <audio ref={ref} src={src} preload="metadata" />
    </div>
  );
}

function Spinner({ small }: { small?: boolean }) {
  const size = small ? "h-4 w-4" : "h-5 w-5";
  return <svg className={`${size} animate-spin`} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" /></svg>;
}
function PlayIcon() { return <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 20 20"><path d="M6.3 2.841A1.5 1.5 0 004 4.11v11.78a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" /></svg>; }
function CheckIcon() { return <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>; }
function DownloadIcon() { return <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5 5m0 0l5-5m-5 5V3" /></svg>; }
function UploadIcon() { return <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M17 8l-5-5m0 0L7 8m5-5v13" /></svg>; }
function MicIcon() { return <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" /><path strokeLinecap="round" strokeLinejoin="round" d="M19 10v2a7 7 0 01-14 0v-2" /><path strokeLinecap="round" strokeLinejoin="round" d="M12 19v4m-4 0h8" /></svg>; }
function CopyIcon() { return <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}><rect x="9" y="9" width="13" height="13" rx="2" ry="2" /><path strokeLinecap="round" strokeLinejoin="round" d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" /></svg>; }
function TrashIcon() { return <svg className="inline h-3.5 w-3.5 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>; }
function PenIcon() { return <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" /></svg>; }
function FileIcon() { return <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" /><polyline points="14 2 14 8 20 8" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" /><line x1="16" y1="13" x2="8" y2="13" stroke="currentColor" strokeLinecap="round" /><line x1="16" y1="17" x2="8" y2="17" stroke="currentColor" strokeLinecap="round" /></svg>; }
function SpellCheckIcon() { return <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4" /><path strokeLinecap="round" strokeLinejoin="round" d="M4 7h16M4 12h8M4 17h6" /></svg>; }
function RefreshIcon() { return <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>; }
function PlusIcon() { return <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" /></svg>; }
function BackIcon() { return <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" /></svg>; }
function BookIcon() { return <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" /></svg>; }
function ListIcon() { return <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" /><circle cx="2" cy="6" r="1" fill="currentColor" /><circle cx="2" cy="12" r="1" fill="currentColor" /><circle cx="2" cy="18" r="1" fill="currentColor" /></svg>; }
function CloseIcon() { return <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>; }
function ChevronIcon({ open }: { open: boolean }) { return <svg className={`h-3.5 w-3.5 transition-transform ${open ? "rotate-90" : ""}`} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" /></svg>; }
function RecordIcon() { return <svg className="h-5 w-5" viewBox="0 0 20 20"><circle cx="10" cy="10" r="6" fill="currentColor" /></svg>; }
function StopIcon() { return <svg className="h-5 w-5" viewBox="0 0 20 20"><rect x="4" y="4" width="12" height="12" rx="1.5" fill="currentColor" /></svg>; }
function PauseIcon() { return <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20"><rect x="5" y="3" width="4" height="14" rx="1" /><rect x="11" y="3" width="4" height="14" rx="1" /></svg>; }
function AudStopIcon() { return <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20"><rect x="4" y="4" width="12" height="12" rx="1" /></svg>; }
function BwdIcon() { return <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20"><path d="M3 10l7-5v10zM10 10l7-5v10z" /></svg>; }
function FwdIcon() { return <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20"><path d="M10 10l-7-5v10zM17 10l-7-5v10z" /></svg>; }
function ScissorsIcon() { return <svg className="inline h-3.5 w-3.5 mr-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}><circle cx="6" cy="6" r="3" /><circle cx="6" cy="18" r="3" /><path strokeLinecap="round" d="M20 4L8.12 15.88M14.47 14.48L20 20M8.12 8.12L12 12" /></svg>; }
function TrimIcon() { return <svg className="inline h-3.5 w-3.5 mr-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9 2v20M15 2v20M4 6h16M4 18h16" /></svg>; }
function SaveIcon() { return <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z" /><polyline points="17 21 17 13 7 13 7 21" stroke="currentColor" /><polyline points="7 3 7 8 15 8" stroke="currentColor" /></svg>; }
function SaveAsIcon() { return <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M17 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v4" /><path strokeLinecap="round" strokeLinejoin="round" d="M17 18l3 3m0 0l3-3m-3 3V14" /></svg>; }
