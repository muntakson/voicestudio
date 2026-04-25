"use client";

import { useState, useEffect, useRef, useCallback } from "react";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface Voice { id: string; name: string; language: string; ref_text: string; source: string; }
interface GenerationStatus { status: "idle" | "loading" | "generating" | "complete" | "error"; message: string; audioUrl: string | null; duration: number | null; }
interface TranscriptSegment { speaker: number; start: number; end: number; text: string; }
interface TranscriptResult { segments: TranscriptSegment[]; full_text: string; duration: number; processing_time: number; }

interface Project {
  id: string; name: string; created_at: string;
  source_audio_filename: string | null; source_audio_original_name: string | null; source_audio_size: number;
  transcript_json: string | null; transcript_text: string | null; num_speakers: number;
  llm_model: string | null; edited_transcript: string | null; rewritten_text: string | null;
  generated_audio_filename: string | null; generated_audio_size: number; generated_audio_duration: number;
  status: string;
  asr_model: string | null; asr_elapsed: number; asr_audio_duration: number; asr_cost: number;
  fix_typos_model: string | null; fix_typos_input_tokens: number; fix_typos_output_tokens: number; fix_typos_elapsed: number; fix_typos_cost: number;
  rewrite_model: string | null; rewrite_input_tokens: number; rewrite_output_tokens: number; rewrite_elapsed: number; rewrite_cost: number;
  tts_engine: string | null; tts_model: string | null; tts_text_chars: number; tts_elapsed: number; tts_cost: number;
  total_cost: number;
}

interface AudioFileInfo { filename: string; size: number; modified: string; }

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
      elements.push(<h2 key={i} className="text-sm font-bold text-[#e8e4f0] mb-1">{line.slice(3)}</h2>);
    } else if (line.startsWith("### ")) {
      elements.push(<h3 key={i} className="text-xs font-semibold text-accent-400 mt-2 mb-0.5">{line.slice(4)}</h3>);
    } else if (line.startsWith("- ")) {
      const content = line.slice(2).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
      elements.push(<p key={i} className="text-[11px] text-[#a09bb5] pl-2" dangerouslySetInnerHTML={{ __html: `• ${content}` }} />);
    } else if (line.startsWith("---")) {
      elements.push(<hr key={i} className="border-[#2e2845] my-1.5" />);
    } else if (line.startsWith("**")) {
      const content = line.replace(/\*\*(.+?)\*\*/g, '<strong class="text-[#e8e4f0]">$1</strong>');
      elements.push(<p key={i} className="text-xs text-[#a09bb5]" dangerouslySetInnerHTML={{ __html: content }} />);
    } else if (line.trim() === "") {
      continue;
    } else {
      elements.push(<p key={i} className="text-[11px] text-[#a09bb5]">{line}</p>);
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
  const [activeTab, setActiveTab] = useState<"recorder" | "source" | "tts" | "asr" | "editor" | "settings">("asr");

  /* TTS */
  const [ttsEngine, setTtsEngine] = useState<"elevenlabs" | "qwen3">("elevenlabs");
  const [text, setText] = useState("");
  const [language, setLanguage] = useState("Auto");
  const [voices, setVoices] = useState<Voice[]>([]);
  const [elVoices, setElVoices] = useState<Voice[]>([]);
  const [selectedVoice, setSelectedVoice] = useState("");
  const [seed, setSeed] = useState("");
  const [gen, setGen] = useState<GenerationStatus>({ status: "idle", message: "", audioUrl: null, duration: null });
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadName, setUploadName] = useState("");
  const [uploadRefText, setUploadRefText] = useState("");
  const [uploadLanguage, setUploadLanguage] = useState("Auto");
  const [uploading, setUploading] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

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

  /* Source tab */
  const [sourceProject, setSourceProject] = useState<Project | null>(null);
  const [srcTranscript, setSrcTranscript] = useState("");
  const [srcEdited, setSrcEdited] = useState("");
  const [srcRewritten, setSrcRewritten] = useState("");
  const [srcSaving, setSrcSaving] = useState<string | null>(null);

  /* Settings */
  const [selectedModel, setSelectedModel] = useState("claude-sonnet-4-6");

  /* ================================================================ */
  /*  Project / Landing logic                                          */
  /* ================================================================ */

  const fetchProjects = useCallback(async () => {
    try { const res = await fetch("/api/projects"); if (res.ok) setProjects(await res.json()); } catch { /* silent */ }
  }, []);

  useEffect(() => { fetchProjects(); }, [fetchProjects]);

  const createProject = async () => {
    if (!newProjectName.trim()) return;
    try {
      const res = await fetch("/api/projects", {
        method: "POST", headers: { "Content-Type": "application/json" },
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
    setAsrFile(null); setAsrStatus({ status: "idle", message: "" }); setTranscript(null); setCopied(false);
    setEditorText(""); setRewrittenText(""); setRewriteStatus({ status: "idle", message: "" });
    setText(""); setGen({ status: "idle", message: "", audioUrl: null, duration: null });
    setIsRecording(false); setRecorderElapsed(0); setRecordingFilename(""); setRecordingSaving(false); setRecordingSaved(false);
    if (audRef.current) audRef.current.pause();
    cancelAnimationFrame(audAnimRef.current);
    if (audBlobRef.current) { URL.revokeObjectURL(audBlobRef.current); audBlobRef.current = null; }
    setAudBuffer(null); setAudUrl(null); setAudPlaying(false); setAudTime(0);
    setAudSelection(null); audSelRef.current = null; setAudSaving(false); setAudSaveMsg("");
    audPeaksRef.current = [];
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
      if (proj.generated_audio_filename) {
        setGen({ status: "complete", message: "Done!", audioUrl: `/api/outputs/${proj.generated_audio_filename}`, duration: null });
      }
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
      const res = await fetch("/api/audio-files");
      if (res.ok) setAudioFiles(await res.json());
    } catch { /* silent */ }
    setShowAudioList(true);
  };

  const selectExistingAudio = async (filename: string) => {
    setShowAudioList(false);
    if (!currentProjectId) return;

    asrAbortRef.current?.abort();
    const controller = new AbortController();
    asrAbortRef.current = controller;
    setAsrFile(null);
    setAsrStatus({ status: "loading", message: "오디오 파일 처리 중..." });
    setTranscript(null); setCopied(false);

    const form = new FormData();
    form.append("existing_file", filename);
    form.append("num_speakers", numSpeakers.toString());

    try {
      const res = await fetch(`/api/projects/${currentProjectId}/transcribe`, {
        method: "POST", body: form, signal: controller.signal,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "요청 실패" }));
        setAsrStatus({ status: "error", message: err.detail || "요청 실패" });
        return;
      }
      await readSSE(res, (event) => {
        if (event.status === "complete") {
          const tr = { segments: event.segments as TranscriptSegment[], full_text: event.full_text as string, duration: event.duration as number, processing_time: event.processing_time as number };
          setTranscript(tr);
          setAsrStatus({ status: "complete", message: "완료!" });
        } else if (event.status === "error") {
          setAsrStatus({ status: "error", message: (event.message as string) || "실패" });
        } else {
          setAsrStatus({ status: event.status as string, message: (event.message as string) || "" });
        }
      });
    } catch (err: unknown) {
      if ((err as Error).name === "AbortError") return;
      setAsrStatus({ status: "error", message: err instanceof Error ? err.message : "알 수 없는 오류" });
    }
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
    ctx.fillStyle = "#1e1a2e";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#2e2845";
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
            ctx.fillStyle = "#1e1a2e";
            ctx.fillRect(0, 0, w, h);
            ctx.strokeStyle = "#2e2845";
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

    ctx.fillStyle = "#1e1a2e";
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
      ctx.fillStyle = peakT <= curTime ? "#8b5cf6" : "#4c3d6e";
      ctx.fillRect(i * barW, (h - barH) / 2, Math.max(1, barW - 0.5), barH);
    }

    ctx.strokeStyle = "#3d3555"; ctx.lineWidth = 0.5;
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
  /*  TTS logic                                                        */
  /* ================================================================ */

  const fetchVoices = useCallback(async () => {
    try {
      const res = await fetch("/api/voices");
      if (!res.ok) return;
      const data = await res.json();
      const list: Voice[] = data.voices ?? [];
      setVoices(list);
    } catch { /* silent */ }
  }, []);

  const fetchElVoices = useCallback(async () => {
    try {
      const res = await fetch("/api/elevenlabs-voices");
      if (!res.ok) return;
      const data = await res.json();
      const list: Voice[] = data.voices ?? [];
      setElVoices(list);
      if (list.length > 0) setSelectedVoice((prev) => (!prev || !prev.startsWith("el_")) ? list[0].id : prev);
    } catch { /* silent */ }
  }, []);

  useEffect(() => { fetchVoices(); fetchElVoices(); }, []);

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

  const generate = async () => {
    if (!text.trim() || !selectedVoice) return;
    let voiceToUse = selectedVoice;
    if (ttsEngine === "elevenlabs" && !voiceToUse.startsWith("el_")) {
      const fallback = elVoices[0]?.id;
      if (!fallback) { setGen({ status: "error", message: "ElevenLabs 음성을 먼저 선택하세요", audioUrl: null, duration: null }); return; }
      voiceToUse = fallback;
      setSelectedVoice(fallback);
    }
    if (ttsEngine === "qwen3" && voiceToUse.startsWith("el_")) {
      const fallback = voices[0]?.id;
      if (!fallback) { setGen({ status: "error", message: "Qwen3 음성을 먼저 선택하세요", audioUrl: null, duration: null }); return; }
      voiceToUse = fallback;
      setSelectedVoice(fallback);
    }
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setGen({ status: "loading", message: ttsEngine === "elevenlabs" ? "ElevenLabs 준비 중..." : "Preparing...", audioUrl: null, duration: null });
    const body: Record<string, unknown> = { text: text.trim(), voice_id: voiceToUse, language, engine: ttsEngine };
    if (seed.trim() && ttsEngine === "qwen3") body.seed = parseInt(seed, 10);
    if (currentProjectId && currentProjectName) {
      body.output_name = currentProjectName;
    }
    try {
      const res = await fetch("/api/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: controller.signal });
      if (!res.ok) { setGen({ status: "error", message: await res.text().catch(() => "Request failed"), audioUrl: null, duration: null }); return; }
      await readSSE(res, (event) => {
        if (event.status === "complete") {
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
        } else if (event.status === "error") { setGen({ status: "error", message: (event.message as string) ?? "Failed", audioUrl: null, duration: null }); }
        else { setGen((p) => ({ ...p, status: event.status as GenerationStatus["status"], message: (event.message as string) ?? p.message })); }
      });
    } catch (err: unknown) {
      if ((err as Error).name === "AbortError") return;
      setGen({ status: "error", message: err instanceof Error ? err.message : "Unknown error", audioUrl: null, duration: null });
    }
  };

  const isGenerating = gen.status === "loading" || gen.status === "generating";
  const activeVoices = ttsEngine === "elevenlabs" ? elVoices : voices;
  const currentVoice = activeVoices.find((v) => v.id === selectedVoice);

  /* ================================================================ */
  /*  ASR logic                                                        */
  /* ================================================================ */

  const transcribe = async () => {
    if (!asrFile) return;
    asrAbortRef.current?.abort();
    const controller = new AbortController();
    asrAbortRef.current = controller;
    setAsrStatus({ status: "loading", message: "오디오 파일 처리 중..." });
    setTranscript(null); setCopied(false);

    const form = new FormData();
    form.append("file", asrFile);
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

  const isTranscribing = asrStatus.status === "loading" || asrStatus.status === "transcribing";
  const copyTranscript = async () => { if (!transcript?.full_text) return; await navigator.clipboard.writeText(transcript.full_text); setCopied(true); setTimeout(() => setCopied(false), 2000); };
  const downloadTranscript = () => { if (!transcript?.full_text) return; const b = new Blob([transcript.full_text], { type: "text/plain;charset=utf-8" }); const u = URL.createObjectURL(b); const a = document.createElement("a"); a.href = u; a.download = "transcript.txt"; a.click(); URL.revokeObjectURL(u); };

  /* ================================================================ */
  /*  Editor logic                                                     */
  /* ================================================================ */

  const loadTranscriptToEditor = useCallback(() => {
    if (!transcript) return;
    setEditorText(transcript.segments.map((s) => `[화자 ${s.speaker}] ${s.text}`).join("\n"));
    setRewrittenText(""); setRewriteStatus({ status: "idle", message: "" });
  }, [transcript]);

  useEffect(() => { if (activeTab === "editor" && transcript && !editorText) loadTranscriptToEditor(); }, [activeTab, transcript, editorText, loadTranscriptToEditor]);

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
        <header className="mb-10 text-center">
          <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
            <span className="bg-gradient-to-r from-accent-400 to-purple-400 bg-clip-text text-transparent">
              Voice Studio
            </span>
          </h1>
          <p className="mt-2 text-lg text-[#a09bb5]">아버지 어머니의 이야기를 기록합니다</p>
          <p className="mt-1 text-sm text-[#6b6580]">AI 음성인식, 문체 변환, 음성 합성으로 만드는 회고록</p>
        </header>

        <div className="mb-8 flex items-center justify-between">
          <h2 className="text-xl font-semibold text-[#e8e4f0]">프로젝트</h2>
          <button className="btn-primary text-sm" onClick={() => { setNewProjectName(""); setShowNewModal(true); }}>
            <PlusIcon /> 새 프로젝트
          </button>
        </div>

        {projects.length === 0 ? (
          <div className="card text-center py-16">
            <BookIcon />
            <p className="mt-4 text-[#a09bb5]">아직 프로젝트가 없습니다</p>
            <p className="mt-1 text-sm text-[#6b6580]">새 프로젝트를 만들어 회고록 작업을 시작하세요</p>
            <button className="btn-primary mt-6" onClick={() => { setNewProjectName(""); setShowNewModal(true); }}>
              <PlusIcon /> 새 프로젝트 만들기
            </button>
          </div>
        ) : (
          <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
            {projects.map((p) => (
              <div key={p.id} className="flex flex-col rounded-2xl border border-[#3d3556] bg-[#252040] p-6 shadow-xl hover:border-[#5b4f8a] transition-colors">
                {/* Header */}
                <div className="mb-4 flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <h3 className="text-xl font-bold text-white leading-tight truncate">{p.name}</h3>
                    <p className="mt-1 text-sm text-[#b8b0cc]">{fmtDate(p.created_at)}</p>
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
                  <div className="mb-3 flex items-center gap-2 rounded-lg bg-[#1e1a30] px-3 py-2">
                    <MicIcon />
                    <span className="truncate text-sm font-medium text-[#ddd8ee]">{p.source_audio_original_name}</span>
                    <span className="ml-auto shrink-0 text-xs text-[#b8b0cc]">{fmtSize(p.source_audio_size)}</span>
                  </div>
                )}

                {/* AI Services usage */}
                {(p.transcript_text || p.fix_typos_model || p.rewrite_model || p.generated_audio_filename) && (() => {
                  const totalCost = (p.asr_cost || 0) + (p.fix_typos_cost || 0) + (p.rewrite_cost || 0) + (p.tts_cost || 0);
                  const fmtCost = (c: number) => c === 0 ? "무료" : c < 0.01 ? `$${c.toFixed(4)}` : `$${c.toFixed(3)}`;

                  return (
                    <div className="mb-3 rounded-lg border border-[#3d3556] bg-[#1a1630] overflow-hidden">
                      <div className="px-3 py-2 border-b border-[#3d3556] flex items-center justify-between">
                        <span className="text-xs font-semibold uppercase tracking-wider text-[#b8b0cc]">AI Services</span>
                        {totalCost > 0 && <span className="text-xs font-bold text-amber-300">총 {fmtCost(totalCost)}</span>}
                      </div>
                      <div className="divide-y divide-[#2e2845]">
                        {/* ASR */}
                        {p.transcript_text && (
                          <div className="px-3 py-2">
                            <div className="flex items-center justify-between">
                              <span className="text-xs font-semibold text-sky-300">🎙 ASR</span>
                              <span className="text-[11px] font-medium text-emerald-400">무료</span>
                            </div>
                            <div className="mt-0.5 text-[11px] text-[#b8b0cc]">
                              Groq Whisper large-v3 + WavLM ({p.num_speakers || 2}명)
                              {p.asr_audio_duration > 0 && <span className="ml-1 text-[#8a84a0]">· {fmtDuration(p.asr_audio_duration)}</span>}
                              {p.asr_elapsed > 0 && <span className="ml-1 text-[#8a84a0]">· {fmtDuration(p.asr_elapsed)}</span>}
                            </div>
                          </div>
                        )}
                        {/* Fix Typos */}
                        {p.fix_typos_model && (
                          <div className="px-3 py-2">
                            <div className="flex items-center justify-between">
                              <span className="text-xs font-semibold text-orange-300">✏️ 오타수정</span>
                              <span className="text-[11px] font-medium text-amber-300">{fmtCost(p.fix_typos_cost || 0)}</span>
                            </div>
                            <div className="mt-0.5 text-[11px] text-[#b8b0cc]">
                              {LLM_MODELS.find((m) => m.id === p.fix_typos_model)?.label ?? p.fix_typos_model}
                              <span className="ml-1.5 text-[#8a84a0]">{(p.fix_typos_input_tokens + p.fix_typos_output_tokens).toLocaleString()} tok</span>
                              <span className="ml-1 text-[#706a85]">({p.fix_typos_input_tokens.toLocaleString()}↓ {p.fix_typos_output_tokens.toLocaleString()}↑)</span>
                            </div>
                          </div>
                        )}
                        {/* Rewrite */}
                        {p.rewrite_model && (
                          <div className="px-3 py-2">
                            <div className="flex items-center justify-between">
                              <span className="text-xs font-semibold text-violet-300">🖊 문체변환</span>
                              <span className="text-[11px] font-medium text-amber-300">{fmtCost(p.rewrite_cost || 0)}</span>
                            </div>
                            <div className="mt-0.5 text-[11px] text-[#b8b0cc]">
                              {LLM_MODELS.find((m) => m.id === p.rewrite_model)?.label ?? p.rewrite_model}
                              <span className="ml-1.5 text-[#8a84a0]">{(p.rewrite_input_tokens + p.rewrite_output_tokens).toLocaleString()} tok</span>
                              <span className="ml-1 text-[#706a85]">({p.rewrite_input_tokens.toLocaleString()}↓ {p.rewrite_output_tokens.toLocaleString()}↑)</span>
                            </div>
                          </div>
                        )}
                        {/* TTS */}
                        {p.generated_audio_filename && (
                          <div className="px-3 py-2">
                            <div className="flex items-center justify-between">
                              <span className="text-xs font-semibold text-emerald-300">🔊 TTS</span>
                              <span className={`text-[11px] font-medium ${p.tts_engine === "elevenlabs" ? "text-amber-300" : "text-emerald-400"}`}>
                                {p.tts_engine === "elevenlabs" ? fmtCost(p.tts_cost || 0) : "로컬 GPU"}
                              </span>
                            </div>
                            <div className="mt-0.5 text-[11px] text-[#b8b0cc]">
                              {p.tts_model || (p.tts_engine === "elevenlabs" ? "eleven_flash_v2_5" : "Qwen3-TTS-1.7B")}
                              {p.tts_engine === "elevenlabs" && p.tts_text_chars > 0 && (
                                <span className="ml-1.5 text-[#8a84a0]">{p.tts_text_chars.toLocaleString()} chars</span>
                              )}
                              {p.generated_audio_duration > 0 && <span className="ml-1 text-[#8a84a0]">· {fmtDuration(p.generated_audio_duration)}</span>}
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
                    <div className="mb-3 flex items-center gap-2 rounded-lg bg-[#1e1a30] px-3 py-2">
                      <span className="text-xs text-[#b8b0cc]">📝 {label}: {bestText.length.toLocaleString()}자 · {wc.toLocaleString()}단어</span>
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
                      <div className="mt-1.5 max-h-[200px] overflow-y-auto rounded-lg border border-sky-500/20 bg-[#1a1630] px-4 py-3 text-sm leading-relaxed text-[#ddd8ee] whitespace-pre-wrap">
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
                      <div className="mt-1.5 max-h-[200px] overflow-y-auto rounded-lg border border-violet-500/20 bg-violet-500/5 px-4 py-3 text-sm leading-relaxed text-[#ddd8ee] whitespace-pre-wrap">
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
                    <div className="mt-1.5 max-h-[300px] overflow-y-auto rounded-lg border border-emerald-500/20 bg-emerald-500/5 px-4 py-3 text-sm leading-relaxed text-[#ddd8ee]">
                      {renderMarkdown(buildProjectSummary(p))}
                    </div>
                  )}
                </div>

                <div className="mt-auto flex gap-3 pt-4 border-t border-[#3d3556]">
                  <button className="btn-primary flex-1 text-base py-2.5" onClick={() => openProject(p.id)}>열기</button>
                  <button className="inline-flex items-center justify-center rounded-lg border border-[#3d3556] bg-[#1e1a30] px-4 py-2.5 text-[#b8b0cc] hover:bg-red-500/15 hover:text-red-300 hover:border-red-500/30 transition-colors" onClick={() => deleteProject(p.id)}><TrashIcon /></button>
                </div>
              </div>
            ))}
          </div>
        )}

        <footer className="mt-12 border-t border-[#2e2845] pt-6 text-center text-xs text-[#6b6580]">
          Voice Studio &mdash; Powered by Qwen3-TTS &amp; Whisper
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
      </div>
    );
  }

  /* ================================================================ */
  /*  Render: Studio View                                              */
  /* ================================================================ */

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      <header className="mb-10 text-center">
        <button onClick={goToLanding} className="mb-4 inline-flex items-center gap-1 text-sm text-[#a09bb5] hover:text-white transition-colors">
          <BackIcon /> 프로젝트 목록
        </button>
        <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
          <span className="bg-gradient-to-r from-accent-400 to-purple-400 bg-clip-text text-transparent">Voice Studio</span>
        </h1>
        <p className="mt-2 text-[#a09bb5]">오디오회고록 제작 서비스 - 음성인식, 녹취록 생성, AI편집, AI 오디오북 생성</p>
        <div className="mt-6 inline-flex rounded-lg border border-[#2e2845] bg-[#1a1726] p-1">
          {(["recorder", "asr", "editor", "source", "tts", "settings"] as const).map((tab) => (
            <button key={tab} onClick={() => setActiveTab(tab)}
              className={`rounded-md px-6 py-2 text-sm font-medium transition-colors ${activeTab === tab ? "bg-accent-600 text-white" : "text-[#a09bb5] hover:text-white"}`}>
              {tab === "recorder" ? "음성녹음" : tab === "source" ? "소스" : tab === "tts" ? "오디오북생성" : tab === "asr" ? "음성인식" : tab === "editor" ? "글편집" : "설정"}
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
                <h2 className="text-lg font-semibold">Text to Speak</h2>
                {text.trim() && <span className="text-xs text-[#b8b0cc]">{text.length.toLocaleString()}자 · {countWords(text).toLocaleString()}단어{ttsEngine === "elevenlabs" && <span className="ml-1 text-amber-300">(≈${(text.length * EL_COST_PER_CHAR).toFixed(3)})</span>}</span>}
              </div>
              <textarea className="input-field min-h-[200px] resize-y text-sm leading-relaxed" placeholder="Type or paste the text you want to convert to speech..."
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
                  <button onClick={() => { setTtsEngine("elevenlabs"); setSelectedVoice(elVoices[0]?.id || ""); }}
                    className={`flex-1 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${ttsEngine === "elevenlabs" ? "bg-accent-600/30 border border-accent-500/50 text-white" : "bg-[#1e1a2e] border border-transparent hover:bg-[#2a2540] text-[#c0bcd0]"}`}>
                    ElevenLabs <span className="text-[10px] text-accent-400/70">Cloud</span>
                  </button>
                  <button onClick={() => { setTtsEngine("qwen3"); setSelectedVoice(voices[0]?.id || ""); }}
                    className={`flex-1 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${ttsEngine === "qwen3" ? "bg-accent-600/30 border border-accent-500/50 text-white" : "bg-[#1e1a2e] border border-transparent hover:bg-[#2a2540] text-[#c0bcd0]"}`}>
                    Qwen3-TTS <span className="text-[10px] text-accent-400/70">Local</span>
                  </button>
                </div>
              </div>
              {ttsEngine === "qwen3" && (
                <div className="grid gap-4 sm:grid-cols-2">
                  <div><label className="label">Language</label><select className="input-field cursor-pointer" value={language} onChange={(e) => setLanguage(e.target.value)}>{LANGUAGES.map((l) => <option key={l} value={l}>{l}</option>)}</select></div>
                  <div><label className="label">Seed (optional)</label><input type="number" className="input-field" placeholder="Random" value={seed} onChange={(e) => setSeed(e.target.value)} /></div>
                </div>
              )}
              <button className="btn-primary mt-6 w-full text-lg" disabled={!text.trim() || !selectedVoice || isGenerating} onClick={generate}>
                {isGenerating ? <><Spinner /> Generating...</> : <><PlayIcon /> Generate Speech</>}
              </button>
              {ttsEngine === "elevenlabs" && <p className="mt-2 text-xs text-[#6b6580]">ElevenLabs 클라우드 TTS (빠름, 다국어 지원)</p>}
            </section>
            {gen.status !== "idle" && (
              <section className="card">
                <h2 className="mb-4 text-lg font-semibold">Output</h2>
                {(gen.status === "loading" || gen.status === "generating") && (
                  <div className="space-y-3"><div className="flex items-center gap-3"><Spinner /><span className="text-sm text-[#a09bb5]">{gen.message}</span></div>
                    <div className="h-2 overflow-hidden rounded-full bg-[#241f33]"><div className="progress-pulse h-full rounded-full bg-gradient-to-r from-accent-600 to-purple-500" style={{ width: gen.status === "loading" ? "40%" : "75%", transition: "width 0.5s ease" }} /></div></div>)}
                {gen.status === "error" && <div className="rounded-lg border border-red-800/50 bg-red-900/20 px-4 py-3 text-sm text-red-300">{gen.message}</div>}
                {gen.status === "complete" && gen.audioUrl && (
                  <div className="space-y-4"><div className="flex items-center gap-2 text-sm text-green-400"><CheckIcon /><span>Done!{gen.duration != null && <span className="ml-2 text-[#6b6580]">({gen.duration.toFixed(1)}s)</span>}</span></div>
                    <AudioPlayer src={gen.audioUrl} /><a href={gen.audioUrl} download className="btn-secondary inline-flex mt-2"><DownloadIcon /> Download</a></div>)}
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
              {activeVoices.length === 0 ? <p className="text-sm text-[#6b6580]">No voices yet.</p> : (
                <div className="space-y-1.5 max-h-[400px] overflow-y-auto pr-1">
                  {activeVoices.map((v) => (
                    <button key={v.id} onClick={() => setSelectedVoice(v.id)}
                      className={`w-full rounded-lg px-3 py-2.5 text-left text-sm transition-colors ${selectedVoice === v.id ? "bg-accent-600/30 border border-accent-500/50 text-white" : "bg-[#1e1a2e] border border-transparent hover:bg-[#2a2540] text-[#c0bcd0]"}`}>
                      <div className="flex items-center justify-between"><span className="font-medium">{v.name}</span><span className="text-xs text-[#6b6580]">{v.language}</span></div>
                      {v.source === "uploaded" && <span className="mt-0.5 inline-block text-[10px] text-accent-400/70">uploaded</span>}
                      {v.source === "elevenlabs" && <span className="mt-0.5 inline-block text-[10px] text-emerald-400/70">{v.ref_text}</span>}
                    </button>
                  ))}
                </div>
              )}
              {ttsEngine === "qwen3" && currentVoice?.ref_text && <div className="mt-3 rounded-lg bg-[#1e1a2e] px-3 py-2"><p className="text-[10px] uppercase tracking-wider text-[#6b6580] mb-1">Reference transcript</p><p className="text-xs text-[#a09bb5] line-clamp-3">{currentVoice.ref_text}</p></div>}
            </section>
          </aside>
        </div>
      )}

      {/* ============ Recorder Tab ============ */}
      {activeTab === "recorder" && (
        <div className="grid gap-6 lg:grid-cols-[1fr_340px]">
          <div className="space-y-6">
            <section className="card">
              <h2 className="mb-3 text-lg font-semibold">음성 녹음</h2>
              <div className="relative rounded-lg border border-[#2e2845] overflow-hidden">
                <canvas ref={recorderCanvasRef} width={800} height={160} className="w-full h-40 block" />
                <div className="absolute top-3 right-3 rounded-lg bg-black/60 px-3 py-1.5">
                  <span className="text-2xl font-mono text-white tabular-nums">
                    {Math.floor(recorderElapsed / 60).toString().padStart(2, "0")}:{(recorderElapsed % 60).toString().padStart(2, "0")}
                  </span>
                </div>
                {!isRecording && recorderElapsed === 0 && !recordingSaved && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <p className="text-[#9590a8] text-sm">녹음 버튼을 눌러 시작하세요</p>
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
                <div className="mt-3 flex items-center gap-2 text-sm text-[#c0bcd0]">
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
                <div className="relative rounded-lg border border-[#2e2845] overflow-hidden cursor-crosshair"
                  onMouseDown={onAudMouseDown}>
                  <canvas ref={audCanvasRef} width={900} height={140} className="w-full h-36 block" />
                </div>

                {/* Time + Selection info */}
                <div className="mt-2 flex items-center justify-between text-xs">
                  <span className="font-mono text-[#c0bcd0] tabular-nums">
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
                <div className="mt-3 flex items-center gap-2 border-t border-[#2e2845] pt-3">
                  <span className="text-xs text-[#9590a8] mr-1">편집:</span>
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
                <div className="mt-3 flex items-center gap-2 border-t border-[#2e2845] pt-3">
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
              <ul className="space-y-2 text-sm text-[#c0bcd0]">
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
              <div className="relative flex min-h-[160px] cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-[#2e2845] bg-[#241f33]/50 p-6 transition-colors hover:border-accent-500/50 hover:bg-[#241f33]"
                onClick={() => asrFileRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
                onDrop={(e) => { e.preventDefault(); e.stopPropagation(); const f = e.dataTransfer.files?.[0]; if (f) { setAsrFile(f); setAsrStatus({ status: "idle", message: "" }); setTranscript(null); } }}>
                {asrFile ? (<><MicIcon /><p className="mt-3 text-sm font-medium text-[#e8e4f0]">{asrFile.name}</p><p className="mt-1 text-xs text-[#6b6580]">{fmtSize(asrFile.size)}</p><p className="mt-2 text-xs text-accent-400">클릭하여 다른 파일 선택</p></>)
                  : (<><UploadIcon /><p className="mt-3 text-sm text-[#a09bb5]">클릭하거나 파일을 드래그하세요</p><p className="mt-1 text-xs text-[#6b6580]">WAV, MP3, M4A, OGG, FLAC, WebM</p></>)}
                <input ref={asrFileRef} type="file" accept=".wav,.m4a,.mp3,.ogg,.flac,.webm,audio/*" className="hidden"
                  onChange={(e) => { const f = e.target.files?.[0]; if (f) { setAsrFile(f); setAsrStatus({ status: "idle", message: "" }); setTranscript(null); } e.target.value = ""; }} />
              </div>
            </section>
            <section className="card">
              <h2 className="mb-4 text-lg font-semibold">설정</h2>
              <div><label className="label">화자 수</label><select className="input-field cursor-pointer" value={numSpeakers} onChange={(e) => setNumSpeakers(Number(e.target.value))}>{[1,2,3,4,5].map((n) => <option key={n} value={n}>{n}명</option>)}</select></div>
              <button className="btn-primary mt-6 w-full text-lg" disabled={!asrFile || isTranscribing} onClick={transcribe}>
                {isTranscribing ? <><Spinner /> 처리 중...</> : <><MicIcon /> 음성 인식 시작</>}
              </button>
            </section>
            {(asrStatus.status === "loading" || asrStatus.status === "transcribing") && (
              <section className="card"><div className="space-y-3"><div className="flex items-center gap-3"><Spinner /><span className="text-sm text-[#a09bb5]">{asrStatus.message}</span></div>
                <div className="h-2 overflow-hidden rounded-full bg-[#241f33]"><div className="progress-pulse h-full rounded-full bg-gradient-to-r from-accent-600 to-purple-500" style={{ width: asrStatus.status === "loading" ? "30%" : "65%", transition: "width 0.5s ease" }} /></div></div></section>)}
            {asrStatus.status === "error" && <section className="card"><div className="rounded-lg border border-red-800/50 bg-red-900/20 px-4 py-3 text-sm text-red-300">{asrStatus.message}</div></section>}
            {transcript && asrStatus.status === "complete" && (
              <section className="card">
                <div className="mb-4 flex items-center justify-between"><h2 className="text-lg font-semibold">인식 결과</h2>
                  <div className="flex gap-2"><button className="btn-secondary text-xs" onClick={copyTranscript}>{copied ? <><CheckIcon /> 복사됨</> : <><CopyIcon /> 복사</>}</button>
                    <button className="btn-secondary text-xs" onClick={downloadTranscript}><DownloadIcon /> 다운로드</button></div></div>
                <div className="mb-4 flex gap-4 text-xs text-[#6b6580]"><span>오디오 길이: {fmtTime(transcript.duration)}</span><span>처리 시간: {transcript.processing_time.toFixed(1)}초</span></div>
                <div className="space-y-2 max-h-[500px] overflow-y-auto pr-1">
                  {transcript.segments.map((seg, i) => { const color = SPEAKER_COLORS[(seg.speaker - 1) % SPEAKER_COLORS.length]; return (
                    <div key={i} className={`rounded-lg border ${color.border} ${color.bg} px-3 py-2`}>
                      <div className="mb-1 flex items-center gap-2"><span className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${color.label}`}>화자 {seg.speaker}</span><span className="text-[10px] text-[#6b6580]">{fmtTime(seg.start)} &ndash; {fmtTime(seg.end)}</span></div>
                      <p className="text-sm text-[#e8e4f0]">{seg.text}</p></div>); })}
                </div>
              </section>
            )}
          </div>
          <aside className="space-y-6">
            <section className="card"><h2 className="mb-3 text-lg font-semibold">사용 방법</h2><ul className="space-y-2 text-sm text-[#a09bb5]">
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
              <textarea className="input-field min-h-[250px] resize-y text-sm leading-relaxed" value={editorText} onChange={(e) => setEditorText(e.target.value)} placeholder="텍스트를 붙여넣거나, 파일을 열거나, 음성인식 결과를 불러오세요..." />
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
                <p className="mt-2 text-xs text-[#6b6580]">모델: {LLM_MODELS.find((m) => m.id === selectedModel)?.label ?? selectedModel} · 설정 탭에서 변경 가능</p>
              </section>
            )}
            {rewriteStatus.status === "rewriting" && (
              <section className="card"><div className="space-y-3"><div className="flex items-center gap-3"><Spinner /><span className="text-sm text-[#a09bb5]">{rewriteStatus.message}</span></div>
                <div className="h-2 overflow-hidden rounded-full bg-[#241f33]"><div className="progress-pulse h-full rounded-full bg-gradient-to-r from-accent-600 to-purple-500" style={{ width: "60%", transition: "width 0.5s ease" }} /></div></div></section>)}
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
                <textarea className="input-field min-h-[250px] resize-y text-sm leading-relaxed border-purple-500/30 bg-purple-500/10 text-[#e8e4f0]" value={rewrittenText} onChange={(e) => setRewrittenText(e.target.value)} />
              </section>
            )}
          </div>
          <aside className="space-y-6">
            <section className="card"><h2 className="mb-3 text-lg font-semibold">사용 방법</h2><ul className="space-y-2 text-sm text-[#a09bb5]">
              <li className="flex gap-2"><span className="text-accent-400">1.</span>음성인식 탭에서 인식을 완료한 후 이 탭으로 이동하세요.</li>
              <li className="flex gap-2"><span className="text-accent-400">2.</span>화자 삭제 버튼으로 특정 화자의 발화를 제거할 수 있습니다.</li>
              <li className="flex gap-2"><span className="text-accent-400">3.</span>&ldquo;박완서 문체&rdquo; 버튼을 클릭하면 AI가 문체를 변환합니다.</li>
              <li className="flex gap-2"><span className="text-accent-400">4.</span>설정 탭에서 사용할 LLM 모델을 변경할 수 있습니다.</li>
            </ul></section>
            <section className="card"><h2 className="mb-3 text-lg font-semibold">박완서 문체란?</h2><p className="text-sm text-[#a09bb5] leading-relaxed">박완서(1931–2011)는 한국 문학의 거장으로, 일상의 세밀한 관찰, 솔직하고 담담한 어조, 깊은 감정 묘사, 사회 비판적 시선이 특징입니다.</p></section>
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
              <div className="flex items-center gap-3 rounded-lg bg-[#1e1a30] px-4 py-3">
                <MicIcon />
                <div className="min-w-0 flex-1">
                  <p className="font-medium text-[#ddd8ee] truncate">{sourceProject.source_audio_original_name}</p>
                  <p className="text-xs text-[#8a84a0]">{fmtSize(sourceProject.source_audio_size)}{sourceProject.asr_audio_duration > 0 && ` · ${fmtDuration(sourceProject.asr_audio_duration)}`}</p>
                </div>
              </div>
            ) : (
              <p className="text-sm text-[#6b6580]">소스 오디오가 없습니다. 음성녹음 또는 음성인식 탭에서 오디오를 추가하세요.</p>
            )}
          </section>

          {/* Raw Transcript */}
          <section className="card">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-lg font-semibold">🎙 음성인식 원본 (ASR)</h2>
              <div className="flex items-center gap-3">
                {srcTranscript && <span className="text-xs text-[#b8b0cc]">{srcTranscript.length.toLocaleString()}자 · {countWords(srcTranscript).toLocaleString()}단어</span>}
                <button className="btn-secondary text-xs" disabled={srcSaving === "transcript_text"}
                  onClick={() => saveSourceField("transcript_text", srcTranscript)}>
                  {srcSaving === "transcript_text" ? "저장 중..." : "저장"}
                </button>
              </div>
            </div>
            <textarea className="input-field min-h-[180px] resize-y text-sm leading-relaxed"
              value={srcTranscript} onChange={(e) => setSrcTranscript(e.target.value)}
              placeholder="음성인식 결과가 여기에 표시됩니다. 비어있으면 직접 입력할 수 있습니다." />
          </section>

          {/* Edited Transcript */}
          <section className="card">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-lg font-semibold">✏️ 편집된 녹취록</h2>
              <div className="flex items-center gap-3">
                {srcEdited && <span className="text-xs text-[#b8b0cc]">{srcEdited.length.toLocaleString()}자 · {countWords(srcEdited).toLocaleString()}단어</span>}
                <button className="btn-secondary text-xs" disabled={srcSaving === "edited_transcript"}
                  onClick={() => saveSourceField("edited_transcript", srcEdited)}>
                  {srcSaving === "edited_transcript" ? "저장 중..." : "저장"}
                </button>
              </div>
            </div>
            <textarea className="input-field min-h-[180px] resize-y text-sm leading-relaxed border-orange-500/20 bg-orange-500/5"
              value={srcEdited} onChange={(e) => setSrcEdited(e.target.value)}
              placeholder="글편집 탭에서 오타수정/화자삭제 등의 편집 결과가 저장됩니다. 직접 입력할 수도 있습니다." />
          </section>

          {/* Rewritten Text */}
          <section className="card">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-lg font-semibold">🖊 LLM 변환 텍스트</h2>
              <div className="flex items-center gap-3">
                {srcRewritten && <span className="text-xs text-[#b8b0cc]">{srcRewritten.length.toLocaleString()}자 · {countWords(srcRewritten).toLocaleString()}단어</span>}
                <button className="btn-secondary text-xs" disabled={srcSaving === "rewritten_text"}
                  onClick={() => saveSourceField("rewritten_text", srcRewritten)}>
                  {srcSaving === "rewritten_text" ? "저장 중..." : "저장"}
                </button>
              </div>
            </div>
            <textarea className="input-field min-h-[180px] resize-y text-sm leading-relaxed border-violet-500/20 bg-violet-500/5"
              value={srcRewritten} onChange={(e) => setSrcRewritten(e.target.value)}
              placeholder="박완서 문체 변환 결과가 여기에 저장됩니다. 직접 입력할 수도 있습니다." />
            {srcRewritten && (
              <p className="mt-2 text-xs text-[#8a84a0]">
                이 텍스트가 오디오북생성 탭의 &quot;Text to Speak&quot;에 사용됩니다.
              </p>
            )}
          </section>
        </div>
      )}

      {/* ============ Settings Tab ============ */}
      {activeTab === "settings" && (
        <div className="mx-auto max-w-2xl space-y-6">
          <section className="card">
            <h2 className="mb-4 text-lg font-semibold">TTS 엔진 설정</h2>
            <p className="mb-4 text-sm text-[#a09bb5]">오디오북 생성에 사용할 TTS 엔진을 선택하세요.</p>
            <div className="space-y-2">
              <button onClick={() => { setTtsEngine("elevenlabs"); if (elVoices.length > 0) setSelectedVoice(elVoices[0].id); }}
                className={`w-full rounded-lg px-4 py-3 text-left text-sm transition-colors ${ttsEngine === "elevenlabs" ? "bg-accent-600/30 border border-accent-500/50 text-white" : "bg-[#1e1a2e] border border-transparent hover:bg-[#2a2540] text-[#c0bcd0]"}`}>
                <div className="flex items-center justify-between">
                  <div><span className="font-medium">ElevenLabs</span><span className="ml-2 rounded px-1.5 py-0.5 text-[10px] bg-emerald-500/20 text-emerald-300">Cloud</span></div>
                  {ttsEngine === "elevenlabs" && <span className="text-xs text-accent-400">선택됨</span>}
                </div>
                <p className="mt-1 text-xs text-[#6b6580]">빠른 클라우드 TTS, 다국어 지원, ~1초 생성</p>
              </button>
              <button onClick={() => { setTtsEngine("qwen3"); if (voices.length > 0) setSelectedVoice(voices[0].id); }}
                className={`w-full rounded-lg px-4 py-3 text-left text-sm transition-colors ${ttsEngine === "qwen3" ? "bg-accent-600/30 border border-accent-500/50 text-white" : "bg-[#1e1a2e] border border-transparent hover:bg-[#2a2540] text-[#c0bcd0]"}`}>
                <div className="flex items-center justify-between">
                  <div><span className="font-medium">Qwen3-TTS 1.7B</span><span className="ml-2 rounded px-1.5 py-0.5 text-[10px] bg-blue-500/20 text-blue-300">Local GPU</span></div>
                  {ttsEngine === "qwen3" && <span className="text-xs text-accent-400">선택됨</span>}
                </div>
                <p className="mt-1 text-xs text-[#6b6580]">로컬 GPU 음성 클론, 무료, 느림 (GB10)</p>
              </button>
            </div>
          </section>
          <section className="card">
            <h2 className="mb-4 text-lg font-semibold">LLM 모델 설정</h2>
            <p className="mb-4 text-sm text-[#a09bb5]">글편집 탭의 오타수정 및 문체 변환에 사용할 LLM 모델을 선택하세요.</p>
            <div className="space-y-2">
              {LLM_MODELS.map((m) => (
                <button key={m.id} onClick={() => setSelectedModel(m.id)}
                  className={`w-full rounded-lg px-4 py-3 text-left text-sm transition-colors ${selectedModel === m.id ? "bg-accent-600/30 border border-accent-500/50 text-white" : "bg-[#1e1a2e] border border-transparent hover:bg-[#2a2540] text-[#c0bcd0]"}`}>
                  <div className="flex items-center justify-between"><div><span className="font-medium">{m.label}</span><span className="ml-2 rounded px-1.5 py-0.5 text-[10px] bg-[#2e2845] text-[#a09bb5]">{m.provider}</span></div>
                    {selectedModel === m.id && <span className="text-xs text-accent-400">선택됨</span>}</div>
                </button>
              ))}
            </div>
          </section>
        </div>
      )}

      <footer className="mt-12 border-t border-[#2e2845] pt-6 text-center text-xs text-[#6b6580]">Voice Studio &mdash; Powered by ElevenLabs, Qwen3-TTS &amp; Whisper</footer>

      {/* ---- Upload Modal (TTS) ---- */}
      {showUploadModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="card mx-4 w-full max-w-md space-y-4">
            <h2 className="text-lg font-semibold">Register Voice</h2>
            <div className="rounded-lg bg-[#1e1a2e] px-3 py-2 text-sm text-[#a09bb5]">File: {uploadFile?.name}</div>
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
              <button className="text-[#6b6580] hover:text-white" onClick={() => setShowAudioList(false)}><CloseIcon /></button>
            </div>
            {audioFiles.length === 0 ? (
              <p className="text-sm text-[#6b6580] py-4 text-center">저장된 음성 파일이 없습니다</p>
            ) : (
              <div className="max-h-[400px] overflow-y-auto space-y-1.5">
                {audioFiles.map((af) => (
                  <button key={af.filename} onClick={() => selectExistingAudio(af.filename)}
                    className="w-full rounded-lg bg-[#1e1a2e] border border-transparent hover:bg-[#2a2540] hover:border-accent-500/30 px-4 py-3 text-left transition-colors">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-[#e8e4f0] truncate">{af.filename}</span>
                      <span className="text-xs text-[#6b6580] ml-2 shrink-0">{fmtSize(af.size)}</span>
                    </div>
                    <p className="text-[10px] text-[#6b6580] mt-0.5">{fmtDate(af.modified)}</p>
                  </button>
                ))}
              </div>
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
    <div className="flex items-center gap-3 rounded-xl bg-[#1a1630] border border-[#3d3556] px-4 py-3">
      <button onClick={toggle} className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-accent-500 text-white hover:bg-accent-400 transition-colors">
        {playing
          ? <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20"><rect x="5" y="3" width="4" height="14" rx="1" /><rect x="11" y="3" width="4" height="14" rx="1" /></svg>
          : <svg className="h-4 w-4 ml-0.5" fill="currentColor" viewBox="0 0 20 20"><path d="M6.3 2.841A1.5 1.5 0 004 4.11v11.78a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" /></svg>}
      </button>
      <span className="w-11 shrink-0 text-xs font-mono text-[#ddd8ee]">{fmt(cur)}</span>
      <div className="relative flex-1 cursor-pointer py-1" onClick={seek}>
        <div className="h-2 rounded-full bg-[#3d3556]">
          <div className="h-full rounded-full bg-accent-400 transition-[width] duration-100" style={{ width: `${pct}%` }} />
        </div>
      </div>
      <span className="w-11 shrink-0 text-xs font-mono text-[#b8b0cc]">{fmt(dur)}</span>
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
function BookIcon() { return <svg className="mx-auto h-12 w-12 text-[#6b6580]" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" /></svg>; }
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
