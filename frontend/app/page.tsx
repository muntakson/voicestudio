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
  llm_model: string | null; rewritten_text: string | null;
  generated_audio_filename: string | null; generated_audio_size: number; status: string;
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

  /* ---- Landing state ---- */
  const [projects, setProjects] = useState<Project[]>([]);
  const [showNewModal, setShowNewModal] = useState(false);
  const [newProjectName, setNewProjectName] = useState("");
  const [expandedTranscripts, setExpandedTranscripts] = useState<Set<string>>(new Set());
  const [expandedRewrites, setExpandedRewrites] = useState<Set<string>>(new Set());

  /* ---- Studio state ---- */
  const [activeTab, setActiveTab] = useState<"tts" | "asr" | "editor" | "settings">("asr");

  /* TTS */
  const [text, setText] = useState("");
  const [language, setLanguage] = useState("Auto");
  const [voices, setVoices] = useState<Voice[]>([]);
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

  /* Editor */
  const [editorText, setEditorText] = useState("");
  const [rewrittenText, setRewrittenText] = useState("");
  const [rewriteStatus, setRewriteStatus] = useState<{ status: string; message: string }>({ status: "idle", message: "" });
  const rewriteAbortRef = useRef<AbortController | null>(null);
  const editorFileRef = useRef<HTMLInputElement | null>(null);

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
  };

  const openProject = async (id: string) => {
    resetStudioState();
    setCurrentProjectId(id);
    try {
      const res = await fetch(`/api/projects/${id}`);
      if (!res.ok) return;
      const proj: Project = await res.json();
      if (proj.transcript_json) {
        try {
          const parsed = JSON.parse(proj.transcript_json);
          setTranscript(parsed as TranscriptResult);
          setAsrStatus({ status: "complete", message: "완료!" });
        } catch { /* skip */ }
      }
      if (proj.transcript_text) {
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
  /*  TTS logic                                                        */
  /* ================================================================ */

  const fetchVoices = useCallback(async () => {
    try {
      const res = await fetch("/api/voices");
      if (!res.ok) return;
      const data = await res.json();
      const list: Voice[] = data.voices ?? [];
      setVoices(list);
      if (list.length > 0 && !list.find((v) => v.id === selectedVoice)) setSelectedVoice(list[0].id);
    } catch { /* silent */ }
  }, [selectedVoice]);

  useEffect(() => { fetchVoices(); }, []);

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
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setGen({ status: "loading", message: "Preparing...", audioUrl: null, duration: null });
    const body: Record<string, unknown> = { text: text.trim(), voice_id: selectedVoice, language };
    if (seed.trim()) body.seed = parseInt(seed, 10);
    try {
      const res = await fetch("/api/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: controller.signal });
      if (!res.ok) { setGen({ status: "error", message: await res.text().catch(() => "Request failed"), audioUrl: null, duration: null }); return; }
      await readSSE(res, (event) => {
        if (event.status === "complete") {
          setGen({ status: "complete", message: "Done!", audioUrl: (event.audio_url as string) ?? null, duration: (event.duration as number) ?? null });
          if (currentProjectId && event.audio_url) {
            const fname = (event.audio_url as string).split("/").pop() || "";
            patchProject({ generated_audio_filename: fname, status: "generated" });
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
  const currentVoice = voices.find((v) => v.id === selectedVoice);

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
          patchProject({ rewritten_text: rw, llm_model: selectedModel, status: "rewritten" });
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
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {projects.map((p) => (
              <div key={p.id} className="card flex flex-col">
                {/* Header */}
                <div className="mb-3 flex items-start justify-between">
                  <div>
                    <h3 className="font-semibold text-[#e8e4f0]">{p.name}</h3>
                    <p className="text-xs text-[#6b6580]">{fmtDate(p.created_at)}</p>
                  </div>
                  <span className={`rounded px-2 py-0.5 text-[10px] font-medium ${
                    p.status === "created" ? "bg-[#2e2845] text-[#a09bb5]" :
                    p.status === "transcribed" ? "bg-blue-500/20 text-blue-300" :
                    p.status === "rewritten" ? "bg-purple-500/20 text-purple-300" :
                    "bg-green-500/20 text-green-300"
                  }`}>
                    {p.status === "created" ? "생성됨" : p.status === "transcribed" ? "전사 완료" : p.status === "rewritten" ? "편집 완료" : p.status === "generated" ? "음성 생성" : p.status}
                  </span>
                </div>

                {/* Audio source */}
                {p.source_audio_original_name && (
                  <div className="mb-2 flex items-center gap-2 text-xs text-[#a09bb5]">
                    <MicIcon /><span className="truncate">{p.source_audio_original_name}</span>
                    <span className="text-[#6b6580]">{fmtSize(p.source_audio_size)}</span>
                  </div>
                )}

                {/* Transcript (collapsible) */}
                {p.transcript_text && (
                  <div className="mb-2">
                    <button className="flex w-full items-center gap-1 text-xs font-medium text-blue-300 hover:text-blue-200"
                      onClick={() => toggleExpanded(expandedTranscripts, p.id, setExpandedTranscripts)}>
                      <ChevronIcon open={expandedTranscripts.has(p.id)} /> 원본 전사
                    </button>
                    {expandedTranscripts.has(p.id) && (
                      <div className="mt-1 max-h-[200px] overflow-y-auto rounded-lg bg-[#1e1a2e] px-3 py-2 text-xs text-[#a09bb5] whitespace-pre-wrap">
                        {p.transcript_text}
                      </div>
                    )}
                  </div>
                )}

                {/* LLM */}
                {p.llm_model && (
                  <div className="mb-2 text-xs text-[#6b6580]">
                    LLM: {LLM_MODELS.find((m) => m.id === p.llm_model)?.label ?? p.llm_model}
                  </div>
                )}

                {/* Rewritten (collapsible) */}
                {p.rewritten_text && (
                  <div className="mb-2">
                    <button className="flex w-full items-center gap-1 text-xs font-medium text-purple-300 hover:text-purple-200"
                      onClick={() => toggleExpanded(expandedRewrites, p.id, setExpandedRewrites)}>
                      <ChevronIcon open={expandedRewrites.has(p.id)} /> 박완서 문체
                    </button>
                    {expandedRewrites.has(p.id) && (
                      <div className="mt-1 max-h-[200px] overflow-y-auto rounded-lg border border-purple-500/20 bg-purple-500/5 px-3 py-2 text-xs text-[#c0bcd0] whitespace-pre-wrap">
                        {p.rewritten_text}
                      </div>
                    )}
                  </div>
                )}

                {/* Generated audio */}
                {p.generated_audio_filename && (
                  <div className="mb-2">
                    <p className="mb-1 text-xs text-[#6b6580]">생성 음성 {p.generated_audio_size > 0 && `(${fmtSize(p.generated_audio_size)})`}</p>
                    <audio controls src={`/api/outputs/${p.generated_audio_filename}`} className="w-full h-8 rounded" />
                  </div>
                )}

                <div className="mt-auto flex gap-2 pt-3 border-t border-[#2e2845]">
                  <button className="btn-primary flex-1 text-xs py-2" onClick={() => openProject(p.id)}>열기</button>
                  <button className="btn-secondary text-xs py-2 px-3" onClick={() => deleteProject(p.id)}><TrashIcon /></button>
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
        <p className="mt-2 text-[#a09bb5]">AI Voice Cloning, TTS &amp; Speech Recognition</p>
        <div className="mt-6 inline-flex rounded-lg border border-[#2e2845] bg-[#1a1726] p-1">
          {(["asr", "editor", "tts", "settings"] as const).map((tab) => (
            <button key={tab} onClick={() => setActiveTab(tab)}
              className={`rounded-md px-6 py-2 text-sm font-medium transition-colors ${activeTab === tab ? "bg-accent-600 text-white" : "text-[#a09bb5] hover:text-white"}`}>
              {tab === "tts" ? "Text to Speech" : tab === "asr" ? "음성인식" : tab === "editor" ? "글편집" : "설정"}
            </button>
          ))}
        </div>
      </header>

      {/* ============ TTS Tab ============ */}
      {activeTab === "tts" && (
        <div className="grid gap-6 lg:grid-cols-[1fr_340px]">
          <div className="space-y-6">
            <section className="card">
              <h2 className="mb-3 text-lg font-semibold">Text to Speak</h2>
              <textarea className="input-field min-h-[200px] resize-y text-sm leading-relaxed" placeholder="Type or paste the text you want to convert to speech..."
                value={text} onChange={(e) => setText(e.target.value)} />
              <div className="mt-3 flex flex-wrap gap-2">
                {EXAMPLES.map((ex) => (<button key={ex.label} className="btn-secondary text-xs" onClick={() => { setText(ex.text); setLanguage(ex.lang); }}>{ex.label}</button>))}
              </div>
            </section>
            <section className="card">
              <h2 className="mb-4 text-lg font-semibold">Settings</h2>
              <div className="grid gap-4 sm:grid-cols-2">
                <div><label className="label">Language</label><select className="input-field cursor-pointer" value={language} onChange={(e) => setLanguage(e.target.value)}>{LANGUAGES.map((l) => <option key={l} value={l}>{l}</option>)}</select></div>
                <div><label className="label">Seed (optional)</label><input type="number" className="input-field" placeholder="Random" value={seed} onChange={(e) => setSeed(e.target.value)} /></div>
              </div>
              <button className="btn-primary mt-6 w-full text-lg" disabled={!text.trim() || !selectedVoice || isGenerating} onClick={generate}>
                {isGenerating ? <><Spinner /> Generating...</> : <><PlayIcon /> Generate Speech</>}
              </button>
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
                    <audio controls src={gen.audioUrl} className="w-full rounded-lg" /><a href={gen.audioUrl} download className="btn-secondary inline-flex"><DownloadIcon /> Download</a></div>)}
              </section>
            )}
          </div>
          <aside className="space-y-6">
            <section className="card">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-lg font-semibold">Voice</h2>
                <button className="btn-secondary text-xs" onClick={() => fileInputRef.current?.click()}><UploadIcon /> Add Voice</button>
                <input ref={fileInputRef} type="file" accept=".wav,.m4a,.mp3,.ogg,.flac,.webm,audio/*" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) onFileSelected(f); e.target.value = ""; }} />
              </div>
              {voices.length === 0 ? <p className="text-sm text-[#6b6580]">No voices yet.</p> : (
                <div className="space-y-1.5 max-h-[400px] overflow-y-auto pr-1">
                  {voices.map((v) => (
                    <button key={v.id} onClick={() => setSelectedVoice(v.id)}
                      className={`w-full rounded-lg px-3 py-2.5 text-left text-sm transition-colors ${selectedVoice === v.id ? "bg-accent-600/30 border border-accent-500/50 text-white" : "bg-[#1e1a2e] border border-transparent hover:bg-[#2a2540] text-[#c0bcd0]"}`}>
                      <div className="flex items-center justify-between"><span className="font-medium">{v.name}</span><span className="text-xs text-[#6b6580]">{v.language}</span></div>
                      {v.source === "uploaded" && <span className="mt-0.5 inline-block text-[10px] text-accent-400/70">uploaded</span>}
                    </button>
                  ))}
                </div>
              )}
              {currentVoice?.ref_text && <div className="mt-3 rounded-lg bg-[#1e1a2e] px-3 py-2"><p className="text-[10px] uppercase tracking-wider text-[#6b6580] mb-1">Reference transcript</p><p className="text-xs text-[#a09bb5] line-clamp-3">{currentVoice.ref_text}</p></div>}
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
                <h2 className="text-lg font-semibold">원본 텍스트</h2>
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
                <div className="mb-3 flex items-center justify-between"><h2 className="text-lg font-semibold">변환 결과</h2>
                  <div className="flex gap-2">
                    <button className="btn-secondary text-xs" onClick={async () => { await navigator.clipboard.writeText(rewrittenText); }}><CopyIcon /> 복사</button>
                    <button className="btn-secondary text-xs" onClick={() => { const b = new Blob([rewrittenText], { type: "text/plain;charset=utf-8" }); const u = URL.createObjectURL(b); const a = document.createElement("a"); a.href = u; a.download = "rewritten.txt"; a.click(); URL.revokeObjectURL(u); }}><DownloadIcon /> 다운로드</button>
                  </div></div>
                <div className="whitespace-pre-wrap rounded-lg border border-purple-500/30 bg-purple-500/10 px-4 py-3 text-sm leading-relaxed text-[#e8e4f0]">{rewrittenText}</div>
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

      {/* ============ Settings Tab ============ */}
      {activeTab === "settings" && (
        <div className="mx-auto max-w-2xl space-y-6">
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

      <footer className="mt-12 border-t border-[#2e2845] pt-6 text-center text-xs text-[#6b6580]">Voice Studio &mdash; Powered by Qwen3-TTS &amp; Whisper</footer>

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
