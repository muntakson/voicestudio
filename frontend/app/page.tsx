"use client";

import { useState, useEffect, useRef, useCallback } from "react";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface Voice {
  id: string;
  name: string;
  language: string;
  ref_text: string;
  source: string;
}

interface GenerationStatus {
  status: "idle" | "loading" | "generating" | "complete" | "error";
  message: string;
  audioUrl: string | null;
  duration: number | null;
}

interface TranscriptSegment {
  speaker: number;
  start: number;
  end: number;
  text: string;
}

interface TranscriptResult {
  segments: TranscriptSegment[];
  full_text: string;
  duration: number;
  processing_time: number;
}

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

const LANGUAGES = ["Auto", "Korean", "English", "Chinese", "Japanese", "French", "German", "Spanish", "Russian", "Portuguese", "Italian"];

const EXAMPLES: { label: string; text: string; lang: string }[] = [
  {
    label: "English",
    text: "The sun dipped below the horizon, painting the sky in shades of amber and violet. A cool breeze swept through the valley, carrying with it the scent of pine and distant rain.",
    lang: "English",
  },
  {
    label: "Korean",
    text: "인공지능 기술이 빠르게 발전하면서 우리 생활의 많은 부분이 변화하고 있습니다. 음성 합성 기술도 그 중 하나로, 이제는 사람의 목소리와 거의 구별할 수 없는 수준에 이르렀습니다.",
    lang: "Korean",
  },
  {
    label: "Chinese",
    text: "人工智能技术正在以前所未有的速度改变我们的世界。从语音合成到自然语言处理，每一项突破都让我们离未来更近一步。",
    lang: "Chinese",
  },
  {
    label: "Japanese",
    text: "人工知能の技術は日々進化を続けています。音声合成の分野でも、人間の声と区別がつかないほどの品質が実現されています。",
    lang: "Japanese",
  },
];

const SPEAKER_COLORS = [
  { bg: "bg-blue-500/15", border: "border-blue-500/30", label: "bg-blue-500/25 text-blue-300" },
  { bg: "bg-emerald-500/15", border: "border-emerald-500/30", label: "bg-emerald-500/25 text-emerald-300" },
  { bg: "bg-amber-500/15", border: "border-amber-500/30", label: "bg-amber-500/25 text-amber-300" },
  { bg: "bg-pink-500/15", border: "border-pink-500/30", label: "bg-pink-500/25 text-pink-300" },
  { bg: "bg-cyan-500/15", border: "border-cyan-500/30", label: "bg-cyan-500/25 text-cyan-300" },
];

function fmtTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function Home() {
  const [activeTab, setActiveTab] = useState<"tts" | "asr">("tts");

  /* ---- TTS state ---- */
  const [text, setText] = useState("");
  const [language, setLanguage] = useState("Auto");
  const [voices, setVoices] = useState<Voice[]>([]);
  const [selectedVoice, setSelectedVoice] = useState("");
  const [seed, setSeed] = useState("");
  const [gen, setGen] = useState<GenerationStatus>({
    status: "idle", message: "", audioUrl: null, duration: null,
  });

  const [showUploadModal, setShowUploadModal] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadName, setUploadName] = useState("");
  const [uploadRefText, setUploadRefText] = useState("");
  const [uploadLanguage, setUploadLanguage] = useState("Auto");
  const [uploading, setUploading] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  /* ---- ASR state ---- */
  const [asrFile, setAsrFile] = useState<File | null>(null);
  const [numSpeakers, setNumSpeakers] = useState(2);
  const [asrStatus, setAsrStatus] = useState<{ status: string; message: string }>({ status: "idle", message: "" });
  const [transcript, setTranscript] = useState<TranscriptResult | null>(null);
  const [copied, setCopied] = useState(false);
  const asrAbortRef = useRef<AbortController | null>(null);
  const asrFileRef = useRef<HTMLInputElement | null>(null);

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
      if (list.length > 0 && !list.find((v) => v.id === selectedVoice)) {
        setSelectedVoice(list[0].id);
      }
    } catch { /* silent */ }
  }, [selectedVoice]);

  useEffect(() => { fetchVoices(); }, []);

  const onFileSelected = (file: File) => {
    setUploadFile(file);
    setUploadName(file.name.replace(/\.[^.]+$/, ""));
    setUploadRefText("");
    setUploadLanguage("Auto");
    setShowUploadModal(true);
  };

  const submitUpload = async () => {
    if (!uploadFile || !uploadName.trim()) return;
    setUploading(true);
    try {
      const form = new FormData();
      form.append("file", uploadFile);
      form.append("name", uploadName.trim());
      form.append("ref_text", uploadRefText.trim());
      form.append("language", uploadLanguage);
      const res = await fetch("/api/upload-voice", { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Upload failed" }));
        throw new Error(err.detail || "Upload failed");
      }
      const result = await res.json();
      await fetchVoices();
      setSelectedVoice(result.id);
      setShowUploadModal(false);
      setUploadFile(null);
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const generate = async () => {
    if (!text.trim() || !selectedVoice) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setGen({ status: "loading", message: "Preparing...", audioUrl: null, duration: null });

    const body: Record<string, unknown> = {
      text: text.trim(),
      voice_id: selectedVoice,
      language,
    };
    if (seed.trim()) body.seed = parseInt(seed, 10);

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!res.ok) {
        const errText = await res.text().catch(() => "Request failed");
        setGen({ status: "error", message: errText, audioUrl: null, duration: null });
        return;
      }

      const reader = res.body?.getReader();
      if (!reader) {
        setGen({ status: "error", message: "No response stream", audioUrl: null, duration: null });
        return;
      }

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
          try {
            const event = JSON.parse(trimmed.slice(5).trim());
            if (event.status === "complete") {
              setGen({
                status: "complete", message: "Done!",
                audioUrl: event.audio_url ?? null, duration: event.duration ?? null,
              });
            } else if (event.status === "error") {
              setGen({ status: "error", message: event.message ?? "Failed", audioUrl: null, duration: null });
            } else {
              setGen((p) => ({ ...p, status: event.status, message: event.message ?? p.message }));
            }
          } catch { /* skip */ }
        }
      }
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
    setTranscript(null);
    setCopied(false);

    const form = new FormData();
    form.append("file", asrFile);
    form.append("num_speakers", numSpeakers.toString());

    try {
      const res = await fetch("/api/transcribe", {
        method: "POST",
        body: form,
        signal: controller.signal,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "요청 실패" }));
        setAsrStatus({ status: "error", message: err.detail || "요청 실패" });
        return;
      }

      const reader = res.body?.getReader();
      if (!reader) {
        setAsrStatus({ status: "error", message: "No response stream" });
        return;
      }

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
          try {
            const event = JSON.parse(trimmed.slice(5).trim());
            if (event.status === "complete") {
              setTranscript({
                segments: event.segments,
                full_text: event.full_text,
                duration: event.duration,
                processing_time: event.processing_time,
              });
              setAsrStatus({ status: "complete", message: "완료!" });
            } else if (event.status === "error") {
              setAsrStatus({ status: "error", message: event.message || "실패" });
            } else {
              setAsrStatus({ status: event.status, message: event.message || "" });
            }
          } catch { /* skip */ }
        }
      }
    } catch (err: unknown) {
      if ((err as Error).name === "AbortError") return;
      setAsrStatus({
        status: "error",
        message: err instanceof Error ? err.message : "알 수 없는 오류",
      });
    }
  };

  const isTranscribing = asrStatus.status === "loading" || asrStatus.status === "transcribing";

  const copyTranscript = async () => {
    if (!transcript?.full_text) return;
    await navigator.clipboard.writeText(transcript.full_text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const downloadTranscript = () => {
    if (!transcript?.full_text) return;
    const blob = new Blob([transcript.full_text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "transcript.txt";
    a.click();
    URL.revokeObjectURL(url);
  };

  /* ================================================================ */
  /*  Render                                                           */
  /* ================================================================ */

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      {/* Header */}
      <header className="mb-10 text-center">
        <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
          <span className="bg-gradient-to-r from-accent-400 to-purple-400 bg-clip-text text-transparent">
            Voice Studio
          </span>
        </h1>
        <p className="mt-2 text-[#a09bb5]">AI Voice Cloning, TTS &amp; Speech Recognition</p>

        {/* Tabs */}
        <div className="mt-6 inline-flex rounded-lg border border-[#2e2845] bg-[#1a1726] p-1">
          <button
            onClick={() => setActiveTab("tts")}
            className={`rounded-md px-6 py-2 text-sm font-medium transition-colors ${
              activeTab === "tts" ? "bg-accent-600 text-white" : "text-[#a09bb5] hover:text-white"
            }`}
          >
            Text to Speech
          </button>
          <button
            onClick={() => setActiveTab("asr")}
            className={`rounded-md px-6 py-2 text-sm font-medium transition-colors ${
              activeTab === "asr" ? "bg-accent-600 text-white" : "text-[#a09bb5] hover:text-white"
            }`}
          >
            음성 인식 (ASR)
          </button>
        </div>
      </header>

      {/* ============================================================ */}
      {/*  TTS Tab                                                      */}
      {/* ============================================================ */}

      {activeTab === "tts" && (
        <div className="grid gap-6 lg:grid-cols-[1fr_340px]">
          {/* ---------- Left column ---------- */}
          <div className="space-y-6">
            {/* Text input */}
            <section className="card">
              <h2 className="mb-3 text-lg font-semibold">Text to Speak</h2>
              <textarea
                className="input-field min-h-[200px] resize-y text-sm leading-relaxed"
                placeholder="Type or paste the text you want to convert to speech..."
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
              <div className="mt-3 flex flex-wrap gap-2">
                {EXAMPLES.map((ex) => (
                  <button
                    key={ex.label}
                    className="btn-secondary text-xs"
                    onClick={() => { setText(ex.text); setLanguage(ex.lang); }}
                  >
                    {ex.label}
                  </button>
                ))}
              </div>
            </section>

            {/* Settings + Generate */}
            <section className="card">
              <h2 className="mb-4 text-lg font-semibold">Settings</h2>
              <div className="grid gap-4 sm:grid-cols-2">
                <div>
                  <label className="label">Language</label>
                  <select className="input-field cursor-pointer" value={language} onChange={(e) => setLanguage(e.target.value)}>
                    {LANGUAGES.map((l) => <option key={l} value={l}>{l}</option>)}
                  </select>
                </div>
                <div>
                  <label className="label">Seed (optional)</label>
                  <input type="number" className="input-field" placeholder="Random" value={seed} onChange={(e) => setSeed(e.target.value)} />
                </div>
              </div>

              <button
                className="btn-primary mt-6 w-full text-lg"
                disabled={!text.trim() || !selectedVoice || isGenerating}
                onClick={generate}
              >
                {isGenerating ? <><Spinner /> Generating...</> : <><PlayIcon /> Generate Speech</>}
              </button>
            </section>

            {/* Output */}
            {gen.status !== "idle" && (
              <section className="card">
                <h2 className="mb-4 text-lg font-semibold">Output</h2>
                {(gen.status === "loading" || gen.status === "generating") && (
                  <div className="space-y-3">
                    <div className="flex items-center gap-3">
                      <Spinner />
                      <span className="text-sm text-[#a09bb5]">{gen.message}</span>
                    </div>
                    <div className="h-2 overflow-hidden rounded-full bg-[#241f33]">
                      <div className="progress-pulse h-full rounded-full bg-gradient-to-r from-accent-600 to-purple-500"
                        style={{ width: gen.status === "loading" ? "40%" : "75%", transition: "width 0.5s ease" }} />
                    </div>
                  </div>
                )}
                {gen.status === "error" && (
                  <div className="rounded-lg border border-red-800/50 bg-red-900/20 px-4 py-3 text-sm text-red-300">{gen.message}</div>
                )}
                {gen.status === "complete" && gen.audioUrl && (
                  <div className="space-y-4">
                    <div className="flex items-center gap-2 text-sm text-green-400">
                      <CheckIcon />
                      <span>Done!{gen.duration != null && <span className="ml-2 text-[#6b6580]">({gen.duration.toFixed(1)}s)</span>}</span>
                    </div>
                    <audio controls src={gen.audioUrl} className="w-full rounded-lg" />
                    <a href={gen.audioUrl} download className="btn-secondary inline-flex"><DownloadIcon /> Download</a>
                  </div>
                )}
              </section>
            )}
          </div>

          {/* ---------- Right sidebar ---------- */}
          <aside className="space-y-6">
            {/* Voice selection */}
            <section className="card">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-lg font-semibold">Voice</h2>
                <button
                  className="btn-secondary text-xs"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <UploadIcon /> Add Voice
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".wav,.m4a,.mp3,.ogg,.flac,.webm,audio/*"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) onFileSelected(f);
                    e.target.value = "";
                  }}
                />
              </div>

              {voices.length === 0 ? (
                <p className="text-sm text-[#6b6580]">No voices yet. Upload a voice sample to get started.</p>
              ) : (
                <div className="space-y-1.5 max-h-[400px] overflow-y-auto pr-1">
                  {voices.map((v) => (
                    <button
                      key={v.id}
                      onClick={() => setSelectedVoice(v.id)}
                      className={`w-full rounded-lg px-3 py-2.5 text-left text-sm transition-colors ${
                        selectedVoice === v.id
                          ? "bg-accent-600/30 border border-accent-500/50 text-white"
                          : "bg-[#1e1a2e] border border-transparent hover:bg-[#2a2540] text-[#c0bcd0]"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-medium">{v.name}</span>
                        <span className="text-xs text-[#6b6580]">{v.language}</span>
                      </div>
                      {v.source === "uploaded" && (
                        <span className="mt-0.5 inline-block text-[10px] text-accent-400/70">uploaded</span>
                      )}
                    </button>
                  ))}
                </div>
              )}

              {currentVoice?.ref_text && (
                <div className="mt-3 rounded-lg bg-[#1e1a2e] px-3 py-2">
                  <p className="text-[10px] uppercase tracking-wider text-[#6b6580] mb-1">Reference transcript</p>
                  <p className="text-xs text-[#a09bb5] line-clamp-3">{currentVoice.ref_text}</p>
                </div>
              )}
            </section>

            {/* Quick reference */}
            <section className="card">
              <h2 className="mb-3 text-lg font-semibold">How it works</h2>
              <ul className="space-y-2 text-sm text-[#a09bb5]">
                <li className="flex gap-2">
                  <span className="text-accent-400">1.</span>
                  Click <b>Add Voice</b> and upload a voice sample (.wav, .m4a, etc).
                </li>
                <li className="flex gap-2">
                  <span className="text-accent-400">2.</span>
                  Enter a speaker name and optionally the transcript of the sample.
                </li>
                <li className="flex gap-2">
                  <span className="text-accent-400">3.</span>
                  Type or paste text, select the voice, and hit Generate.
                </li>
                <li className="flex gap-2">
                  <span className="text-accent-400">4.</span>
                  Providing a reference transcript improves voice cloning quality.
                </li>
              </ul>
            </section>
          </aside>
        </div>
      )}

      {/* ============================================================ */}
      {/*  ASR Tab                                                      */}
      {/* ============================================================ */}

      {activeTab === "asr" && (
        <div className="grid gap-6 lg:grid-cols-[1fr_340px]">
          {/* ---------- Left column ---------- */}
          <div className="space-y-6">
            {/* Upload */}
            <section className="card">
              <h2 className="mb-3 text-lg font-semibold">음성 파일 업로드</h2>
              <div
                className="relative flex min-h-[160px] cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-[#2e2845] bg-[#241f33]/50 p-6 transition-colors hover:border-accent-500/50 hover:bg-[#241f33]"
                onClick={() => asrFileRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
                onDrop={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  const f = e.dataTransfer.files?.[0];
                  if (f) { setAsrFile(f); setAsrStatus({ status: "idle", message: "" }); setTranscript(null); }
                }}
              >
                {asrFile ? (
                  <>
                    <MicIcon />
                    <p className="mt-3 text-sm font-medium text-[#e8e4f0]">{asrFile.name}</p>
                    <p className="mt-1 text-xs text-[#6b6580]">{(asrFile.size / 1024 / 1024).toFixed(1)} MB</p>
                    <p className="mt-2 text-xs text-accent-400">클릭하여 다른 파일 선택</p>
                  </>
                ) : (
                  <>
                    <UploadIcon />
                    <p className="mt-3 text-sm text-[#a09bb5]">클릭하거나 파일을 드래그하세요</p>
                    <p className="mt-1 text-xs text-[#6b6580]">WAV, MP3, M4A, OGG, FLAC, WebM</p>
                  </>
                )}
                <input
                  ref={asrFileRef}
                  type="file"
                  accept=".wav,.m4a,.mp3,.ogg,.flac,.webm,audio/*"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) { setAsrFile(f); setAsrStatus({ status: "idle", message: "" }); setTranscript(null); }
                    e.target.value = "";
                  }}
                />
              </div>
            </section>

            {/* Settings + Transcribe */}
            <section className="card">
              <h2 className="mb-4 text-lg font-semibold">설정</h2>
              <div>
                <label className="label">화자 수</label>
                <select
                  className="input-field cursor-pointer"
                  value={numSpeakers}
                  onChange={(e) => setNumSpeakers(Number(e.target.value))}
                >
                  {[1, 2, 3, 4, 5].map((n) => (
                    <option key={n} value={n}>{n}명</option>
                  ))}
                </select>
              </div>

              <button
                className="btn-primary mt-6 w-full text-lg"
                disabled={!asrFile || isTranscribing}
                onClick={transcribe}
              >
                {isTranscribing ? <><Spinner /> 처리 중...</> : <><MicIcon /> 음성 인식 시작</>}
              </button>
            </section>

            {/* Progress */}
            {(asrStatus.status === "loading" || asrStatus.status === "transcribing") && (
              <section className="card">
                <div className="space-y-3">
                  <div className="flex items-center gap-3">
                    <Spinner />
                    <span className="text-sm text-[#a09bb5]">{asrStatus.message}</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-[#241f33]">
                    <div
                      className="progress-pulse h-full rounded-full bg-gradient-to-r from-accent-600 to-purple-500"
                      style={{ width: asrStatus.status === "loading" ? "30%" : "65%", transition: "width 0.5s ease" }}
                    />
                  </div>
                </div>
              </section>
            )}

            {/* Error */}
            {asrStatus.status === "error" && (
              <section className="card">
                <div className="rounded-lg border border-red-800/50 bg-red-900/20 px-4 py-3 text-sm text-red-300">
                  {asrStatus.message}
                </div>
              </section>
            )}

            {/* Results */}
            {transcript && asrStatus.status === "complete" && (
              <section className="card">
                <div className="mb-4 flex items-center justify-between">
                  <h2 className="text-lg font-semibold">인식 결과</h2>
                  <div className="flex gap-2">
                    <button className="btn-secondary text-xs" onClick={copyTranscript}>
                      {copied ? <><CheckIcon /> 복사됨</> : <><CopyIcon /> 복사</>}
                    </button>
                    <button className="btn-secondary text-xs" onClick={downloadTranscript}>
                      <DownloadIcon /> 다운로드
                    </button>
                  </div>
                </div>

                <div className="mb-4 flex gap-4 text-xs text-[#6b6580]">
                  <span>오디오 길이: {fmtTime(transcript.duration)}</span>
                  <span>처리 시간: {transcript.processing_time.toFixed(1)}초</span>
                </div>

                <div className="space-y-2 max-h-[500px] overflow-y-auto pr-1">
                  {transcript.segments.map((seg, i) => {
                    const color = SPEAKER_COLORS[(seg.speaker - 1) % SPEAKER_COLORS.length];
                    return (
                      <div key={i} className={`rounded-lg border ${color.border} ${color.bg} px-3 py-2`}>
                        <div className="mb-1 flex items-center gap-2">
                          <span className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${color.label}`}>
                            화자 {seg.speaker}
                          </span>
                          <span className="text-[10px] text-[#6b6580]">
                            {fmtTime(seg.start)} &ndash; {fmtTime(seg.end)}
                          </span>
                        </div>
                        <p className="text-sm text-[#e8e4f0]">{seg.text}</p>
                      </div>
                    );
                  })}
                </div>
              </section>
            )}
          </div>

          {/* ---------- Right sidebar ---------- */}
          <aside className="space-y-6">
            <section className="card">
              <h2 className="mb-3 text-lg font-semibold">사용 방법</h2>
              <ul className="space-y-2 text-sm text-[#a09bb5]">
                <li className="flex gap-2">
                  <span className="text-accent-400">1.</span>
                  한국어 인터뷰 음성 파일을 업로드하세요 (최대 15분).
                </li>
                <li className="flex gap-2">
                  <span className="text-accent-400">2.</span>
                  대화 참여 화자 수를 선택하세요 (기본: 2명).
                </li>
                <li className="flex gap-2">
                  <span className="text-accent-400">3.</span>
                  &ldquo;음성 인식 시작&rdquo; 버튼을 클릭하면 자동으로 전사됩니다.
                </li>
                <li className="flex gap-2">
                  <span className="text-accent-400">4.</span>
                  화자 구분이 포함된 한국어 자막이 생성됩니다.
                </li>
              </ul>
            </section>

            <section className="card">
              <h2 className="mb-3 text-lg font-semibold">파일 요구사항</h2>
              <ul className="space-y-1.5 text-sm text-[#a09bb5]">
                <li>지원 형식: WAV, MP3, M4A, OGG, FLAC, WebM</li>
                <li>최대 파일 크기: 100 MB</li>
                <li>최대 오디오 길이: 10분</li>
                <li>최적 품질: 깨끗한 녹음, 최소한의 배경 소음</li>
              </ul>
            </section>
          </aside>
        </div>
      )}

      <footer className="mt-12 border-t border-[#2e2845] pt-6 text-center text-xs text-[#6b6580]">
        Voice Studio &mdash; Powered by Qwen3-TTS &amp; Whisper
      </footer>

      {/* ---- Upload Modal (TTS) ---- */}
      {showUploadModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="card mx-4 w-full max-w-md space-y-4">
            <h2 className="text-lg font-semibold">Register Voice</h2>

            <div className="rounded-lg bg-[#1e1a2e] px-3 py-2 text-sm text-[#a09bb5]">
              File: {uploadFile?.name}
            </div>

            <div>
              <label className="label">Speaker Name *</label>
              <input
                type="text"
                className="input-field"
                placeholder="e.g. John, Narrator, My Voice"
                value={uploadName}
                onChange={(e) => setUploadName(e.target.value)}
                autoFocus
              />
            </div>

            <div>
              <label className="label">Reference Transcript (optional)</label>
              <textarea
                className="input-field min-h-[80px] resize-y text-sm"
                placeholder="Type what is spoken in the audio sample for better cloning quality..."
                value={uploadRefText}
                onChange={(e) => setUploadRefText(e.target.value)}
              />
              <p className="mt-1 text-xs text-[#6b6580]">Providing the transcript of your audio dramatically improves voice cloning.</p>
            </div>

            <div>
              <label className="label">Language</label>
              <select className="input-field cursor-pointer" value={uploadLanguage} onChange={(e) => setUploadLanguage(e.target.value)}>
                {LANGUAGES.map((l) => <option key={l} value={l}>{l}</option>)}
              </select>
            </div>

            <div className="flex gap-3 pt-2">
              <button
                className="btn-secondary flex-1"
                onClick={() => { setShowUploadModal(false); setUploadFile(null); }}
                disabled={uploading}
              >
                Cancel
              </button>
              <button
                className="btn-primary flex-1"
                disabled={!uploadName.trim() || uploading}
                onClick={submitUpload}
              >
                {uploading ? <><Spinner small /> Registering...</> : "Register Voice"}
              </button>
            </div>
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
  return (
    <svg className={`${size} animate-spin`} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
    </svg>
  );
}

function PlayIcon() {
  return <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 20 20"><path d="M6.3 2.841A1.5 1.5 0 004 4.11v11.78a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" /></svg>;
}

function CheckIcon() {
  return <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>;
}

function DownloadIcon() {
  return <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5 5m0 0l5-5m-5 5V3" /></svg>;
}

function UploadIcon() {
  return <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M17 8l-5-5m0 0L7 8m5-5v13" /></svg>;
}

function MicIcon() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M19 10v2a7 7 0 01-14 0v-2" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 19v4m-4 0h8" />
    </svg>
  );
}

function CopyIcon() {
  return (
    <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
    </svg>
  );
}
