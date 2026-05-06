import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Voice Studio - 오픈소스 오디오북 생성기 | AI TTS & 음성인식",
  description: "오픈소스 오디오 자서전, 시낭독 영상 제작 소프트웨어. AI 음성인식(Whisper), 문체 변환(Claude), 음성 합성(ElevenLabs, Qwen3-TTS)으로 오디오북을 만들어보세요. Open-source audiobook generator with AI transcription, LLM rewriting, and text-to-speech.",
  keywords: ["Voice Studio", "오디오북", "audiobook", "TTS", "음성합성", "text-to-speech", "AI", "음성인식", "ASR", "Whisper", "ElevenLabs", "Qwen3-TTS", "오픈소스", "open-source", "시낭독", "회고록", "자서전"],
  authors: [{ name: "Voice Studio", url: "https://github.com/muntakson/voicestudio" }],
  openGraph: {
    title: "Voice Studio - 오픈소스 오디오북 생성기",
    description: "AI 음성인식, 문체 변환, 음성 합성으로 오디오북과 시낭독 영상을 만드는 오픈소스 플랫폼",
    url: "https://voice.iotok.org",
    siteName: "Voice Studio",
    locale: "ko_KR",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Voice Studio - 오픈소스 오디오북 생성기",
    description: "AI 음성인식, 문체 변환, 음성 합성으로 오디오북과 시낭독 영상을 만드는 오픈소스 플랫폼",
  },
  metadataBase: new URL("https://voice.iotok.org"),
  alternates: {
    canonical: "https://voice.iotok.org",
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
    },
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko" className="dark">
      <head />
      <body className="min-h-screen antialiased">
        {children}
      </body>
    </html>
  );
}
