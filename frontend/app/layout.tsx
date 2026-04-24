import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Voice Studio - AI Voice Cloning & TTS",
  description: "AI Voice Cloning & Text-to-Speech powered by VibeVoice",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen antialiased">
        {children}
      </body>
    </html>
  );
}
