import type { Metadata } from "next";
import "./globals.css";
import { Toaster } from "react-hot-toast";

export const metadata: Metadata = {
  title: "Unfold - AI-Assisted Reading Platform",
  description:
    "Bridge the gap between dense academic texts and genuine understanding with AI-powered reading assistance.",
  keywords: ["reading", "learning", "AI", "academic", "education"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="font-sans antialiased">
        {children}
        <Toaster
          position="bottom-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: "#1e293b",
              color: "#f8fafc",
              borderRadius: "0.75rem",
            },
            success: {
              iconTheme: {
                primary: "#10b981",
                secondary: "#f8fafc",
              },
            },
            error: {
              iconTheme: {
                primary: "#ef4444",
                secondary: "#f8fafc",
              },
            },
          }}
        />
      </body>
    </html>
  );
}
