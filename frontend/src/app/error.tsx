"use client";

import { useEffect } from "react";
import { ErrorFallback } from "@/components/ErrorBoundary";

// Catches errors thrown by nested layouts such as (dashboard)/layout.tsx, which
// sit above the per-page PageErrorBoundary and would otherwise blank the app.
export default function AppError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("Unhandled application error:", error);
  }, [error]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50 dark:bg-slate-900">
      <ErrorFallback error={error} resetError={reset} />
    </div>
  );
}
