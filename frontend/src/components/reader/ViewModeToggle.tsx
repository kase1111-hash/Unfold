"use client";

import { useReadingStore } from "@/store";
import { cn } from "@/utils/cn";
import { FileText, Lightbulb, Columns } from "lucide-react";

const viewModes = [
  {
    value: "technical" as const,
    label: "Technical",
    icon: FileText,
    description: "Original text",
  },
  {
    value: "hybrid" as const,
    label: "Hybrid",
    icon: Columns,
    description: "Side by side",
  },
  {
    value: "conceptual" as const,
    label: "Conceptual",
    icon: Lightbulb,
    description: "Simplified",
  },
];

export function ViewModeToggle() {
  const { viewMode, setViewMode } = useReadingStore();

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-sm border border-slate-200 dark:border-slate-700">
      <span className="text-sm font-medium text-slate-900 dark:text-white mb-3 block">
        View Mode
      </span>

      <div className="flex gap-2">
        {viewModes.map((mode) => (
          <button
            key={mode.value}
            onClick={() => setViewMode(mode.value)}
            className={cn(
              "flex-1 flex flex-col items-center gap-1 py-2 px-3 rounded-lg transition-all",
              viewMode === mode.value
                ? "bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 ring-2 ring-primary-500"
                : "bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-600"
            )}
          >
            <mode.icon className="w-4 h-4" />
            <span className="text-xs font-medium">{mode.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
