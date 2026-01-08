"use client";

import { useReadingStore } from "@/store";
import { cn } from "@/utils/cn";
import { Baby, GraduationCap, Sparkles } from "lucide-react";

const complexityLabels = [
  { value: 0, label: "ELI5", icon: Baby },
  { value: 25, label: "Beginner", icon: null },
  { value: 50, label: "Intermediate", icon: null },
  { value: 75, label: "Advanced", icon: null },
  { value: 100, label: "Expert", icon: GraduationCap },
];

export function ComplexitySlider() {
  const { complexityLevel, setComplexity, fetchParaphrase, isParaphrasing } =
    useReadingStore();

  const handleChange = (value: number) => {
    setComplexity(value);
  };

  const handleApply = () => {
    fetchParaphrase();
  };

  const getCurrentLabel = () => {
    if (complexityLevel <= 10) return "Explain Like I'm 5";
    if (complexityLevel <= 30) return "Beginner Friendly";
    if (complexityLevel <= 50) return "Intermediate";
    if (complexityLevel <= 70) return "Advanced";
    if (complexityLevel <= 90) return "Technical";
    return "Expert Level";
  };

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-sm border border-slate-200 dark:border-slate-700">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-primary-500" />
          <span className="font-medium text-slate-900 dark:text-white text-sm">
            Complexity Level
          </span>
        </div>
        <span className="text-xs font-medium px-2 py-1 bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded-full">
          {getCurrentLabel()}
        </span>
      </div>

      <div className="relative mt-2">
        <input
          type="range"
          min="0"
          max="100"
          value={complexityLevel}
          onChange={(e) => handleChange(Number(e.target.value))}
          className={cn(
            "w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer",
            "slider-thumb"
          )}
          style={{
            background: `linear-gradient(to right, #7c3aed ${complexityLevel}%, #e2e8f0 ${complexityLevel}%)`,
          }}
        />

        {/* Markers */}
        <div className="flex justify-between mt-2">
          {complexityLabels.map((item) => (
            <button
              key={item.value}
              onClick={() => handleChange(item.value)}
              className={cn(
                "flex flex-col items-center transition-colors",
                complexityLevel === item.value
                  ? "text-primary-600"
                  : "text-slate-400 hover:text-slate-600"
              )}
            >
              <div
                className={cn(
                  "w-2 h-2 rounded-full mb-1",
                  complexityLevel >= item.value
                    ? "bg-primary-500"
                    : "bg-slate-300 dark:bg-slate-600"
                )}
              />
              {item.icon && <item.icon className="w-3 h-3" />}
              <span className="text-xs mt-0.5">{item.label}</span>
            </button>
          ))}
        </div>
      </div>

      <button
        onClick={handleApply}
        disabled={isParaphrasing}
        className={cn(
          "mt-4 w-full py-2 px-4 rounded-lg font-medium text-sm transition-colors",
          "bg-primary-600 hover:bg-primary-700 text-white",
          "disabled:opacity-50 disabled:cursor-not-allowed"
        )}
      >
        {isParaphrasing ? "Generating..." : "Apply Complexity"}
      </button>
    </div>
  );
}
