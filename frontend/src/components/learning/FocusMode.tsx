"use client";

import { useState, useEffect } from "react";
import { ChevronDown, ChevronRight, Target, Sparkles, Loader2 } from "lucide-react";
import { cn } from "@/utils/cn";
import { api } from "@/services/api";

interface Section {
  id: string;
  title: string;
  content: string;
  focus_score?: number;
  relevance_level?: string;
  should_expand?: boolean;
  focus_order?: number;
}

interface FocusModeProps {
  sections: Section[];
  learningGoal?: string;
  onSectionView?: (sectionId: string, dwellTimeMs: number) => void;
}

export function FocusMode({
  sections: initialSections,
  learningGoal,
  onSectionView,
}: FocusModeProps) {
  const [sections, setSections] = useState<Section[]>(initialSections);
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [goal, setGoal] = useState(learningGoal || "");
  const [viewStartTimes, setViewStartTimes] = useState<Record<string, number>>({});

  // Apply focus optimization when goal changes
  const optimizeForGoal = async () => {
    if (!goal.trim() || sections.length === 0) return;

    setIsOptimizing(true);
    try {
      // In production, this would call the API
      // For now, simulate optimization
      const optimizedSections = sections.map((section, i) => ({
        ...section,
        focus_score: Math.random() * 0.5 + 0.3, // Simulated score
        relevance_level: Math.random() > 0.5 ? "relevant" : "somewhat_relevant",
        should_expand: Math.random() > 0.6,
        focus_order: i + 1,
      }));

      // Sort by focus score
      optimizedSections.sort((a, b) => (b.focus_score || 0) - (a.focus_score || 0));

      // Re-assign focus order after sorting
      optimizedSections.forEach((s, i) => {
        s.focus_order = i + 1;
      });

      setSections(optimizedSections);

      // Auto-expand relevant sections
      const toExpand = new Set(
        optimizedSections
          .filter((s) => s.should_expand)
          .map((s) => s.id)
      );
      setExpandedSections(toExpand);
    } catch (error) {
      console.error("Focus optimization failed:", error);
    } finally {
      setIsOptimizing(false);
    }
  };

  const toggleSection = (sectionId: string) => {
    setExpandedSections((prev) => {
      const next = new Set(prev);
      if (next.has(sectionId)) {
        next.delete(sectionId);
        // Record dwell time when collapsing
        const startTime = viewStartTimes[sectionId];
        if (startTime && onSectionView) {
          onSectionView(sectionId, Date.now() - startTime);
        }
      } else {
        next.add(sectionId);
        // Record start time when expanding
        setViewStartTimes((prev) => ({
          ...prev,
          [sectionId]: Date.now(),
        }));
      }
      return next;
    });
  };

  const getRelevanceBadgeColor = (level?: string) => {
    switch (level) {
      case "highly_relevant":
        return "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300";
      case "relevant":
        return "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300";
      case "somewhat_relevant":
        return "bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300";
      default:
        return "bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-400";
    }
  };

  return (
    <div className="space-y-4">
      {/* Learning Goal Input */}
      <div className="card p-4">
        <div className="flex items-center gap-2 mb-3">
          <Target className="w-4 h-4 text-primary-500" />
          <span className="font-medium text-slate-900 dark:text-white text-sm">
            Learning Goal
          </span>
        </div>

        <div className="flex gap-2">
          <input
            type="text"
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
            placeholder="What do you want to learn from this document?"
            className="flex-1 px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-900 dark:text-white text-sm focus:outline-none focus:ring-2 focus:ring-primary-500/50"
          />
          <button
            onClick={optimizeForGoal}
            disabled={isOptimizing || !goal.trim()}
            className={cn(
              "px-4 py-2 rounded-lg font-medium text-sm transition-colors flex items-center gap-2",
              "bg-primary-600 hover:bg-primary-700 text-white",
              "disabled:opacity-50 disabled:cursor-not-allowed"
            )}
          >
            {isOptimizing ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Sparkles className="w-4 h-4" />
            )}
            Optimize
          </button>
        </div>

        <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">
          Enter your learning goal to highlight and prioritize relevant sections
        </p>
      </div>

      {/* Progressive Reveal Accordion */}
      <div className="space-y-2">
        {sections.map((section) => {
          const isExpanded = expandedSections.has(section.id);

          return (
            <div
              key={section.id}
              className={cn(
                "card overflow-hidden transition-all",
                section.should_expand && "ring-2 ring-primary-500/50"
              )}
            >
              {/* Section Header */}
              <button
                onClick={() => toggleSection(section.id)}
                className="w-full px-4 py-3 flex items-center justify-between hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors"
              >
                <div className="flex items-center gap-3">
                  {isExpanded ? (
                    <ChevronDown className="w-4 h-4 text-slate-400" />
                  ) : (
                    <ChevronRight className="w-4 h-4 text-slate-400" />
                  )}

                  {section.focus_order && (
                    <span className="w-6 h-6 rounded-full bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 text-xs font-medium flex items-center justify-center">
                      {section.focus_order}
                    </span>
                  )}

                  <span className="font-medium text-slate-900 dark:text-white text-sm text-left">
                    {section.title}
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  {section.relevance_level && (
                    <span
                      className={cn(
                        "px-2 py-0.5 rounded-full text-xs font-medium",
                        getRelevanceBadgeColor(section.relevance_level)
                      )}
                    >
                      {section.relevance_level.replace(/_/g, " ")}
                    </span>
                  )}

                  {section.focus_score !== undefined && (
                    <span className="text-xs text-slate-500 dark:text-slate-400">
                      {Math.round(section.focus_score * 100)}%
                    </span>
                  )}
                </div>
              </button>

              {/* Section Content */}
              <div
                className={cn(
                  "overflow-hidden transition-all duration-300",
                  isExpanded ? "max-h-[2000px] opacity-100" : "max-h-0 opacity-0"
                )}
              >
                <div className="px-4 pb-4 pt-2 border-t border-slate-200 dark:border-slate-700">
                  <div className="prose prose-slate dark:prose-invert prose-sm max-w-none">
                    <p className="text-slate-700 dark:text-slate-300 leading-relaxed whitespace-pre-wrap">
                      {section.content}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Focus Summary */}
      {sections.some((s) => s.focus_score !== undefined) && (
        <div className="card p-4 bg-primary-50 dark:bg-primary-900/20 border-primary-200 dark:border-primary-800">
          <h4 className="font-medium text-primary-900 dark:text-primary-100 text-sm mb-2">
            Focus Summary
          </h4>
          <ul className="space-y-1 text-sm text-primary-700 dark:text-primary-300">
            <li>
              {sections.filter((s) => s.should_expand).length} sections highly
              relevant to your goal
            </li>
            <li>
              Recommended reading order: Start with section{" "}
              {sections.find((s) => s.focus_order === 1)?.title || "1"}
            </li>
          </ul>
        </div>
      )}
    </div>
  );
}
