"use client";

import { useState, useEffect, useCallback } from "react";
import { useScholarStore } from "@/store";
import { cn } from "@/utils/cn";
import type { ReadingSnapshot, LearningJourney } from "@/types";
import {
  BookOpen,
  RefreshCw,
  Eye,
  Lightbulb,
  HelpCircle,
  Link,
  Clock,
  TrendingUp,
  TrendingDown,
  Minus,
  ChevronRight,
  Plus,
  Save,
  Loader2,
} from "lucide-react";

interface ReflectionJournalProps {
  documentId: string;
  className?: string;
}

const REFLECTION_TYPES = [
  { value: "initial_reading", label: "Initial Reading", icon: BookOpen },
  { value: "re_reading", label: "Re-reading", icon: RefreshCw },
  { value: "review", label: "Review", icon: Eye },
  { value: "insight", label: "Insight", icon: Lightbulb },
];

function ProgressRing({ value, size = 60, strokeWidth = 6 }: { value: number; size?: number; strokeWidth?: number }) {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (value / 100) * circumference;

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle
          className="text-slate-200 dark:text-slate-700"
          strokeWidth={strokeWidth}
          stroke="currentColor"
          fill="transparent"
          r={radius}
          cx={size / 2}
          cy={size / 2}
        />
        <circle
          className="text-primary-500"
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          stroke="currentColor"
          fill="transparent"
          r={radius}
          cx={size / 2}
          cy={size / 2}
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-sm font-bold text-slate-700 dark:text-slate-300">
          {Math.round(value)}%
        </span>
      </div>
    </div>
  );
}

function SnapshotCard({ snapshot }: { snapshot: ReadingSnapshot }) {
  const typeConfig = REFLECTION_TYPES.find((t) => t.value === snapshot.reflection_type);
  const Icon = typeConfig?.icon || BookOpen;

  return (
    <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <Icon className="w-4 h-4 text-primary-500" />
          <span className="text-sm font-medium text-slate-900 dark:text-white">
            {typeConfig?.label || "Reading"}
          </span>
        </div>
        <span className="text-xs text-slate-500">
          {new Date(snapshot.created_at).toLocaleDateString()}
        </span>
      </div>

      {/* Summary */}
      {snapshot.summary && (
        <p className="text-sm text-slate-600 dark:text-slate-400 mb-3">{snapshot.summary}</p>
      )}

      {/* Metrics */}
      <div className="grid grid-cols-3 gap-4 text-center">
        <div>
          <div className="text-lg font-bold text-slate-900 dark:text-white">
            {Math.round(snapshot.comprehension_score * 100)}%
          </div>
          <div className="text-xs text-slate-500">Comprehension</div>
        </div>
        <div>
          <div className="text-lg font-bold text-slate-900 dark:text-white">
            {Math.round(snapshot.time_spent_minutes)}m
          </div>
          <div className="text-xs text-slate-500">Time Spent</div>
        </div>
        <div>
          <div className="text-lg font-bold text-slate-900 dark:text-white">
            {Math.round(snapshot.scroll_depth * 100)}%
          </div>
          <div className="text-xs text-slate-500">Progress</div>
        </div>
      </div>

      {/* Takeaways */}
      {snapshot.key_takeaways.length > 0 && (
        <div className="mt-3 pt-3 border-t border-slate-100 dark:border-slate-700">
          <div className="text-xs font-medium text-slate-700 dark:text-slate-300 mb-1">
            Key Takeaways
          </div>
          <ul className="text-xs text-slate-600 dark:text-slate-400 space-y-1">
            {snapshot.key_takeaways.slice(0, 3).map((takeaway, i) => (
              <li key={i} className="flex items-start gap-1">
                <ChevronRight className="w-3 h-3 mt-0.5 text-primary-500" />
                {takeaway}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Questions */}
      {snapshot.questions.length > 0 && (
        <div className="mt-2">
          <div className="text-xs font-medium text-slate-700 dark:text-slate-300 mb-1">
            Open Questions ({snapshot.questions.length})
          </div>
          <div className="flex items-center gap-1 text-xs text-slate-500">
            <HelpCircle className="w-3 h-3" />
            {snapshot.questions[0]}
          </div>
        </div>
      )}
    </div>
  );
}

export function ReflectionJournal({ documentId, className }: ReflectionJournalProps) {
  const [showForm, setShowForm] = useState(false);
  const [saving, setSaving] = useState(false);
  const [formData, setFormData] = useState({
    reflectionType: "initial_reading",
    summary: "",
    keyTakeaways: [""],
    questions: [""],
    connections: [""],
    timeSpentMinutes: 0,
    scrollDepth: 0,
  });

  const {
    journey,
    reflectionPrompts,
    loadJourney,
    loadPrompts,
    createSnapshot,
  } = useScholarStore();

  useEffect(() => {
    if (documentId) {
      loadJourney(documentId);
      loadPrompts(documentId);
    }
  }, [documentId, loadJourney, loadPrompts]);

  const handleAddItem = (field: "keyTakeaways" | "questions" | "connections") => {
    setFormData((prev) => ({
      ...prev,
      [field]: [...prev[field], ""],
    }));
  };

  const handleRemoveItem = (field: "keyTakeaways" | "questions" | "connections", index: number) => {
    setFormData((prev) => ({
      ...prev,
      [field]: prev[field].filter((_, i) => i !== index),
    }));
  };

  const handleItemChange = (
    field: "keyTakeaways" | "questions" | "connections",
    index: number,
    value: string
  ) => {
    setFormData((prev) => ({
      ...prev,
      [field]: prev[field].map((item, i) => (i === index ? value : item)),
    }));
  };

  const handleSubmit = useCallback(async () => {
    setSaving(true);
    await createSnapshot(documentId, {
      reflectionType: formData.reflectionType,
      summary: formData.summary || undefined,
      keyTakeaways: formData.keyTakeaways.filter((t) => t.trim()),
      questions: formData.questions.filter((q) => q.trim()),
      connections: formData.connections.filter((c) => c.trim()),
      timeSpentMinutes: formData.timeSpentMinutes,
      scrollDepth: formData.scrollDepth / 100,
    });
    setSaving(false);
    setShowForm(false);
    setFormData({
      reflectionType: "re_reading",
      summary: "",
      keyTakeaways: [""],
      questions: [""],
      connections: [""],
      timeSpentMinutes: 0,
      scrollDepth: 0,
    });
    loadJourney(documentId);
  }, [formData, documentId, createSnapshot, loadJourney]);

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case "increasing":
        return <TrendingUp className="w-4 h-4 text-green-500" />;
      case "decreasing":
        return <TrendingDown className="w-4 h-4 text-red-500" />;
      default:
        return <Minus className="w-4 h-4 text-slate-400" />;
    }
  };

  return (
    <div
      className={cn(
        "bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden",
        className
      )}
    >
      {/* Header */}
      <div className="px-4 py-3 bg-slate-50 dark:bg-slate-900 border-b border-slate-200 dark:border-slate-700">
        <div className="flex items-center justify-between">
          <h3 className="font-medium text-slate-900 dark:text-white">Learning Journey</h3>
          <button
            onClick={() => setShowForm(!showForm)}
            className="flex items-center gap-1 px-3 py-1 text-sm bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors"
          >
            <Plus className="w-4 h-4" />
            Add Reflection
          </button>
        </div>
      </div>

      {/* Summary stats */}
      {journey && journey.summary && (
        <div className="p-4 border-b border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-6">
            <ProgressRing value={journey.summary.current_comprehension * 100} />
            <div className="flex-1 grid grid-cols-2 gap-4 text-sm">
              <div>
                <div className="text-slate-500 dark:text-slate-400">Total Time</div>
                <div className="font-medium text-slate-900 dark:text-white">
                  {Math.round(journey.summary.total_time_minutes)} min
                </div>
              </div>
              <div>
                <div className="text-slate-500 dark:text-slate-400">Sessions</div>
                <div className="font-medium text-slate-900 dark:text-white">
                  {journey.snapshot_count}
                </div>
              </div>
              <div>
                <div className="text-slate-500 dark:text-slate-400">Takeaways</div>
                <div className="font-medium text-slate-900 dark:text-white">
                  {journey.summary.total_takeaways}
                </div>
              </div>
              <div>
                <div className="text-slate-500 dark:text-slate-400">Open Questions</div>
                <div className="font-medium text-slate-900 dark:text-white">
                  {journey.summary.questions_remaining}
                </div>
              </div>
            </div>
          </div>

          {/* Growth indicator */}
          {journey.summary.comprehension_growth !== undefined && (
            <div className="mt-3 pt-3 border-t border-slate-100 dark:border-slate-700 flex items-center gap-2">
              {getTrendIcon(
                journey.summary.comprehension_growth > 0.1
                  ? "increasing"
                  : journey.summary.comprehension_growth < -0.1
                  ? "decreasing"
                  : "stable"
              )}
              <span className="text-sm text-slate-600 dark:text-slate-400">
                Comprehension{" "}
                {journey.summary.comprehension_growth > 0 ? "+" : ""}
                {Math.round(journey.summary.comprehension_growth * 100)}% since first read
              </span>
            </div>
          )}
        </div>
      )}

      {/* Reflection prompts */}
      {reflectionPrompts.length > 0 && (
        <div className="p-4 bg-primary-50 dark:bg-primary-900/20 border-b border-slate-200 dark:border-slate-700">
          <div className="text-xs font-medium text-primary-700 dark:text-primary-300 mb-2">
            Reflection Prompts
          </div>
          <ul className="text-sm text-primary-600 dark:text-primary-400 space-y-1">
            {reflectionPrompts.map((prompt, i) => (
              <li key={i} className="flex items-start gap-2">
                <Lightbulb className="w-4 h-4 mt-0.5 flex-shrink-0" />
                {prompt}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* New reflection form */}
      {showForm && (
        <div className="p-4 bg-slate-50 dark:bg-slate-900 border-b border-slate-200 dark:border-slate-700 space-y-4">
          {/* Reflection type */}
          <div>
            <label className="text-xs font-medium text-slate-700 dark:text-slate-300 mb-1 block">
              Type
            </label>
            <select
              value={formData.reflectionType}
              onChange={(e) => setFormData({ ...formData, reflectionType: e.target.value })}
              className="w-full px-3 py-2 text-sm bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-primary-500"
            >
              {REFLECTION_TYPES.map((type) => (
                <option key={type.value} value={type.value}>
                  {type.label}
                </option>
              ))}
            </select>
          </div>

          {/* Summary */}
          <div>
            <label className="text-xs font-medium text-slate-700 dark:text-slate-300 mb-1 block">
              Summary
            </label>
            <textarea
              value={formData.summary}
              onChange={(e) => setFormData({ ...formData, summary: e.target.value })}
              placeholder="What did you understand from this reading?"
              className="w-full px-3 py-2 text-sm bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-primary-500 resize-none"
              rows={3}
            />
          </div>

          {/* Key takeaways */}
          <div>
            <label className="text-xs font-medium text-slate-700 dark:text-slate-300 mb-1 block">
              Key Takeaways
            </label>
            {formData.keyTakeaways.map((takeaway, i) => (
              <div key={i} className="flex gap-2 mb-2">
                <input
                  type="text"
                  value={takeaway}
                  onChange={(e) => handleItemChange("keyTakeaways", i, e.target.value)}
                  placeholder="What did you learn?"
                  className="flex-1 px-3 py-2 text-sm bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-primary-500"
                />
                {formData.keyTakeaways.length > 1 && (
                  <button
                    onClick={() => handleRemoveItem("keyTakeaways", i)}
                    className="text-red-500 hover:text-red-600"
                  >
                    ×
                  </button>
                )}
              </div>
            ))}
            <button
              onClick={() => handleAddItem("keyTakeaways")}
              className="text-xs text-primary-500 hover:underline"
            >
              + Add takeaway
            </button>
          </div>

          {/* Questions */}
          <div>
            <label className="text-xs font-medium text-slate-700 dark:text-slate-300 mb-1 block">
              Questions
            </label>
            {formData.questions.map((question, i) => (
              <div key={i} className="flex gap-2 mb-2">
                <input
                  type="text"
                  value={question}
                  onChange={(e) => handleItemChange("questions", i, e.target.value)}
                  placeholder="What questions do you have?"
                  className="flex-1 px-3 py-2 text-sm bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-primary-500"
                />
                {formData.questions.length > 1 && (
                  <button
                    onClick={() => handleRemoveItem("questions", i)}
                    className="text-red-500 hover:text-red-600"
                  >
                    ×
                  </button>
                )}
              </div>
            ))}
            <button
              onClick={() => handleAddItem("questions")}
              className="text-xs text-primary-500 hover:underline"
            >
              + Add question
            </button>
          </div>

          {/* Time spent */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-xs font-medium text-slate-700 dark:text-slate-300 mb-1 block">
                Time Spent (min)
              </label>
              <input
                type="number"
                value={formData.timeSpentMinutes}
                onChange={(e) =>
                  setFormData({ ...formData, timeSpentMinutes: parseInt(e.target.value) || 0 })
                }
                className="w-full px-3 py-2 text-sm bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-primary-500"
                min="0"
              />
            </div>
            <div>
              <label className="text-xs font-medium text-slate-700 dark:text-slate-300 mb-1 block">
                Progress (%)
              </label>
              <input
                type="number"
                value={formData.scrollDepth}
                onChange={(e) =>
                  setFormData({ ...formData, scrollDepth: parseInt(e.target.value) || 0 })
                }
                className="w-full px-3 py-2 text-sm bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-primary-500"
                min="0"
                max="100"
              />
            </div>
          </div>

          {/* Actions */}
          <div className="flex justify-end gap-2">
            <button
              onClick={() => setShowForm(false)}
              className="px-4 py-2 text-sm text-slate-600 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200"
            >
              Cancel
            </button>
            <button
              onClick={handleSubmit}
              disabled={saving}
              className="flex items-center gap-2 px-4 py-2 text-sm bg-primary-500 text-white rounded-lg hover:bg-primary-600 disabled:opacity-50 transition-colors"
            >
              {saving ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Save className="w-4 h-4" />
              )}
              Save Reflection
            </button>
          </div>
        </div>
      )}

      {/* Snapshots list */}
      <div className="p-4 space-y-4 max-h-96 overflow-y-auto">
        {journey?.first_read ? (
          <>
            {journey.latest_read && journey.latest_read.snapshot_id !== journey.first_read.snapshot_id && (
              <SnapshotCard snapshot={journey.latest_read} />
            )}
            <SnapshotCard snapshot={journey.first_read} />
          </>
        ) : (
          <div className="text-center py-8 text-slate-500 dark:text-slate-400">
            No reading sessions yet. Start reading and add your first reflection!
          </div>
        )}
      </div>
    </div>
  );
}
