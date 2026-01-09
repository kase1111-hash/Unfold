"use client";

import { useState, useEffect } from "react";
import { useScholarStore } from "@/store";
import { cn } from "@/utils/cn";
import type { CredibilityScore } from "@/types";
import {
  Shield,
  ShieldCheck,
  ShieldAlert,
  ShieldQuestion,
  AlertTriangle,
  Info,
  ChevronDown,
  ChevronUp,
  ExternalLink,
} from "lucide-react";

interface CredibilityBadgeProps {
  doi: string;
  className?: string;
  showDetails?: boolean;
}

const LEVEL_CONFIG = {
  high: {
    icon: ShieldCheck,
    color: "text-green-500",
    bg: "bg-green-50 dark:bg-green-900/20",
    border: "border-green-200 dark:border-green-800",
    label: "High Credibility",
  },
  medium: {
    icon: Shield,
    color: "text-yellow-500",
    bg: "bg-yellow-50 dark:bg-yellow-900/20",
    border: "border-yellow-200 dark:border-yellow-800",
    label: "Medium Credibility",
  },
  low: {
    icon: ShieldAlert,
    color: "text-red-500",
    bg: "bg-red-50 dark:bg-red-900/20",
    border: "border-red-200 dark:border-red-800",
    label: "Low Credibility",
  },
  unknown: {
    icon: ShieldQuestion,
    color: "text-slate-400",
    bg: "bg-slate-50 dark:bg-slate-900/20",
    border: "border-slate-200 dark:border-slate-800",
    label: "Unknown",
  },
};

function ScoreBar({ label, score, color }: { label: string; score: number; color: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-slate-500 w-20">{label}</span>
      <div className="flex-1 h-2 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
        <div
          className={cn("h-full rounded-full transition-all", color)}
          style={{ width: `${score}%` }}
        />
      </div>
      <span className="text-xs text-slate-600 dark:text-slate-400 w-8 text-right">
        {Math.round(score)}
      </span>
    </div>
  );
}

export function CredibilityBadge({ doi, className, showDetails = false }: CredibilityBadgeProps) {
  const [expanded, setExpanded] = useState(showDetails);
  const [loading, setLoading] = useState(false);
  const { credibilityScores, loadCredibilityScore } = useScholarStore();

  const score = credibilityScores.get(doi);

  useEffect(() => {
    if (!score && doi) {
      setLoading(true);
      loadCredibilityScore(doi).finally(() => setLoading(false));
    }
  }, [doi, score, loadCredibilityScore]);

  if (loading) {
    return (
      <div
        className={cn(
          "flex items-center gap-2 px-3 py-2 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800",
          className
        )}
      >
        <div className="w-4 h-4 border-2 border-slate-300 border-t-primary-500 rounded-full animate-spin" />
        <span className="text-xs text-slate-500">Checking credibility...</span>
      </div>
    );
  }

  if (!score) {
    return null;
  }

  const config = LEVEL_CONFIG[score.level];
  const Icon = config.icon;

  return (
    <div
      className={cn(
        "rounded-lg border transition-all",
        config.bg,
        config.border,
        className
      )}
    >
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-3 py-2"
      >
        <div className="flex items-center gap-2">
          <Icon className={cn("w-5 h-5", config.color)} />
          <div className="text-left">
            <div className="text-sm font-medium text-slate-900 dark:text-white">
              {config.label}
            </div>
            <div className="text-xs text-slate-500">
              Score: {Math.round(score.overall_score)}/100
            </div>
          </div>
        </div>
        {expanded ? (
          <ChevronUp className="w-4 h-4 text-slate-400" />
        ) : (
          <ChevronDown className="w-4 h-4 text-slate-400" />
        )}
      </button>

      {/* Expanded details */}
      {expanded && (
        <div className="px-3 pb-3 border-t border-slate-200/50 dark:border-slate-700/50 pt-3 space-y-4">
          {/* Warnings */}
          {score.warnings.length > 0 && (
            <div className="space-y-1">
              {score.warnings.map((warning, i) => (
                <div
                  key={i}
                  className="flex items-start gap-2 text-xs text-red-600 dark:text-red-400"
                >
                  <AlertTriangle className="w-3 h-3 mt-0.5 flex-shrink-0" />
                  <span>{warning}</span>
                </div>
              ))}
            </div>
          )}

          {/* Notes */}
          {score.notes.length > 0 && (
            <div className="space-y-1">
              {score.notes.map((note, i) => (
                <div
                  key={i}
                  className="flex items-start gap-2 text-xs text-slate-600 dark:text-slate-400"
                >
                  <Info className="w-3 h-3 mt-0.5 flex-shrink-0" />
                  <span>{note}</span>
                </div>
              ))}
            </div>
          )}

          {/* Component scores */}
          <div className="space-y-2">
            <div className="text-xs font-medium text-slate-700 dark:text-slate-300">
              Component Scores
            </div>
            <ScoreBar
              label="Citations"
              score={score.components.citation_score}
              color="bg-blue-500"
            />
            <ScoreBar
              label="Venue"
              score={score.components.venue_score}
              color="bg-purple-500"
            />
            <ScoreBar
              label="Author"
              score={score.components.author_score}
              color="bg-green-500"
            />
            <ScoreBar
              label="Recency"
              score={score.components.recency_score}
              color="bg-yellow-500"
            />
            <ScoreBar
              label="Altmetric"
              score={score.components.altmetric_score}
              color="bg-pink-500"
            />
          </div>

          {/* Metadata */}
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="flex justify-between">
              <span className="text-slate-500">Citations:</span>
              <span className="text-slate-700 dark:text-slate-300">
                {score.metadata.citation_count.toLocaleString()}
              </span>
            </div>
            {score.metadata.journal_impact_factor && (
              <div className="flex justify-between">
                <span className="text-slate-500">Impact Factor:</span>
                <span className="text-slate-700 dark:text-slate-300">
                  {score.metadata.journal_impact_factor.toFixed(1)}
                </span>
              </div>
            )}
            {score.metadata.altmetric_attention && (
              <div className="flex justify-between">
                <span className="text-slate-500">Altmetric:</span>
                <span className="text-slate-700 dark:text-slate-300">
                  {score.metadata.altmetric_attention}
                </span>
              </div>
            )}
            <div className="flex justify-between">
              <span className="text-slate-500">Peer Reviewed:</span>
              <span className="text-slate-700 dark:text-slate-300">
                {score.metadata.is_peer_reviewed ? "Yes" : "No"}
              </span>
            </div>
          </div>

          {/* DOI link */}
          <a
            href={`https://doi.org/${doi}`}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 text-xs text-primary-500 hover:underline"
          >
            View on DOI.org
            <ExternalLink className="w-3 h-3" />
          </a>
        </div>
      )}
    </div>
  );
}
