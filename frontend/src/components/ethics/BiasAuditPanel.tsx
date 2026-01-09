"use client";

import { useState, useCallback } from "react";
import { cn } from "@/utils/cn";
import type { BiasAuditReport, BiasFinding, BiasCategory, Severity } from "@/types";
import {
  AlertTriangle,
  CheckCircle,
  Info,
  ArrowRight,
  Loader2,
  BarChart2,
  FileText,
  TrendingUp,
} from "lucide-react";

interface BiasAuditPanelProps {
  documentId: string;
  className?: string;
  onAudit?: (report: BiasAuditReport) => void;
}

const CATEGORY_LABELS: Record<BiasCategory, string> = {
  gender: "Gender Bias",
  race: "Racial Bias",
  age: "Age Bias",
  ability: "Ableism",
  religion: "Religious Bias",
  nationality: "Nationality Bias",
  political: "Political Bias",
  other: "Other Bias",
};

const CATEGORY_COLORS: Record<BiasCategory, string> = {
  gender: "bg-pink-100 text-pink-700 dark:bg-pink-900/30 dark:text-pink-400",
  race: "bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400",
  age: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400",
  ability: "bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400",
  religion: "bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-400",
  nationality: "bg-teal-100 text-teal-700 dark:bg-teal-900/30 dark:text-teal-400",
  political: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400",
  other: "bg-slate-100 text-slate-700 dark:bg-slate-700 dark:text-slate-300",
};

const SEVERITY_CONFIG: Record<Severity, { icon: typeof AlertTriangle; color: string; label: string }> = {
  high: { icon: AlertTriangle, color: "text-red-500", label: "High" },
  medium: { icon: Info, color: "text-yellow-500", label: "Medium" },
  low: { icon: Info, color: "text-blue-500", label: "Low" },
};

function InclusivityMeter({ score }: { score: number }) {
  const getColor = (s: number) => {
    if (s >= 80) return "bg-green-500";
    if (s >= 60) return "bg-yellow-500";
    return "bg-red-500";
  };

  const getLabel = (s: number) => {
    if (s >= 80) return "Excellent";
    if (s >= 60) return "Good";
    if (s >= 40) return "Fair";
    return "Needs Improvement";
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-sm text-slate-600 dark:text-slate-400">Inclusivity Score</span>
        <span className="text-sm font-medium text-slate-900 dark:text-white">
          {Math.round(score)}% - {getLabel(score)}
        </span>
      </div>
      <div className="h-3 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
        <div
          className={cn("h-full rounded-full transition-all", getColor(score))}
          style={{ width: `${score}%` }}
        />
      </div>
    </div>
  );
}

function SentimentBadge({ label, score }: { label: string; score: number }) {
  const config = {
    positive: { color: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400" },
    negative: { color: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400" },
    neutral: { color: "bg-slate-100 text-slate-700 dark:bg-slate-700 dark:text-slate-300" },
  };

  return (
    <div className={cn("px-3 py-1.5 rounded-full text-sm font-medium", config[label as keyof typeof config]?.color)}>
      {label.charAt(0).toUpperCase() + label.slice(1)} ({Math.round(score * 100)}%)
    </div>
  );
}

function FindingCard({ finding }: { finding: BiasFinding }) {
  const [expanded, setExpanded] = useState(false);
  const severityConfig = SEVERITY_CONFIG[finding.severity];
  const SeverityIcon = severityConfig.icon;

  return (
    <div className="border border-slate-200 dark:border-slate-700 rounded-lg overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-start gap-3 p-3 text-left hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
      >
        <SeverityIcon className={cn("w-5 h-5 flex-shrink-0 mt-0.5", severityConfig.color)} />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className={cn("px-2 py-0.5 text-xs rounded-full", CATEGORY_COLORS[finding.category])}>
              {CATEGORY_LABELS[finding.category]}
            </span>
            <span className={cn("text-xs", severityConfig.color)}>
              {severityConfig.label} severity
            </span>
          </div>
          <p className="mt-1 text-sm text-slate-700 dark:text-slate-300 line-clamp-2">
            &ldquo;{finding.text}&rdquo;
          </p>
        </div>
        <ArrowRight className={cn("w-4 h-4 text-slate-400 transition-transform", expanded && "rotate-90")} />
      </button>

      {expanded && (
        <div className="px-3 pb-3 pt-0 space-y-2 border-t border-slate-100 dark:border-slate-700">
          <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
            <p className="text-xs font-medium text-green-700 dark:text-green-400 mb-1">Suggestion</p>
            <p className="text-sm text-green-800 dark:text-green-300">{finding.suggestion}</p>
          </div>
          {finding.position.section && (
            <p className="text-xs text-slate-500">
              Found in section: {finding.position.section}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function CategoryBreakdown({ summary }: { summary: BiasAuditReport["summary"] }) {
  const categories = Object.entries(summary.by_category).filter(([, count]) => count > 0);

  if (categories.length === 0) {
    return null;
  }

  return (
    <div className="space-y-2">
      <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300">By Category</h4>
      <div className="flex flex-wrap gap-2">
        {categories.map(([category, count]) => (
          <div
            key={category}
            className={cn("px-3 py-1 rounded-full text-sm", CATEGORY_COLORS[category as BiasCategory])}
          >
            {CATEGORY_LABELS[category as BiasCategory]}: {count}
          </div>
        ))}
      </div>
    </div>
  );
}

export function BiasAuditPanel({ documentId, className, onAudit }: BiasAuditPanelProps) {
  const [report, setReport] = useState<BiasAuditReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runAudit = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      // Mock data for demonstration - replace with actual API call
      const mockReport: BiasAuditReport = {
        report_id: "report_" + Date.now(),
        document_id: documentId,
        generated_at: new Date().toISOString(),
        findings: [
          {
            finding_id: "f1",
            category: "gender",
            severity: "medium",
            text: "The chairman announced the new policy",
            suggestion: "Consider using gender-neutral term 'chairperson' or 'chair'",
            position: { start: 150, end: 190, section: "Introduction" },
          },
          {
            finding_id: "f2",
            category: "ability",
            severity: "low",
            text: "crazy deadline",
            suggestion: "Consider using 'tight' or 'demanding' instead of 'crazy'",
            position: { start: 500, end: 515, section: "Methods" },
          },
        ],
        sentiment: {
          label: "neutral",
          score: 0.65,
          confidence: 0.89,
        },
        inclusivity_score: 78,
        reading_level: 12.5,
        summary: {
          total_findings: 2,
          by_category: { gender: 1, ability: 1 },
          by_severity: { medium: 1, low: 1 },
        },
        recommendations: [
          "Review and update gendered language to be more inclusive",
          "Consider using person-first language when discussing disabilities",
        ],
      };

      setReport(mockReport);
      onAudit?.(mockReport);
    } catch {
      setError("Failed to run bias audit");
    } finally {
      setLoading(false);
    }
  }, [documentId, onAudit]);

  if (!report && !loading) {
    return (
      <div className={cn("bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6", className)}>
        <div className="text-center space-y-4">
          <div className="w-16 h-16 mx-auto bg-primary-100 dark:bg-primary-900/30 rounded-full flex items-center justify-center">
            <BarChart2 className="w-8 h-8 text-primary-500" />
          </div>
          <div>
            <h3 className="font-medium text-slate-900 dark:text-white">Bias Audit</h3>
            <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
              Analyze this document for potential bias and inclusivity issues
            </p>
          </div>
          <button
            onClick={runAudit}
            className="px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors"
          >
            Run Audit
          </button>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className={cn("bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6", className)}>
        <div className="flex flex-col items-center justify-center py-8 space-y-3">
          <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
          <p className="text-slate-500 dark:text-slate-400">Analyzing document for bias...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={cn("bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6", className)}>
        <div className="text-center text-red-500">
          <AlertTriangle className="w-8 h-8 mx-auto mb-2" />
          {error}
        </div>
      </div>
    );
  }

  if (!report) return null;

  return (
    <div className={cn("bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden", className)}>
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-700 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <BarChart2 className="w-5 h-5 text-primary-500" />
          <h3 className="font-medium text-slate-900 dark:text-white">Bias Audit Report</h3>
        </div>
        <button
          onClick={runAudit}
          className="text-sm text-primary-500 hover:underline"
        >
          Re-run
        </button>
      </div>

      <div className="p-4 space-y-6">
        {/* Summary Stats */}
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center p-3 bg-slate-50 dark:bg-slate-700/50 rounded-lg">
            <p className="text-2xl font-bold text-slate-900 dark:text-white">
              {report.summary.total_findings}
            </p>
            <p className="text-xs text-slate-500 dark:text-slate-400">Findings</p>
          </div>
          <div className="text-center p-3 bg-slate-50 dark:bg-slate-700/50 rounded-lg">
            <p className="text-2xl font-bold text-slate-900 dark:text-white">
              {Math.round(report.inclusivity_score)}%
            </p>
            <p className="text-xs text-slate-500 dark:text-slate-400">Inclusivity</p>
          </div>
          <div className="text-center p-3 bg-slate-50 dark:bg-slate-700/50 rounded-lg">
            <p className="text-2xl font-bold text-slate-900 dark:text-white">
              {report.reading_level.toFixed(1)}
            </p>
            <p className="text-xs text-slate-500 dark:text-slate-400">Reading Level</p>
          </div>
        </div>

        {/* Inclusivity Meter */}
        <InclusivityMeter score={report.inclusivity_score} />

        {/* Sentiment */}
        <div className="flex items-center justify-between">
          <span className="text-sm text-slate-600 dark:text-slate-400">Overall Sentiment</span>
          <SentimentBadge label={report.sentiment.label} score={report.sentiment.score} />
        </div>

        {/* Category Breakdown */}
        <CategoryBreakdown summary={report.summary} />

        {/* Findings */}
        {report.findings.length > 0 && (
          <div className="space-y-3">
            <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4" />
              Findings ({report.findings.length})
            </h4>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {report.findings.map((finding) => (
                <FindingCard key={finding.finding_id} finding={finding} />
              ))}
            </div>
          </div>
        )}

        {report.findings.length === 0 && (
          <div className="text-center py-6">
            <CheckCircle className="w-12 h-12 mx-auto text-green-500 mb-2" />
            <p className="text-slate-900 dark:text-white font-medium">No bias issues detected</p>
            <p className="text-sm text-slate-500 dark:text-slate-400">
              This document appears to use inclusive language
            </p>
          </div>
        )}

        {/* Recommendations */}
        {report.recommendations.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 flex items-center gap-2">
              <TrendingUp className="w-4 h-4" />
              Recommendations
            </h4>
            <ul className="space-y-1">
              {report.recommendations.map((rec, i) => (
                <li key={i} className="text-sm text-slate-600 dark:text-slate-400 flex items-start gap-2">
                  <span className="text-primary-500">-</span>
                  {rec}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Footer */}
        <p className="text-xs text-slate-400 text-center pt-2 border-t border-slate-100 dark:border-slate-700">
          Generated {new Date(report.generated_at).toLocaleString()}
        </p>
      </div>
    </div>
  );
}
