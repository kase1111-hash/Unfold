"use client";

import { useState, useEffect, useCallback } from "react";
import { cn } from "@/utils/cn";
import type { EthicsDashboard as EthicsDashboardType, AIOperation } from "@/types";
import {
  Shield,
  BarChart3,
  Clock,
  AlertTriangle,
  TrendingUp,
  ChevronRight,
  Loader2,
  Bot,
  FileText,
  Search,
  Lightbulb,
  RefreshCw,
} from "lucide-react";

interface EthicsDashboardProps {
  className?: string;
}

const OPERATION_ICONS: Record<string, typeof Bot> = {
  document_upload: FileText,
  ai_summary: Bot,
  ai_extraction: Search,
  ai_flashcard: Lightbulb,
  ai_recommendation: TrendingUp,
  bias_check: AlertTriangle,
  default: Bot,
};

function PrivacyScoreRing({ score }: { score: number }) {
  const radius = 45;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;

  const getScoreColor = (s: number) => {
    if (s >= 80) return "text-green-500";
    if (s >= 60) return "text-yellow-500";
    return "text-red-500";
  };

  return (
    <div className="relative w-28 h-28">
      <svg className="w-28 h-28 -rotate-90">
        <circle
          cx="56"
          cy="56"
          r={radius}
          stroke="currentColor"
          strokeWidth="8"
          fill="none"
          className="text-slate-200 dark:text-slate-700"
        />
        <circle
          cx="56"
          cy="56"
          r={radius}
          stroke="currentColor"
          strokeWidth="8"
          fill="none"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className={getScoreColor(score)}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className={cn("text-2xl font-bold", getScoreColor(score))}>
          {Math.round(score)}
        </span>
        <span className="text-xs text-slate-500 dark:text-slate-400">Privacy</span>
      </div>
    </div>
  );
}

function StatCard({
  icon: Icon,
  label,
  value,
  trend,
  color = "primary",
}: {
  icon: typeof Shield;
  label: string;
  value: string | number;
  trend?: string;
  color?: "primary" | "yellow" | "green" | "red";
}) {
  const colorClasses = {
    primary: "bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400",
    yellow: "bg-yellow-100 dark:bg-yellow-900/30 text-yellow-600 dark:text-yellow-400",
    green: "bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400",
    red: "bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400",
  };

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4">
      <div className="flex items-center gap-3">
        <div className={cn("p-2 rounded-lg", colorClasses[color])}>
          <Icon className="w-5 h-5" />
        </div>
        <div>
          <p className="text-sm text-slate-500 dark:text-slate-400">{label}</p>
          <p className="text-xl font-semibold text-slate-900 dark:text-white">{value}</p>
          {trend && (
            <p className="text-xs text-green-500 flex items-center gap-1">
              <TrendingUp className="w-3 h-3" />
              {trend}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function OperationItem({ operation }: { operation: AIOperation }) {
  const Icon = OPERATION_ICONS[operation.operation_type] || OPERATION_ICONS.default;

  return (
    <div className="flex items-center gap-3 py-2 border-b border-slate-100 dark:border-slate-700 last:border-0">
      <div className="p-2 bg-slate-100 dark:bg-slate-700 rounded-lg">
        <Icon className="w-4 h-4 text-slate-600 dark:text-slate-400" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-slate-900 dark:text-white truncate">
          {operation.purpose}
        </p>
        <p className="text-xs text-slate-500 dark:text-slate-400">
          {operation.model_used || "System"} - {operation.operation_type}
        </p>
      </div>
      <div className="text-right">
        <p className="text-xs text-slate-500 dark:text-slate-400">
          {new Date(operation.timestamp).toLocaleTimeString()}
        </p>
        {operation.confidence_score !== undefined && (
          <p className={cn(
            "text-xs",
            operation.confidence_score >= 0.8 ? "text-green-500" :
            operation.confidence_score >= 0.6 ? "text-yellow-500" : "text-red-500"
          )}>
            {Math.round(operation.confidence_score * 100)}% confidence
          </p>
        )}
      </div>
    </div>
  );
}

function RecommendationCard({ recommendation }: { recommendation: string }) {
  return (
    <div className="flex items-start gap-3 p-3 bg-primary-50 dark:bg-primary-900/20 rounded-lg">
      <Lightbulb className="w-5 h-5 text-primary-500 flex-shrink-0 mt-0.5" />
      <p className="text-sm text-slate-700 dark:text-slate-300">{recommendation}</p>
    </div>
  );
}

export function EthicsDashboard({ className }: EthicsDashboardProps) {
  const [dashboard, setDashboard] = useState<EthicsDashboardType | null>(null);
  const [loading, setLoading] = useState(true);
  const [periodDays, setPeriodDays] = useState(30);

  const loadDashboard = useCallback(async () => {
    setLoading(true);
    try {
      // Mock data for demonstration - replace with actual API call
      const mockDashboard: EthicsDashboardType = {
        user_id: "user_123",
        generated_at: new Date().toISOString(),
        period_start: new Date(Date.now() - periodDays * 24 * 60 * 60 * 1000).toISOString(),
        period_end: new Date().toISOString(),
        summary: {
          ai_operations_count: 47,
          documents_analyzed: 12,
          bias_findings_count: 3,
          privacy_score: 92,
        },
        operations_by_type: {
          ai_summary: 20,
          ai_extraction: 15,
          ai_flashcard: 8,
          document_upload: 4,
        },
        operations_by_day: {},
        recent_operations: [
          {
            operation_id: "op_1",
            operation_type: "ai_summary",
            timestamp: new Date().toISOString(),
            model_used: "claude-3",
            input_tokens: 1500,
            output_tokens: 500,
            purpose: "Generated document summary",
            data_accessed: ["document_content"],
            confidence_score: 0.95,
            human_review_required: false,
          },
          {
            operation_id: "op_2",
            operation_type: "ai_extraction",
            timestamp: new Date(Date.now() - 3600000).toISOString(),
            model_used: "claude-3",
            input_tokens: 2000,
            output_tokens: 800,
            purpose: "Extracted key concepts for knowledge graph",
            data_accessed: ["document_content", "existing_graph"],
            confidence_score: 0.87,
            human_review_required: false,
          },
          {
            operation_id: "op_3",
            operation_type: "bias_check",
            timestamp: new Date(Date.now() - 7200000).toISOString(),
            purpose: "Bias audit on uploaded document",
            data_accessed: ["document_content"],
            input_tokens: 0,
            output_tokens: 0,
            confidence_score: 0.72,
            human_review_required: true,
          },
        ],
        metrics: [],
        recommendations: [
          "Your ethics profile looks good! Keep monitoring the dashboard for transparency into AI operations.",
          "Review the 3 bias findings detected in your recent documents.",
        ],
      };

      setDashboard(mockDashboard);
    } finally {
      setLoading(false);
    }
  }, [periodDays]);

  useEffect(() => {
    loadDashboard();
  }, [loadDashboard]);

  if (loading) {
    return (
      <div className={cn("flex items-center justify-center py-12", className)}>
        <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
      </div>
    );
  }

  if (!dashboard) {
    return (
      <div className={cn("text-center py-12 text-slate-500", className)}>
        Failed to load ethics dashboard
      </div>
    );
  }

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-slate-900 dark:text-white flex items-center gap-2">
            <Shield className="w-6 h-6 text-primary-500" />
            Ethics Dashboard
          </h2>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            AI operations transparency and privacy metrics
          </p>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={periodDays}
            onChange={(e) => setPeriodDays(Number(e.target.value))}
            className="px-3 py-2 text-sm bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg"
          >
            <option value={7}>Last 7 days</option>
            <option value={30}>Last 30 days</option>
            <option value={90}>Last 90 days</option>
          </select>
          <button
            onClick={loadDashboard}
            className="p-2 text-slate-500 hover:text-slate-700 dark:hover:text-slate-300"
          >
            <RefreshCw className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="col-span-2 md:col-span-1 flex items-center justify-center bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4">
          <PrivacyScoreRing score={dashboard.summary.privacy_score} />
        </div>
        <StatCard
          icon={Bot}
          label="AI Operations"
          value={dashboard.summary.ai_operations_count}
          color="primary"
        />
        <StatCard
          icon={FileText}
          label="Documents Analyzed"
          value={dashboard.summary.documents_analyzed}
          color="green"
        />
        <StatCard
          icon={AlertTriangle}
          label="Bias Findings"
          value={dashboard.summary.bias_findings_count}
          color={dashboard.summary.bias_findings_count > 5 ? "red" : "yellow"}
        />
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Recent Operations */}
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-700 flex items-center justify-between">
            <h3 className="font-medium text-slate-900 dark:text-white flex items-center gap-2">
              <Clock className="w-4 h-4 text-slate-400" />
              Recent AI Operations
            </h3>
            <button className="text-sm text-primary-500 hover:underline flex items-center gap-1">
              View all
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
          <div className="p-4 max-h-80 overflow-y-auto">
            {dashboard.recent_operations.length === 0 ? (
              <p className="text-center text-slate-500 dark:text-slate-400 py-4">
                No recent operations
              </p>
            ) : (
              dashboard.recent_operations.map((op) => (
                <OperationItem key={op.operation_id} operation={op} />
              ))
            )}
          </div>
        </div>

        {/* Operations by Type */}
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-700">
            <h3 className="font-medium text-slate-900 dark:text-white flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-slate-400" />
              Operations by Type
            </h3>
          </div>
          <div className="p-4 space-y-3">
            {Object.entries(dashboard.operations_by_type).map(([type, count]) => {
              const Icon = OPERATION_ICONS[type] || OPERATION_ICONS.default;
              const maxCount = Math.max(...Object.values(dashboard.operations_by_type));
              const percentage = (count / maxCount) * 100;

              return (
                <div key={type} className="space-y-1">
                  <div className="flex items-center justify-between text-sm">
                    <span className="flex items-center gap-2 text-slate-700 dark:text-slate-300">
                      <Icon className="w-4 h-4" />
                      {type.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                    </span>
                    <span className="text-slate-500 dark:text-slate-400">{count}</span>
                  </div>
                  <div className="h-2 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-primary-500 rounded-full transition-all"
                      style={{ width: `${percentage}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Recommendations */}
      {dashboard.recommendations.length > 0 && (
        <div className="space-y-3">
          <h3 className="font-medium text-slate-900 dark:text-white flex items-center gap-2">
            <Lightbulb className="w-4 h-4 text-yellow-500" />
            Recommendations
          </h3>
          <div className="space-y-2">
            {dashboard.recommendations.map((rec, i) => (
              <RecommendationCard key={i} recommendation={rec} />
            ))}
          </div>
        </div>
      )}

      {/* Generated timestamp */}
      <p className="text-xs text-slate-400 text-center">
        Dashboard generated {new Date(dashboard.generated_at).toLocaleString()}
      </p>
    </div>
  );
}
