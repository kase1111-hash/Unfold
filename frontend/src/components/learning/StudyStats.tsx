"use client";

import { useEffect, useState } from "react";
import {
  Brain,
  Target,
  Clock,
  TrendingUp,
  Calendar,
  Award,
  Loader2,
} from "lucide-react";
import { cn } from "@/utils/cn";
import { api } from "@/services/api";

interface StudyStatsData {
  total_cards: number;
  due_now: number;
  due_today: number;
  average_ef: number;
  average_retention: number;
  mature_cards: number;
  learning_cards: number;
}

interface EngagementProfile {
  total_reading_time_minutes: number;
  documents_read: number;
  avg_session_duration_minutes: number;
  avg_scroll_depth: number;
  preferred_complexity: number;
  total_highlights: number;
  total_flashcards: number;
  comprehension_score: number;
}

interface StudyStatsProps {
  className?: string;
}

export function StudyStats({ className }: StudyStatsProps) {
  const [studyStats, setStudyStats] = useState<StudyStatsData | null>(null);
  const [engagementProfile, setEngagementProfile] = useState<EngagementProfile | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        // In production, these would be real API calls
        // For now, use mock data
        setStudyStats({
          total_cards: 42,
          due_now: 8,
          due_today: 15,
          average_ef: 2.35,
          average_retention: 78.5,
          mature_cards: 28,
          learning_cards: 14,
        });

        setEngagementProfile({
          total_reading_time_minutes: 245,
          documents_read: 12,
          avg_session_duration_minutes: 18.5,
          avg_scroll_depth: 72.3,
          preferred_complexity: 55,
          total_highlights: 34,
          total_flashcards: 42,
          comprehension_score: 68.5,
        });
      } catch (error) {
        console.error("Failed to fetch stats:", error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchStats();
  }, []);

  if (isLoading) {
    return (
      <div className={cn("card p-8 flex items-center justify-center", className)}>
        <Loader2 className="w-6 h-6 animate-spin text-primary-500" />
      </div>
    );
  }

  return (
    <div className={cn("space-y-6", className)}>
      {/* Flashcard Stats */}
      <div className="card">
        <div className="p-4 border-b border-slate-200 dark:border-slate-700">
          <h3 className="font-semibold text-slate-900 dark:text-white flex items-center gap-2">
            <Brain className="w-5 h-5 text-primary-500" />
            Flashcard Progress
          </h3>
        </div>

        <div className="p-4 grid grid-cols-2 gap-4">
          {/* Due Now */}
          <div className="text-center p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
            <p className="text-3xl font-bold text-red-600 dark:text-red-400">
              {studyStats?.due_now || 0}
            </p>
            <p className="text-sm text-red-600/80 dark:text-red-400/80">Due Now</p>
          </div>

          {/* Due Today */}
          <div className="text-center p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
            <p className="text-3xl font-bold text-yellow-600 dark:text-yellow-400">
              {studyStats?.due_today || 0}
            </p>
            <p className="text-sm text-yellow-600/80 dark:text-yellow-400/80">Due Today</p>
          </div>

          {/* Retention */}
          <div className="text-center p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
            <p className="text-3xl font-bold text-green-600 dark:text-green-400">
              {studyStats?.average_retention || 0}%
            </p>
            <p className="text-sm text-green-600/80 dark:text-green-400/80">Retention</p>
          </div>

          {/* Total Cards */}
          <div className="text-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <p className="text-3xl font-bold text-blue-600 dark:text-blue-400">
              {studyStats?.total_cards || 0}
            </p>
            <p className="text-sm text-blue-600/80 dark:text-blue-400/80">Total Cards</p>
          </div>
        </div>

        {/* Card Distribution */}
        <div className="px-4 pb-4">
          <div className="flex items-center justify-between text-sm text-slate-600 dark:text-slate-400 mb-2">
            <span>Learning</span>
            <span>Mature</span>
          </div>
          <div className="h-3 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-yellow-500 to-green-500 rounded-full"
              style={{
                width: `${
                  studyStats
                    ? (studyStats.mature_cards / studyStats.total_cards) * 100
                    : 0
                }%`,
              }}
            />
          </div>
          <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 mt-1">
            <span>{studyStats?.learning_cards || 0} cards</span>
            <span>{studyStats?.mature_cards || 0} cards</span>
          </div>
        </div>
      </div>

      {/* Reading Stats */}
      <div className="card">
        <div className="p-4 border-b border-slate-200 dark:border-slate-700">
          <h3 className="font-semibold text-slate-900 dark:text-white flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-primary-500" />
            Reading Progress
          </h3>
        </div>

        <div className="p-4 space-y-4">
          {/* Reading Time */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center">
                <Clock className="w-5 h-5 text-purple-600" />
              </div>
              <div>
                <p className="font-medium text-slate-900 dark:text-white">
                  Total Reading Time
                </p>
                <p className="text-sm text-slate-500 dark:text-slate-400">
                  Avg {engagementProfile?.avg_session_duration_minutes?.toFixed(0) || 0} min/session
                </p>
              </div>
            </div>
            <p className="text-lg font-bold text-slate-900 dark:text-white">
              {Math.round((engagementProfile?.total_reading_time_minutes || 0) / 60)}h{" "}
              {(engagementProfile?.total_reading_time_minutes || 0) % 60}m
            </p>
          </div>

          {/* Documents Read */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center">
                <Calendar className="w-5 h-5 text-blue-600" />
              </div>
              <div>
                <p className="font-medium text-slate-900 dark:text-white">
                  Documents Read
                </p>
                <p className="text-sm text-slate-500 dark:text-slate-400">
                  {engagementProfile?.total_highlights || 0} highlights
                </p>
              </div>
            </div>
            <p className="text-lg font-bold text-slate-900 dark:text-white">
              {engagementProfile?.documents_read || 0}
            </p>
          </div>

          {/* Comprehension Score */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-green-100 dark:bg-green-900/30 flex items-center justify-center">
                <Award className="w-5 h-5 text-green-600" />
              </div>
              <div>
                <p className="font-medium text-slate-900 dark:text-white">
                  Comprehension Score
                </p>
                <p className="text-sm text-slate-500 dark:text-slate-400">
                  Based on engagement
                </p>
              </div>
            </div>
            <p className="text-lg font-bold text-slate-900 dark:text-white">
              {engagementProfile?.comprehension_score?.toFixed(0) || 0}%
            </p>
          </div>

          {/* Preferred Complexity */}
          <div className="pt-4 border-t border-slate-200 dark:border-slate-700">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-slate-600 dark:text-slate-400">
                Preferred Complexity
              </span>
              <span className="text-sm font-medium text-slate-900 dark:text-white">
                {engagementProfile?.preferred_complexity || 50}%
              </span>
            </div>
            <div className="h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-primary-500 rounded-full"
                style={{
                  width: `${engagementProfile?.preferred_complexity || 50}%`,
                }}
              />
            </div>
            <div className="flex justify-between text-xs text-slate-400 mt-1">
              <span>Simple</span>
              <span>Expert</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
