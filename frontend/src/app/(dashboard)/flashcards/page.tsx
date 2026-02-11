"use client";

import { useState, useEffect, useCallback } from "react";
import { Brain, Plus, Download, Play, Loader2, AlertCircle } from "lucide-react";
import { FlashcardReview, StudyStats } from "@/components/learning";
import { Button } from "@/components/ui";
import { api } from "@/services/api";
import toast from "react-hot-toast";

interface Flashcard {
  card_id: string;
  question: string;
  answer: string;
  hint?: string;
  type?: string;
  difficulty?: string;
  key_concepts?: string[];
}

export default function FlashcardsPage() {
  const [isReviewing, setIsReviewing] = useState(false);
  const [flashcards, setFlashcards] = useState<Flashcard[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadFlashcards = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await api.getFlashcardsDue(50);
      const cards: Flashcard[] = result.due_cards.map((card) => ({
        card_id: card.card_id,
        question: `Card ${card.card_id}`,
        answer: "",
        difficulty: card.easiness_factor > 2.5 ? "easy" : card.easiness_factor > 1.5 ? "intermediate" : "hard",
      }));
      setFlashcards(cards);
    } catch {
      setError("Failed to load flashcards. Make sure the backend is running.");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadFlashcards();
  }, [loadFlashcards]);

  const handleReview = useCallback(async (cardId: string, quality: number) => {
    try {
      const result = await api.reviewFlashcard(cardId, quality);
      toast.success(
        `Next review in ${result.interval_days} day${result.interval_days !== 1 ? "s" : ""}`
      );
    } catch {
      toast.error("Failed to save review. Your progress may not be recorded.");
    }
  }, []);

  const handleReviewComplete = useCallback(
    (results: { card_id: string; quality: number; time_ms: number }[]) => {
      setIsReviewing(false);
      const correctCount = results.filter((r) => r.quality >= 3).length;
      toast.success(`Session complete: ${correctCount}/${results.length} correct`);
      loadFlashcards();
    },
    [loadFlashcards]
  );

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
          <span className="text-slate-500 dark:text-slate-400">
            Loading flashcards...
          </span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="flex flex-col items-center gap-3 text-center">
          <AlertCircle className="w-8 h-8 text-red-500" />
          <span className="text-red-500 font-medium">{error}</span>
          <Button variant="secondary" onClick={loadFlashcards}>
            Retry
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
            Flashcards
          </h1>
          <p className="text-slate-600 dark:text-slate-400 mt-1">
            Review and reinforce your learning with spaced repetition
          </p>
        </div>

        <div className="flex gap-2">
          <Button variant="secondary" leftIcon={<Download className="w-4 h-4" />}>
            Export
          </Button>
          <Button leftIcon={<Plus className="w-4 h-4" />}>
            Create Cards
          </Button>
        </div>
      </div>

      {isReviewing ? (
        <div className="max-w-2xl mx-auto">
          <FlashcardReview
            flashcards={flashcards}
            onReview={handleReview}
            onComplete={handleReviewComplete}
          />
        </div>
      ) : (
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Main content */}
          <div className="lg:col-span-2 space-y-6">
            {/* Start Review Card */}
            <div className="card p-8 text-center">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center">
                <Brain className="w-8 h-8 text-primary-600" />
              </div>

              <h2 className="text-xl font-bold text-slate-900 dark:text-white mb-2">
                {flashcards.length > 0 ? "Ready to Review?" : "No Cards Due"}
              </h2>

              <p className="text-slate-600 dark:text-slate-400 mb-6">
                {flashcards.length > 0
                  ? `You have ${flashcards.length} card${flashcards.length !== 1 ? "s" : ""} ready for review. Regular reviews help strengthen your memory.`
                  : "Upload a document and generate flashcards to start studying."}
              </p>

              {flashcards.length > 0 && (
                <Button
                  size="lg"
                  onClick={() => setIsReviewing(true)}
                  leftIcon={<Play className="w-5 h-5" />}
                >
                  Start Review Session
                </Button>
              )}
            </div>

            {/* Card Preview */}
            {flashcards.length > 0 && (
              <div className="card">
                <div className="p-4 border-b border-slate-200 dark:border-slate-700">
                  <h3 className="font-semibold text-slate-900 dark:text-white">
                    Due Cards
                  </h3>
                </div>

                <div className="divide-y divide-slate-200 dark:divide-slate-700">
                  {flashcards.slice(0, 5).map((card) => (
                    <div key={card.card_id} className="p-4">
                      <p className="font-medium text-slate-900 dark:text-white text-sm mb-1">
                        {card.question}
                      </p>
                      <div className="flex items-center gap-2 mt-2">
                        {card.type && (
                          <span className="text-xs px-2 py-0.5 bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded">
                            {card.type}
                          </span>
                        )}
                        {card.difficulty && (
                          <span className="text-xs px-2 py-0.5 bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-400 rounded">
                            {card.difficulty}
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Sidebar */}
          <div>
            <StudyStats />
          </div>
        </div>
      )}
    </div>
  );
}
