"use client";

import { useState, useEffect, useCallback } from "react";
import { Brain, Plus, Play, Loader2, AlertCircle } from "lucide-react";
import { FlashcardReview, StudyStats } from "@/components/learning";
import { Button } from "@/components/ui";
import { api, getErrorMessage } from "@/services/api";
import toast from "react-hot-toast";
import type { Document } from "@/types";

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
  const [isCreating, setIsCreating] = useState(false);
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
        question: card.question,
        answer: card.answer,
        hint: card.hint ?? undefined,
        type: card.card_type,
        difficulty: card.difficulty,
      }));
      setFlashcards(cards);
    } catch (err) {
      setError(`Failed to load flashcards: ${getErrorMessage(err)}`);
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
    } catch (err) {
      toast.error(`Failed to save review: ${getErrorMessage(err)}`);
    }
  }, []);

  // The reviewer stays mounted to show its summary; reload when the user leaves it
  const handleReviewComplete = useCallback(
    (results: { card_id: string; quality: number; time_ms: number }[]) => {
      const correctCount = results.filter((r) => r.quality >= 3).length;
      toast.success(`Session complete: ${correctCount}/${results.length} correct`);
    },
    []
  );

  const handleReviewExit = useCallback(() => {
    setIsReviewing(false);
    loadFlashcards();
  }, [loadFlashcards]);

  const handleCardsCreated = useCallback(
    (count: number) => {
      setIsCreating(false);
      toast.success(`Created ${count} flashcard${count === 1 ? "" : "s"}`);
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
        <div role="alert" className="flex flex-col items-center gap-3 text-center">
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

        {!isReviewing && (
          <Button
            onClick={() => setIsCreating(true)}
            leftIcon={<Plus className="w-4 h-4" />}
          >
            Create Cards
          </Button>
        )}
      </div>

      {isCreating && (
        <CreateCardsDialog
          onCancel={() => setIsCreating(false)}
          onCreated={handleCardsCreated}
        />
      )}

      {isReviewing ? (
        <div className="max-w-2xl mx-auto">
          <FlashcardReview
            flashcards={flashcards}
            onReview={handleReview}
            onComplete={handleReviewComplete}
            onExit={handleReviewExit}
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
                  : "Use Create Cards to generate flashcards from one of your documents."}
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

// Picks one of the user's documents and generates flashcards from its stored text
function CreateCardsDialog({
  onCancel,
  onCreated,
}: {
  onCancel: () => void;
  onCreated: (count: number) => void;
}) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDocId, setSelectedDocId] = useState("");
  const [isLoadingDocs, setIsLoadingDocs] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);
  const [dialogError, setDialogError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getDocuments(1, 100)
      .then((res) => {
        setDocuments(res.data);
        if (res.data.length > 0) setSelectedDocId(res.data[0].doc_id);
      })
      .catch((err) => setDialogError(getErrorMessage(err)))
      .finally(() => setIsLoadingDocs(false));
  }, []);

  const handleGenerate = async () => {
    if (!selectedDocId) return;
    setIsGenerating(true);
    setDialogError(null);
    try {
      const result = await api.generateFlashcards(selectedDocId);
      onCreated(result.count);
    } catch (err) {
      setDialogError(getErrorMessage(err));
      setIsGenerating(false);
    }
  };

  return (
    <div
      role="dialog"
      aria-labelledby="create-cards-title"
      className="card p-6 max-w-xl space-y-4"
    >
      <h2
        id="create-cards-title"
        className="text-lg font-semibold text-slate-900 dark:text-white"
      >
        Generate flashcards from a document
      </h2>

      {isLoadingDocs ? (
        <div className="flex items-center gap-2 text-sm text-slate-500">
          <Loader2 className="w-4 h-4 animate-spin" />
          Loading documents...
        </div>
      ) : documents.length === 0 ? (
        <p className="text-sm text-slate-600 dark:text-slate-400">
          Upload a document first, then generate flashcards from it.
        </p>
      ) : (
        <div>
          <label
            htmlFor="create-cards-document"
            className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1.5"
          >
            Document
          </label>
          <select
            id="create-cards-document"
            value={selectedDocId}
            onChange={(e) => setSelectedDocId(e.target.value)}
            className="w-full px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-primary-500/50"
          >
            {documents.map((doc) => (
              <option key={doc.doc_id} value={doc.doc_id}>
                {doc.title}
              </option>
            ))}
          </select>
        </div>
      )}

      {dialogError && (
        <p role="alert" className="text-sm text-red-600 dark:text-red-400">
          {dialogError}
        </p>
      )}

      <div className="flex justify-end gap-2">
        <Button variant="secondary" onClick={onCancel} disabled={isGenerating}>
          Cancel
        </Button>
        <Button
          onClick={handleGenerate}
          isLoading={isGenerating}
          disabled={!selectedDocId}
        >
          Generate
        </Button>
      </div>
    </div>
  );
}
