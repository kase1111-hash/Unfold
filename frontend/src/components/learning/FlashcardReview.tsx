"use client";

import { useState, useCallback } from "react";
import {
  RotateCcw,
  Check,
  X,
  Brain,
  ChevronLeft,
  ChevronRight,
  Lightbulb,
} from "lucide-react";
import { cn } from "@/utils/cn";
import { Button } from "@/components/ui";

interface Flashcard {
  card_id: string;
  question: string;
  answer: string;
  hint?: string;
  type?: string;
  difficulty?: string;
  key_concepts?: string[];
}

interface FlashcardReviewProps {
  flashcards: Flashcard[];
  onReview?: (cardId: string, quality: number) => void;
  onComplete?: (results: ReviewResult[]) => void;
}

interface ReviewResult {
  card_id: string;
  quality: number;
  time_ms: number;
}

const QUALITY_BUTTONS = [
  { quality: 0, label: "Blackout", color: "bg-red-600 hover:bg-red-700", icon: X },
  { quality: 1, label: "Wrong", color: "bg-orange-600 hover:bg-orange-700", icon: X },
  { quality: 3, label: "Hard", color: "bg-yellow-600 hover:bg-yellow-700", icon: Brain },
  { quality: 4, label: "Good", color: "bg-blue-600 hover:bg-blue-700", icon: Check },
  { quality: 5, label: "Easy", color: "bg-green-600 hover:bg-green-700", icon: Check },
];

export function FlashcardReview({
  flashcards,
  onReview,
  onComplete,
}: FlashcardReviewProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isFlipped, setIsFlipped] = useState(false);
  const [showHint, setShowHint] = useState(false);
  const [results, setResults] = useState<ReviewResult[]>([]);
  const [cardStartTime, setCardStartTime] = useState(Date.now());
  const [isComplete, setIsComplete] = useState(false);

  const currentCard = flashcards[currentIndex];
  const progress = ((currentIndex) / flashcards.length) * 100;

  const handleFlip = useCallback(() => {
    setIsFlipped((prev) => !prev);
  }, []);

  const handleRate = useCallback(
    (quality: number) => {
      const timeMs = Date.now() - cardStartTime;

      const result: ReviewResult = {
        card_id: currentCard.card_id,
        quality,
        time_ms: timeMs,
      };

      setResults((prev) => [...prev, result]);
      onReview?.(currentCard.card_id, quality);

      // Move to next card or complete
      if (currentIndex < flashcards.length - 1) {
        setCurrentIndex((prev) => prev + 1);
        setIsFlipped(false);
        setShowHint(false);
        setCardStartTime(Date.now());
      } else {
        setIsComplete(true);
        onComplete?.([...results, result]);
      }
    },
    [currentCard, currentIndex, flashcards.length, cardStartTime, results, onReview, onComplete]
  );

  const handlePrevious = useCallback(() => {
    if (currentIndex > 0) {
      setCurrentIndex((prev) => prev - 1);
      setIsFlipped(false);
      setShowHint(false);
      setCardStartTime(Date.now());
    }
  }, [currentIndex]);

  const handleRestart = useCallback(() => {
    setCurrentIndex(0);
    setIsFlipped(false);
    setShowHint(false);
    setResults([]);
    setCardStartTime(Date.now());
    setIsComplete(false);
  }, []);

  if (isComplete) {
    const avgQuality = results.reduce((sum, r) => sum + r.quality, 0) / results.length;
    const correctCount = results.filter((r) => r.quality >= 3).length;

    return (
      <div className="card p-8 text-center">
        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center">
          <Check className="w-8 h-8 text-green-600" />
        </div>

        <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-2">
          Review Complete!
        </h2>

        <p className="text-slate-600 dark:text-slate-400 mb-6">
          You reviewed {flashcards.length} cards
        </p>

        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="p-4 bg-slate-50 dark:bg-slate-700/50 rounded-lg">
            <p className="text-2xl font-bold text-slate-900 dark:text-white">
              {correctCount}
            </p>
            <p className="text-sm text-slate-500 dark:text-slate-400">Correct</p>
          </div>
          <div className="p-4 bg-slate-50 dark:bg-slate-700/50 rounded-lg">
            <p className="text-2xl font-bold text-slate-900 dark:text-white">
              {flashcards.length - correctCount}
            </p>
            <p className="text-sm text-slate-500 dark:text-slate-400">To Review</p>
          </div>
          <div className="p-4 bg-slate-50 dark:bg-slate-700/50 rounded-lg">
            <p className="text-2xl font-bold text-slate-900 dark:text-white">
              {avgQuality.toFixed(1)}
            </p>
            <p className="text-sm text-slate-500 dark:text-slate-400">Avg Score</p>
          </div>
        </div>

        <Button onClick={handleRestart} leftIcon={<RotateCcw className="w-4 h-4" />}>
          Review Again
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Progress bar */}
      <div className="flex items-center gap-3">
        <span className="text-sm text-slate-500 dark:text-slate-400">
          {currentIndex + 1} / {flashcards.length}
        </span>
        <div className="flex-1 h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-primary-500 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Flashcard */}
      <div
        onClick={handleFlip}
        className={cn(
          "card min-h-[300px] cursor-pointer transition-all duration-300 perspective-1000",
          "hover:shadow-lg"
        )}
      >
        <div
          className={cn(
            "relative w-full h-full transition-transform duration-500 preserve-3d",
            isFlipped && "rotate-y-180"
          )}
        >
          {/* Front - Question */}
          <div
            className={cn(
              "absolute inset-0 p-6 backface-hidden",
              isFlipped && "invisible"
            )}
          >
            <div className="flex items-center justify-between mb-4">
              <span className="text-xs font-medium px-2 py-1 bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded-full">
                {currentCard.type || "Question"}
              </span>
              {currentCard.difficulty && (
                <span className="text-xs text-slate-500 dark:text-slate-400">
                  {currentCard.difficulty}
                </span>
              )}
            </div>

            <div className="flex-1 flex items-center justify-center min-h-[150px]">
              <p className="text-lg text-slate-900 dark:text-white text-center">
                {currentCard.question}
              </p>
            </div>

            {currentCard.hint && (
              <div className="mt-4">
                {showHint ? (
                  <p className="text-sm text-slate-500 dark:text-slate-400 text-center italic">
                    Hint: {currentCard.hint}
                  </p>
                ) : (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setShowHint(true);
                    }}
                    className="flex items-center gap-1 mx-auto text-sm text-primary-600 hover:text-primary-700"
                  >
                    <Lightbulb className="w-4 h-4" />
                    Show hint
                  </button>
                )}
              </div>
            )}

            <p className="text-center text-sm text-slate-400 dark:text-slate-500 mt-4">
              Click to reveal answer
            </p>
          </div>

          {/* Back - Answer */}
          <div
            className={cn(
              "absolute inset-0 p-6 backface-hidden rotate-y-180",
              !isFlipped && "invisible"
            )}
          >
            <div className="flex items-center justify-between mb-4">
              <span className="text-xs font-medium px-2 py-1 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 rounded-full">
                Answer
              </span>
            </div>

            <div className="flex-1 flex items-center justify-center min-h-[150px]">
              <p className="text-lg text-slate-900 dark:text-white text-center">
                {currentCard.answer}
              </p>
            </div>

            {currentCard.key_concepts && currentCard.key_concepts.length > 0 && (
              <div className="mt-4 flex flex-wrap gap-2 justify-center">
                {currentCard.key_concepts.map((concept, i) => (
                  <span
                    key={i}
                    className="text-xs px-2 py-1 bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-400 rounded"
                  >
                    {concept}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Rating buttons (only shown when flipped) */}
      {isFlipped && (
        <div className="space-y-3">
          <p className="text-center text-sm text-slate-600 dark:text-slate-400">
            How well did you know this?
          </p>
          <div className="flex justify-center gap-2">
            {QUALITY_BUTTONS.map((btn) => (
              <button
                key={btn.quality}
                onClick={() => handleRate(btn.quality)}
                className={cn(
                  "flex flex-col items-center gap-1 px-4 py-2 rounded-lg text-white transition-colors",
                  btn.color
                )}
              >
                <btn.icon className="w-4 h-4" />
                <span className="text-xs">{btn.label}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Navigation */}
      <div className="flex justify-between">
        <button
          onClick={handlePrevious}
          disabled={currentIndex === 0}
          className="flex items-center gap-1 text-sm text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-300 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <ChevronLeft className="w-4 h-4" />
          Previous
        </button>

        <button
          onClick={handleRestart}
          className="flex items-center gap-1 text-sm text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-300"
        >
          <RotateCcw className="w-4 h-4" />
          Restart
        </button>
      </div>
    </div>
  );
}
