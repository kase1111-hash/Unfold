"use client";

import { useState } from "react";
import { Brain, Plus, Download, Play } from "lucide-react";
import { FlashcardReview, StudyStats } from "@/components/learning";
import { Button } from "@/components/ui";

// Mock flashcards for demo
const DEMO_FLASHCARDS = [
  {
    card_id: "fc_1",
    question: "What is the primary function of a knowledge graph in document comprehension?",
    answer: "A knowledge graph connects concepts, entities, and relationships from documents, enabling users to visualize how ideas relate to each other and explore contextual connections that aid understanding.",
    hint: "Think about how concepts connect to each other",
    type: "concept",
    difficulty: "intermediate",
    key_concepts: ["knowledge graph", "concepts", "relationships"],
  },
  {
    card_id: "fc_2",
    question: "How does spaced repetition improve long-term memory retention?",
    answer: "Spaced repetition schedules reviews at increasing intervals based on how well you remember each item. Items you struggle with are shown more frequently, while well-remembered items are shown less often, optimizing study time.",
    hint: "Consider the timing of reviews",
    type: "concept",
    difficulty: "intermediate",
    key_concepts: ["spaced repetition", "memory", "intervals"],
  },
  {
    card_id: "fc_3",
    question: "What is the SuperMemo 2 (SM2) algorithm?",
    answer: "SM2 is a spaced repetition algorithm that calculates optimal review intervals using an 'easiness factor' that adjusts based on recall quality (rated 0-5). It's the foundation for many modern flashcard applications.",
    type: "definition",
    difficulty: "advanced",
    key_concepts: ["SM2", "algorithm", "easiness factor"],
  },
  {
    card_id: "fc_4",
    question: "What is the difference between TF-IDF and semantic similarity for text relevance?",
    answer: "TF-IDF measures relevance based on term frequency and document frequency (keyword matching), while semantic similarity uses embeddings to capture meaning, allowing it to identify relevant content even without exact keyword matches.",
    hint: "One uses keywords, one uses meaning",
    type: "comparison",
    difficulty: "advanced",
    key_concepts: ["TF-IDF", "semantic similarity", "embeddings"],
  },
  {
    card_id: "fc_5",
    question: "What does 'ELI5' mean in the context of text complexity?",
    answer: "'Explain Like I'm 5' - a request to simplify complex information to a level that a young child could understand, using simple words, analogies, and avoiding technical jargon.",
    type: "definition",
    difficulty: "beginner",
    key_concepts: ["ELI5", "simplification", "complexity"],
  },
];

export default function FlashcardsPage() {
  const [isReviewing, setIsReviewing] = useState(false);
  const [flashcards] = useState(DEMO_FLASHCARDS);

  const handleReviewComplete = (results: { card_id: string; quality: number; time_ms: number }[]) => {
    console.log("Review results:", results);
    setIsReviewing(false);
  };

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
                Ready to Review?
              </h2>

              <p className="text-slate-600 dark:text-slate-400 mb-6">
                You have {flashcards.length} cards ready for review. Regular reviews
                help strengthen your memory.
              </p>

              <Button
                size="lg"
                onClick={() => setIsReviewing(true)}
                leftIcon={<Play className="w-5 h-5" />}
              >
                Start Review Session
              </Button>
            </div>

            {/* Card Preview */}
            <div className="card">
              <div className="p-4 border-b border-slate-200 dark:border-slate-700">
                <h3 className="font-semibold text-slate-900 dark:text-white">
                  Your Flashcards
                </h3>
              </div>

              <div className="divide-y divide-slate-200 dark:divide-slate-700">
                {flashcards.slice(0, 5).map((card) => (
                  <div key={card.card_id} className="p-4">
                    <p className="font-medium text-slate-900 dark:text-white text-sm mb-1">
                      {card.question}
                    </p>
                    <div className="flex items-center gap-2 mt-2">
                      <span className="text-xs px-2 py-0.5 bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded">
                        {card.type}
                      </span>
                      <span className="text-xs px-2 py-0.5 bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-400 rounded">
                        {card.difficulty}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
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
