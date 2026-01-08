"use client";

import Link from "next/link";
import { BookOpen, Brain, Sparkles, GraduationCap } from "lucide-react";

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      {/* Header */}
      <header className="border-b border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-2">
              <BookOpen className="h-8 w-8 text-primary-600" />
              <span className="text-xl font-bold text-slate-900 dark:text-white">
                Unfold
              </span>
            </div>
            <nav className="flex items-center space-x-4">
              <Link
                href="/login"
                className="text-slate-600 hover:text-slate-900 dark:text-slate-300 dark:hover:text-white transition-colors"
              >
                Sign In
              </Link>
              <Link href="/register" className="btn-primary">
                Get Started
              </Link>
            </nav>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="text-center">
          <h1 className="text-5xl font-bold text-slate-900 dark:text-white mb-6">
            Read Smarter, Not Harder
          </h1>
          <p className="text-xl text-slate-600 dark:text-slate-300 max-w-3xl mx-auto mb-10">
            Unfold transforms dense academic and technical texts into
            comprehensible content with AI-powered paraphrasing, knowledge
            graphs, and adaptive learning.
          </p>
          <div className="flex justify-center space-x-4">
            <Link href="/register" className="btn-primary text-lg px-8 py-3">
              Start Reading for Free
            </Link>
            <Link
              href="/demo"
              className="btn-secondary text-lg px-8 py-3"
            >
              Try Demo
            </Link>
          </div>
        </div>

        {/* Features */}
        <div className="mt-24 grid md:grid-cols-2 lg:grid-cols-4 gap-8">
          <FeatureCard
            icon={<Sparkles className="h-8 w-8" />}
            title="AI Paraphrasing"
            description="Adjust complexity from 'Explain Like I'm 5' to expert-level with our sliding scale."
          />
          <FeatureCard
            icon={<Brain className="h-8 w-8" />}
            title="Knowledge Graphs"
            description="Visualize connections between concepts and explore ideas in context."
          />
          <FeatureCard
            icon={<GraduationCap className="h-8 w-8" />}
            title="Adaptive Learning"
            description="Spaced repetition flashcards that adapt to your comprehension level."
          />
          <FeatureCard
            icon={<BookOpen className="h-8 w-8" />}
            title="Dual-View Reading"
            description="Toggle between technical text and simplified explanations seamlessly."
          />
        </div>

        {/* Demo Section */}
        <div className="mt-24 card p-8">
          <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-4">
            See It In Action
          </h2>
          <div className="aspect-video bg-slate-100 dark:bg-slate-700 rounded-lg flex items-center justify-center">
            <p className="text-slate-500 dark:text-slate-400">
              Interactive demo coming soon
            </p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 dark:border-slate-700 mt-20 py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-2">
              <BookOpen className="h-6 w-6 text-primary-600" />
              <span className="font-semibold text-slate-900 dark:text-white">
                Unfold
              </span>
            </div>
            <p className="text-slate-500 dark:text-slate-400 text-sm">
              &copy; 2026 Unfold. Open Source under AGPL v3.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

function FeatureCard({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="card p-6 text-center hover:shadow-lg transition-shadow">
      <div className="inline-flex items-center justify-center w-14 h-14 rounded-xl bg-primary-100 dark:bg-primary-900/30 text-primary-600 mb-4">
        {icon}
      </div>
      <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">
        {title}
      </h3>
      <p className="text-slate-600 dark:text-slate-300 text-sm">
        {description}
      </p>
    </div>
  );
}
