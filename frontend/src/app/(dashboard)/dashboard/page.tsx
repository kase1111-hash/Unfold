"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { FileText, Brain, TrendingUp, Clock, Plus } from "lucide-react";
import { api } from "@/services/api";
import { useAuthStore } from "@/store";
import type { Document } from "@/types";

export default function DashboardPage() {
  const { user } = useAuthStore();
  const [recentDocs, setRecentDocs] = useState<Document[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    api
      .getDocuments(1, 5)
      .then((res) => setRecentDocs(res.data))
      .catch(() => setRecentDocs([]))
      .finally(() => setIsLoading(false));
  }, []);

  const stats = [
    {
      label: "Documents",
      value: recentDocs.length,
      icon: FileText,
      color: "bg-blue-100 dark:bg-blue-900/30 text-blue-600",
    },
    {
      label: "Concepts Learned",
      value: 0,
      icon: Brain,
      color: "bg-purple-100 dark:bg-purple-900/30 text-purple-600",
    },
    {
      label: "Reading Streak",
      value: "0 days",
      icon: TrendingUp,
      color: "bg-green-100 dark:bg-green-900/30 text-green-600",
    },
    {
      label: "Time Reading",
      value: "0 hrs",
      icon: Clock,
      color: "bg-orange-100 dark:bg-orange-900/30 text-orange-600",
    },
  ];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
          Welcome back, {user?.full_name || user?.username || "Reader"}!
        </h1>
        <p className="text-slate-600 dark:text-slate-400 mt-1">
          Continue your learning journey
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((stat) => (
          <div
            key={stat.label}
            className="card p-5 flex items-center gap-4"
          >
            <div
              className={`w-12 h-12 rounded-xl flex items-center justify-center ${stat.color}`}
            >
              <stat.icon className="w-6 h-6" />
            </div>
            <div>
              <p className="text-2xl font-bold text-slate-900 dark:text-white">
                {stat.value}
              </p>
              <p className="text-sm text-slate-500 dark:text-slate-400">
                {stat.label}
              </p>
            </div>
          </div>
        ))}
      </div>

      {/* Recent Documents */}
      <div className="card">
        <div className="p-5 border-b border-slate-200 dark:border-slate-700 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-slate-900 dark:text-white">
            Recent Documents
          </h2>
          <Link
            href="/upload"
            className="btn-primary text-sm flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Upload New
          </Link>
        </div>

        {isLoading ? (
          <div className="p-8 text-center text-slate-500 dark:text-slate-400">
            Loading...
          </div>
        ) : recentDocs.length === 0 ? (
          <div className="p-8 text-center">
            <FileText className="w-12 h-12 mx-auto text-slate-300 dark:text-slate-600 mb-3" />
            <p className="text-slate-600 dark:text-slate-400 mb-4">
              No documents yet. Upload your first paper to get started!
            </p>
            <Link href="/upload" className="btn-primary">
              Upload Document
            </Link>
          </div>
        ) : (
          <div className="divide-y divide-slate-200 dark:divide-slate-700">
            {recentDocs.map((doc) => (
              <Link
                key={doc.doc_id}
                href={`/read/${doc.doc_id}`}
                className="block p-4 hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors"
              >
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="font-medium text-slate-900 dark:text-white">
                      {doc.title}
                    </h3>
                    {doc.authors.length > 0 && (
                      <p className="text-sm text-slate-500 dark:text-slate-400 mt-0.5">
                        {doc.authors.slice(0, 3).join(", ")}
                        {doc.authors.length > 3 && " et al."}
                      </p>
                    )}
                  </div>
                  <span
                    className={`text-xs px-2 py-1 rounded-full ${
                      doc.status === "indexed"
                        ? "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300"
                        : doc.status === "processing"
                        ? "bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300"
                        : "bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-400"
                    }`}
                  >
                    {doc.status}
                  </span>
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>

      {/* Quick Actions */}
      <div className="grid md:grid-cols-2 gap-4">
        <Link
          href="/upload"
          className="card p-6 hover:shadow-lg transition-shadow group"
        >
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center group-hover:scale-110 transition-transform">
              <Plus className="w-6 h-6 text-primary-600" />
            </div>
            <div>
              <h3 className="font-semibold text-slate-900 dark:text-white">
                Upload Document
              </h3>
              <p className="text-sm text-slate-500 dark:text-slate-400">
                Add a new paper or article to read
              </p>
            </div>
          </div>
        </Link>

        <Link
          href="/graph"
          className="card p-6 hover:shadow-lg transition-shadow group"
        >
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center group-hover:scale-110 transition-transform">
              <Brain className="w-6 h-6 text-purple-600" />
            </div>
            <div>
              <h3 className="font-semibold text-slate-900 dark:text-white">
                Explore Knowledge Graph
              </h3>
              <p className="text-sm text-slate-500 dark:text-slate-400">
                Visualize concept connections
              </p>
            </div>
          </div>
        </Link>
      </div>
    </div>
  );
}
