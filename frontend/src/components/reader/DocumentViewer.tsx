"use client";

import { useEffect } from "react";
import { useReadingStore } from "@/store";
import { cn } from "@/utils/cn";
import { Loader2, AlertCircle, FileText } from "lucide-react";

interface DocumentViewerProps {
  documentId: string;
}

export function DocumentViewer({ documentId }: DocumentViewerProps) {
  const {
    document,
    isLoading,
    error,
    viewMode,
    paraphrasedContent,
    isParaphrasing,
    loadDocument,
  } = useReadingStore();

  useEffect(() => {
    loadDocument(documentId);
  }, [documentId, loadDocument]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
          <span className="text-slate-500 dark:text-slate-400">
            Loading document...
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
          <span className="text-red-500 font-medium">Error loading document</span>
          <span className="text-slate-500 dark:text-slate-400 text-sm">
            {error}
          </span>
        </div>
      </div>
    );
  }

  if (!document) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="flex flex-col items-center gap-3 text-center">
          <FileText className="w-8 h-8 text-slate-400" />
          <span className="text-slate-500 dark:text-slate-400">
            No document selected
          </span>
        </div>
      </div>
    );
  }

  // Mock content for demo - in production this would come from the document
  const technicalContent = document.abstract || "Document content would appear here...";
  const simplifiedContent = paraphrasedContent || technicalContent;

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 overflow-hidden">
      {/* Document header */}
      <div className="border-b border-slate-200 dark:border-slate-700 p-4">
        <h1 className="text-xl font-bold text-slate-900 dark:text-white mb-2">
          {document.title}
        </h1>
        {document.authors.length > 0 && (
          <p className="text-sm text-slate-600 dark:text-slate-400">
            {document.authors.join(", ")}
          </p>
        )}
        {document.doi && (
          <a
            href={`https://doi.org/${document.doi}`}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-primary-600 hover:text-primary-700 mt-1 inline-block"
          >
            DOI: {document.doi}
          </a>
        )}
      </div>

      {/* Document content */}
      <div className="p-6">
        {viewMode === "hybrid" ? (
          <div className="grid grid-cols-2 gap-6">
            {/* Technical view */}
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-sm font-medium text-slate-600 dark:text-slate-400 pb-2 border-b border-slate-200 dark:border-slate-700">
                <FileText className="w-4 h-4" />
                Original Text
              </div>
              <div className="prose prose-slate dark:prose-invert prose-sm max-w-none">
                <p className="text-slate-700 dark:text-slate-300 leading-relaxed">
                  {technicalContent}
                </p>
              </div>
            </div>

            {/* Conceptual view */}
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-sm font-medium text-primary-600 dark:text-primary-400 pb-2 border-b border-primary-200 dark:border-primary-900">
                <span className="relative">
                  Simplified
                  {isParaphrasing && (
                    <Loader2 className="w-3 h-3 animate-spin absolute -right-5 top-0.5" />
                  )}
                </span>
              </div>
              <div className="prose prose-slate dark:prose-invert prose-sm max-w-none">
                {isParaphrasing ? (
                  <div className="flex items-center gap-2 text-slate-500">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Generating simplified version...</span>
                  </div>
                ) : (
                  <p className="text-slate-700 dark:text-slate-300 leading-relaxed bg-primary-50 dark:bg-primary-900/20 p-4 rounded-lg">
                    {simplifiedContent}
                  </p>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="prose prose-slate dark:prose-invert max-w-none">
            {viewMode === "technical" ? (
              <p className="text-slate-700 dark:text-slate-300 leading-relaxed">
                {technicalContent}
              </p>
            ) : (
              <>
                {isParaphrasing ? (
                  <div className="flex items-center gap-2 text-slate-500">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Generating simplified version...</span>
                  </div>
                ) : (
                  <p className="text-slate-700 dark:text-slate-300 leading-relaxed bg-primary-50 dark:bg-primary-900/20 p-4 rounded-lg">
                    {simplifiedContent}
                  </p>
                )}
              </>
            )}
          </div>
        )}
      </div>

      {/* Document footer */}
      <div className="border-t border-slate-200 dark:border-slate-700 p-4 bg-slate-50 dark:bg-slate-900/50">
        <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
          <div className="flex items-center gap-4">
            {document.word_count && (
              <span>{document.word_count.toLocaleString()} words</span>
            )}
            {document.page_count && <span>{document.page_count} pages</span>}
          </div>
          <span className="capitalize">{document.status}</span>
        </div>
      </div>
    </div>
  );
}
