"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { AlertCircle, GraduationCap, Loader2 } from "lucide-react";
import toast from "react-hot-toast";
import { DocumentViewer, ComplexitySlider, ViewModeToggle } from "@/components/reader";
import { DocumentGraphPanel, NodeDetails } from "@/components/graph";
import { Button } from "@/components/ui";
import { api, getErrorMessage } from "@/services/api";
import { useGraphStore, useReadingStore } from "@/store";

// App Router passes dynamic segments still URL-encoded (doc ids may contain ':')
function decodeParam(value: string): string {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

export default function ReadPage() {
  const params = useParams();
  const docId = decodeParam(params.docId as string);
  const selectedNodeId = useGraphStore((s) => s.selectedNodeId);
  const loadDocument = useReadingStore((s) => s.loadDocument);
  const updateDocument = useReadingStore((s) => s.updateDocument);
  // Only state of this page's document counts: until loadDocument runs, the
  // store still holds the previously opened one
  const isCurrent = useReadingStore((s) => s.requestedDocId === docId);
  const docStatus = useReadingStore((s) =>
    s.document?.doc_id === docId ? s.document.status : undefined
  );
  const loadError = useReadingStore((s) => (s.requestedDocId === docId ? s.error : null));
  const notFound = useReadingStore((s) => s.requestedDocId === docId && s.notFound);
  const [isGenerating, setIsGenerating] = useState(false);
  // Cards stored by the last "Generate flashcards" (0: they all existed already)
  const [generatedCount, setGeneratedCount] = useState<number | null>(null);

  useEffect(() => {
    loadDocument(docId);
  }, [docId, loadDocument]);

  useEffect(() => {
    setGeneratedCount(null);
  }, [docId]);

  const handleGenerateFlashcards = async () => {
    setIsGenerating(true);
    try {
      const result = await api.generateFlashcards(docId);
      setGeneratedCount(result.count);
      if (result.count > 0) {
        toast.success(
          `Created ${result.count} new flashcard${result.count === 1 ? "" : "s"}`
        );
      }
    } catch (error) {
      toast.error(getErrorMessage(error));
    } finally {
      setIsGenerating(false);
    }
  };

  // A missing (or someone else's) document: one message, nothing to act on
  if (loadError) {
    return (
      <div className="flex items-center justify-center py-24">
        <div
          role="alert"
          className="card p-8 max-w-md w-full flex flex-col items-center gap-3 text-center"
        >
          <AlertCircle className="w-8 h-8 text-red-500" />
          <h1 className="text-lg font-semibold text-slate-900 dark:text-white">
            {notFound ? "Document not found" : "Could not load this document"}
          </h1>
          <p className="text-sm text-slate-600 dark:text-slate-400 break-words">
            {notFound
              ? "It may have been deleted, or it belongs to another account."
              : loadError}
          </p>
          <div className="flex items-center gap-4 mt-2">
            {!notFound && (
              <Button variant="secondary" onClick={() => loadDocument(docId)}>
                Retry
              </Button>
            )}
            <Link
              href="/documents"
              className="text-sm font-medium text-primary-600 hover:text-primary-700"
            >
              Back to Documents
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
            Reading View
          </h1>
          <p className="text-slate-600 dark:text-slate-400 mt-1">
            Adjust complexity and explore connected concepts
          </p>
        </div>
        {docStatus && (
          <div className="flex flex-col items-end gap-2">
            {generatedCount === null ? (
              <Button
                onClick={handleGenerateFlashcards}
                isLoading={isGenerating}
                leftIcon={<GraduationCap className="w-4 h-4" />}
              >
                Generate flashcards
              </Button>
            ) : (
              <>
                <Link
                  href="/flashcards"
                  className="btn-primary inline-flex items-center gap-2"
                >
                  <GraduationCap className="w-4 h-4" />
                  Review flashcards
                </Link>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                  {generatedCount > 0
                    ? `${generatedCount} new card${generatedCount === 1 ? "" : "s"} created.`
                    : "Flashcards for this document already exist."}
                </p>
              </>
            )}
          </div>
        )}
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Main content area */}
        <div className="lg:col-span-2 space-y-6">
          {/* Document viewer */}
          {isCurrent && <DocumentViewer />}

          {/* Knowledge graph: waits for the document, whose status says
              whether the graph is complete or still being built */}
          <div>
            <h2 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
              Knowledge Graph
            </h2>
            {docStatus ? (
              <DocumentGraphPanel
                docId={docId}
                status={docStatus}
                onDocumentUpdate={updateDocument}
                className="h-[500px]"
              />
            ) : (
              <div className="h-[500px] flex items-center justify-center bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
                <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
              </div>
            )}
          </div>
        </div>

        {/* Sidebar */}
        {docStatus && (
          <div className="space-y-4">
            {/* Complexity slider */}
            <ComplexitySlider />

            {/* View mode toggle */}
            <ViewModeToggle />

            {/* Node details */}
            {selectedNodeId && <NodeDetails />}
          </div>
        )}
      </div>
    </div>
  );
}
