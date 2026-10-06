"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { GraduationCap } from "lucide-react";
import toast from "react-hot-toast";
import { DocumentViewer, ComplexitySlider, ViewModeToggle } from "@/components/reader";
import { KnowledgeGraph, NodeDetails } from "@/components/graph";
import { Button } from "@/components/ui";
import { api, getErrorMessage } from "@/services/api";
import { useGraphStore, useReadingStore } from "@/store";

// After upload the graph is built by a background task; the document becomes
// "indexed" when it is done. Poll for that (bounded: a failed build never gets there).
const GRAPH_POLL_INTERVAL_MS = 3000;
const GRAPH_POLL_ATTEMPTS = 20;
const NOT_YET_INDEXED = ["pending", "processing", "validated"];

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
  const loadGraphForDocument = useGraphStore((s) => s.loadGraphForDocument);
  const updateDocument = useReadingStore((s) => s.updateDocument);
  const docStatus = useReadingStore((s) =>
    s.document?.doc_id === docId ? s.document.status : undefined
  );
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedCount, setGeneratedCount] = useState<number | null>(null);

  useEffect(() => {
    setGeneratedCount(null);
  }, [docId]);

  useEffect(() => {
    if (!docStatus || !NOT_YET_INDEXED.includes(docStatus)) return;

    let cancelled = false;
    let attempts = 0;
    const timer = setInterval(async () => {
      attempts += 1;
      if (attempts >= GRAPH_POLL_ATTEMPTS) clearInterval(timer);
      try {
        const doc = await api.getDocument(docId);
        if (cancelled || doc.status === docStatus) return;
        updateDocument(doc); // changes docStatus, which stops this poll
        if (doc.status === "indexed" && useGraphStore.getState().nodes.length === 0) {
          loadGraphForDocument(docId);
        }
      } catch {
        // Transient failure: try again on the next tick
      }
    }, GRAPH_POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [docId, docStatus, updateDocument, loadGraphForDocument]);

  const handleGenerateFlashcards = async () => {
    setIsGenerating(true);
    try {
      const result = await api.generateFlashcards(docId);
      setGeneratedCount(result.count);
      toast.success(`Created ${result.count} flashcard${result.count === 1 ? "" : "s"}`);
    } catch (error) {
      toast.error(getErrorMessage(error));
    } finally {
      setIsGenerating(false);
    }
  };

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
        <div className="flex flex-col items-end gap-2">
          <Button
            onClick={handleGenerateFlashcards}
            isLoading={isGenerating}
            leftIcon={<GraduationCap className="w-4 h-4" />}
          >
            Generate flashcards
          </Button>
          {generatedCount !== null && (
            <Link
              href="/flashcards"
              className="text-sm text-primary-600 hover:text-primary-700"
            >
              {generatedCount} flashcard{generatedCount === 1 ? "" : "s"} created. Review now
            </Link>
          )}
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Main content area */}
        <div className="lg:col-span-2 space-y-6">
          {/* Document viewer */}
          <DocumentViewer documentId={docId} />

          {/* Knowledge graph */}
          <div>
            <h2 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
              Knowledge Graph
            </h2>
            <KnowledgeGraph documentId={docId} className="h-[500px]" />
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          {/* Complexity slider */}
          <ComplexitySlider />

          {/* View mode toggle */}
          <ViewModeToggle />

          {/* Node details */}
          {selectedNodeId && <NodeDetails />}
        </div>
      </div>
    </div>
  );
}
