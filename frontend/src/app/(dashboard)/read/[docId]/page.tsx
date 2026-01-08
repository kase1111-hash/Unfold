"use client";

import { useParams } from "next/navigation";
import { DocumentViewer, ComplexitySlider, ViewModeToggle } from "@/components/reader";
import { KnowledgeGraph, NodeDetails } from "@/components/graph";
import { useGraphStore } from "@/store";

export default function ReadPage() {
  const params = useParams();
  const docId = params.docId as string;
  const { selectedNodeId } = useGraphStore();

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
          Reading View
        </h1>
        <p className="text-slate-600 dark:text-slate-400 mt-1">
          Adjust complexity and explore connected concepts
        </p>
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
