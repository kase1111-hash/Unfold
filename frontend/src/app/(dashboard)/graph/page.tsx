"use client";

import { useEffect, useState } from "react";
import { KnowledgeGraph, NodeDetails } from "@/components/graph";
import { useGraphStore } from "@/store";
import { api, getErrorMessage } from "@/services/api";
import { Loader2 } from "lucide-react";
import toast from "react-hot-toast";
import type { Document } from "@/types";

export default function GraphPage() {
  const selectedNodeId = useGraphStore((s) => s.selectedNodeId);
  const nodeCount = useGraphStore((s) => s.nodes.length);
  const linkCount = useGraphStore((s) => s.links.length);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [documentsLoaded, setDocumentsLoaded] = useState(false);
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);

  useEffect(() => {
    api
      .getDocuments(1, 100)
      .then((res) => {
        setDocuments(res.data);
        if (res.data.length > 0) {
          setSelectedDocId(res.data[0].doc_id);
        }
      })
      .catch((error) => toast.error(getErrorMessage(error)))
      .finally(() => setDocumentsLoaded(true));
  }, []);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
            Knowledge Graph
          </h1>
          <p className="text-slate-600 dark:text-slate-400 mt-1">
            Explore concept connections across your documents
          </p>
        </div>
      </div>

      {/* Controls */}
      <div className="card p-4">
        <div className="flex flex-wrap gap-4">
          {/* Document selector */}
          <div className="flex-1 min-w-[200px]">
            <label
              htmlFor="graph-document"
              className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1.5"
            >
              Document
            </label>
            <select
              id="graph-document"
              value={selectedDocId || ""}
              onChange={(e) => setSelectedDocId(e.target.value || null)}
              className="w-full px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-primary-500/50"
            >
              <option value="">All documents</option>
              {documents.map((doc) => (
                <option key={doc.doc_id} value={doc.doc_id}>
                  {doc.title}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Graph and details */}
      <div className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          {/* Wait for the document list so the first document is loaded directly
              (instead of loading "All documents" first and then replacing it) */}
          {documentsLoaded ? (
            <KnowledgeGraph
              documentId={selectedDocId || undefined}
              className="h-[600px]"
            />
          ) : (
            <div className="h-[600px] flex items-center justify-center bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
              <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
            </div>
          )}
        </div>

        <div className="space-y-4">
          {selectedNodeId ? (
            <NodeDetails />
          ) : (
            <div className="card p-6 text-center">
              <p className="text-slate-500 dark:text-slate-400">
                Click on a node to see details
              </p>
              <p className="text-sm text-slate-400 dark:text-slate-500 mt-2">
                Double-click to expand related nodes
              </p>
            </div>
          )}

          {/* Graph stats */}
          <div className="card p-4">
            <h3 className="font-medium text-slate-900 dark:text-white mb-3">
              Graph Statistics
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-slate-600 dark:text-slate-400">Total Nodes</span>
                <span className="font-medium text-slate-900 dark:text-white">
                  {nodeCount}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-600 dark:text-slate-400">Total Links</span>
                <span className="font-medium text-slate-900 dark:text-white">
                  {linkCount}
                </span>
              </div>
            </div>
          </div>

          {/* Instructions */}
          <div className="card p-4 bg-primary-50 dark:bg-primary-900/20 border-primary-200 dark:border-primary-800">
            <h3 className="font-medium text-primary-900 dark:text-primary-100 mb-2">
              Navigation Tips
            </h3>
            <ul className="text-sm text-primary-700 dark:text-primary-300 space-y-1">
              <li>• Drag nodes to reposition</li>
              <li>• Scroll to zoom in/out</li>
              <li>• Click node to select</li>
              <li>• Double-click to expand</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
