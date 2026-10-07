"use client";

import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import { api } from "@/services/api";
import { useGraphStore } from "@/store";
import { cn } from "@/utils/cn";
import type { Document, DocumentStatus } from "@/types";
import { KnowledgeGraph } from "./KnowledgeGraph";

// A document's graph is built by a background task after upload (and on
// demand). While the build runs the document is "processing" and its nodes are
// readable as they are written, relations last; it becomes "indexed" once the
// build is done, and goes back to "validated" if it fails. So the graph is only
// loaded once the document is indexed, and the document is polled until then.
const POLL_INTERVAL_MS = 3000;
// A scheduled build normally starts within seconds. When a "validated" document
// shows no sign of one for this long, the normal panel (with its Build button)
// is shown instead.
const BUILD_START_TIMEOUT_MS = 15_000;
// Stop polling when the status has not changed for this long
const MAX_POLL_MS = 10 * 60_000;

const NOT_STARTED: DocumentStatus[] = ["pending", "validated"];

interface DocumentGraphPanelProps {
  docId: string;
  // The document's last known status
  status: DocumentStatus;
  // Receives the document whenever polling finds a new status
  onDocumentUpdate: (document: Document) => void;
  className?: string;
}

// The knowledge graph of one document, or a "Building knowledge graph…" state
// while its build is pending or running
export function DocumentGraphPanel(props: DocumentGraphPanelProps) {
  // Fresh build-watch state for every document
  return <DocumentGraphPanelInner key={props.docId} {...props} />;
}

function DocumentGraphPanelInner({
  docId,
  status,
  onDocumentUpdate,
  className,
}: DocumentGraphPanelProps) {
  const serverBuild = useGraphStore((s) => s.serverBuildDocIds.includes(docId));
  const endServerBuild = useGraphStore((s) => s.endServerBuild);
  const clearGraph = useGraphStore((s) => s.clearGraph);
  // A build was seen running, so a later "validated" means it ended without a graph
  const [sawProcessing, setSawProcessing] = useState(status === "processing");
  const [startTimedOut, setStartTimedOut] = useState(false);
  const [pollExpired, setPollExpired] = useState(false);

  // The server reported a build in progress (409): treat the document as
  // processing until polling has fetched its current status
  const liveStatus: DocumentStatus = serverBuild ? "processing" : status;
  const building =
    liveStatus === "processing" ||
    (NOT_STARTED.includes(liveStatus) && !sawProcessing && !startTimedOut);

  useEffect(() => {
    if (status === "processing") setSawProcessing(true);
  }, [status]);

  useEffect(() => {
    const timer = setTimeout(() => setStartTimedOut(true), BUILD_START_TIMEOUT_MS);
    return () => clearTimeout(timer);
  }, []);

  // Whatever graph the store holds is stale (or partial) while a build runs;
  // the graph is loaded afresh when KnowledgeGraph mounts after the build
  useEffect(() => {
    if (building) clearGraph();
  }, [building, clearGraph]);

  useEffect(() => {
    if (!building) return;

    let cancelled = false;
    const startedAt = Date.now();
    setPollExpired(false);
    const timer = setInterval(async () => {
      if (Date.now() - startedAt > MAX_POLL_MS) {
        clearInterval(timer);
        setPollExpired(true);
        return;
      }
      try {
        const doc = await api.getDocument(docId);
        if (cancelled) return;
        // A new status re-renders this panel: "indexed" mounts KnowledgeGraph,
        // which loads the complete graph
        if (doc.status !== status) onDocumentUpdate(doc);
        endServerBuild(docId);
      } catch {
        // Transient failure: try again on the next tick
      }
    }, POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [building, docId, status, onDocumentUpdate, endServerBuild]);

  if (building) return <GraphBuilding className={className} slow={pollExpired} />;
  return <KnowledgeGraph documentId={docId} className={className} />;
}

function GraphBuilding({ className, slow }: { className?: string; slow: boolean }) {
  return (
    <div
      className={cn(
        "flex items-center justify-center h-96 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700",
        className
      )}
    >
      <div role="status" className="flex flex-col items-center gap-3 text-center px-6 max-w-sm">
        <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
        <span className="font-medium text-slate-700 dark:text-slate-200">
          Building knowledge graph…
        </span>
        <span className="text-sm text-slate-500 dark:text-slate-400">
          {slow
            ? "This is taking longer than usual. Reload the page to check again."
            : "Concepts and their connections are being extracted. The graph appears here when it is ready."}
        </span>
      </div>
    </div>
  );
}
