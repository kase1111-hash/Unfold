"use client";

import { useEffect, useState } from "react";
import { useGraphStore } from "@/store";
import { api } from "@/services/api";
import { cn } from "@/utils/cn";
import {
  X,
  ExternalLink,
  Loader2,
  Book,
  Globe,
  ChevronRight,
} from "lucide-react";
import type { GraphNode } from "@/types";

interface WikipediaLink {
  entity: string;
  title: string | null;
  url: string | null;
  extract: string | null;
  found: boolean;
}

export function NodeDetails() {
  const { selectedNodeId, nodes, selectNode, loadRelatedNodes } = useGraphStore();
  const [wikipediaData, setWikipediaData] = useState<WikipediaLink | null>(null);
  const [isLoadingWikipedia, setIsLoadingWikipedia] = useState(false);

  const selectedNode = nodes.find((n) => n.node_id === selectedNodeId);

  useEffect(() => {
    if (selectedNode?.label) {
      setIsLoadingWikipedia(true);
      api
        .linkToWikipedia(selectedNode.label)
        .then(setWikipediaData)
        .catch(() => setWikipediaData(null))
        .finally(() => setIsLoadingWikipedia(false));
    } else {
      setWikipediaData(null);
    }
  }, [selectedNode?.label]);

  if (!selectedNode) {
    return null;
  }

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 overflow-hidden">
      {/* Header */}
      <div className="flex items-start justify-between p-4 border-b border-slate-200 dark:border-slate-700">
        <div>
          <h3 className="font-semibold text-slate-900 dark:text-white">
            {selectedNode.label}
          </h3>
          <span className="text-xs text-slate-500 dark:text-slate-400 mt-0.5 inline-block">
            {selectedNode.type}
          </span>
        </div>
        <button
          onClick={() => selectNode(null)}
          className="p-1 rounded hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
        >
          <X className="w-4 h-4 text-slate-500" />
        </button>
      </div>

      {/* Content */}
      <div className="p-4 space-y-4">
        {/* Description */}
        {selectedNode.description && (
          <div>
            <h4 className="text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-1">
              Description
            </h4>
            <p className="text-sm text-slate-700 dark:text-slate-300">
              {selectedNode.description}
            </p>
          </div>
        )}

        {/* Confidence */}
        <div>
          <h4 className="text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-1">
            Confidence
          </h4>
          <div className="flex items-center gap-2">
            <div className="flex-1 h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-primary-500 rounded-full"
                style={{ width: `${selectedNode.confidence * 100}%` }}
              />
            </div>
            <span className="text-sm text-slate-600 dark:text-slate-400">
              {Math.round(selectedNode.confidence * 100)}%
            </span>
          </div>
        </div>

        {/* External links */}
        {Object.keys(selectedNode.external_links).length > 0 && (
          <div>
            <h4 className="text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-2">
              External Links
            </h4>
            <div className="space-y-1">
              {Object.entries(selectedNode.external_links).map(([name, url]) => (
                <a
                  key={name}
                  href={url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 text-sm text-primary-600 hover:text-primary-700 py-1"
                >
                  <ExternalLink className="w-3 h-3" />
                  {name}
                </a>
              ))}
            </div>
          </div>
        )}

        {/* Wikipedia */}
        <div>
          <h4 className="text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-2 flex items-center gap-1">
            <Globe className="w-3 h-3" />
            Wikipedia
          </h4>
          {isLoadingWikipedia ? (
            <div className="flex items-center gap-2 text-sm text-slate-500">
              <Loader2 className="w-3 h-3 animate-spin" />
              Loading...
            </div>
          ) : wikipediaData?.found ? (
            <div className="bg-slate-50 dark:bg-slate-700/50 rounded-lg p-3">
              <a
                href={wikipediaData.url || "#"}
                target="_blank"
                rel="noopener noreferrer"
                className="font-medium text-sm text-primary-600 hover:text-primary-700"
              >
                {wikipediaData.title}
              </a>
              {wikipediaData.extract && (
                <p className="text-xs text-slate-600 dark:text-slate-400 mt-1 line-clamp-3">
                  {wikipediaData.extract}
                </p>
              )}
            </div>
          ) : (
            <p className="text-sm text-slate-500">No Wikipedia article found</p>
          )}
        </div>

        {/* Actions */}
        <div className="pt-2">
          <button
            onClick={() => loadRelatedNodes(selectedNode.node_id)}
            className="w-full flex items-center justify-center gap-2 py-2 px-4 bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded-lg text-sm font-medium hover:bg-primary-100 dark:hover:bg-primary-900/50 transition-colors"
          >
            <ChevronRight className="w-4 h-4" />
            Expand Related Nodes
          </button>
        </div>
      </div>
    </div>
  );
}
