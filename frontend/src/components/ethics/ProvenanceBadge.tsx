"use client";

import { useState, useCallback } from "react";
import { cn } from "@/utils/cn";
import type { ContentCredential, ProvenanceManifest, ProvenanceAssertion } from "@/types";
import {
  Shield,
  ShieldCheck,
  ShieldAlert,
  ShieldX,
  ChevronDown,
  Clock,
  User,
  FileText,
  Link2,
  Loader2,
  Info,
  Copy,
  Check,
} from "lucide-react";

interface ProvenanceBadgeProps {
  documentId: string;
  compact?: boolean;
  className?: string;
}

const STATUS_CONFIG = {
  valid: {
    icon: ShieldCheck,
    color: "text-green-500",
    bgColor: "bg-green-100 dark:bg-green-900/30",
    label: "Verified",
  },
  invalid: {
    icon: ShieldAlert,
    color: "text-yellow-500",
    bgColor: "bg-yellow-100 dark:bg-yellow-900/30",
    label: "Unverified",
  },
  pending: {
    icon: Shield,
    color: "text-blue-500",
    bgColor: "bg-blue-100 dark:bg-blue-900/30",
    label: "Pending",
  },
  tampered: {
    icon: ShieldX,
    color: "text-red-500",
    bgColor: "bg-red-100 dark:bg-red-900/30",
    label: "Tampered",
  },
};

const ASSERTION_ICONS: Record<string, typeof FileText> = {
  created: FileText,
  ai_processed: Shield,
  user_edited: User,
  exported: Link2,
  default: FileText,
};

function AssertionTimeline({ assertions }: { assertions: ProvenanceAssertion[] }) {
  return (
    <div className="space-y-3">
      {assertions.map((assertion, index) => {
        const Icon = ASSERTION_ICONS[assertion.type] || ASSERTION_ICONS.default;
        return (
          <div key={assertion.assertion_id} className="flex gap-3">
            <div className="flex flex-col items-center">
              <div className="p-1.5 bg-slate-100 dark:bg-slate-700 rounded-full">
                <Icon className="w-3 h-3 text-slate-600 dark:text-slate-400" />
              </div>
              {index < assertions.length - 1 && (
                <div className="w-px flex-1 bg-slate-200 dark:bg-slate-700 my-1" />
              )}
            </div>
            <div className="flex-1 pb-3">
              <div className="flex items-center justify-between">
                <p className="text-sm font-medium text-slate-900 dark:text-white">
                  {assertion.type.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                </p>
                <span className="text-xs text-slate-400">
                  {new Date(assertion.timestamp).toLocaleString()}
                </span>
              </div>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                {assertion.description || `By ${assertion.actor}`}
              </p>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function HashDisplay({ hash }: { hash: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(hash);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [hash]);

  return (
    <div className="flex items-center gap-2 p-2 bg-slate-100 dark:bg-slate-700 rounded-lg">
      <code className="text-xs text-slate-600 dark:text-slate-400 font-mono truncate flex-1">
        {hash}
      </code>
      <button
        onClick={handleCopy}
        className="p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
        title="Copy hash"
      >
        {copied ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
      </button>
    </div>
  );
}

function ManifestDetails({ manifest }: { manifest: ProvenanceManifest }) {
  const [expandedCredential, setExpandedCredential] = useState<string | null>(null);

  return (
    <div className="space-y-4">
      {/* Validation Summary */}
      <div className="grid grid-cols-4 gap-2 text-center">
        <div className="p-2 bg-slate-50 dark:bg-slate-700/50 rounded">
          <p className="text-lg font-bold text-slate-900 dark:text-white">
            {manifest.validation_summary.total_credentials}
          </p>
          <p className="text-xs text-slate-500">Total</p>
        </div>
        <div className="p-2 bg-green-50 dark:bg-green-900/20 rounded">
          <p className="text-lg font-bold text-green-600 dark:text-green-400">
            {manifest.validation_summary.valid}
          </p>
          <p className="text-xs text-green-600 dark:text-green-400">Valid</p>
        </div>
        <div className="p-2 bg-yellow-50 dark:bg-yellow-900/20 rounded">
          <p className="text-lg font-bold text-yellow-600 dark:text-yellow-400">
            {manifest.validation_summary.invalid}
          </p>
          <p className="text-xs text-yellow-600 dark:text-yellow-400">Invalid</p>
        </div>
        <div className="p-2 bg-red-50 dark:bg-red-900/20 rounded">
          <p className="text-lg font-bold text-red-600 dark:text-red-400">
            {manifest.validation_summary.tampered}
          </p>
          <p className="text-xs text-red-600 dark:text-red-400">Tampered</p>
        </div>
      </div>

      {/* Chain Status */}
      <div className={cn(
        "flex items-center gap-2 p-3 rounded-lg",
        manifest.chain_valid
          ? "bg-green-50 dark:bg-green-900/20"
          : "bg-red-50 dark:bg-red-900/20"
      )}>
        {manifest.chain_valid ? (
          <ShieldCheck className="w-5 h-5 text-green-500" />
        ) : (
          <ShieldX className="w-5 h-5 text-red-500" />
        )}
        <span className={cn(
          "text-sm font-medium",
          manifest.chain_valid
            ? "text-green-700 dark:text-green-400"
            : "text-red-700 dark:text-red-400"
        )}>
          {manifest.chain_valid
            ? "Provenance chain verified"
            : "Provenance chain broken"}
        </span>
      </div>

      {/* Credentials */}
      <div className="space-y-2">
        <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300">
          Credentials ({manifest.credentials.length})
        </h4>
        {manifest.credentials.map((credential) => {
          const config = STATUS_CONFIG[credential.validation_status];
          const StatusIcon = config.icon;
          const isExpanded = expandedCredential === credential.credential_id;

          return (
            <div
              key={credential.credential_id}
              className="border border-slate-200 dark:border-slate-700 rounded-lg overflow-hidden"
            >
              <button
                onClick={() => setExpandedCredential(isExpanded ? null : credential.credential_id)}
                className="w-full flex items-center gap-3 p-3 text-left hover:bg-slate-50 dark:hover:bg-slate-800"
              >
                <StatusIcon className={cn("w-5 h-5", config.color)} />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-slate-900 dark:text-white truncate">
                    {credential.credential_id}
                  </p>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    Created by {credential.actor} on {new Date(credential.created_at).toLocaleDateString()}
                  </p>
                </div>
                <span className={cn("px-2 py-0.5 text-xs rounded-full", config.bgColor, config.color)}>
                  {config.label}
                </span>
                <ChevronDown className={cn("w-4 h-4 text-slate-400 transition-transform", isExpanded && "rotate-180")} />
              </button>

              {isExpanded && (
                <div className="p-3 border-t border-slate-100 dark:border-slate-700 space-y-3">
                  <div>
                    <p className="text-xs text-slate-500 mb-1">Content Hash (SHA-256)</p>
                    <HashDisplay hash={credential.content_hash} />
                  </div>
                  {credential.assertions.length > 0 && (
                    <div>
                      <p className="text-xs text-slate-500 mb-2">Processing History</p>
                      <AssertionTimeline assertions={credential.assertions} />
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function ProvenanceBadge({ documentId, compact = false, className }: ProvenanceBadgeProps) {
  const [manifest, setManifest] = useState<ProvenanceManifest | null>(null);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState(false);

  const loadManifest = useCallback(async () => {
    if (manifest) {
      setExpanded(!expanded);
      return;
    }

    setLoading(true);
    try {
      // Mock data - replace with actual API call
      const mockManifest: ProvenanceManifest = {
        manifest_id: "manifest_" + Date.now(),
        document_id: documentId,
        created_at: new Date().toISOString(),
        credentials: [
          {
            credential_id: "cred_abc123",
            document_id: documentId,
            content_hash: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            created_at: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
            actor: "user@example.com",
            validation_status: "valid",
            assertions: [
              {
                assertion_id: "a1",
                type: "created",
                timestamp: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
                actor: "user@example.com",
                description: "Document uploaded and processed",
              },
              {
                assertion_id: "a2",
                type: "ai_processed",
                timestamp: new Date(Date.now() - 6 * 24 * 60 * 60 * 1000).toISOString(),
                actor: "AI System",
                description: "Extracted entities and generated summary",
              },
              {
                assertion_id: "a3",
                type: "user_edited",
                timestamp: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
                actor: "user@example.com",
                description: "Added annotations",
              },
            ],
          },
        ],
        validation_summary: {
          total_credentials: 1,
          valid: 1,
          invalid: 0,
          tampered: 0,
        },
        chain_valid: true,
      };

      setManifest(mockManifest);
      setExpanded(true);
    } finally {
      setLoading(false);
    }
  }, [documentId, manifest, expanded]);

  // Determine overall status
  const overallStatus: keyof typeof STATUS_CONFIG = manifest
    ? manifest.chain_valid
      ? "valid"
      : manifest.validation_summary.tampered > 0
        ? "tampered"
        : "invalid"
    : "pending";

  const config = STATUS_CONFIG[overallStatus];
  const StatusIcon = config.icon;

  if (compact) {
    return (
      <button
        onClick={loadManifest}
        disabled={loading}
        className={cn(
          "flex items-center gap-1.5 px-2 py-1 rounded-full text-xs transition-colors",
          config.bgColor,
          config.color,
          className
        )}
        title={`Content ${config.label}`}
      >
        {loading ? (
          <Loader2 className="w-3 h-3 animate-spin" />
        ) : (
          <StatusIcon className="w-3 h-3" />
        )}
        <span>{config.label}</span>
      </button>
    );
  }

  return (
    <div className={cn("bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden", className)}>
      {/* Header */}
      <button
        onClick={loadManifest}
        disabled={loading}
        className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-slate-50 dark:hover:bg-slate-700/50"
      >
        <div className={cn("p-2 rounded-lg", config.bgColor)}>
          {loading ? (
            <Loader2 className={cn("w-5 h-5 animate-spin", config.color)} />
          ) : (
            <StatusIcon className={cn("w-5 h-5", config.color)} />
          )}
        </div>
        <div className="flex-1">
          <p className="font-medium text-slate-900 dark:text-white">Content Provenance</p>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            {manifest
              ? `${manifest.credentials.length} credential(s) - ${config.label}`
              : "Click to verify authenticity"}
          </p>
        </div>
        {manifest && (
          <ChevronDown className={cn("w-5 h-5 text-slate-400 transition-transform", expanded && "rotate-180")} />
        )}
      </button>

      {/* Expanded Details */}
      {expanded && manifest && (
        <div className="px-4 pb-4 border-t border-slate-100 dark:border-slate-700 pt-4">
          <ManifestDetails manifest={manifest} />

          {/* Info */}
          <div className="flex items-start gap-2 mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <Info className="w-4 h-4 text-blue-500 flex-shrink-0 mt-0.5" />
            <p className="text-xs text-blue-700 dark:text-blue-300">
              Content provenance tracks the origin and history of this document using
              cryptographic hashes. Any modifications to the content will be detected.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
