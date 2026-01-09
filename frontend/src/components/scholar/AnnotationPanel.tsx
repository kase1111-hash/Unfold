"use client";

import { useState, useEffect, useCallback } from "react";
import { useScholarStore, useAuthStore } from "@/store";
import { cn } from "@/utils/cn";
import type { Annotation, AnnotationType, AnnotationVisibility } from "@/types";
import {
  MessageSquare,
  Highlighter,
  HelpCircle,
  Link2,
  Tag,
  Check,
  Send,
  Trash2,
  MoreVertical,
  Globe,
  Users,
  Lock,
  Reply,
  ThumbsUp,
  Heart,
  Loader2,
} from "lucide-react";

interface AnnotationPanelProps {
  documentId: string;
  className?: string;
}

const TYPE_CONFIG: Record<AnnotationType, { icon: typeof MessageSquare; label: string; color: string }> = {
  highlight: { icon: Highlighter, label: "Highlight", color: "text-yellow-500" },
  comment: { icon: MessageSquare, label: "Comment", color: "text-blue-500" },
  question: { icon: HelpCircle, label: "Question", color: "text-purple-500" },
  answer: { icon: Check, label: "Answer", color: "text-green-500" },
  link: { icon: Link2, label: "Link", color: "text-cyan-500" },
  tag: { icon: Tag, label: "Tag", color: "text-orange-500" },
};

const VISIBILITY_CONFIG: Record<AnnotationVisibility, { icon: typeof Globe; label: string }> = {
  private: { icon: Lock, label: "Private" },
  group: { icon: Users, label: "Group" },
  public: { icon: Globe, label: "Public" },
};

const REACTIONS = ["👍", "❤️", "💡", "🤔", "👏"];

interface AnnotationCardProps {
  annotation: Annotation;
  onReply: (id: string) => void;
  onDelete: (id: string) => void;
  onReaction: (id: string, emoji: string) => void;
}

function AnnotationCard({ annotation, onReply, onDelete, onReaction }: AnnotationCardProps) {
  const [showActions, setShowActions] = useState(false);
  const typeConfig = TYPE_CONFIG[annotation.type];
  const visConfig = VISIBILITY_CONFIG[annotation.visibility];
  const Icon = typeConfig.icon;
  const VisIcon = visConfig.icon;
  const { user } = useAuthStore();
  const isOwner = user?.user_id === annotation.user_id;

  return (
    <div
      className="p-3 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 space-y-2"
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
    >
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2">
          <Icon className={cn("w-4 h-4", typeConfig.color)} />
          <span className="text-sm font-medium text-slate-900 dark:text-white">
            {annotation.user_name}
          </span>
          <span title={visConfig.label}>
            <VisIcon className="w-3 h-3 text-slate-400" />
          </span>
        </div>
        {showActions && (
          <div className="flex items-center gap-1">
            <button
              onClick={() => onReply(annotation.annotation_id)}
              className="p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
              title="Reply"
            >
              <Reply className="w-4 h-4" />
            </button>
            {isOwner && (
              <button
                onClick={() => onDelete(annotation.annotation_id)}
                className="p-1 text-slate-400 hover:text-red-500"
                title="Delete"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            )}
          </div>
        )}
      </div>

      {/* Selected text */}
      {annotation.selected_text && (
        <div className="pl-3 border-l-2 border-yellow-400 text-sm text-slate-600 dark:text-slate-400 italic">
          &ldquo;{annotation.selected_text}&rdquo;
        </div>
      )}

      {/* Content */}
      {annotation.content && (
        <p className="text-sm text-slate-700 dark:text-slate-300">{annotation.content}</p>
      )}

      {/* Tags */}
      {annotation.tags.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {annotation.tags.map((tag) => (
            <span
              key={tag}
              className="px-2 py-0.5 text-xs bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-400 rounded-full"
            >
              #{tag}
            </span>
          ))}
        </div>
      )}

      {/* Footer */}
      <div className="flex items-center justify-between pt-2 border-t border-slate-100 dark:border-slate-700">
        {/* Reactions */}
        <div className="flex items-center gap-1">
          {Object.entries(annotation.reactions).map(([emoji, users]) => (
            <button
              key={emoji}
              onClick={() => onReaction(annotation.annotation_id, emoji)}
              className={cn(
                "flex items-center gap-1 px-2 py-0.5 text-xs rounded-full",
                users.includes(user?.user_id || "")
                  ? "bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400"
                  : "bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-400"
              )}
            >
              <span>{emoji}</span>
              <span>{users.length}</span>
            </button>
          ))}
          <div className="relative group">
            <button className="p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300">
              <ThumbsUp className="w-4 h-4" />
            </button>
            <div className="hidden group-hover:flex absolute bottom-full left-0 mb-1 p-1 bg-white dark:bg-slate-800 rounded-lg shadow-lg border border-slate-200 dark:border-slate-700 gap-1">
              {REACTIONS.map((emoji) => (
                <button
                  key={emoji}
                  onClick={() => onReaction(annotation.annotation_id, emoji)}
                  className="p-1 hover:bg-slate-100 dark:hover:bg-slate-700 rounded"
                >
                  {emoji}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Timestamp */}
        <span className="text-xs text-slate-400">
          {new Date(annotation.created_at).toLocaleDateString()}
        </span>
      </div>
    </div>
  );
}

export function AnnotationPanel({ documentId, className }: AnnotationPanelProps) {
  const [newContent, setNewContent] = useState("");
  const [selectedType, setSelectedType] = useState<AnnotationType>("comment");
  const [selectedVisibility, setSelectedVisibility] = useState<AnnotationVisibility>("private");
  const [replyingTo, setReplyingTo] = useState<string | null>(null);
  const [filter, setFilter] = useState<AnnotationType | "all">("all");

  const {
    annotations,
    annotationsLoading,
    loadAnnotations,
    createAnnotation,
    deleteAnnotation,
    addReaction,
  } = useScholarStore();
  const { user } = useAuthStore();

  useEffect(() => {
    if (documentId) {
      loadAnnotations(documentId);
    }
  }, [documentId, loadAnnotations]);

  const handleSubmit = useCallback(async () => {
    if (!newContent.trim()) return;

    await createAnnotation(documentId, {
      annotationType: selectedType,
      content: newContent,
      visibility: selectedVisibility,
      parentId: replyingTo || undefined,
    });

    setNewContent("");
    setReplyingTo(null);
  }, [newContent, selectedType, selectedVisibility, replyingTo, documentId, createAnnotation]);

  const handleDelete = useCallback(
    async (annotationId: string) => {
      await deleteAnnotation(documentId, annotationId);
    },
    [documentId, deleteAnnotation]
  );

  const handleReaction = useCallback(
    async (annotationId: string, emoji: string) => {
      await addReaction(documentId, annotationId, emoji);
    },
    [documentId, addReaction]
  );

  const filteredAnnotations = annotations.filter(
    (a) => !a.is_deleted && (filter === "all" || a.type === filter)
  );

  const rootAnnotations = filteredAnnotations.filter((a) => !a.parent_id);

  return (
    <div
      className={cn(
        "flex flex-col bg-slate-50 dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden",
        className
      )}
    >
      {/* Header */}
      <div className="px-4 py-3 bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700">
        <div className="flex items-center justify-between">
          <h3 className="font-medium text-slate-900 dark:text-white">Annotations</h3>
          <span className="text-sm text-slate-500">{annotations.length} total</span>
        </div>

        {/* Filter tabs */}
        <div className="flex gap-1 mt-2 overflow-x-auto">
          <button
            onClick={() => setFilter("all")}
            className={cn(
              "px-2 py-1 text-xs rounded-full transition-colors",
              filter === "all"
                ? "bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400"
                : "bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-600"
            )}
          >
            All
          </button>
          {Object.entries(TYPE_CONFIG).map(([type, config]) => {
            const Icon = config.icon;
            return (
              <button
                key={type}
                onClick={() => setFilter(type as AnnotationType)}
                className={cn(
                  "flex items-center gap-1 px-2 py-1 text-xs rounded-full transition-colors",
                  filter === type
                    ? "bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400"
                    : "bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-600"
                )}
              >
                <Icon className={cn("w-3 h-3", config.color)} />
                {config.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Annotations list */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {annotationsLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="w-6 h-6 animate-spin text-slate-400" />
          </div>
        ) : rootAnnotations.length === 0 ? (
          <div className="text-center py-8 text-slate-500 dark:text-slate-400">
            No annotations yet. Be the first to add one!
          </div>
        ) : (
          rootAnnotations.map((annotation) => (
            <div key={annotation.annotation_id}>
              <AnnotationCard
                annotation={annotation}
                onReply={setReplyingTo}
                onDelete={handleDelete}
                onReaction={handleReaction}
              />
              {/* Replies */}
              {filteredAnnotations
                .filter((a) => a.parent_id === annotation.annotation_id)
                .map((reply) => (
                  <div key={reply.annotation_id} className="ml-6 mt-2">
                    <AnnotationCard
                      annotation={reply}
                      onReply={setReplyingTo}
                      onDelete={handleDelete}
                      onReaction={handleReaction}
                    />
                  </div>
                ))}
            </div>
          ))
        )}
      </div>

      {/* New annotation form */}
      <div className="p-4 bg-white dark:bg-slate-800 border-t border-slate-200 dark:border-slate-700">
        {replyingTo && (
          <div className="flex items-center justify-between mb-2 text-xs text-slate-500">
            <span>
              Replying to{" "}
              {annotations.find((a) => a.annotation_id === replyingTo)?.user_name}
            </span>
            <button
              onClick={() => setReplyingTo(null)}
              className="text-red-500 hover:underline"
            >
              Cancel
            </button>
          </div>
        )}

        {/* Type & visibility selectors */}
        <div className="flex gap-2 mb-2">
          <select
            value={selectedType}
            onChange={(e) => setSelectedType(e.target.value as AnnotationType)}
            className="px-2 py-1 text-xs bg-slate-100 dark:bg-slate-700 border-0 rounded-lg focus:ring-2 focus:ring-primary-500"
          >
            {Object.entries(TYPE_CONFIG).map(([type, config]) => (
              <option key={type} value={type}>
                {config.label}
              </option>
            ))}
          </select>
          <select
            value={selectedVisibility}
            onChange={(e) => setSelectedVisibility(e.target.value as AnnotationVisibility)}
            className="px-2 py-1 text-xs bg-slate-100 dark:bg-slate-700 border-0 rounded-lg focus:ring-2 focus:ring-primary-500"
          >
            {Object.entries(VISIBILITY_CONFIG).map(([vis, config]) => (
              <option key={vis} value={vis}>
                {config.label}
              </option>
            ))}
          </select>
        </div>

        {/* Input */}
        <div className="flex gap-2">
          <textarea
            value={newContent}
            onChange={(e) => setNewContent(e.target.value)}
            placeholder="Add your annotation..."
            className="flex-1 px-3 py-2 text-sm bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-600 rounded-lg resize-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            rows={2}
          />
          <button
            onClick={handleSubmit}
            disabled={!newContent.trim()}
            className="px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
