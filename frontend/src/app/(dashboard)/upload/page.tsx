"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { Upload, FileText, X, CheckCircle, AlertCircle } from "lucide-react";
import { api, getErrorMessage } from "@/services/api";
import { Button } from "@/components/ui";
import { cn } from "@/utils/cn";
import toast from "react-hot-toast";

type UploadStatus = "idle" | "uploading" | "success" | "error";

export default function UploadPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<UploadStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && isValidFile(droppedFile)) {
      setFile(droppedFile);
      setError(null);
    } else {
      setError("Please upload a PDF file");
    }
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile && isValidFile(selectedFile)) {
      setFile(selectedFile);
      setError(null);
    } else {
      setError("Please upload a PDF file");
    }
  }, []);

  const isValidFile = (file: File) => {
    return file.type === "application/pdf" || file.name.endsWith(".pdf");
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  const handleUpload = async () => {
    if (!file) return;

    setStatus("uploading");
    setError(null);

    try {
      const document = await api.uploadDocument(file);
      setStatus("success");
      toast.success("Document uploaded successfully!");

      // Redirect to reading view after a short delay
      setTimeout(() => {
        router.push(`/read/${document.doc_id}`);
      }, 1500);
    } catch (err) {
      setStatus("error");
      setError(getErrorMessage(err));
      toast.error("Failed to upload document");
    }
  };

  const handleRemoveFile = () => {
    setFile(null);
    setStatus("idle");
    setError(null);
  };

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
          Upload Document
        </h1>
        <p className="text-slate-600 dark:text-slate-400 mt-1">
          Upload a PDF to start reading with AI assistance
        </p>
      </div>

      {/* Upload area */}
      <div className="card p-8">
        {!file ? (
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={cn(
              "border-2 border-dashed rounded-xl p-12 text-center transition-colors",
              isDragging
                ? "border-primary-500 bg-primary-50 dark:bg-primary-900/20"
                : "border-slate-300 dark:border-slate-600 hover:border-primary-400"
            )}
          >
            <Upload
              className={cn(
                "w-12 h-12 mx-auto mb-4",
                isDragging ? "text-primary-500" : "text-slate-400"
              )}
            />
            <p className="text-lg font-medium text-slate-900 dark:text-white mb-2">
              Drag and drop your PDF here
            </p>
            <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">
              or click to browse
            </p>
            <label className="btn-primary cursor-pointer">
              <input
                type="file"
                accept=".pdf,application/pdf"
                onChange={handleFileSelect}
                className="hidden"
              />
              Select File
            </label>
            <p className="text-xs text-slate-400 dark:text-slate-500 mt-4">
              Supported format: PDF (max 50MB)
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {/* File preview */}
            <div className="flex items-center gap-4 p-4 bg-slate-50 dark:bg-slate-700/50 rounded-lg">
              <div className="w-12 h-12 rounded-lg bg-red-100 dark:bg-red-900/30 flex items-center justify-center">
                <FileText className="w-6 h-6 text-red-600" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="font-medium text-slate-900 dark:text-white truncate">
                  {file.name}
                </p>
                <p className="text-sm text-slate-500 dark:text-slate-400">
                  {formatFileSize(file.size)}
                </p>
              </div>
              {status === "idle" && (
                <button
                  onClick={handleRemoveFile}
                  className="p-2 text-slate-400 hover:text-red-500 transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              )}
              {status === "success" && (
                <CheckCircle className="w-6 h-6 text-green-500" />
              )}
              {status === "error" && (
                <AlertCircle className="w-6 h-6 text-red-500" />
              )}
            </div>

            {/* Upload button */}
            {status === "idle" && (
              <Button
                onClick={handleUpload}
                className="w-full"
                size="lg"
                leftIcon={<Upload className="w-4 h-4" />}
              >
                Upload Document
              </Button>
            )}

            {/* Progress */}
            {status === "uploading" && (
              <div className="space-y-2">
                <div className="h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                  <div className="h-full bg-primary-500 rounded-full animate-pulse w-2/3" />
                </div>
                <p className="text-sm text-slate-500 dark:text-slate-400 text-center">
                  Uploading and processing...
                </p>
              </div>
            )}

            {/* Success */}
            {status === "success" && (
              <p className="text-sm text-green-600 dark:text-green-400 text-center">
                Upload successful! Redirecting to reading view...
              </p>
            )}

            {/* Error */}
            {status === "error" && (
              <div className="space-y-3">
                <p className="text-sm text-red-600 dark:text-red-400 text-center">
                  {error || "Upload failed. Please try again."}
                </p>
                <Button
                  onClick={handleUpload}
                  className="w-full"
                  variant="secondary"
                >
                  Retry Upload
                </Button>
              </div>
            )}
          </div>
        )}

        {/* Error message */}
        {error && !file && (
          <p className="mt-4 text-sm text-red-500 text-center">{error}</p>
        )}
      </div>

      {/* Tips */}
      <div className="card p-6">
        <h2 className="font-semibold text-slate-900 dark:text-white mb-3">
          Tips for best results
        </h2>
        <ul className="space-y-2 text-sm text-slate-600 dark:text-slate-400">
          <li className="flex items-start gap-2">
            <span className="text-primary-500">•</span>
            Upload text-based PDFs for best text extraction
          </li>
          <li className="flex items-start gap-2">
            <span className="text-primary-500">•</span>
            Academic papers with clear structure work best
          </li>
          <li className="flex items-start gap-2">
            <span className="text-primary-500">•</span>
            Processing may take a few minutes for long documents
          </li>
          <li className="flex items-start gap-2">
            <span className="text-primary-500">•</span>
            Scanned PDFs may have reduced accuracy
          </li>
        </ul>
      </div>
    </div>
  );
}
