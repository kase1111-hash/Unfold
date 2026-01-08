"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { FileText, Search, Trash2, ExternalLink, Plus } from "lucide-react";
import { api, getErrorMessage } from "@/services/api";
import { Input, Button } from "@/components/ui";
import toast from "react-hot-toast";
import type { Document } from "@/types";

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);

  const fetchDocuments = async (page: number) => {
    setIsLoading(true);
    try {
      const response = await api.getDocuments(page, 10);
      setDocuments(response.data);
      setTotalPages(response.pagination.total_pages);
    } catch (error) {
      toast.error(getErrorMessage(error));
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments(currentPage);
  }, [currentPage]);

  const handleDelete = async (docId: string) => {
    if (!confirm("Are you sure you want to delete this document?")) return;

    try {
      await api.deleteDocument(docId);
      toast.success("Document deleted");
      fetchDocuments(currentPage);
    } catch (error) {
      toast.error(getErrorMessage(error));
    }
  };

  const filteredDocuments = documents.filter(
    (doc) =>
      doc.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      doc.authors.some((a) =>
        a.toLowerCase().includes(searchQuery.toLowerCase())
      )
  );

  const statusColors: Record<string, string> = {
    indexed: "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300",
    processing: "bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300",
    validated: "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300",
    pending: "bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-400",
    failed: "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300",
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
            Documents
          </h1>
          <p className="text-slate-600 dark:text-slate-400 mt-1">
            Manage your uploaded documents
          </p>
        </div>
        <Link href="/upload" className="btn-primary flex items-center gap-2">
          <Plus className="w-4 h-4" />
          Upload New
        </Link>
      </div>

      {/* Search */}
      <div className="card p-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
          <input
            type="text"
            placeholder="Search documents..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2.5 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-primary-500/50"
          />
        </div>
      </div>

      {/* Document list */}
      <div className="card overflow-hidden">
        {isLoading ? (
          <div className="p-8 text-center text-slate-500 dark:text-slate-400">
            Loading documents...
          </div>
        ) : filteredDocuments.length === 0 ? (
          <div className="p-8 text-center">
            <FileText className="w-12 h-12 mx-auto text-slate-300 dark:text-slate-600 mb-3" />
            <p className="text-slate-600 dark:text-slate-400">
              {searchQuery ? "No documents match your search" : "No documents uploaded yet"}
            </p>
          </div>
        ) : (
          <>
            <table className="w-full">
              <thead className="bg-slate-50 dark:bg-slate-700/50">
                <tr>
                  <th className="text-left px-4 py-3 text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide">
                    Title
                  </th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide">
                    Authors
                  </th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide">
                    Status
                  </th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide">
                    Source
                  </th>
                  <th className="text-right px-4 py-3 text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
                {filteredDocuments.map((doc) => (
                  <tr
                    key={doc.doc_id}
                    className="hover:bg-slate-50 dark:hover:bg-slate-700/50"
                  >
                    <td className="px-4 py-4">
                      <Link
                        href={`/read/${doc.doc_id}`}
                        className="font-medium text-slate-900 dark:text-white hover:text-primary-600 dark:hover:text-primary-400"
                      >
                        {doc.title}
                      </Link>
                      {doc.doi && (
                        <p className="text-xs text-slate-500 mt-0.5">
                          DOI: {doc.doi}
                        </p>
                      )}
                    </td>
                    <td className="px-4 py-4 text-sm text-slate-600 dark:text-slate-400">
                      {doc.authors.slice(0, 2).join(", ")}
                      {doc.authors.length > 2 && " et al."}
                    </td>
                    <td className="px-4 py-4">
                      <span
                        className={`text-xs px-2 py-1 rounded-full ${
                          statusColors[doc.status] || statusColors.pending
                        }`}
                      >
                        {doc.status}
                      </span>
                    </td>
                    <td className="px-4 py-4 text-sm text-slate-600 dark:text-slate-400">
                      {doc.source}
                    </td>
                    <td className="px-4 py-4">
                      <div className="flex items-center justify-end gap-2">
                        <Link
                          href={`/read/${doc.doc_id}`}
                          className="p-2 text-slate-400 hover:text-primary-600 dark:hover:text-primary-400 transition-colors"
                          title="Read"
                        >
                          <ExternalLink className="w-4 h-4" />
                        </Link>
                        <button
                          onClick={() => handleDelete(doc.doc_id)}
                          className="p-2 text-slate-400 hover:text-red-600 dark:hover:text-red-400 transition-colors"
                          title="Delete"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-between px-4 py-3 border-t border-slate-200 dark:border-slate-700">
                <Button
                  variant="secondary"
                  size="sm"
                  disabled={currentPage === 1}
                  onClick={() => setCurrentPage((p) => p - 1)}
                >
                  Previous
                </Button>
                <span className="text-sm text-slate-600 dark:text-slate-400">
                  Page {currentPage} of {totalPages}
                </span>
                <Button
                  variant="secondary"
                  size="sm"
                  disabled={currentPage === totalPages}
                  onClick={() => setCurrentPage((p) => p + 1)}
                >
                  Next
                </Button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
