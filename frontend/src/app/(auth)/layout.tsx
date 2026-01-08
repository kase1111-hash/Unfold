import { BookOpen } from "lucide-react";
import Link from "next/link";

export default function AuthLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      {/* Header */}
      <header className="p-4">
        <Link href="/" className="inline-flex items-center space-x-2">
          <BookOpen className="h-8 w-8 text-primary-600" />
          <span className="text-xl font-bold text-slate-900 dark:text-white">
            Unfold
          </span>
        </Link>
      </header>

      {/* Main content */}
      <main className="flex-1 flex items-center justify-center p-4">
        <div className="w-full max-w-md">{children}</div>
      </main>

      {/* Footer */}
      <footer className="p-4 text-center">
        <p className="text-sm text-slate-500 dark:text-slate-400">
          &copy; 2026 Unfold. Open Source under AGPL v3.
        </p>
      </footer>
    </div>
  );
}
