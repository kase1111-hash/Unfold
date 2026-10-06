"use client";

import { useEffect } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  BookOpen,
  Home,
  FileText,
  Brain,
  LogOut,
  Upload,
  User,
  GraduationCap,
  Loader2,
} from "lucide-react";
import { cn } from "@/utils/cn";
import { useAuthStore } from "@/store";
import { PageErrorBoundary } from "@/components/ErrorBoundary";

const navItems = [
  { href: "/dashboard", label: "Dashboard", icon: Home },
  { href: "/documents", label: "Documents", icon: FileText },
  { href: "/graph", label: "Knowledge Graph", icon: Brain },
  { href: "/flashcards", label: "Flashcards", icon: GraduationCap },
  { href: "/upload", label: "Upload", icon: Upload },
];

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const router = useRouter();
  const { user, isInitialized, logout, initializeAuth } = useAuthStore();

  // Validate the stored access token once on mount (initializeAuth never throws;
  // it sets user=null + isInitialized=true when the token is missing/invalid).
  useEffect(() => {
    initializeAuth();
  }, [initializeAuth]);

  // Redirect only after validation has finished and no user was found.
  useEffect(() => {
    if (isInitialized && !user) {
      router.replace("/login");
    }
  }, [isInitialized, user, router]);

  const handleLogout = async () => {
    await logout();
    router.replace("/login");
  };

  // Protected pages fire API calls on mount, so don't render them until the
  // stored token has been validated (and while redirecting to /login).
  if (!isInitialized || !user) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50 dark:bg-slate-900">
        <Loader2 className="w-8 h-8 animate-spin text-primary-500" aria-label="Loading" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900">
      {/* Sidebar */}
      <aside className="fixed inset-y-0 left-0 w-64 bg-white dark:bg-slate-800 border-r border-slate-200 dark:border-slate-700 z-30">
        {/* Logo */}
        <div className="h-16 flex items-center px-6 border-b border-slate-200 dark:border-slate-700">
          <Link href="/dashboard" className="flex items-center space-x-2">
            <BookOpen className="h-8 w-8 text-primary-600" />
            <span className="text-xl font-bold text-slate-900 dark:text-white">
              Unfold
            </span>
          </Link>
        </div>

        {/* Navigation */}
        <nav className="p-4 space-y-1">
          {navItems.map((item) => {
            const isActive =
              pathname === item.href ||
              (item.href !== "/dashboard" && pathname.startsWith(item.href));

            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 px-4 py-2.5 rounded-lg transition-colors",
                  isActive
                    ? "bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300"
                    : "text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700"
                )}
              >
                <item.icon className="w-5 h-5" />
                <span className="font-medium">{item.label}</span>
              </Link>
            );
          })}
        </nav>

        {/* User section */}
        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-full bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center">
              <User className="w-5 h-5 text-primary-600" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-slate-900 dark:text-white truncate">
                {user?.full_name || user?.username || "User"}
              </p>
              <p className="text-xs text-slate-500 dark:text-slate-400 truncate">
                {user?.email}
              </p>
            </div>
          </div>
          <button
            onClick={handleLogout}
            className="flex items-center gap-2 w-full px-4 py-2 text-sm text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-lg transition-colors"
          >
            <LogOut className="w-4 h-4" />
            Sign Out
          </button>
        </div>
      </aside>

      {/* Main content */}
      <main className="pl-64">
        <div className="min-h-screen p-8">
          <PageErrorBoundary key={pathname}>{children}</PageErrorBoundary>
        </div>
      </main>
    </div>
  );
}
