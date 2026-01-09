"use client";

import { useState, useEffect, useCallback } from "react";
import { cn } from "@/utils/cn";
import type { ConsentRecord, ConsentType, UserEthicsProfile, TransparencyLevel } from "@/types";
import {
  Shield,
  Eye,
  EyeOff,
  BarChart3,
  Users,
  Target,
  FlaskConical,
  Check,
  X,
  ChevronDown,
  Download,
  Trash2,
  Loader2,
  AlertTriangle,
  Info,
  Settings,
} from "lucide-react";

interface PrivacyCenterProps {
  className?: string;
}

const CONSENT_CONFIG: Record<ConsentType, { icon: typeof Shield; label: string; description: string }> = {
  analytics: {
    icon: BarChart3,
    label: "Analytics",
    description: "Allow usage analytics to improve the service",
  },
  personalization: {
    icon: Target,
    label: "Personalization",
    description: "Enable AI-powered recommendations and learning paths",
  },
  third_party: {
    icon: Users,
    label: "Third-Party Sharing",
    description: "Share anonymized data with research partners",
  },
  marketing: {
    icon: Users,
    label: "Marketing Communications",
    description: "Receive updates about new features and tips",
  },
  research: {
    icon: FlaskConical,
    label: "Research Participation",
    description: "Contribute to educational research studies",
  },
};

const TRANSPARENCY_LEVELS: { value: TransparencyLevel; label: string; description: string }[] = [
  { value: "full", label: "Full Transparency", description: "See complete details of all AI operations" },
  { value: "summary", label: "Summary Only", description: "View aggregated information and trends" },
  { value: "minimal", label: "Minimal", description: "Basic acknowledgment of operations" },
  { value: "redacted", label: "Redacted", description: "Hide sensitive operation details" },
];

function ConsentToggle({
  consentType,
  consent,
  onToggle,
  loading,
}: {
  consentType: ConsentType;
  consent?: ConsentRecord;
  onToggle: (type: ConsentType, granted: boolean) => void;
  loading: boolean;
}) {
  const config = CONSENT_CONFIG[consentType];
  const Icon = config.icon;
  const isGranted = consent?.status === "granted";

  return (
    <div className="flex items-start gap-4 p-4 bg-slate-50 dark:bg-slate-700/50 rounded-lg">
      <div className="p-2 bg-white dark:bg-slate-800 rounded-lg">
        <Icon className="w-5 h-5 text-slate-600 dark:text-slate-400" />
      </div>
      <div className="flex-1">
        <div className="flex items-center justify-between">
          <h4 className="font-medium text-slate-900 dark:text-white">{config.label}</h4>
          <button
            onClick={() => onToggle(consentType, !isGranted)}
            disabled={loading}
            className={cn(
              "relative w-12 h-6 rounded-full transition-colors",
              isGranted ? "bg-green-500" : "bg-slate-300 dark:bg-slate-600",
              loading && "opacity-50"
            )}
          >
            <span
              className={cn(
                "absolute top-1 w-4 h-4 bg-white rounded-full transition-transform",
                isGranted ? "left-7" : "left-1"
              )}
            />
          </button>
        </div>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">{config.description}</p>
        {consent?.granted_at && (
          <p className="text-xs text-slate-400 mt-2">
            {isGranted ? "Granted" : "Withdrawn"} on {new Date(consent.granted_at).toLocaleDateString()}
          </p>
        )}
      </div>
    </div>
  );
}

function TransparencySelector({
  value,
  onChange,
}: {
  value: TransparencyLevel;
  onChange: (level: TransparencyLevel) => void;
}) {
  const [open, setOpen] = useState(false);
  const selected = TRANSPARENCY_LEVELS.find((l) => l.value === value) || TRANSPARENCY_LEVELS[0];

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-3 bg-slate-50 dark:bg-slate-700/50 rounded-lg text-left"
      >
        <div className="flex items-center gap-3">
          <Eye className="w-5 h-5 text-slate-600 dark:text-slate-400" />
          <div>
            <p className="font-medium text-slate-900 dark:text-white">{selected.label}</p>
            <p className="text-sm text-slate-500 dark:text-slate-400">{selected.description}</p>
          </div>
        </div>
        <ChevronDown className={cn("w-5 h-5 text-slate-400 transition-transform", open && "rotate-180")} />
      </button>

      {open && (
        <div className="absolute top-full left-0 right-0 mt-2 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg shadow-lg z-10">
          {TRANSPARENCY_LEVELS.map((level) => (
            <button
              key={level.value}
              onClick={() => {
                onChange(level.value);
                setOpen(false);
              }}
              className={cn(
                "w-full flex items-center justify-between px-4 py-3 text-left hover:bg-slate-50 dark:hover:bg-slate-700 first:rounded-t-lg last:rounded-b-lg",
                level.value === value && "bg-primary-50 dark:bg-primary-900/20"
              )}
            >
              <div>
                <p className="font-medium text-slate-900 dark:text-white">{level.label}</p>
                <p className="text-sm text-slate-500 dark:text-slate-400">{level.description}</p>
              </div>
              {level.value === value && <Check className="w-5 h-5 text-primary-500" />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function DataActionButton({
  icon: Icon,
  label,
  description,
  variant,
  onClick,
  loading,
}: {
  icon: typeof Download;
  label: string;
  description: string;
  variant: "primary" | "danger";
  onClick: () => void;
  loading: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={loading}
      className={cn(
        "flex items-start gap-3 p-4 rounded-lg text-left transition-colors w-full",
        variant === "primary"
          ? "bg-primary-50 dark:bg-primary-900/20 hover:bg-primary-100 dark:hover:bg-primary-900/30"
          : "bg-red-50 dark:bg-red-900/20 hover:bg-red-100 dark:hover:bg-red-900/30",
        loading && "opacity-50"
      )}
    >
      <Icon className={cn("w-5 h-5", variant === "primary" ? "text-primary-500" : "text-red-500")} />
      <div>
        <p className={cn("font-medium", variant === "primary" ? "text-primary-700 dark:text-primary-400" : "text-red-700 dark:text-red-400")}>
          {label}
        </p>
        <p className={cn("text-sm", variant === "primary" ? "text-primary-600 dark:text-primary-300" : "text-red-600 dark:text-red-300")}>
          {description}
        </p>
      </div>
    </button>
  );
}

export function PrivacyCenter({ className }: PrivacyCenterProps) {
  const [consents, setConsents] = useState<ConsentRecord[]>([]);
  const [profile, setProfile] = useState<UserEthicsProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      // Mock data - replace with actual API calls
      const mockConsents: ConsentRecord[] = [
        {
          consent_id: "c1",
          user_id: "user_123",
          consent_type: "analytics",
          status: "granted",
          granted_at: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
        },
        {
          consent_id: "c2",
          user_id: "user_123",
          consent_type: "personalization",
          status: "granted",
          granted_at: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
        },
        {
          consent_id: "c3",
          user_id: "user_123",
          consent_type: "third_party",
          status: "denied",
        },
      ];

      const mockProfile: UserEthicsProfile = {
        user_id: "user_123",
        created_at: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000).toISOString(),
        transparency_level: "full",
        receive_ethics_reports: true,
        allow_aggregated_analytics: true,
        total_ai_operations: 47,
        total_documents_processed: 12,
        bias_alerts_received: 3,
        privacy_actions_taken: 5,
      };

      setConsents(mockConsents);
      setProfile(mockProfile);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleConsentToggle = useCallback(async (type: ConsentType, granted: boolean) => {
    setActionLoading(true);
    try {
      // Mock API call - replace with actual implementation
      await new Promise((resolve) => setTimeout(resolve, 500));

      setConsents((prev) => {
        const existing = prev.find((c) => c.consent_type === type);
        if (existing) {
          return prev.map((c) =>
            c.consent_type === type
              ? { ...c, status: granted ? "granted" : "withdrawn", granted_at: new Date().toISOString() }
              : c
          );
        }
        return [
          ...prev,
          {
            consent_id: `c_${Date.now()}`,
            user_id: "user_123",
            consent_type: type,
            status: granted ? "granted" : "denied",
            granted_at: granted ? new Date().toISOString() : undefined,
          } as ConsentRecord,
        ];
      });
    } finally {
      setActionLoading(false);
    }
  }, []);

  const handleTransparencyChange = useCallback(async (level: TransparencyLevel) => {
    setActionLoading(true);
    try {
      await new Promise((resolve) => setTimeout(resolve, 500));
      setProfile((prev) => (prev ? { ...prev, transparency_level: level } : null));
    } finally {
      setActionLoading(false);
    }
  }, []);

  const handleExportData = useCallback(async () => {
    setActionLoading(true);
    try {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      // Mock download - in real implementation, this would download a file
      alert("Data export initiated. You will receive an email with the download link.");
    } finally {
      setActionLoading(false);
    }
  }, []);

  const handleDeleteData = useCallback(async () => {
    setActionLoading(true);
    try {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      alert("Data deletion request submitted. Your data will be removed within 30 days.");
      setShowDeleteConfirm(false);
    } finally {
      setActionLoading(false);
    }
  }, []);

  if (loading) {
    return (
      <div className={cn("flex items-center justify-center py-12", className)}>
        <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
      </div>
    );
  }

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div>
        <h2 className="text-xl font-semibold text-slate-900 dark:text-white flex items-center gap-2">
          <Shield className="w-6 h-6 text-primary-500" />
          Privacy Center
        </h2>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Manage your privacy settings and data preferences
        </p>
      </div>

      {/* GDPR Notice */}
      <div className="flex items-start gap-3 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
        <Info className="w-5 h-5 text-blue-500 flex-shrink-0 mt-0.5" />
        <div className="text-sm text-blue-800 dark:text-blue-200">
          <p className="font-medium">Your Privacy Rights</p>
          <p className="mt-1">
            Under GDPR, you have the right to access, correct, export, and delete your personal data.
            You can also withdraw consent at any time.
          </p>
        </div>
      </div>

      {/* Consent Management */}
      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-700">
          <h3 className="font-medium text-slate-900 dark:text-white flex items-center gap-2">
            <Settings className="w-4 h-4 text-slate-400" />
            Consent Preferences
          </h3>
        </div>
        <div className="p-4 space-y-3">
          {(Object.keys(CONSENT_CONFIG) as ConsentType[]).map((type) => (
            <ConsentToggle
              key={type}
              consentType={type}
              consent={consents.find((c) => c.consent_type === type)}
              onToggle={handleConsentToggle}
              loading={actionLoading}
            />
          ))}
        </div>
      </div>

      {/* Transparency Level */}
      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-700">
          <h3 className="font-medium text-slate-900 dark:text-white flex items-center gap-2">
            <Eye className="w-4 h-4 text-slate-400" />
            Transparency Level
          </h3>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            Control how much detail you see about AI operations
          </p>
        </div>
        <div className="p-4">
          <TransparencySelector
            value={profile?.transparency_level || "full"}
            onChange={handleTransparencyChange}
          />
        </div>
      </div>

      {/* Data Actions */}
      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-700">
          <h3 className="font-medium text-slate-900 dark:text-white">Your Data</h3>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            Export or delete your personal data
          </p>
        </div>
        <div className="p-4 space-y-3">
          <DataActionButton
            icon={Download}
            label="Export My Data"
            description="Download all your personal data in JSON format"
            variant="primary"
            onClick={handleExportData}
            loading={actionLoading}
          />

          {!showDeleteConfirm ? (
            <DataActionButton
              icon={Trash2}
              label="Delete My Data"
              description="Permanently remove all your data from our systems"
              variant="danger"
              onClick={() => setShowDeleteConfirm(true)}
              loading={actionLoading}
            />
          ) : (
            <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                <div className="flex-1">
                  <p className="font-medium text-red-700 dark:text-red-400">
                    Are you sure you want to delete all your data?
                  </p>
                  <p className="text-sm text-red-600 dark:text-red-300 mt-1">
                    This action cannot be undone. All your documents, annotations, and learning history will be permanently deleted.
                  </p>
                  <div className="flex gap-2 mt-3">
                    <button
                      onClick={handleDeleteData}
                      disabled={actionLoading}
                      className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors disabled:opacity-50"
                    >
                      {actionLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : "Yes, Delete Everything"}
                    </button>
                    <button
                      onClick={() => setShowDeleteConfirm(false)}
                      className="px-4 py-2 bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Profile Stats */}
      {profile && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-3 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
            <p className="text-xl font-bold text-slate-900 dark:text-white">{profile.total_ai_operations}</p>
            <p className="text-xs text-slate-500 dark:text-slate-400">AI Operations</p>
          </div>
          <div className="text-center p-3 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
            <p className="text-xl font-bold text-slate-900 dark:text-white">{profile.total_documents_processed}</p>
            <p className="text-xs text-slate-500 dark:text-slate-400">Documents</p>
          </div>
          <div className="text-center p-3 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
            <p className="text-xl font-bold text-slate-900 dark:text-white">{profile.bias_alerts_received}</p>
            <p className="text-xs text-slate-500 dark:text-slate-400">Bias Alerts</p>
          </div>
          <div className="text-center p-3 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
            <p className="text-xl font-bold text-slate-900 dark:text-white">{profile.privacy_actions_taken}</p>
            <p className="text-xs text-slate-500 dark:text-slate-400">Privacy Actions</p>
          </div>
        </div>
      )}
    </div>
  );
}
