import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { User } from "@/types";
import { api, getErrorMessage, isUnauthorized } from "@/services/api";

// "Authenticated" is simply `user !== null` once isInitialized is true. (It used
// to be a JS getter in the state object, which zustand's set() flattens into a
// static false, so derive it from `user` instead of storing it.)
interface AuthState {
  user: User | null;
  isLoading: boolean;
  isInitialized: boolean;
  error: string | null;
  // Why the session could not be checked (429, 5xx, network); retryable, and
  // the session is kept. isInitialized stays false meanwhile.
  initError: string | null;

  // Actions
  login: (email: string, password: string) => Promise<void>;
  register: (
    email: string,
    username: string,
    password: string,
    fullName?: string
  ) => Promise<void>;
  logout: () => Promise<void>;
  initializeAuth: () => Promise<void>;
  clearError: () => void;
}

// The check in flight, shared by concurrent callers (React runs mount effects
// twice in development)
let initPromise: Promise<void> | null = null;

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      isLoading: false,
      isInitialized: false,
      error: null,
      initError: null,

      login: async (email: string, password: string) => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.login(email, password);
          set({
            user: response.user,
            isLoading: false,
          });
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : "Login failed",
            isLoading: false,
          });
          throw error;
        }
      },

      register: async (
        email: string,
        username: string,
        password: string,
        fullName?: string
      ) => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.register(email, username, password, fullName);
          set({
            user: response.user,
            isLoading: false,
          });
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : "Registration failed",
            isLoading: false,
          });
          throw error;
        }
      },

      logout: async () => {
        await api.logout();
        set({
          user: null,
          error: null,
        });
      },

      // Initialize auth state on app load - validates the session. Never throws.
      initializeAuth: () => {
        // Skip if already initialized
        if (get().isInitialized) return Promise.resolve();

        initPromise ??= (async () => {
          set({ isLoading: true, initError: null });
          try {
            if (!api.isAuthenticated()) {
              // Nobody is remembered as signed in on this browser: probing the
              // refresh cookie would only 401 and use up the strict auth limit
              if (!get().user) {
                set({ user: null, isLoading: false, isInitialized: true });
                return;
              }
              // The httpOnly refresh cookie may still hold a valid session:
              // try it once before giving up
              await api.refreshAccessToken();
            }
            // Validate the token by fetching the current user (a 401 is
            // refreshed and retried by the API client)
            const user = await api.getCurrentUser();
            set({ user, isLoading: false, isInitialized: true });
          } catch (error) {
            if (isUnauthorized(error)) {
              // No valid session, even after a refresh
              api.clearTokens();
              set({ user: null, isLoading: false, isInitialized: true });
            } else {
              // 429, 5xx or no connection: says nothing about the session, so
              // keep the token and let the user retry
              set({ isLoading: false, initError: getErrorMessage(error) });
            }
          } finally {
            initPromise = null;
          }
        })();
        return initPromise;
      },

      clearError: () => set({ error: null }),
    }),
    {
      name: "auth-storage",
      // Only persist user data, not authentication state
      // Authentication is validated on app initialization
      partialize: (state) => ({
        user: state.user,
      }),
      // On rehydration, mark as not initialized to trigger validation
      onRehydrateStorage: () => (state) => {
        if (state) {
          state.isInitialized = false;
        }
      },
    }
  )
);

// A 401 that a token refresh could not fix ends the session; dropping the user
// lets the dashboard layout redirect to /login.
api.onAuthFailure(() => {
  useAuthStore.setState({ user: null });
});
