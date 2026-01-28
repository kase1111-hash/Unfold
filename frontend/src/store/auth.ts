import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { User } from "@/types";
import { api } from "@/services/api";

interface AuthState {
  user: User | null;
  isLoading: boolean;
  isInitialized: boolean;
  error: string | null;

  // Computed property - derived from user state
  isAuthenticated: boolean;

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

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      isLoading: false,
      isInitialized: false,
      error: null,

      // Computed: authenticated if we have a user
      get isAuthenticated() {
        return get().user !== null;
      },

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

      // Initialize auth state on app load - validates stored token
      initializeAuth: async () => {
        // Skip if already initialized
        if (get().isInitialized) return;

        // Check if we have an access token stored
        if (!api.isAuthenticated()) {
          set({ isInitialized: true, user: null });
          return;
        }

        set({ isLoading: true });
        try {
          // Validate the token by fetching current user
          const user = await api.getCurrentUser();
          set({
            user,
            isLoading: false,
            isInitialized: true,
          });
        } catch {
          // Token is invalid, clear stored state
          api.clearTokens();
          set({
            user: null,
            isLoading: false,
            isInitialized: true,
          });
        }
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
