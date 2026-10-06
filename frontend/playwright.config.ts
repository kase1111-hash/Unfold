import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright E2E test configuration for Unfold frontend.
 *
 * No backend is needed: every test mocks the API with page.route (see
 * e2e/fixtures.ts), so the suite runs the same locally and in CI.
 * @see https://playwright.dev/docs/test-configuration
 */
const PORT = Number(process.env.PLAYWRIGHT_PORT || 3000);
const BASE_URL = process.env.PLAYWRIGHT_BASE_URL || `http://localhost:${PORT}`;

// Optional, only read when set: use an already installed Chromium instead of
// the revision this @playwright/test version would download.
const chromiumExecutable = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH;
const chromiumLaunch = chromiumExecutable
  ? { launchOptions: { executablePath: chromiumExecutable } }
  : {};

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [["list"], ["html", { open: "never" }]],

  // `next dev` compiles each route on its first visit, which can take a while
  timeout: 60_000,
  expect: { timeout: 15_000 },

  use: {
    baseURL: BASE_URL,
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },

  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"], ...chromiumLaunch },
    },
    {
      name: "firefox",
      use: { ...devices["Desktop Firefox"] },
    },
    {
      name: "webkit",
      use: { ...devices["Desktop Safari"] },
    },
    {
      name: "Mobile Chrome",
      use: { ...devices["Pixel 5"], ...chromiumLaunch },
    },
  ],

  // Run local dev server before tests
  webServer: {
    command: `npm run dev -- --port ${PORT}`,
    url: BASE_URL,
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
  },
});
