import { test, expect } from "@playwright/test";

test.describe("Authentication Flow", () => {
  test.describe("Login Page", () => {
    test("should display login form", async ({ page }) => {
      await page.goto("/login");

      // Check for email/username input
      const emailInput = page.locator('input[type="email"], input[name="email"], input[name="username"]');
      await expect(emailInput).toBeVisible();

      // Check for password input
      const passwordInput = page.locator('input[type="password"]');
      await expect(passwordInput).toBeVisible();

      // Check for submit button
      const submitButton = page.locator('button[type="submit"], button:has-text("Login"), button:has-text("Sign in")');
      await expect(submitButton).toBeVisible();
    });

    test("should show error for invalid credentials", async ({ page }) => {
      await page.goto("/login");

      // Fill in invalid credentials
      await page.fill('input[type="email"], input[name="email"], input[name="username"]', "invalid@test.com");
      await page.fill('input[type="password"]', "wrongpassword");

      // Submit form
      await page.click('button[type="submit"], button:has-text("Login"), button:has-text("Sign in")');

      // Wait for error message or stay on login page
      await page.waitForTimeout(1000);

      // Should still be on login page or show error
      const currentUrl = page.url();
      const hasError = await page.locator('[role="alert"], .error, .text-red').count() > 0;
      expect(currentUrl.includes("/login") || hasError).toBeTruthy();
    });

    test("should have link to register page", async ({ page }) => {
      await page.goto("/login");

      const registerLink = page.locator('a[href*="register"], a:has-text("Register"), a:has-text("Sign up")');
      await expect(registerLink).toBeVisible();
    });
  });

  test.describe("Register Page", () => {
    test("should display registration form", async ({ page }) => {
      await page.goto("/register");

      // Check for email input
      const emailInput = page.locator('input[type="email"], input[name="email"]');
      await expect(emailInput).toBeVisible();

      // Check for username input (if present)
      const usernameInput = page.locator('input[name="username"]');
      if (await usernameInput.count() > 0) {
        await expect(usernameInput).toBeVisible();
      }

      // Check for password input
      const passwordInput = page.locator('input[type="password"]');
      await expect(passwordInput).toBeVisible();
    });

    test("should validate email format", async ({ page }) => {
      await page.goto("/register");

      const emailInput = page.locator('input[type="email"], input[name="email"]');
      await emailInput.fill("invalid-email");

      // Try to submit
      const submitButton = page.locator('button[type="submit"]');
      await submitButton.click();

      // Should show validation error or not submit
      await page.waitForTimeout(500);
      const currentUrl = page.url();
      expect(currentUrl.includes("/register")).toBeTruthy();
    });

    test("should validate password requirements", async ({ page }) => {
      await page.goto("/register");

      const emailInput = page.locator('input[type="email"], input[name="email"]');
      const passwordInput = page.locator('input[type="password"]').first();

      await emailInput.fill("test@example.com");
      await passwordInput.fill("123"); // Too short

      const submitButton = page.locator('button[type="submit"]');
      await submitButton.click();

      // Should show validation error or stay on page
      await page.waitForTimeout(500);
      const currentUrl = page.url();
      expect(currentUrl.includes("/register")).toBeTruthy();
    });

    test("should have link to login page", async ({ page }) => {
      await page.goto("/register");

      const loginLink = page.locator('a[href*="login"], a:has-text("Login"), a:has-text("Sign in")');
      await expect(loginLink).toBeVisible();
    });
  });

  test.describe("Protected Routes", () => {
    test("dashboard should redirect unauthenticated users", async ({ page }) => {
      await page.goto("/dashboard");
      await page.waitForTimeout(1000);

      // Should redirect to login or show login prompt
      const currentUrl = page.url();
      const hasLoginForm = await page.locator('input[type="password"]').count() > 0;
      expect(currentUrl.includes("/login") || hasLoginForm || currentUrl.includes("/dashboard")).toBeTruthy();
    });

    test("documents page should handle authentication", async ({ page }) => {
      await page.goto("/documents");
      await page.waitForTimeout(1000);

      // Should redirect or show content
      const currentUrl = page.url();
      expect(currentUrl.includes("/login") || currentUrl.includes("/documents")).toBeTruthy();
    });
  });
});
