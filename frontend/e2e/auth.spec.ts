import {
  test,
  expect,
  appAlerts,
  toasts,
  USER,
  RATE_LIMITED,
  TOKEN_EXPIRED,
  UNHANDLED_500,
} from "./fixtures";

test.describe("Login page", () => {
  test.use({ authenticated: false });

  test("shows the sign-in form", async ({ page }) => {
    await page.goto("/login");

    await expect(page.getByRole("heading", { name: "Welcome Back" })).toBeVisible();
    await expect(page.getByLabel("Email")).toBeVisible();
    await expect(page.getByLabel("Password")).toBeVisible();
    await expect(page.getByRole("button", { name: "Sign In" })).toBeVisible();
    // The page links to registration and no longer to a missing /forgot-password page
    await expect(page.getByRole("link", { name: "Sign up" })).toHaveAttribute("href", "/register");
    await expect(page.locator('a[href="/forgot-password"]')).toHaveCount(0);
  });

  test("validates required fields without calling the API", async ({ page, api }) => {
    await page.goto("/login");
    await page.getByRole("button", { name: "Sign In" }).click();

    await expect(page.getByText("Email is required")).toBeVisible();
    await expect(page.getByText("Password is required")).toBeVisible();
    expect(api.callsTo("POST", "/auth/login")).toHaveLength(0);
  });

  test("shows the backend message for invalid credentials and stays on /login", async ({
    page,
    api,
  }) => {
    api.on("POST", "/auth/login", {
      status: 401,
      body: { detail: { code: "INVALID_CREDENTIALS", message: "Invalid email or password" } },
    });
    await page.goto("/login");
    await page.getByLabel("Email").fill("alice@example.com");
    await page.getByLabel("Password").fill("wrong-password");
    await page.getByRole("button", { name: "Sign In" }).click();

    await expect(page.getByText("Invalid email or password")).toBeVisible();
    await expect(page).toHaveURL(/\/login$/);
    expect(api.callsTo("POST", "/auth/login")[0].body).toEqual({
      email: "alice@example.com",
      password: "wrong-password",
    });
    // A failed login must not trigger a token refresh
    expect(api.callsTo("POST", "/auth/refresh")).toHaveLength(0);
  });

  test("signs in and lands on the dashboard", async ({ page, api }) => {
    api.on("POST", "/auth/login", {
      body: { user: USER, access_token: "fresh-token", token_type: "bearer", expires_in: 1800 },
    });
    await page.goto("/login");
    await page.getByLabel("Email").fill("alice@example.com");
    await page.getByLabel("Password").fill("Correct-horse-1");
    await page.getByRole("button", { name: "Sign In" }).click();

    await expect(page).toHaveURL(/\/dashboard$/);
    await expect(page.getByRole("heading", { name: "Welcome back, Alice Example!" })).toBeVisible();
    expect(await page.evaluate(() => localStorage.getItem("access_token"))).toBe("fresh-token");
    expect(api.callsTo("GET", "/documents/")[0].headers["authorization"]).toBe("Bearer fresh-token");
  });
});

test.describe("Register page", () => {
  test.use({ authenticated: false });

  test("shows the registration form", async ({ page }) => {
    await page.goto("/register");

    await expect(page.getByRole("heading", { name: "Create Your Account" })).toBeVisible();
    await expect(page.getByLabel("Email")).toBeVisible();
    await expect(page.getByLabel("Username")).toBeVisible();
    await expect(page.getByLabel("Password", { exact: true })).toBeVisible();
    await expect(page.getByLabel("Confirm Password")).toBeVisible();
    await expect(page.getByRole("link", { name: "Sign in" })).toHaveAttribute("href", "/login");
    // Terms and privacy are plain text: those pages do not exist
    await expect(page.getByText("I agree to the Terms of Service and Privacy Policy")).toBeVisible();
    await expect(page.locator('a[href="/terms"], a[href="/privacy"]')).toHaveCount(0);
  });

  test("validates the form without calling the API", async ({ page, api }) => {
    await page.goto("/register");
    // Valid for the browser's type="email" check, invalid for the app's rule
    await page.getByLabel("Email").fill("alice@example");
    await page.getByLabel("Username").fill("al");
    await page.getByLabel("Password", { exact: true }).fill("123");
    await page.getByLabel("Confirm Password").fill("456");
    await page.getByRole("button", { name: "Create Account" }).click();

    await expect(page.getByText("Invalid email address")).toBeVisible();
    await expect(page.getByText("Username must be at least 3 characters")).toBeVisible();
    await expect(page.getByText("Password must be at least 8 characters")).toBeVisible();
    await expect(page.getByText("Passwords do not match")).toBeVisible();
    await expect(page.getByText("You must agree to the terms and conditions")).toBeVisible();
    await expect(page).toHaveURL(/\/register$/);
    expect(api.callsTo("POST", "/auth/register")).toHaveLength(0);
  });

  test("creates an account and lands on the dashboard", async ({ page, api }) => {
    api.on("POST", "/auth/register", {
      status: 201,
      body: { user: USER, access_token: "new-user-token", token_type: "bearer", expires_in: 1800 },
    });
    await page.goto("/register");
    await page.getByLabel("Email").fill("alice@example.com");
    await page.getByLabel("Username").fill("alice");
    await page.getByLabel("Password", { exact: true }).fill("Correct-horse-1");
    await page.getByLabel("Confirm Password").fill("Correct-horse-1");
    await page.getByRole("checkbox").check();
    await page.getByRole("button", { name: "Create Account" }).click();

    await expect(page).toHaveURL(/\/dashboard$/);
    expect(api.callsTo("POST", "/auth/register")[0].body).toEqual({
      email: "alice@example.com",
      username: "alice",
      password: "Correct-horse-1",
    });
  });
});

test.describe("Protected routes", () => {
  test.describe("without a token", () => {
    test.use({ authenticated: false });

    for (const path of ["/dashboard", "/documents", "/graph", "/flashcards", "/read/doc-1"]) {
      test(`${path} redirects to /login without loading data`, async ({ page, api }) => {
        await page.goto(path);

        await expect(page).toHaveURL(/\/login$/);
        await expect(page.getByRole("heading", { name: "Welcome Back" })).toBeVisible();
        // Never signed in on this browser: no refresh probe (it would only use
        // up the auth rate limit) and protected pages fire no requests
        expect(api.calls.map((c) => `${c.method} ${c.path}`)).toEqual([]);
      });
    }

    test("a valid refresh cookie restores the session without signing in", async ({
      page,
      api,
    }) => {
      api.on("POST", "/auth/refresh", {
        body: { access_token: "restored-token", token_type: "bearer", expires_in: 1800 },
      });
      // Signed in before on this browser (the store remembers the user), but
      // the access token is gone
      await page.addInitScript((user) => {
        window.localStorage.setItem(
          "auth-storage",
          JSON.stringify({ state: { user }, version: 0 })
        );
      }, USER);
      await page.goto("/documents");

      await expect(page.getByRole("heading", { level: 1, name: "Documents" })).toBeVisible();
      await expect(page).toHaveURL(/\/documents$/);
      expect(api.callsTo("POST", "/auth/refresh")).toHaveLength(1);
      expect(api.callsTo("GET", "/auth/me")[0].headers["authorization"]).toBe(
        "Bearer restored-token"
      );
      expect(await page.evaluate(() => localStorage.getItem("access_token"))).toBe(
        "restored-token"
      );
    });
  });

  for (const [name, failure, message] of [
    ["rate limited (429)", RATE_LIMITED, "Too many requests. Please try again later."],
    [
      "a server error (500)",
      UNHANDLED_500,
      "Could not reach the server, or it ran into an error. Please try again.",
    ],
  ] as const) {
    test(`a session check that is ${name} keeps the session and can be retried`, async ({
      page,
      api,
    }) => {
      let fail = true;
      api.on("GET", "/auth/me", () => (fail ? failure : { body: USER }));
      await page.goto("/dashboard");

      const alert = appAlerts(page);
      await expect(alert.getByRole("heading", { name: "Could not check your session" })).toBeVisible();
      await expect(alert).toContainText(message);
      await expect(page).toHaveURL(/\/dashboard$/);
      expect(await page.evaluate(() => localStorage.getItem("access_token"))).toBe("test-token");

      fail = false;
      await alert.getByRole("button", { name: "Retry" }).click();

      await expect(page.getByRole("heading", { name: "Welcome back, Alice Example!" })).toBeVisible();
      await expect(page).toHaveURL(/\/dashboard$/);
    });
  }

  test("an expired token whose refresh is rate limited does not sign the user out", async ({
    page,
    api,
  }) => {
    api.on("GET", "/auth/me", (call) =>
      call.headers["authorization"] === "Bearer refreshed-token" ? { body: USER } : TOKEN_EXPIRED
    );
    let refreshFails = true;
    api.on("POST", "/auth/refresh", () =>
      refreshFails
        ? RATE_LIMITED
        : { body: { access_token: "refreshed-token", token_type: "bearer", expires_in: 1800 } }
    );
    await page.goto("/dashboard");

    await expect(appAlerts(page)).toContainText("Too many requests. Please try again later.");
    await expect(page).toHaveURL(/\/dashboard$/);
    expect(await page.evaluate(() => localStorage.getItem("access_token"))).toBe("test-token");

    refreshFails = false;
    await appAlerts(page).getByRole("button", { name: "Retry" }).click();

    await expect(page.getByRole("heading", { name: "Welcome back, Alice Example!" })).toBeVisible();
    expect(await page.evaluate(() => localStorage.getItem("access_token"))).toBe("refreshed-token");
  });

  test("an expired session that cannot be refreshed redirects to /login", async ({
    page,
    api,
  }) => {
    api.on("GET", "/auth/me", TOKEN_EXPIRED);
    await page.goto("/dashboard");

    await expect(page).toHaveURL(/\/login$/);
    expect(await page.evaluate(() => localStorage.getItem("access_token"))).toBeNull();
    expect(api.callsTo("POST", "/auth/refresh").length).toBeGreaterThanOrEqual(1);
  });

  test("concurrent 401s share a single token refresh", async ({ page, api }) => {
    const isFresh = (headers: Record<string, string>) =>
      headers["authorization"] === "Bearer refreshed-token";
    // /auth/me still accepts the old token; the page's own requests do not
    for (const path of [
      "/learning/flashcards/due",
      "/learning/flashcards/stats",
      "/learning/engagement/profile",
    ]) {
      const original = path;
      api.on("GET", original, (call) => {
        if (!isFresh(call.headers)) return TOKEN_EXPIRED;
        if (original.endsWith("/due")) {
          return { body: { due_cards: api.data.dueCards, total_due: api.data.dueCards.length } };
        }
        if (original.endsWith("/stats")) return { body: api.data.stats };
        return { body: api.data.engagement };
      });
    }
    api.on("POST", "/auth/refresh", {
      delayMs: 500,
      body: { access_token: "refreshed-token", token_type: "bearer", expires_in: 1800 },
    });

    await page.goto("/flashcards");

    await expect(page.getByText("Which element did Marie Curie name after Poland?")).toBeVisible();
    await expect(page.getByText("Flashcard Progress")).toBeVisible();
    expect(api.callsTo("POST", "/auth/refresh")).toHaveLength(1);
    expect(await page.evaluate(() => localStorage.getItem("access_token"))).toBe("refreshed-token");
  });

  test("a session that expires mid-use redirects to /login on the next request", async ({
    page,
    api,
  }) => {
    await page.goto("/dashboard");
    await expect(page.getByRole("heading", { name: "Welcome back, Alice Example!" })).toBeVisible();
    // Wait for the dashboard's own requests to finish before expiring the session
    await expect(page.getByTestId("stat-card").filter({ hasText: "Flashcards" })).toContainText("5");

    // From now on every request is rejected and the refresh cookie is gone
    api.on("GET", /^\/learning\//, TOKEN_EXPIRED);
    await page.getByRole("navigation").getByRole("link", { name: "Flashcards" }).click();

    await expect(page).toHaveURL(/\/login$/);
    expect(await page.evaluate(() => localStorage.getItem("access_token"))).toBeNull();
  });

  test("an expired session says so instead of showing the refresh error", async ({
    page,
    api,
  }) => {
    await page.goto("/dashboard");
    await expect(page.getByTestId("stat-card").filter({ hasText: "Flashcards" })).toContainText("5");

    api.on("GET", "/documents/", TOKEN_EXPIRED);
    await page.getByRole("navigation").getByRole("link", { name: "Documents" }).click();

    await expect(page).toHaveURL(/\/login$/);
    // (In development React loads the page twice, hence possibly two toasts)
    await expect(
      toasts(page).filter({ hasText: "Your session has expired. Please sign in again." }).first()
    ).toBeVisible();
    await expect(page.getByText("Refresh token required")).toHaveCount(0);
  });

  test("signing out clears the session and returns to /login", async ({ page, api }) => {
    await page.goto("/dashboard");
    await expect(page.getByText("alice@example.com")).toBeVisible();

    await page.getByRole("button", { name: "Sign Out" }).click();

    await expect(page).toHaveURL(/\/login$/);
    expect(api.callsTo("POST", "/auth/logout")).toHaveLength(1);
    expect(await page.evaluate(() => localStorage.getItem("access_token"))).toBeNull();
  });
});
