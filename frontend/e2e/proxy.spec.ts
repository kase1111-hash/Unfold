import http from "node:http";
import type { AddressInfo } from "node:net";
import { test, expect } from "@playwright/test";

// The same normalized backend URL the dev server was started with (Playwright
// starts it with this process's environment)
const nextConfig = require("../next.config.js") as { env: { NEXT_PUBLIC_API_URL: string } };

test.describe("Same-origin API proxy", () => {
  test("/api/v1/<path> reaches the backend's /api/v1/<path>", async ({ request }) => {
    const apiUrl = new URL(nextConfig.env.NEXT_PUBLIC_API_URL);
    test.skip(
      !["localhost", "127.0.0.1"].includes(apiUrl.hostname),
      `NEXT_PUBLIC_API_URL (${apiUrl}) is not a local backend`
    );

    // A stand-in backend on the configured port
    const seen: string[] = [];
    const backend = http.createServer((req, res) => {
      seen.push(req.url ?? "");
      const ok = req.url === "/api/v1/health";
      res.writeHead(ok ? 200 : 404, { "content-type": "application/json" });
      res.end(JSON.stringify(ok ? { status: "healthy" } : { detail: "Not Found" }));
    });
    const listening = await new Promise<boolean>((resolve) => {
      backend.once("error", () => resolve(false));
      // All interfaces: "localhost" may resolve to ::1 or 127.0.0.1 for the proxy
      backend.listen(Number(apiUrl.port || 80), () => resolve(true));
    });
    test.skip(!listening, `port ${apiUrl.port} is in use (a real backend is running?)`);

    try {
      expect((backend.address() as AddressInfo).port).toBe(Number(apiUrl.port || 80));
      const response = await request.get("/api/v1/health");

      expect(seen).toEqual(["/api/v1/health"]);
      expect(response.status()).toBe(200);
      expect(await response.json()).toEqual({ status: "healthy" });
    } finally {
      await new Promise((resolve) => backend.close(resolve));
    }
  });

  test("the frontend's own /api/health is not proxied", async ({ request }) => {
    const response = await request.get("/api/health");

    expect(response.status()).toBe(200);
    expect(await response.json()).toMatchObject({ status: "healthy", service: "unfold-frontend" });
  });
});
