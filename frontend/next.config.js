/** @type {import('next').NextConfig} */

// Same rule as normalizeApiUrl in src/services/api.ts: NEXT_PUBLIC_API_URL may be
// given with or without the "/api/v1" prefix.
function normalizeApiUrl(raw) {
  const base = (raw || 'http://localhost:8000').trim().replace(/\/+$/, '');
  return base.endsWith('/api/v1') ? base : `${base}/api/v1`;
}

const API_URL = normalizeApiUrl(process.env.NEXT_PUBLIC_API_URL);

const nextConfig = {
  reactStrictMode: true,

  // Enable standalone output for Docker production builds
  output: 'standalone',

  // API proxy to backend
  async rewrites() {
    // Only for an absolute backend URL: a same-origin "/api/v1" (nginx in front)
    // would rewrite /api/* onto itself.
    if (!/^https?:\/\//.test(API_URL)) return [];
    return [
      {
        source: '/api/:path*',
        destination: `${API_URL}/:path*`,
      },
    ];
  },

  // Environment variables
  env: {
    NEXT_PUBLIC_API_URL: API_URL,
  },
};

module.exports = nextConfig;
