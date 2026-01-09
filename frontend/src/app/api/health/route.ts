/**
 * Frontend Health Check Endpoint
 * Used by Docker and load balancers to verify the frontend is running
 */

import { NextResponse } from 'next/server';

export async function GET() {
  return NextResponse.json(
    {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      service: 'unfold-frontend',
    },
    { status: 200 }
  );
}
