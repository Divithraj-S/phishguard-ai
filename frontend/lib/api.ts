/**
 * API client helper for PhishGuard AI frontend.
 * Provides strongly-typed functions for health checks and email prediction.
 */

export interface PredictionPayload {
  email_text: string;
  threshold?: number;
}

export interface PredictionResponse {
  prediction_label: "Phishing Email" | "Safe Email" | string;
  prediction_int: 0 | 1;
  phishing_probability: number;
  safe_probability: number;
  decision_threshold: number;
  security_disclaimer: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_path: string;
  default_threshold: number;
  model_name: string;
}

/**
 * Resolve API endpoint URL supporting both local development and Vercel serverless.
 * Defensively handles whether NEXT_PUBLIC_API_URL includes or omits the trailing '/api'.
 */
export function getApiUrl(endpoint: string): string {
  const cleanEndpoint = endpoint.replace(/^\//, "");
  const rawBase = process.env.NEXT_PUBLIC_API_URL?.trim().replace(/\/+$/, "");

  if (rawBase) {
    // Avoid double '/api/api' if the user configured NEXT_PUBLIC_API_URL with a trailing '/api'
    const baseWithApi = rawBase.endsWith("/api") ? rawBase : `${rawBase}/api`;
    return `${baseWithApi}/${cleanEndpoint}`;
  }

  // When deployed on Vercel using Next.js proxy rewrites or relative paths
  return `/api/${cleanEndpoint}`;
}

/**
 * Check backend health and model loading status.
 */
export async function checkBackendHealth(): Promise<HealthResponse> {
  const url = getApiUrl("health");
  const res = await fetch(url, {
    method: "GET",
    headers: {
      Accept: "application/json",
    },
    cache: "no-store",
  });

  if (!res.ok) {
    throw new Error(`Health check failed with status ${res.status}`);
  }

  return res.json();
}

/**
 * Submit email text for classification with an optional decision threshold.
 */
export async function predictEmail(
  payload: PredictionPayload
): Promise<PredictionResponse> {
  const url = getApiUrl("predict");

  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify({
      email_text: payload.email_text,
      threshold: payload.threshold ?? 0.60,
    }),
  });

  if (!res.ok) {
    let errorDetail = "Failed to evaluate email.";
    try {
      const errJson = await res.json();
      if (errJson.detail) {
        if (typeof errJson.detail === "string") {
          errorDetail = errJson.detail;
        } else if (Array.isArray(errJson.detail)) {
          errorDetail = errJson.detail.map((d: any) => d.msg).join(", ");
        }
      }
    } catch {
      errorDetail = `Server responded with status code ${res.status} (${res.statusText})`;
    }
    throw new Error(errorDetail);
  }

  return res.json();
}
