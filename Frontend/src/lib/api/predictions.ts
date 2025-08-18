import { API_BASE } from './datasets';

export interface PredictionResult {
  // Add prediction result fields based on your API response
  // Example:
  // prediction: string;
  // confidence: number;
  // embeddings: number[];
  [key: string]: unknown;
}

export interface BatchPredictionResult {
  results: Record<string, PredictionResult | null>;
}

export async function getBatchPredictions(
  model: string,
  hashes: string[]
): Promise<BatchPredictionResult> {
  const response = await fetch(`${API_BASE}/results/${model}/batch`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    credentials: 'include',
    body: JSON.stringify({ hashes }),
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch batch predictions: ${response.statusText}`);
  }

  // Backend returns: { ok: boolean, payloads: Record<string, any | null> }
  // Transform to frontend shape: { results: Record<string, PredictionResult | null> }
  const raw = await response.json();
  const payloads = (raw && typeof raw === 'object' && raw.payloads) ? raw.payloads : {};
  return { results: payloads } as BatchPredictionResult;
}
