import { API_BASE } from './datasets';

export type RunInferenceParams = {
  ds_id?: string | null;
  h?: string;
  file_path?: string;
  signal?: AbortSignal;
};

export interface InferenceResponse {
  label?: string;
  confidence?: number;
  probs?: Record<string, number>;
  text?: string;
  [key: string]: unknown;
}

export type FlushScope = 'all' | 'whisper-base' | 'whisper-large' | 'whisper-large-v3' | 'wav2vec2';
export type FlushResponse =
  | { ok: boolean; scope: 'all'; summary: { asr_removed: number; emotion_removed: boolean } }
  | { ok: boolean; scope: 'whisper-base' | 'whisper-large' | 'whisper-large-v3'; asr_removed: number }
  | { ok: boolean; scope: 'wav2vec2'; emotion_removed: boolean };

export type CacheStatus = {
  device: string;
  asr_loaded: string[];
  emotion_loaded: boolean;
  total_loaded: number;
};

// -------------------------------
// Global loading event utilities
// -------------------------------
type LoadingCounts = Record<string, number>;
const inflight: LoadingCounts = {};

function inc(model: string) {
  inflight[model] = (inflight[model] ?? 0) + 1;
  dispatchLoading(model);
}

function dec(model: string) {
  inflight[model] = Math.max(0, (inflight[model] ?? 0) - 1);
  dispatchLoading(model);
}

function dispatchLoading(model: string) {
  const count = inflight[model] ?? 0;
  const isLoading = count > 0;
  try {
    window.dispatchEvent(
      new CustomEvent('inference:loading', { detail: { model, count, isLoading } })
    );
  } catch (_e) { /* noop for non-browser envs */ }
}

// Per-model concurrency limiter
type Limiter = { active: number; queue: Array<() => void>; max: number };
const limiters: Record<string, Limiter> = {};

function getMaxForModel(model: string): number {
  // Whisper large can be very memory-heavy; run strictly serially
  if (model === 'whisper-large' || model === 'whisper-large-v3' || model === 'openai/whisper-large-v3') return 1;
  // Default: allow small parallelism
  return 2;
}

function getLimiter(model: string): Limiter {
  if (!limiters[model]) {
    limiters[model] = { active: 0, queue: [], max: getMaxForModel(model) };
  }
  return limiters[model];
}

function acquire(model: string): Promise<void> {
  const limiter = getLimiter(model);
  return new Promise((resolve) => {
    const tryStart = () => {
      if (limiter.active < limiter.max) {
        limiter.active += 1;
        resolve();
      } else {
        limiter.queue.push(tryStart);
      }
    };
    tryStart();
  });
}

function release(model: string) {
  const limiter = getLimiter(model);
  limiter.active = Math.max(0, limiter.active - 1);
  const next = limiter.queue.shift();
  if (next) next();
}

export async function runInference(model: string, params: RunInferenceParams): Promise<InferenceResponse | string> {
  await acquire(model);
  try {
    const url = new URL(`${API_BASE}/inferences/run`);
    url.searchParams.set('model', model);
    if (params.ds_id) url.searchParams.set('ds_id', String(params.ds_id));
    if (params.h) url.searchParams.set('h', String(params.h));
    if (params.file_path) url.searchParams.set('file_path', String(params.file_path));

    // Lightweight debug log for visibility during development
    console.log('[inferences] runInference', { model, params });

    // Mark as actively loading for this model
    inc(model);

    const res = await fetch(url.toString(), { credentials: 'include', signal: params.signal });
    if (!res.ok) {
      let detail = '';
      try {
        const j: unknown = await res.json();
        if (typeof j === 'object' && j !== null && 'detail' in j && typeof (j as { detail?: unknown }).detail === 'string') {
          detail = String((j as { detail?: string }).detail);
        } else {
          detail = JSON.stringify(j);
        }
      } catch {
        // ignore parse error
      }
      throw new Error(`Inference failed: ${res.status} ${detail}`);
    }
    return await res.json();
  } finally {
    dec(model);
    release(model);
  }
}

export async function flushModels(model?: FlushScope): Promise<FlushResponse> {
  const payload = { model: model ?? 'all' };
  const res = await fetch(`${API_BASE}/inferences/flush`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let detail = '';
    try {
      const j = await res.json();
      detail = j?.detail || JSON.stringify(j);
    } catch (_e) { void _e; /* ignore parse error */ }
    throw new Error(`Flush failed: ${res.status} ${detail}`);
  }
  return await res.json() as FlushResponse;
}

export async function getCacheStatus(): Promise<CacheStatus> {
  const res = await fetch(`${API_BASE}/inferences/cache_status`, { credentials: 'include' });
  if (!res.ok) throw new Error(`Failed to get cache status: ${res.status}`);
  const data = await res.json();
  return {
    device: String(data.device ?? ''),
    asr_loaded: Array.isArray(data.asr_loaded) ? data.asr_loaded.map(String) : [],
    emotion_loaded: Boolean(data.emotion_loaded),
    total_loaded: Number(data.total_loaded ?? 0),
  } as CacheStatus;
}
