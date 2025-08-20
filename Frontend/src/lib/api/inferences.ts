import { API_BASE } from './datasets';

export type RunInferenceParams = {
  ds_id?: string | null;
  h?: string;
  file_path?: string;
};

// Simple concurrency limiter (max 2 concurrent requests)
const MAX_CONCURRENCY = 2;
let active = 0;
const queue: Array<() => void> = [];

function acquire(): Promise<void> {
  return new Promise((resolve) => {
    const tryStart = () => {
      if (active < MAX_CONCURRENCY) {
        active += 1;
        resolve();
      } else {
        queue.push(tryStart);
      }
    };
    tryStart();
  });
}

function release() {
  active = Math.max(0, active - 1);
  const next = queue.shift();
  if (next) next();
}

export async function runInference(model: string, params: RunInferenceParams): Promise<any> {
  await acquire();
  try {
    const url = new URL(`${API_BASE}/inferences/run`);
    url.searchParams.set('model', model);
    if (params.ds_id) url.searchParams.set('ds_id', String(params.ds_id));
    if (params.h) url.searchParams.set('h', String(params.h));
    if (params.file_path) url.searchParams.set('file_path', String(params.file_path));

    const res = await fetch(url.toString(), { credentials: 'include' });
    if (!res.ok) {
      let detail = '';
      try {
        const j = await res.json();
        detail = j?.detail || JSON.stringify(j);
      } catch {
        // ignore parse error
      }
      throw new Error(`Inference failed: ${res.status} ${detail}`);
    }
    return await res.json();
  } finally {
    release();
  }
}
