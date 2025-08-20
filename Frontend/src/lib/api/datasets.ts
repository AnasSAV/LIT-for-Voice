export type Dataset = {
  id: string;
  name: string;
  available: boolean;
  path: string;
};

export type DatasetFile = {
  id: string;
  filename: string;
  relpath: string;
  size: number;
  duration: number;
  sample_rate?: number;
  label: string;
  h: string;
  meta?: Record<string, string>;
};

// Avoid `any` by using a typed cast to an object with only the env we need
const envBase = (import.meta as unknown as { env?: { VITE_API_BASE?: string } }).env?.VITE_API_BASE;
export const API_BASE = envBase ?? "http://localhost:8000";

export async function listDatasets(): Promise<Dataset[]> {
  const res = await fetch(`${API_BASE}/datasets`, { credentials: "include" });
  if (!res.ok) throw new Error(`Failed to list datasets: ${res.status}`);
  const data = await res.json();
  return data.datasets as Dataset[];
}

export async function getActiveDataset(): Promise<string | null> {
  const res = await fetch(`${API_BASE}/datasets/active`, { credentials: "include" });
  if (!res.ok) throw new Error(`Failed to get active dataset: ${res.status}`);
  const data = await res.json();
  return (data.active as string | null) ?? null;
}

export async function setActiveDataset(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/datasets/select`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ id }),
  });
  if (!res.ok) {
    let detail = "";
    try {
      const j = await res.json();
      detail = j?.error || JSON.stringify(j);
    } catch (_e) {
      // Ignore JSON parse errors when the body isn't JSON
      void _e;
    }
    throw new Error(`Failed to set active dataset: ${res.status} ${detail}`);
  }
}

export async function listDatasetFiles(limit = 100, offset = 0): Promise<{ total: number; files: DatasetFile[]; active: string | null }>{
  const url = new URL(`${API_BASE}/datasets/files`);
  url.searchParams.set("limit", String(limit));
  url.searchParams.set("offset", String(offset));
  const res = await fetch(url, { credentials: "include" });
  if (!res.ok) throw new Error(`Failed to list dataset files: ${res.status}`);
  const data = await res.json();
  return { total: data.total as number, files: data.files as DatasetFile[], active: data.active as string | null };
}

export function datasetFileUrl(relpath: string, id?: string): string {
  const url = new URL(`${API_BASE}/datasets/file`);
  url.searchParams.set("relpath", relpath);
  if (id) url.searchParams.set("id", id);
  return url.toString();
}
