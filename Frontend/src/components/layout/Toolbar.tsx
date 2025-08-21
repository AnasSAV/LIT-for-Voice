import React, { useEffect, useState, useRef } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Settings, Upload, Download, Pin, Filter, Loader2, AlertTriangle } from "lucide-react";
import { listDatasets, getActiveDataset, setActiveDataset } from "@/lib/api/datasets";
import { flushModels, getCacheStatus, runInference } from "@/lib/api/inferences";
import type { CacheStatus, InferenceResponse } from "@/lib/api/inferences";

interface UploadedFile {
  file_id: string;
  filename: string;
  file_path: string;
  message: string;
  size?: number;
  duration?: number;
  sample_rate?: number;
}

// Define a more specific type for the API response if possible
// For now, we'll use a generic type that matches the expected structure
interface ApiData {
  transcription?: string;
  error?: string;
  // Add other expected fields from your API response
  [key: string]: unknown; // This is needed for dynamic properties
}

interface DatasetInfo {
  id: string;
  available: boolean;
  [key: string]: unknown;
}

interface ToolbarProps {
  apiData: ApiData | null;
  setApiData: (data: ApiData | null) => void;
  selectedFile?: UploadedFile | null;
  uploadedFiles?: UploadedFile[];
  onFileSelect?: (file: UploadedFile) => void;
  model: string;
  setModel: (model: string) => void;
}

const modelDatasetMap: Record<string, string[]> = {
  "whisper-tiny": ["common-voice", "custom"],
  "whisper-base": ["common-voice", "custom"],
  "whisper-large": ["common-voice", "custom"],
  "wav2vec2": ["ravdess", "custom"],
};

export const Toolbar = ({
  apiData,
  setApiData,
  selectedFile,
  uploadedFiles = [],
  onFileSelect,
  model,
  setModel,
}: ToolbarProps) => {
  // Dataset UI value is one of: "ravdess" | "common-voice" | "custom"
  const [dataset, setDataset] = useState(() => {
    const allowed = (model in modelDatasetMap ? modelDatasetMap[model] : ["custom"]) as string[];
    return allowed[0] ?? "custom";
  });
  const [hasRavdessSubset, setHasRavdessSubset] = useState(false);
  const [hasCommonVoice, setHasCommonVoice] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const [cacheStatus, setCacheStatus] = useState<CacheStatus | null>(null);
  const [isFlushing, setIsFlushing] = useState(false);
  const [reloadIn, setReloadIn] = useState<number>(0);
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [loadingCounts, setLoadingCounts] = useState<Record<string, number>>({});
  const [isAnyLoading, setIsAnyLoading] = useState(false);

  // Load available datasets on component mount
  useEffect(() => {
    const loadDatasets = async () => {
      try {
        const datasets = await listDatasets() as DatasetInfo[];
        const hasSubset = !!datasets.find(d => d.id === "ravdess_subset" && d.available);
        const hasCv = !!datasets.find(d => d.id === "common_voice_en_dev" && d.available);
        setHasRavdessSubset(hasSubset);
        setHasCommonVoice(hasCv);

        const active = await getActiveDataset();
        const activeUi = active === "ravdess_subset"
          ? "ravdess"
          : active === "common_voice_en_dev"
            ? "common-voice"
            : active
              ? "custom"
              : null;

        const allowed = (model in modelDatasetMap ? modelDatasetMap[model] : ["custom"]) as string[];
        if (activeUi && allowed.includes(activeUi)) {
          setDataset(activeUi);
          // Broadcast current dataset to listeners
          window.dispatchEvent(new CustomEvent("dataset-changed", { detail: { uiDataset: activeUi } }));
        } else {
          // Active dataset not compatible with current model; switch to a valid default
          const target = allowed[0] ?? "custom";
          setDataset(target);
          try {
            if (target === "ravdess") {
              if (hasSubset) {
                await setActiveDataset("ravdess_subset");
              }
            } else if (target === "common-voice") {
              if (hasCv) await setActiveDataset("common_voice_en_dev");
            } else {
              // custom: leave backend active as-is
            }
            window.dispatchEvent(new CustomEvent("dataset-changed", { detail: { uiDataset: target } }));
          } catch (e) {
            console.error("Failed to set dataset on init", e);
          }
        }
      } catch (e) {
        console.warn("Dataset API not available", e);
      }
    };

    loadDatasets();
  }, [model]);

  // Countdown for delayed reload indicator
  useEffect(() => {
    if (reloadIn <= 0) return;
    const id = window.setInterval(() => {
      setReloadIn((s) => (s > 0 ? s - 1 : 0));
    }, 1000);
    return () => window.clearInterval(id);
  }, [reloadIn]);

  // Poll backend cache status to know how many models are resident
  useEffect(() => {
    let timer: number | undefined;
    let cancelled = false;
    const tick = async () => {
      try {
        const s = await getCacheStatus();
        if (!cancelled) setCacheStatus(s);
      } catch (e) {
        // ignore status errors in UI
        void e;
      } finally {
        if (!cancelled) {
          timer = window.setTimeout(tick, 4000);
        }
      }
    };
    tick();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, []);

  // Subscribe to global inference loading events to reflect true loading state
  useEffect(() => {
    const onLoading = (e: Event) => {
      const ce = e as CustomEvent<{ model: string; count: number; isLoading: boolean; anyCount?: number; anyLoading?: boolean }>;
      // Track current model's loading state for existing UX bits
      if (ce?.detail?.model === model) {
        setIsModelLoading(Boolean(ce.detail.isLoading));
      }
      // Track aggregate state and per-model counts to decide which models can be flushed
      if (typeof ce?.detail?.anyLoading === 'boolean') {
        setIsAnyLoading(ce.detail.anyLoading);
      }
      if (ce?.detail?.model) {
        const m = String(ce.detail.model);
        const c = Math.max(0, Number(ce.detail.count ?? 0));
        setLoadingCounts((prev) => ({ ...prev, [m]: c }));
      }
    };
    window.addEventListener("inference:loading", onLoading as EventListener);
    return () => window.removeEventListener("inference:loading", onLoading as EventListener);
  }, [model]);

  const onModelChange = async (value: string) => {
    // If a delayed reload countdown is active, avoid triggering loads
    if (reloadIn > 0) {
      console.log("Deferring model change actions during reload countdown (hiding timer)");
      // Hide the timer warning if user switches model during countdown
      setReloadIn(0);
      setModel(value);
      return;
    }
    setModel(value);
    console.log("Model selected:", value);
    // Ensure dataset matches the selected model
    const allowed = (value in modelDatasetMap ? modelDatasetMap[value] : ["custom"]) as string[];
    if (!allowed.includes(dataset)) {
      const target = allowed[0] ?? "custom";
      setDataset(target);
      try {
        if (target === "ravdess") {
          if (hasRavdessSubset) await setActiveDataset("ravdess_subset");
        } else if (target === "common-voice") {
          if (hasCommonVoice) await setActiveDataset("common_voice_en_dev");
        } else {
          // custom: leave backend active as-is
        }
        // Notify listeners
        window.dispatchEvent(new CustomEvent("dataset-changed", { detail: { uiDataset: target } }));
      } catch (e) {
        console.error("Failed to switch dataset for model", e);
      }
    }
    
    // Abort any ongoing requests
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    
    // Create a new abort controller for this request
    const controller = new AbortController();
    abortControllerRef.current = controller;
    
    try {
      // Reset API data when model changes
      setApiData(null);

      // If there's a selected file, run inference with the new model via backend /inferences/run
      if (selectedFile?.file_path) {
        const raw = await runInference(value, { file_path: selectedFile.file_path, signal: controller.signal });
        const data: ApiData = typeof raw === "string"
          ? { transcription: raw }
          : ((raw as InferenceResponse)?.text
              ? { transcription: String((raw as InferenceResponse).text), ...(raw as Record<string, unknown>) }
              : (raw as ApiData));
        setApiData(data);
        console.log("API response:", data);
      }
    } catch (error) {
      // Don't log aborted requests as errors
      if (error instanceof Error && error.name !== 'AbortError') {
        console.error("Failed to run inference:", error);
        // Optionally update UI to show error to user
      }
    }
  };

  const onDatasetChange = async (value: string) => {
    setDataset(value);
    console.log("Dataset selected:", value);
    
    try {
      if (value === "ravdess") {
        // Only use subset
        if (hasRavdessSubset) {
          await setActiveDataset("ravdess_subset");
        }
      } else if (value === "common-voice") {
        await setActiveDataset("common_voice_en_dev");
      } else if (value === "custom") {
        // Handle custom dataset selection
        console.log("Custom dataset selected");
      }
      
      // Notify listeners to reload dataset files for any change
      window.dispatchEvent(new CustomEvent("dataset-changed", { detail: { uiDataset: value } }));
    } catch (e) {
      console.error("Failed to set active dataset", e);
    }
  };

  // Get datasets allowed for current model with type safety
  const allowedDatasets = (model in modelDatasetMap ? modelDatasetMap[model] : ["custom"]) as string[];
  
  return (
    <div className="h-14 panel-header border-b panel-border px-4 flex items-center justify-between">
      {/* Left side: Model and Dataset selectors */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-foreground">LIT for Voice</span>
          <Badge variant="outline" className="text-xs">
            v1.0
          </Badge>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Model:</span>
            <Select value={model} onValueChange={onModelChange}>
              <SelectTrigger className="w-32 h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="whisper-tiny">Whisper Tiny</SelectItem>
                <SelectItem value="whisper-base">Whisper Base</SelectItem>
                <SelectItem value="whisper-large">Whisper Large</SelectItem>
                <SelectItem value="wav2vec2">Wav2Vec2</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Dataset:</span>
            <Select value={dataset} onValueChange={onDatasetChange}>
              <SelectTrigger className="w-32 h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {allowedDatasets.map((ds) => {
                  let label = ds;
                  if (ds === "common-voice") label = "Common Voice";
                  else if (ds === "ravdess") label = "RAVDESS";
                  else if (ds === "custom") label = "Custom";
                  return (
                    <SelectItem key={ds} value={ds}>
                      {label}
                    </SelectItem>
                  );
                })}
              </SelectContent>
            </Select>
          </div>

          {reloadIn > 0 && (
            <div
              className="flex items-center gap-2 rounded-full bg-amber-100 text-amber-900 border border-amber-300 px-3 py-1 shadow-sm animate-pulse"
              title="Models flushed. We will avoid auto-reloading briefly to free RAM/VRAM."
            >
              <AlertTriangle className="h-4 w-4" />
              <span className="text-xs font-semibold tracking-wide">
                Model will reload in {reloadIn}s
              </span>
            </div>
          )}

          {uploadedFiles && uploadedFiles.length > 0 && (
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">File:</span>
              <Select 
                value={selectedFile?.file_id || ""} 
                onValueChange={(fileId) => {
                  const file = (uploadedFiles || []).find((f: UploadedFile) => f.file_id === fileId);
                  if (file && onFileSelect) {
                    onFileSelect(file);
                  }
                }}
              >
                <SelectTrigger className="w-48 h-8">
                  <SelectValue placeholder="Select uploaded file" />
                </SelectTrigger>
                <SelectContent>
                  {uploadedFiles.map((file: UploadedFile) => (
                    <SelectItem key={file.file_id} value={file.file_id}>
                      {file.filename}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
        </div>
      </div>

      {/* Right side: Action buttons */}
      <div className="flex items-center gap-2">
        <Button variant="outline" size="sm" className="h-8">
          <Filter className="h-4 w-4 mr-2" />
          Filters
        </Button>

        <Button variant="outline" size="sm" className="h-8">
          <Pin className="h-4 w-4 mr-2" />
          Pin
        </Button>

        <Button variant="outline" size="sm" className="h-8">
          <Upload className="h-4 w-4 mr-2" />
          Upload
        </Button>

        <Button variant="outline" size="sm" className="h-8">
          <Download className="h-4 w-4 mr-2" />
          Export
        </Button>

        <Button 
          variant="outline" 
          size="sm" 
          className="h-8"
          onClick={async () => {
            try {
              const response = await fetch('http://localhost:8000/upload/test');
              const data = await response.json();
              console.log('Backend test:', data);
              alert(`Backend is ${response.ok ? 'working' : 'not working'}: ${JSON.stringify(data)}`);
            } catch (error) {
              const errorMessage = error instanceof Error ? error.message : 'Unknown error';
              console.error('Backend test failed:', error);
              alert(`Backend test failed: ${errorMessage}`);
            }
          }}
        >
          Test Backend
        </Button>

        {(() => {
          const loadedCount = cacheStatus?.total_loaded ?? 0;
          // Map backend-loaded model identifiers to UI model tokens
          const loadedTokens = (() => {
            const asrIds = cacheStatus?.asr_loaded ?? [];
            const tokens: string[] = [];
            for (const id of asrIds) {
              if (id.includes('whisper-large-v3') || id.includes('openai/whisper-large-v3')) tokens.push('whisper-large-v3');
              else if (id.includes('whisper-large')) tokens.push('whisper-large');
              else if (id.includes('whisper-tiny')) tokens.push('whisper-tiny');
              else if (id.includes('whisper-base')) tokens.push('whisper-base');
            }
            if (cacheStatus?.emotion_loaded) tokens.push('wav2vec2');
            return Array.from(new Set(tokens));
          })();
          // Determine which loaded models are currently busy
          const busyTokens = loadedTokens.filter(t => (loadingCounts[t] ?? 0) > 0);
          const idleTokens = loadedTokens.filter(t => (loadingCounts[t] ?? 0) === 0);
          const isPlural = loadedCount > 1;
          const baseLabel = `Flush Model${isPlural ? 's' : ''}`;
          const label = isFlushing ? 'Flushing…' : (isAnyLoading && idleTokens.length === 0) ? 'Busy' : baseLabel;
          // Enable flushing whenever there is at least one idle loaded model
          const disabled = isFlushing || loadedTokens.length === 0 || idleTokens.length === 0;
          const title = isFlushing
            ? 'Flush in progress...'
            : loadedTokens.length === 0
              ? 'No cached models to flush'
              : idleTokens.length === 0
                ? 'All loaded models are busy; wait for inferences to finish'
                : `Idle models available to flush: ${idleTokens.join(', ')}`;
          const busy = isFlushing || (isAnyLoading && idleTokens.length === 0);
          const colorClasses = isModelLoading
            ? 'bg-green-600 hover:bg-green-600 text-white'
            : isFlushing
              ? 'bg-blue-600 hover:bg-blue-600 text-white'
              : '';
          const disabledClasses = disabled
            ? `cursor-not-allowed ${busy ? '' : 'opacity-70'}`
            : '';
          return (
            <div className="flex items-center gap-3">
              {/* Loaded models badges */}
              {loadedTokens.length > 0 && (
                <div className="hidden md:flex items-center gap-2" title={`Loaded models: ${loadedTokens.join(', ')}`}>
                  <span className="text-sm text-muted-foreground">Loaded:</span>
                  {loadedTokens.map((t) => {
                    const label = t === 'whisper-tiny'
                      ? 'Whisper Tiny'
                      : t === 'whisper-base'
                        ? 'Whisper Base'
                        : (t === 'whisper-large' || t === 'whisper-large-v3')
                          ? 'Whisper Large'
                        : t === 'wav2vec2'
                          ? 'Wav2Vec2'
                          : t; // legacy tokens like whisper-large, whisper-large-v3
                    const isBusy = busyTokens.includes(t);
                    // Match Flush button shape for Whisper Tiny only
                    const shapeClass = t === 'whisper-tiny' ? 'rounded-md' : 'rounded-full';
                    return (
                      isBusy ? (
                        <Button
                          key={t}
                          variant="secondary"
                          size="sm"
                          disabled
                          className={`h-8 px-3 ${shapeClass} bg-blue-600 hover:bg-blue-600 text-white shadow-sm pointer-events-none`}
                          title={label}
                        >
                          <span className="text-xs font-medium">{label}</span>
                        </Button>
                      ) : (
                        t === 'whisper-tiny' ? (
                          <Button
                            key={t}
                            variant="outline"
                            size="sm"
                            disabled
                            className={`h-8 px-3 ${shapeClass}`}
                            title={label}
                          >
                            <span className="text-xs font-medium">{label}</span>
                          </Button>
                        ) : (
                          <Badge key={t} variant="outline" className="text-xs">
                            {label}
                          </Badge>
                        )
                      )
                    );
                  })}
                </div>
              )}
              <Button
                variant={busy ? "secondary" : "destructive"}
                size="sm"
                className={`h-8 ${colorClasses} ${disabledClasses}`}
                disabled={disabled}
                title={title}
                onClick={async () => {
                // Determine which model to flush: prefer selected idle model, else prompt
                const loadedTokensNow = (() => {
                  const asrIds = cacheStatus?.asr_loaded ?? [];
                  const tokens: string[] = [];
                  for (const id of asrIds) {
                    if (id.includes('whisper-large-v3') || id.includes('openai/whisper-large-v3')) tokens.push('whisper-large-v3');
                    else if (id.includes('whisper-large')) tokens.push('whisper-large');
                    else if (id.includes('whisper-tiny')) tokens.push('whisper-tiny');
                    else if (id.includes('whisper-base')) tokens.push('whisper-base');
                  }
                  if (cacheStatus?.emotion_loaded) tokens.push('wav2vec2');
                  return Array.from(new Set(tokens));
                })();
                const busySet = new Set(loadedTokensNow.filter(t => (loadingCounts[t] ?? 0) > 0));
                const idleNow = loadedTokensNow.filter(t => !busySet.has(t));
                if (idleNow.length === 0) {
                  alert('All loaded models are currently busy. Try again when inferences finish.');
                  return;
                }
                let scope: 'whisper-tiny' | 'whisper-base' | 'whisper-large' | 'whisper-large-v3' | 'wav2vec2' | 'all' | undefined;
                if (model && idleNow.includes(model)) {
                  scope = model as 'whisper-tiny' | 'whisper-base' | 'whisper-large' | 'whisper-large-v3' | 'wav2vec2';
                } else if (idleNow.length === 1) {
                  scope = idleNow[0] as 'whisper-tiny' | 'whisper-base' | 'whisper-large' | 'whisper-large-v3' | 'wav2vec2';
                } else {
                  const choice = window.prompt(`Multiple idle models detected. Enter one to flush: ${idleNow.join(', ')}`);
                  if (!choice) return;
                  const normalized = choice.trim();
                  if (!idleNow.includes(normalized)) {
                    alert(`Invalid choice. Allowed: ${idleNow.join(', ')}`);
                    return;
                  }
                  scope = normalized as 'whisper-tiny' | 'whisper-base' | 'whisper-large' | 'whisper-large-v3' | 'wav2vec2';
                }
                const confirmed = window.confirm(`Flush cached model? This frees RAM/VRAM.\nScope: ${scope}`);
                if (!confirmed || !scope) return;
                setIsFlushing(true);
                try {
                  const result = await flushModels(scope);
                  console.log('Flushed models:', result);
                  // Refresh cache status after flush
                  try { setCacheStatus(await getCacheStatus()); } catch (e) { void e; }
                  // Start countdown to inform the user and prevent immediate reloads
                  setReloadIn(12);
                } catch (e) {
                  const msg = e instanceof Error ? e.message : String(e);
                  alert(`Flush failed: ${msg}`);
                } finally {
                  setIsFlushing(false);
                }
                }}
              >
                {(isFlushing || isModelLoading) && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                {label}
              </Button>
            </div>
          );
        })()}

        <Button variant="outline" size="sm" className="h-8">
          <Settings className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
};
