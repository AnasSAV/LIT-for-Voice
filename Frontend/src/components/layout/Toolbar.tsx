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
import { Settings, Upload, Download, Pin, Filter } from "lucide-react";
import { listDatasets, getActiveDataset, setActiveDataset, API_BASE } from "@/lib/api/datasets";

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

  const onModelChange = async (value: string) => {
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
        const url = new URL(`${API_BASE}/inferences/run`);
        url.searchParams.set("model", value);
        url.searchParams.set("file_path", selectedFile.file_path);

        const response = await fetch(url.toString(), {
          credentials: "include",
          signal: controller.signal,
        });

        if (!response.ok) {
          let detail = "";
          try {
            const j = await response.json();
            detail = (j && typeof j === "object" && "detail" in j) ? (j as { detail?: string }).detail || "" : "";
          } catch (e) {
            // ignore JSON parse errors when response is not JSON
          }
          throw new Error(`API error: ${response.status} ${detail || response.statusText}`);
        }

        // Backend may return string (text) for Whisper or object for others
        const raw = await response.json();
        const data: ApiData = typeof raw === "string" ? { transcription: raw } : (raw as ApiData);
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

        <Button variant="outline" size="sm" className="h-8">
          <Settings className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
};
