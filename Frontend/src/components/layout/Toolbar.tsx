import React from "react";
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

interface UploadedFile {
  file_id: string;
  filename: string;
  file_path: string;
  message: string;
  size?: number;
  duration?: number;
  sample_rate?: number;
}

interface ToolbarProps {
  apiData: unknown;
  setApiData: (data: unknown) => void;
  selectedFile?: UploadedFile | null;
  uploadedFiles?: UploadedFile[];
  onFileSelect?: (file: UploadedFile) => void;
  model: string;
  setModel: (model: string) => void; // important for lifting state
  dataset: string;
  setDataset: (dataset: string) => void;
}
const modelDatasetMap: Record<string, string[]> = {
  "whisper-base": ["common-voice", "custom"],
  "whisper-large": ["common-voice", "custom"],
  "wav2vec2": ["ravdess", "custom"],
};

const defaultDatasetForModel: Record<string, string> = {
  "whisper-base": "common-voice",
  "whisper-large": "common-voice",
  "wav2vec2": "ravdess",
};

export const Toolbar = ({apiData, setApiData, selectedFile, uploadedFiles, onFileSelect, model, setModel, dataset, setDataset}: ToolbarProps) => {

let abortController: AbortController | null = null;
const onModelChange = async (value: string) => {
  setModel(value);
    if (abortController) {
    abortController.abort();
  }
  abortController = new AbortController();
  
  // Update dataset based on model
  const allowedDatasets = modelDatasetMap[value] || ["custom"];
  const defaultDataset = defaultDatasetForModel[value] || "custom";

  if (!allowedDatasets.includes(dataset)) {
    setDataset(defaultDataset);
  }

  console.log("Model selected:", value);

  try {
    let url = `http://localhost:8000/inferences/run?model=${value}`;
    if (selectedFile) {
      if (dataset !== 'custom') {
        const filename = encodeURIComponent(selectedFile.filename);
        url += `&dataset=${encodeURIComponent(dataset)}&dataset_file=${filename}`;
        console.log("Using dataset file:", dataset, selectedFile.filename);
      } else {
        url += `&file_path=${encodeURIComponent(selectedFile.file_path)}`;
        console.log("Using uploaded file:", selectedFile.filename);
      }
    }
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`API error: ${res.status}`);
    }
    const data = await res.json();
    setApiData(data);
    console.log("API response:", data);
  } catch (error) {
    console.error("Failed to run inference:", error);
  }
};

  const onDatasetChange = (value: string) => {
    setDataset(value);
    console.log("Dataset selected:", value);
  };

  // Get datasets allowed for current model
  const allowedDatasets = modelDatasetMap[model] || ["custom"];

  // Auto-run inference when a file is selected/changes
  React.useEffect(() => {
    if (!selectedFile) return;
    const ac = new AbortController();
    (async () => {
      try {
        let url = `http://localhost:8000/inferences/run?model=${model}`;
        if (dataset !== 'custom') {
          const filename = encodeURIComponent(selectedFile.filename);
          url += `&dataset=${encodeURIComponent(dataset)}&dataset_file=${filename}`;
        } else if (selectedFile.file_path) {
          url += `&file_path=${encodeURIComponent(selectedFile.file_path)}`;
        }
        const res = await fetch(url, { signal: ac.signal });
        if (!res.ok) {
          throw new Error(`API error: ${res.status}`);
        }
        const data = await res.json();
        setApiData(data);
        console.log("Auto inference (file change) response:", data);
      } catch (error: unknown) {
        if (typeof error === 'object' && error && 'name' in error && (error as { name?: string }).name === 'AbortError') return;
        console.error("Auto inference failed:", error as unknown);
      }
    })();
    return () => ac.abort();
  }, [selectedFile, selectedFile?.file_path, model, dataset, setApiData]);

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
                  const file = uploadedFiles.find(f => f.file_id === fileId);
                  if (file && onFileSelect) {
                    onFileSelect(file);
                  }
                }}
              >
                <SelectTrigger className="w-48 h-8">
                  <SelectValue placeholder="Select uploaded file" />
                </SelectTrigger>
                <SelectContent>
                  {uploadedFiles.map((file) => (
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
              console.error('Backend test failed:', error);
              alert(`Backend test failed: ${error.message}`);
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