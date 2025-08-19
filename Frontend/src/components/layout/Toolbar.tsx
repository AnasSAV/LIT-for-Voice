import React, { useEffect, useState } from "react";
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
import { listDatasets, getActiveDataset, setActiveDataset } from "@/lib/api/datasets";

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
  apiData: any;
  setApiData: (data: any) => void;
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
  const [dataset, setDataset] = useState("ravdess");
  const [hasRavdessSubset, setHasRavdessSubset] = useState(false);
  const [hasRavdessFull, setHasRavdessFull] = useState(false);
  const [hasCommonVoice, setHasCommonVoice] = useState(false);

  // Load available datasets on component mount
  useEffect(() => {
    (async () => {
      try {
        const ds = await listDatasets();
        setHasRavdessSubset(!!ds.find((d) => d.id === "ravdess_subset" && d.available));
        setHasRavdessFull(!!ds.find((d) => d.id === "ravdess_full" && d.available));
        setHasCommonVoice(!!ds.find((d) => d.id === "common_voice_en" && d.available));

        const active = await getActiveDataset();
        if (active === "ravdess_subset" || active === "ravdess_full") {
          setDataset("ravdess");
        } else if (active === "common_voice_en") {
          setDataset("common-voice");
        } else if (active) {
          setDataset("custom");
        }
      } catch (e) {
        console.warn("Dataset API not available", e);
      }
    })();
  }, []);

  const onModelChange = async (value: string) => {
    setModel(value);
  console.log("Model selected:", value);

  try {
    const res = await fetch(`http://localhost:8000/inferences/run?model=${value}`);
=======
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
  apiData: any;
  setApiData: (data: any) => void;
  selectedFile?: UploadedFile | null;
  uploadedFiles?: UploadedFile[];
  onFileSelect?: (file: UploadedFile) => void;
  model: string;
  setModel: (model: string) => void; // important for lifting state
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

;

export const Toolbar = ({apiData, setApiData, selectedFile, uploadedFiles, onFileSelect,model,setModel}: ToolbarProps) => {
  const [dataset, setDataset] = useState(defaultDatasetForModel[model]);

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
    
    // If there's a selected file, include it in the API call
    if (selectedFile) {
      url += `&file_path=${encodeURIComponent(selectedFile.file_path)}`;
      console.log("Using uploaded file:", selectedFile.filename);
    } else {
      console.log("Using default sample file");
    }
    
    const res = await fetch(url);
>>>>>>> f6a55f1124ca672be22712f60d706bedafdb4de7
    if (!res.ok) {
      throw new Error(`API error: ${res.status}`);
    }
    const data = await res.json();
<<<<<<< HEAD
=======
    setApiData(data);
>>>>>>> f6a55f1124ca672be22712f60d706bedafdb4de7
    console.log("API response:", data);
  } catch (error) {
    console.error("Failed to run inference:", error);
  }
};

<<<<<<< HEAD
  const onDatasetChange = async (value: string) => {
    setDataset(value);
    console.log("Dataset selected:", value);
    
    // Handle dataset selection based on the UI value
    try {
      if (value === "ravdess") {
        // Default to subset if available, otherwise use full
        const targetDataset = hasRavdessSubset ? "ravdess_subset" : "ravdess_full";
        await setActiveDataset(targetDataset);
      } else if (value === "common-voice") {
        await setActiveDataset("common_voice_en");
      } else if (value === "custom") {
        // Handle custom dataset selection
        // This could be extended to show a file picker or other UI
        console.log("Custom dataset selected");
      }
    } catch (e) {
      console.error("Failed to set active dataset", e);
    }

    try {
      if (value === "ravdess") {
        const id = hasRavdessSubset ? "ravdess_subset" : hasRavdessFull ? "ravdess_full" : "ravdess_subset";
        await setActiveDataset(id);
      } else if (value === "common-voice") {
        // Set Common Voice even if not installed; backend will return empty lists
        await setActiveDataset("common_voice_en");
      } else {
        // For custom, we currently don't persist a specific id
        // Clear selection by setting to a non-existing id could be added in backend if needed
      }
      // Notify listeners to reload dataset files for any change
      window.dispatchEvent(new Event("dataset-changed"));
    } catch (e) {
      console.error("Failed to set active dataset:", e);
    }
  };

  // Dataset options are independent of model
  const availableDatasetOptions = ["ravdess", "common-voice", "custom"];
=======
  const onDatasetChange = (value: string) => {
    setDataset(value);
    console.log("Dataset selected:", value);
  };

  // Get datasets allowed for current model
  const allowedDatasets = modelDatasetMap[model] || ["custom"];

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
<<<<<<< HEAD
                {availableDatasetOptions.map((ds) => {
=======
                {allowedDatasets.map((ds) => {
>>>>>>> f6a55f1124ca672be22712f60d706bedafdb4de7
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

<<<<<<< HEAD
=======
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

>>>>>>> f6a55f1124ca672be22712f60d706bedafdb4de7
        <Button variant="outline" size="sm" className="h-8">
          <Settings className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
};
