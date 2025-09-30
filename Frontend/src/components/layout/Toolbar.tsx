import React, { useState, useEffect } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Upload } from "lucide-react";
import { API_BASE } from '@/lib/api';
import { CustomDatasetManager } from '@/components/dataset/CustomDatasetManager';

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
  onBatchInference?: (model: string, dataset: string) => void; // New callback for batch inference
}

interface CustomDataset {
  dataset_name: string;
  formatted_name: string;
  total_files: number;
}

const modelDatasetMap: Record<string, string[]> = {
  "whisper-base": ["common-voice", "ravdess", "custom"],
  "whisper-large": ["common-voice", "ravdess", "custom"],
  "wav2vec2": ["common-voice", "ravdess", "custom"],
};

const defaultDatasetForModel: Record<string, string> = {
  "whisper-base": "common-voice",
  "whisper-large": "common-voice",
  "wav2vec2": "ravdess",
};

export const Toolbar = ({apiData, setApiData, selectedFile, uploadedFiles, onFileSelect, model, setModel, dataset, setDataset, onBatchInference}: ToolbarProps) => {
  const [customDatasets, setCustomDatasets] = useState<CustomDataset[]>([]);

  const fetchCustomDatasets = async () => {
    try {
      const response = await fetch(`${API_BASE}/upload/dataset/list`, {
        credentials: 'include'
      });
      
      if (response.ok) {
        const data = await response.json();
        setCustomDatasets(data.datasets || []);
      }
    } catch (err) {
      console.error('Error fetching custom datasets:', err);
    }
  };

  useEffect(() => {
    fetchCustomDatasets();
  }, []);

  const handleDatasetCreated = (datasetName: string) => {
    fetchCustomDatasets(); // Refresh the list
    setDataset(datasetName); // Automatically select the new dataset
  };

  const handleDatasetSelected = (datasetName: string) => {
    setDataset(datasetName);
  };

const onModelChange = (value: string) => {
  setModel(value);
  
  // Update dataset based on model
  const allowedDatasets = modelDatasetMap[value] || ["custom"];
  const defaultDataset = defaultDatasetForModel[value] || "custom";

  console.log("Model selected:", value);
  
  // Check if current dataset is a custom dataset
  const isCurrentCustomDataset = dataset.startsWith('custom:');

  if (!allowedDatasets.includes(dataset) && !isCurrentCustomDataset) {
    // Use the canonical handler so all side effects fire (metadata loading)
    onDatasetChange(defaultDataset);
  } else if (!isCurrentCustomDataset && dataset !== 'custom' && onBatchInference) {
    // Dataset is already valid and not custom, fire batch inference directly
    onBatchInference(value, dataset);
  }
};

  const onDatasetChange = (value: string) => {
    setDataset(value);
    console.log("Dataset selected:", value);
    
    // Check if this is a custom dataset (formatted as custom:session_id:dataset_name)
    const isCustomDataset = value.startsWith('custom:');
    
    // Trigger batch inference when dataset changes (except for custom datasets)
    if (!isCustomDataset && value !== 'custom' && onBatchInference) {
      onBatchInference(model, value);
    }
  };

  // Get datasets allowed for current model
  const allowedDatasets = modelDatasetMap[model] || ["custom"];

  return (
    <div className="h-14 panel-header border-b panel-border px-4 flex items-center justify-between">
      {/* Left side: Model and Dataset selectors */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <span className="text-sm font-bold text-foreground">LIT for Voice</span>
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
              <SelectTrigger className="w-40 h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {/* Built-in datasets */}
                {allowedDatasets.filter(ds => ds !== 'custom').map((ds) => {
                  let label = ds;
                  if (ds === "common-voice") label = "Common Voice";
                  else if (ds === "ravdess") label = "RAVDESS";
                  return (
                    <SelectItem key={ds} value={ds}>
                      {label}
                    </SelectItem>
                  );
                })}
                
                {/* Custom datasets */}
                {customDatasets.length > 0 && (
                  <>
                    <SelectItem disabled value="separator">
                      ── Custom Datasets ──
                    </SelectItem>
                    {customDatasets.map((customDataset) => (
                      <SelectItem key={customDataset.formatted_name} value={customDataset.formatted_name}>
                        {customDataset.dataset_name} ({customDataset.total_files} files)
                      </SelectItem>
                    ))}
                  </>
                )}
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
        <CustomDatasetManager 
          onDatasetCreated={handleDatasetCreated}
          onDatasetSelected={handleDatasetSelected}
        />

        <Button variant="outline" size="sm" className="h-8">
          <Upload className="h-4 w-4 mr-2" />
          Upload
        </Button>

        <Button 
          variant="outline" 
          size="sm" 
          className="h-8"
          onClick={async () => {
            try {
              const response = await fetch(`${API_BASE}/upload/test`, { credentials: 'include' });
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
      </div>
    </div>
  );
};