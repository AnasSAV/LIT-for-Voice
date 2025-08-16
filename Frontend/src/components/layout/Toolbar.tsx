import React, { useState, useRef } from "react";
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
import { useAudio } from "@/contexts/AudioContext";
import { toast } from "sonner";

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

export const Toolbar = () => {
  const [model, setModel] = useState("Select");
  const [dataset, setDataset] = useState(defaultDatasetForModel[model]);
  const { currentAudio, runPrediction, addAudioFile } = useAudio();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const onModelChange = async (value: string) => {
    setModel(value);

    // Update dataset based on model
    const allowedDatasets = modelDatasetMap[value] || ["custom"];
    const defaultDataset = defaultDatasetForModel[value] || "custom";

    if (!allowedDatasets.includes(dataset)) {
      setDataset(defaultDataset);
    }

    console.log("Model selected:", value);

    // Run prediction on current audio if available
    if (currentAudio) {
      try {
        await runPrediction(currentAudio.id, value);
        toast.success(`${value} prediction completed`);
      } catch (error) {
        toast.error(`Failed to run ${value} prediction`);
        console.error("Failed to run inference:", error);
      }
    } else {
      // Run inference on sample data for demonstration
      try {
        const res = await fetch(`http://localhost:8000/inferences/run?model=${value}`);
        if (!res.ok) {
          throw new Error(`API error: ${res.status}`);
        }
        const data = await res.json();
        console.log("API response:", data);
        toast.success(`${value} inference completed on sample data`);
      } catch (error) {
        console.error("Failed to run inference:", error);
        toast.error(`Failed to run ${value} inference`);
      }
    }
  };

  const onDatasetChange = (value: string) => {
    setDataset(value);
    console.log("Dataset selected:", value);
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files) return;

    for (const file of Array.from(files)) {
      if (file.type.startsWith('audio/')) {
        try {
          await addAudioFile(file);
          toast.success(`Uploaded: ${file.name}`);
        } catch (error) {
          toast.error(`Failed to upload: ${file.name}`);
          console.error('Upload error:', error);
        }
      } else {
        toast.error(`Invalid file type: ${file.name}`);
      }
    }
    
    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Get datasets allowed for current model
  const allowedDatasets = modelDatasetMap[model] || ["custom"];

  return (
    <div className="h-14 panel-header border-b panel-border px-4 flex items-center justify-between">
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept="audio/*"
        onChange={handleFileSelect}
        className="hidden"
      />
      
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

        <Button 
          variant="outline" 
          size="sm" 
          className="h-8"
          onClick={handleUploadClick}
        >
          <Upload className="h-4 w-4 mr-2" />
          Upload
        </Button>

        <Button variant="outline" size="sm" className="h-8">
          <Download className="h-4 w-4 mr-2" />
          Export
        </Button>

        <Button variant="outline" size="sm" className="h-8">
          <Settings className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
};