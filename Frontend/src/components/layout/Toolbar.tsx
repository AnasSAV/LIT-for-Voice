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

export const Toolbar = () => {
  const [model, setModel] = useState("whisper-base");
  // Dataset UI value is one of: "ravdess" | "common-voice" | "custom"
  const [dataset, setDataset] = useState("ravdess");
  const [hasRavdessSubset, setHasRavdessSubset] = useState(false);
  const [hasRavdessFull, setHasRavdessFull] = useState(false);
  const [hasCommonVoice, setHasCommonVoice] = useState(false);

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
    if (!res.ok) {
      throw new Error(`API error: ${res.status}`);
    }
    const data = await res.json();
    console.log("API response:", data);
  } catch (error) {
    console.error("Failed to run inference:", error);
  }
};

  const onDatasetChange = async (value: string) => {
    setDataset(value);
    console.log("Dataset selected:", value);

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
                {availableDatasetOptions.map((ds) => {
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

        <Button variant="outline" size="sm" className="h-8">
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
