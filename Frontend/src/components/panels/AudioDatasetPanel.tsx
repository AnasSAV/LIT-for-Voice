import { useState, useRef, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Upload, Search, Play, Pause, Zap } from "lucide-react";
import { AudioUploader } from "../audio/AudioUploader";
import { AudioDataTable } from "../audio/AudioDataTable";
import { toast } from "sonner";

interface UploadedFile {
  file_id: string;
  filename: string;
  file_path: string;
  message: string;
  size?: number;
  duration?: number;
  sample_rate?: number;
  prediction?:string
}

interface AudioDatasetPanelProps {
  apiData?: unknown;
  model: string | null;
  dataset: string;
  uploadedFiles?: UploadedFile[];
  selectedFile?: UploadedFile | null;
  onFileSelect?: (file: UploadedFile) => void;
  onUploadSuccess?: (uploadResponse: UploadedFile) => void;
  onPredictionResults?: (results: any[]) => void;
}

export const AudioDatasetPanel = ({ 
  apiData, 
  model,
  dataset,
  selectedFile, 
  onFileSelect, 
  onUploadSuccess,
  onPredictionResults
}: AudioDatasetPanelProps) => {
  const [selectedRow, setSelectedRow] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [datasetMetadata, setDatasetMetadata] = useState<Record<string, string | number>[]>([]);
  const [isRunningBatchPredictions, setIsRunningBatchPredictions] = useState(false);

  const runBatchPredictions = async () => {
    if (!model || dataset === "custom") {
      toast.error("Please select a model and dataset first");
      return;
    }

    setIsRunningBatchPredictions(true);
    try {
      const response = await fetch(`http://localhost:8000/predictions/batch?model=${model}&dataset=${dataset}&limit=10`);
      if (!response.ok) {
        throw new Error(`Failed to run batch predictions: ${response.status}`);
      }
      
      const results = await response.json();
      toast.success(`Batch predictions completed! Processed ${results.processed_files} files`);
      
      // Pass results to parent component
      if (onPredictionResults && results.predictions) {
        onPredictionResults(results.predictions);
      }
      
      // Display results in console for now - you could show this in a modal or update the UI
      console.log('Batch prediction results:', results);
      
      // Show first few results as an example
      if (results.predictions && results.predictions.length > 0) {
        const firstResult = results.predictions[0];
        if (firstResult.prediction_type === "emotion") {
          alert(`Sample Result:\nFile: ${firstResult.filename}\nPredicted Emotion: ${firstResult.emotion_prediction}\nGround Truth: ${firstResult.ground_truth_emotion}\nTranscript: ${firstResult.transcript}`);
        }
      }
    } catch (error) {
      console.error('Batch prediction error:', error);
      toast.error(`Batch predictions failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsRunningBatchPredictions(false);
    }
  };

  // Fetch dataset metadata when dataset changes (for non-custom datasets)
  useEffect(() => {
    const allowed = ["common-voice", "ravdess"];
    if (!allowed.includes(dataset)) {
      setDatasetMetadata([]);
      return;
    }
    const ac = new AbortController();
    (async () => {
      try {
        const res = await fetch(`http://localhost:8000/datasets/${dataset}/metadata`, { signal: ac.signal });
        if (!res.ok) throw new Error(`Failed to fetch metadata: ${res.status}`);
        const data = await res.json();
        if (Array.isArray(data)) setDatasetMetadata(data as Record<string, string | number>[]);
        else setDatasetMetadata([]);
      } catch (e) {
        const name = (e as { name?: string } | null)?.name;
        if (name !== 'AbortError') console.error(e);
      }
    })();
    return () => ac.abort();
  }, [dataset]);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        if (file.type.startsWith('audio/')) {
          try {
            await uploadFile(file, model ?? "");
          } catch (error) {
            console.error('Upload error:', error);
          }
        } else {
          toast.error(`Invalid file type: ${file.name}. Only audio files are supported.`);
        }
      }
    }
    // Reset the input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const uploadFile = async (file: File, model: string) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', model);

    try {
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      const data = await response.json();
      setUploadedFiles(prevFiles => [...prevFiles, data]);
      toast.success(`Uploaded: ${file.name}`);
      
      if (onUploadSuccess) {
        onUploadSuccess(data);
      }
      
      return data;
    } catch (error) {
      console.error('Upload error:', error);
      const msg = error instanceof Error ? error.message : 'Unknown error';
      toast.error(`Failed to upload ${file.name}: ${msg}`);
      throw error;
    }
  };

  return (
    <div className="h-full panel-background flex flex-col">
      <div className="panel-header p-3 border-b panel-border">
        <div className="flex items-center justify-between">
          <h3 className="font-medium text-sm">Audio Dataset</h3>
          <div className="flex items-center gap-2">
            <Badge variant="secondary" className="text-xs">
              {uploadedFiles ? `${uploadedFiles.length} uploaded` : "0 files"}
            </Badge>
            {dataset !== "custom" && (
              <Button 
                size="sm" 
                variant="outline" 
                className="h-7" 
                onClick={runBatchPredictions}
                disabled={isRunningBatchPredictions || !model}
              >
                <Zap className="h-3 w-3 mr-1" />
                {isRunningBatchPredictions ? "Running..." : "Run Predictions"}
              </Button>
            )}
            <Button size="sm" variant="outline" className="h-7" onClick={handleUploadClick}>
              <Upload className="h-3 w-3 mr-1" />
              Upload
            </Button>
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*"
              multiple
              onChange={handleFileSelect}
              className="hidden"
            />
          </div>
        </div>
        
        {/* Search bar */}
        <div className="mt-2 relative">
          <Search className="absolute left-2 top-1/2 transform -translate-y-1/2 h-3 w-3 text-muted-foreground" />
          <Input
            placeholder="Search audio files..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-7 h-8 text-xs"
          />
        </div>
      </div>
      
      <div className="flex-1 overflow-hidden">
        <Card className="h-full rounded-none border-0">
          <CardContent className="p-0 h-full">
            <AudioDataTable 
              selectedRow={selectedRow}
              onRowSelect={setSelectedRow}
              searchQuery={searchQuery}
              apiData={apiData}
              model={model ?? ""}
              dataset={dataset}
              datasetMetadata={datasetMetadata}
              uploadedFiles={uploadedFiles}
              onFilePlay={(file) => {
                console.log('AudioDatasetPanel - File selected for play:', file);
                console.log('AudioDatasetPanel - onFileSelect callback exists:', !!onFileSelect);
                if (onFileSelect) {
                  onFileSelect(file);
                  console.log('AudioDatasetPanel - Called onFileSelect with file');
                }
              }}
            />
          </CardContent>
        </Card>
      </div>
      
      {/* Upload overlay */}
      <AudioUploader onUploadSuccess={onUploadSuccess} />
    </div>
  );
};