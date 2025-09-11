import { useState, useRef, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Upload, Search, Play, Pause } from "lucide-react";
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
  batchInferenceStatus?: 'idle' | 'running' | 'done';
  onBatchInferenceStart?: () => void;
  onBatchInferenceComplete?: () => void;
}

export const AudioDatasetPanel = ({ 
  apiData, 
  model,
  dataset,
  selectedFile, 
  onFileSelect, 
  onUploadSuccess,
  batchInferenceStatus,
  onBatchInferenceStart,
  onBatchInferenceComplete
}: AudioDatasetPanelProps) => {
  const [selectedRow, setSelectedRow] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [datasetMetadata, setDatasetMetadata] = useState<Record<string, string | number>[]>([]);
  const [predictionMap, setPredictionMap] = useState<Record<string, string>>({});
  const [inferenceStatus, setInferenceStatus] = useState<Record<string, 'idle' | 'loading' | 'done' | 'error'>>({});
  
  // Batch inference state
  const [currentInferenceIndex, setCurrentInferenceIndex] = useState(0);
  const [batchInferenceQueue, setBatchInferenceQueue] = useState<string[]>([]);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Stable handlers to prevent downstream re-renders
  const handleRowSelect = useCallback((id: string) => {
    setSelectedRow(id);
    
    // When a row is selected, just propagate the file selection for UI/audio playback
    // No inference should be triggered here
    if (dataset === "custom") return;
    if (!onFileSelect) return;

    const findMatch = () => {
      for (const row of datasetMetadata) {
        const rowId = row["id"]; 
        const path = row["path"] || row["filepath"] || row["file"] || row["filename"];
        if (typeof rowId === "string" && rowId === id) return row;
        if (typeof path === "string" && (path === id || path.endsWith(`/${id}`) || path.endsWith(`\\${id}`))) return row;
      }
      return null;
    };

    const match = findMatch();
    if (!match) return;

    const pathVal = (match["path"] || match["filepath"] || match["file"] || match["filename"]) as string | undefined;
    const filename = pathVal ? (pathVal.split("/").pop() || pathVal.split("\\").pop() || String(id)) : String(id);

    const fileLike: UploadedFile = {
      file_id: String(id),
      filename,
      file_path: pathVal || filename,
      message: "",
    };

    // Just select the file for UI purposes, no inference
    onFileSelect(fileLike);
  }, [dataset, datasetMetadata, onFileSelect]);

  const handleFilePlay = useCallback((file: UploadedFile) => {
    console.log('AudioDatasetPanel - File selected for play:', file);
    console.log('AudioDatasetPanel - onFileSelect callback exists:', !!onFileSelect);
    if (onFileSelect) {
      onFileSelect(file);
      console.log('AudioDatasetPanel - Called onFileSelect with file');
    }
  }, [onFileSelect]);

  const handleVisibleRowIdsChange = useCallback((ids: string[]) => {
    // This is now just for pagination, no inference triggering
  }, []);

  // Batch inference for entire dataset when model/dataset changes
  useEffect(() => {
    if (dataset === "custom" || !model) return;
    if (datasetMetadata.length === 0) return;
    
    console.log(`Starting batch inference for ${model} on ${datasetMetadata.length} files in ${dataset} dataset`);
    
    // Abort any ongoing inference
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();
    
    // Create queue of all dataset files
    const fileIds = datasetMetadata.map((row, index) => {
      const id = row["id"] || row["path"] || row["filepath"] || row["file"] || row["filename"] || String(index);
      return String(id);
    });
    
    setBatchInferenceQueue(fileIds);
    setCurrentInferenceIndex(0);
    setPredictionMap({}); // Clear previous predictions
    setInferenceStatus({}); // Clear previous status
    
    if (onBatchInferenceStart) {
      onBatchInferenceStart();
    }
  }, [model, dataset, datasetMetadata, onBatchInferenceStart]);

  // Process batch inference queue
  useEffect(() => {
    if (batchInferenceQueue.length === 0) return;
    if (currentInferenceIndex >= batchInferenceQueue.length) {
      // Batch inference complete
      console.log('Batch inference completed');
      if (onBatchInferenceComplete) {
        onBatchInferenceComplete();
      }
      return;
    }

    const currentFileId = batchInferenceQueue[currentInferenceIndex];
    const currentRow = datasetMetadata.find(row => {
      const id = row["id"] || row["path"] || row["filepath"] || row["file"] || row["filename"];
      return String(id) === currentFileId;
    });

    if (!currentRow) {
      // Skip this file and continue
      setCurrentInferenceIndex(prev => prev + 1);
      return;
    }

    const runInference = async () => {
      try {
        setInferenceStatus(prev => ({ ...prev, [currentFileId]: 'loading' }));
        
        const pathVal = (currentRow["path"] || currentRow["filepath"] || currentRow["file"] || currentRow["filename"]) as string;
        const filename = pathVal ? (pathVal.split("/").pop() || pathVal.split("\\").pop() || currentFileId) : currentFileId;

        const requestBody = {
          model,
          dataset,
          dataset_file: filename
        };

        const response = await fetch(`http://localhost:8000/inferences/run`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestBody),
          signal: abortControllerRef.current?.signal,
        });

        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }

        const prediction = await response.json();
        const predictionText = typeof prediction === 'string' ? prediction : prediction?.text || JSON.stringify(prediction);

        setPredictionMap(prev => ({ ...prev, [currentFileId]: predictionText }));
        setInferenceStatus(prev => ({ ...prev, [currentFileId]: 'done' }));
        
        console.log(`Inference complete for ${filename}: ${predictionText}`);
        
      } catch (error: any) {
        if (error.name === 'AbortError') return;
        console.error(`Inference failed for ${currentFileId}:`, error);
        setInferenceStatus(prev => ({ ...prev, [currentFileId]: 'error' }));
      }
      
      // Move to next file
      setCurrentInferenceIndex(prev => prev + 1);
    };

    // Add small delay to prevent overwhelming the server
    const timeoutId = setTimeout(runInference, 100);
    
    return () => clearTimeout(timeoutId);
  }, [batchInferenceQueue, currentInferenceIndex, datasetMetadata, model, dataset, onBatchInferenceComplete]);

  // Cleanup on unmount or when dataset changes
  useEffect(() => {
    abortControllerRef.current = new AbortController();
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [model, dataset]);

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
        const res = await fetch(`http://localhost:8000/${dataset}/metadata`, { signal: ac.signal });
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
            {dataset !== 'custom' && batchInferenceStatus === 'running' && (
              <Badge variant="outline" className="text-xs">
                Inferencing... {currentInferenceIndex}/{batchInferenceQueue.length}
              </Badge>
            )}
            {dataset !== 'custom' && batchInferenceStatus === 'done' && (
              <Badge variant="default" className="text-xs">
                ✓ Inference Complete
              </Badge>
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
              onRowSelect={handleRowSelect}
              searchQuery={searchQuery}
              apiData={apiData}
              model={model ?? ""}
              dataset={dataset}
              datasetMetadata={datasetMetadata}
              uploadedFiles={uploadedFiles}
              onFilePlay={handleFilePlay}
              predictionMap={predictionMap}
              inferenceStatus={inferenceStatus}
              onVisibleRowIdsChange={handleVisibleRowIdsChange}
            />
          </CardContent>
        </Card>
      </div>
      
      {/* Upload overlay */}
      <AudioUploader onUploadSuccess={onUploadSuccess} />
    </div>
  );
};