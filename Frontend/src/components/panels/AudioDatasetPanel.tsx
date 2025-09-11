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
}

export const AudioDatasetPanel = ({ 
  apiData, 
  model,
  dataset,
  selectedFile, 
  onFileSelect, 
  onUploadSuccess 
}: AudioDatasetPanelProps) => {
  const [selectedRow, setSelectedRow] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [datasetMetadata, setDatasetMetadata] = useState<Record<string, string | number>[]>([]);
  const [predictionMap, setPredictionMap] = useState<Record<string, string>>({});
  const [inferenceStatus, setInferenceStatus] = useState<Record<string, 'idle' | 'loading' | 'done' | 'error'>>({});
  // Gate inferences so responses map to the correct row
  const inFlightRowRef = useRef<string | null>(null);
  const [queuedRowId, setQueuedRowId] = useState<string | null>(null);
  // Minimal queue to process visible rows sequentially (per page)
  const [pendingRowQueue, setPendingRowQueue] = useState<string[]>([]);

  // Stable handlers to prevent downstream re-renders
  const handleRowSelect = useCallback((id: string) => {
    setSelectedRow(id);
  }, []);

  const handleFilePlay = useCallback((file: UploadedFile) => {
    console.log('AudioDatasetPanel - File selected for play:', file);
    console.log('AudioDatasetPanel - onFileSelect callback exists:', !!onFileSelect);
    if (onFileSelect) {
      onFileSelect(file);
      console.log('AudioDatasetPanel - Called onFileSelect with file');
    }
  }, [onFileSelect]);

  // When a dataset row is selected (non-custom), map it to a file-like object and propagate selection.
  // Only start a new inference if none is in-flight; otherwise queue the selection.
  useEffect(() => {
    if (dataset === "custom") return;
    if (!selectedRow) return;
    if (!onFileSelect) return;

    // If an inference is already running for another row, queue this selection and return
    if (inFlightRowRef.current && inFlightRowRef.current !== selectedRow) {
      // Ensure selected row appears next in queue if not already present
      setPendingRowQueue((prev) => (prev.includes(selectedRow) ? prev : [...prev, selectedRow]));
      return;
    }

    // Try to find the row in metadata matching the selectedRow by common keys
    const findMatch = () => {
      for (const row of datasetMetadata) {
        const id = row["id"]; // may exist
        const path = row["path"] || row["filepath"] || row["file"] || row["filename"]; // prefer path-like keys
        if (typeof id === "string" && id === selectedRow) return row;
        if (typeof path === "string" && (path === selectedRow || path.endsWith(`/${selectedRow}`) || path.endsWith(`\\${selectedRow}`))) return row;
      }
      return null;
    };

    const match = findMatch();
    if (!match) return;

    const pathVal = (match["path"] || match["filepath"] || match["file"] || match["filename"]) as string | undefined;
    const filename = pathVal ? (pathVal.split("/").pop() || pathVal.split("\\").pop() || String(selectedRow)) : String(selectedRow);

    const fileLike: UploadedFile = {
      file_id: String(selectedRow),
      filename,
      file_path: pathVal || filename, // assume backend can resolve this path
      message: "",
    };

    // Mark this row as in-flight and set status to loading, then trigger inference
    if (!inFlightRowRef.current) {
      inFlightRowRef.current = selectedRow;
      setInferenceStatus(prev => ({ ...prev, [selectedRow]: 'loading' }));
      onFileSelect(fileLike);
    }
  }, [dataset, selectedRow, datasetMetadata, onFileSelect]);

  // Persist latest inference result against the row that initiated the request.
  useEffect(() => {
    if (typeof apiData === "string") {
      const rowId = inFlightRowRef.current;
      if (rowId) {
        setPredictionMap((prev) => ({ ...prev, [rowId]: String(apiData) }));
        setInferenceStatus((prev) => ({ ...prev, [rowId]: 'done' }));
        inFlightRowRef.current = null;
        // Dequeue and trigger next item if present
        setPendingRowQueue((prev) => {
          const nextQueue = prev[0] === rowId ? prev.slice(1) : prev;
          if (nextQueue.length > 0) {
            // trigger next
            const nextId = nextQueue[0];
            setSelectedRow(nextId);
          }
          return nextQueue;
        });
      }
    }
  }, [apiData, queuedRowId]);

  // Process queue when visible rows change or when idle
  useEffect(() => {
    if (inFlightRowRef.current) return;
    if (pendingRowQueue.length === 0) return;
    const nextId = pendingRowQueue[0];
    setSelectedRow(nextId);
  }, [pendingRowQueue]);

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
              onVisibleRowIdsChange={(ids) => {
                // Skip rows already completed; preserve simple ordering as provided
                const next = ids.filter((id) => inferenceStatus[id] !== 'done');
                setPendingRowQueue(next);
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