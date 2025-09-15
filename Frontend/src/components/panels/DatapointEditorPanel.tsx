import { useState, useRef, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { AudioPlayer } from "../audio/AudioPlayer";
import { WaveformViewer } from "../audio/WaveformViewer";
import { Play, Pause, RotateCcw, Trash2, Plus } from "lucide-react";
import WaveSurfer from "wavesurfer.js";

interface UploadedFile {
  file_id: string;
  filename: string;
  file_path: string;
  message: string;
  size?: number;
  duration?: number;
  sample_rate?: number;
}

interface PerturbationResult {
  perturbed_file: string;
  filename: string;
  duration_ms: number;
  sample_rate: number;
  applied_perturbations: Array<{
    type: string;
    params: Record<string, any>;
    status: string;
    error?: string;
  }>;
  success: boolean;
  error?: string;
}

interface DatapointEditorPanelProps {
  selectedFile?: UploadedFile | null;
  dataset?: string; // "custom" | dataset key
  perturbationResult?: PerturbationResult | null;
  predictionMap?: Record<string, string>;
}

export const DatapointEditorPanel = ({ selectedFile, dataset = "custom", perturbationResult, predictionMap }: DatapointEditorPanelProps) => {
  const [selectedLabel, setSelectedLabel] = useState<string>("neutral");
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [showPerturbed, setShowPerturbed] = useState(false);
  const wavesurferRef = useRef<WaveSurfer | null>(null);

  const audioUrl = (() => {
    // If showing perturbed audio and it's available
    if (showPerturbed && perturbationResult?.success) {
      const filename = perturbationResult.filename;
      return `http://localhost:8000/upload/file/${filename}`;
    }
    
    // Otherwise show original audio
    if (!selectedFile) return undefined;
    
    // Check if this is an uploaded file - more precise detection
    // Uploaded files have file_path that contains "uploads/" OR have specific message types
    // Dataset files have message "Selected from embeddings" or no uploads/ in path
    const isUploadedFile = selectedFile.file_path && (
      selectedFile.file_path.includes('uploads/') || 
      selectedFile.file_path.startsWith('uploads/') ||
      selectedFile.message === "Perturbed file" ||
      selectedFile.message === "File uploaded successfully" ||
      selectedFile.message === "File uploaded and processed successfully"
    ) && selectedFile.message !== "Selected from embeddings" && selectedFile.message !== "Selected from dataset";
    
    if (isUploadedFile) {
      // This is an uploaded file, use the upload endpoint
      return `http://localhost:8000/upload/file/${selectedFile.file_id}`;
    } else if (dataset && dataset !== "custom") {
      // This is a dataset file
      const filename = encodeURIComponent(selectedFile.filename);
      return `http://localhost:8000/${dataset}/file/${filename}`;
    } else {
      // Fallback to upload endpoint
      return `http://localhost:8000/upload/file/${selectedFile.file_id}`;
    }
  })();

  // Get current file info (original or perturbed)
  const currentFileInfo = (() => {
    if (showPerturbed && perturbationResult?.success) {
      return {
        filename: perturbationResult.filename,
        duration: perturbationResult.duration_ms / 1000,
        sample_rate: perturbationResult.sample_rate,
        size: undefined
      };
    }
    return selectedFile;
  })();

  // Debug logging for selectedFile and audioUrl
  useEffect(() => {
    console.log('DatapointEditorPanel - selectedFile changed:', selectedFile);
    console.log('DatapointEditorPanel - audioUrl:', audioUrl);
  }, [selectedFile, audioUrl]);

  // Reset playback when file changes or when switching between original/perturbed
  useEffect(() => {
    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(0);
    
    // Reset wavesurfer instance if it exists
    if (wavesurferRef.current) {
      wavesurferRef.current.stop();
    }
  }, [selectedFile?.file_id, dataset, showPerturbed, perturbationResult?.filename]);
  
  return (
    <div className="h-full panel-background border-l panel-border flex flex-col">
      <div className="panel-header p-3 border-b panel-border">
        <h3 className="font-medium text-sm">Datapoint Editor</h3>
      </div>
      
      <div className="flex-1 p-3 overflow-auto space-y-4">
        {/* File Info */}
        <Card>
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm">Sample Info</CardTitle>
              {perturbationResult?.success && (
                <div className="flex items-center gap-2">
                  <Button
                    variant={showPerturbed ? "default" : "outline"}
                    size="sm"
                    onClick={() => setShowPerturbed(false)}
                    className="text-xs h-6 px-2"
                  >
                    Original
                  </Button>
                  <Button
                    variant={!showPerturbed ? "default" : "outline"}
                    size="sm"
                    onClick={() => setShowPerturbed(true)}
                    className="text-xs h-6 px-2"
                  >
                    Perturbed
                  </Button>
                </div>
              )}
            </div>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="text-xs">
              <span className="text-muted-foreground">File:</span>
              <span className="ml-2 font-mono">{currentFileInfo?.filename || "No file selected"}</span>
              {showPerturbed && (
                <Badge variant="secondary" className="ml-2 text-[10px]">P</Badge>
              )}
            </div>
            <div className="text-xs">
              <span className="text-muted-foreground">Duration:</span>
              <span className="ml-2">{currentFileInfo?.duration ? `${currentFileInfo.duration.toFixed(1)}s` : "N/A"}</span>
            </div>
            <div className="text-xs">
              <span className="text-muted-foreground">Sample Rate:</span>
              <span className="ml-2">{currentFileInfo?.sample_rate ? `${(currentFileInfo.sample_rate / 1000).toFixed(1)}kHz` : "N/A"}</span>
            </div>
            {currentFileInfo?.size && (
              <div className="text-xs">
                <span className="text-muted-foreground">Size:</span>
                <span className="ml-2">{(currentFileInfo.size / 1024 / 1024).toFixed(2)} MB</span>
              </div>
            )}
            {showPerturbed && perturbationResult?.applied_perturbations && (
              <div className="text-xs">
                <span className="text-muted-foreground">Applied:</span>
                <div className="ml-2 mt-1 space-y-1">
                  {perturbationResult.applied_perturbations.map((pert, idx) => (
                    <Badge key={idx} variant="outline" className="text-[10px] mr-1">
                      {pert.type.replace('_', ' ')}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
            {showPerturbed && perturbationResult?.filename && predictionMap && (
              <div className="text-xs mt-2">
                <span className="text-muted-foreground">Perturbed Prediction:</span>
                <div className="ml-2 mt-1">
                  <Badge variant="secondary" className="text-[10px]">
                    {predictionMap[perturbationResult.filename] || "Loading..."}
                  </Badge>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Audio Player & Waveform */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Audio Playback</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <WaveformViewer 
              audioUrl={audioUrl}
              isPlaying={isPlaying}
              onReady={(wavesurfer) => {
                console.log('WaveformViewer ready callback in DatapointEditorPanel');
                wavesurferRef.current = wavesurfer;
                const duration = wavesurfer.getDuration();
                console.log('Duration from WaveSurfer:', duration);
                setDuration(duration);
              }}
              onProgress={(time, dur) => {
                setCurrentTime(time);
                setDuration(dur);
              }}
            />
            <AudioPlayer 
              isPlaying={isPlaying}
              onPlayPause={() => {
                setIsPlaying(!isPlaying);
                if (wavesurferRef.current) {
                  if (isPlaying) {
                    wavesurferRef.current.pause();
                  } else {
                    wavesurferRef.current.play();
                  }
                }
              }}
              currentTime={currentTime}
              duration={duration}
              onSeek={(time) => {
                if (wavesurferRef.current) {
                  wavesurferRef.current.seekTo(time / duration);
                }
              }}
              onVolumeChange={(volume) => {
                if (wavesurferRef.current) {
                  wavesurferRef.current.setVolume(volume);
                }
              }}
            />
          </CardContent>
        </Card>
      </div>
    </div>
  );
};