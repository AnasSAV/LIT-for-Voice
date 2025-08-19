import { useState, useRef, useEffect, useMemo, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { AudioPlayer } from "../audio/AudioPlayer";
import { WaveformViewer } from "../audio/WaveformViewer";
import { Play, Pause, RotateCcw, Trash2, Plus } from "lucide-react";
import WaveSurfer from "wavesurfer.js";
import { datasetFileUrl } from "@/lib/api/datasets";

interface UploadedFile {
  file_id: string;
  filename: string;
  file_path: string;
  message: string;
  size?: number;
  duration?: number;
  sample_rate?: number;
  label?: string;
  dataset_id?: string | null;
  prediction?: {
    text?: string;
    label?: string;
    confidence?: number;
  };
}

interface DatapointEditorPanelProps {
  selectedFile?: UploadedFile | null;
}

export const DatapointEditorPanel = ({ selectedFile }: DatapointEditorPanelProps) => {
  const [selectedLabel, setSelectedLabel] = useState<string>("neutral");
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const handleReady = useCallback((wavesurfer: WaveSurfer) => {
    console.log('WaveformViewer ready callback in DatapointEditorPanel');
    wavesurferRef.current = wavesurfer;
    const d = wavesurfer.getDuration();
    console.log('Duration from WaveSurfer:', d);
    setDuration(d);
  }, []);
  const handleProgress = useCallback((time: number, dur: number) => {
    setCurrentTime(time);
    setDuration(dur);
  }, []);

  const audioUrl = useMemo(() => {
    if (!selectedFile) return undefined;
    // If coming from dataset table, use dataset file endpoint; otherwise use upload endpoint
    if (selectedFile.message === "dataset") {
      return datasetFileUrl(selectedFile.file_path, selectedFile.dataset_id ?? undefined);
    }
    return `http://localhost:8000/upload/file/${selectedFile.file_id}`;
  }, [selectedFile]);

  // Debug logging for selectedFile and audioUrl
  useEffect(() => {
    console.log('DatapointEditorPanel - selectedFile changed:', selectedFile);
    console.log('DatapointEditorPanel - audioUrl:', audioUrl);
  }, [selectedFile, audioUrl]);

  // When a new file with a cached label is selected, reflect it in the selector
  useEffect(() => {
    if (selectedFile?.label) {
      setSelectedLabel(selectedFile.label);
    }
  }, [selectedFile?.label]);

  // Reset playback when file changes
  useEffect(() => {
    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(0);
    
    // Reset wavesurfer instance if it exists
    if (wavesurferRef.current) {
      wavesurferRef.current.stop();
    }
  }, [selectedFile?.file_id]);
  
  return (
    <div className="h-full panel-background border-l panel-border flex flex-col">
      <div className="panel-header p-3 border-b panel-border">
        <h3 className="font-medium text-sm">Datapoint Editor</h3>
      </div>
      
      <div className="flex-1 p-3 overflow-auto space-y-4">
        {/* File Info */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Sample Info</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="text-xs">
              <span className="text-muted-foreground">File:</span>
              <span className="ml-2 font-mono">{selectedFile?.filename || "No file selected"}</span>
            </div>
            <div className="text-xs">
              <span className="text-muted-foreground">Duration:</span>
              <span className="ml-2">{selectedFile?.duration ? `${selectedFile.duration.toFixed(1)}s` : "N/A"}</span>
            </div>
            <div className="text-xs">
              <span className="text-muted-foreground">Sample Rate:</span>
              <span className="ml-2">{selectedFile?.sample_rate ? `${(selectedFile.sample_rate / 1000).toFixed(1)}kHz` : "N/A"}</span>
            </div>
            {selectedFile?.label && (
              <div className="text-xs">
                <span className="text-muted-foreground">Ground Truth:</span>
                <Badge variant="outline" className="ml-2 text-xs">{selectedFile.label}</Badge>
              </div>
            )}
            {selectedFile?.size && (
              <div className="text-xs">
                <span className="text-muted-foreground">Size:</span>
                <span className="ml-2">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</span>
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
              onReady={handleReady}
              onProgress={handleProgress}
            />
            <AudioPlayer 
              isPlaying={isPlaying}
              onPlayPause={() => {
                setIsPlaying((prev) => {
                  const next = !prev;
                  if (wavesurferRef.current) {
                    try {
                      if (next) {
                        wavesurferRef.current.play();
                      } else {
                        wavesurferRef.current.pause();
                      }
                    } catch (e) {
                      console.error('Play/pause error:', e);
                    }
                  }
                  return next;
                });
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

        {/* Label Editor */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Label Editor</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div>
              <label className="text-xs text-muted-foreground mb-1 block">
                Predicted Label
              </label>
              <Badge variant="outline" className="text-xs">
                {selectedFile?.prediction?.label
                  ? `${selectedFile.prediction.label}${
                      typeof selectedFile.prediction.confidence === 'number'
                        ? ` (${selectedFile.prediction.confidence.toFixed(2)})`
                        : ''
                    }`
                  : 'N/A'}
              </Badge>
            </div>
            
            <div>
              <label className="text-xs text-muted-foreground mb-1 block">
                Ground Truth
              </label>
              <Select value={selectedLabel} onValueChange={setSelectedLabel}>
                <SelectTrigger className="h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="neutral">Neutral</SelectItem>
                  <SelectItem value="happy">Happy</SelectItem>
                  <SelectItem value="sad">Sad</SelectItem>
                  <SelectItem value="angry">Angry</SelectItem>
                  <SelectItem value="fearful">Fearful</SelectItem>
                  <SelectItem value="surprised">Surprised</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="pt-2 space-y-2">
              <Button size="sm" className="w-full h-7">
                <Plus className="h-3 w-3 mr-1" />
                Add and Compare
              </Button>
              
              <div className="flex gap-2">
                <Button size="sm" variant="outline" className="flex-1 h-7">
                  <RotateCcw className="h-3 w-3 mr-1" />
                  Reset
                </Button>
                <Button size="sm" variant="outline" className="flex-1 h-7">
                  <Trash2 className="h-3 w-3 mr-1" />
                  Clear
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Transcript (for ASR models) */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Transcript</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xs space-y-2">
              <div>
                <span className="text-muted-foreground">Predicted:</span>
                <p className="mt-1 p-2 bg-muted rounded text-foreground">
                  "The quick brown fox jumps over the lazy dog"
                </p>
              </div>
              <div>
                <span className="text-muted-foreground">Ground Truth:</span>
                <p className="mt-1 p-2 bg-muted rounded text-foreground">
                  "The quick brown fox jumps over the lazy dog"
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};