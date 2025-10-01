import { useState, useRef, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { AudioPlayer } from "../audio/AudioPlayer";
import { WaveformViewer } from "../audio/WaveformViewer";
import { PredictionDisplay } from "../predictions/PredictionDisplay";
import { Play, Pause, RotateCcw, Trash2, Plus } from "lucide-react";
import WaveSurfer from "wavesurfer.js";
import { API_BASE } from '@/lib/api';

interface UploadedFile {
  file_id: string;
  filename: string;
  file_path: string;
  message: string;
  size?: number;
  duration?: number;
  sample_rate?: number;
}

interface Wav2Vec2Prediction {
  predicted_emotion: string;
  probabilities: Record<string, number>;
  confidence: number;
  ground_truth_emotion?: string;
}

interface WhisperPrediction {
  predicted_transcript: string;
  ground_truth: string;
  accuracy_percentage: number | null;
  word_error_rate: number | null;
  character_error_rate: number | null;
  levenshtein_distance: number | null;
  exact_match: number | null;
  character_similarity: number | null;
  word_count_predicted: number;
  word_count_truth: number;
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
  selectedEmbeddingFile?: string | null;
  dataset?: string; // "custom" | dataset key (effective dataset)
  originalDataset?: string; // Original dataset selection from toolbar
  perturbationResult?: PerturbationResult | null;
  predictionMap?: Record<string, string>;
  model?: string;
  wav2vecPrediction?: Wav2Vec2Prediction | null;
  whisperPrediction?: WhisperPrediction | null;
  perturbedPredictions?: Wav2Vec2Prediction | WhisperPrediction | null;
  isLoadingPredictions?: boolean;
  isLoadingPerturbed?: boolean;
  predictionError?: string | null;
}

export const DatapointEditorPanel = ({ 
  selectedFile, 
  selectedEmbeddingFile,
  dataset = "custom", 
  originalDataset,
  perturbationResult, 
  predictionMap,
  model,
  wav2vecPrediction,
  whisperPrediction,
  perturbedPredictions,
  isLoadingPredictions,
  isLoadingPerturbed,
  predictionError
}: DatapointEditorPanelProps) => {
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
      return `${API_BASE}/upload/file/${filename}`;
    }
    
    // Otherwise show original audio
    if (!selectedFile) return undefined;
    
    // Check if this is an uploaded file - more precise detection
    const isUploadedFile = selectedFile.file_path && (
      selectedFile.file_path.includes('uploads/') || 
      selectedFile.file_path.startsWith('uploads/') ||
      selectedFile.message === "Perturbed file" ||
      selectedFile.message === "File uploaded successfully" ||
      selectedFile.message === "File uploaded and processed successfully"
    ) && selectedFile.message !== "Selected from embeddings" && selectedFile.message !== "Selected from dataset";
    
    if (isUploadedFile) {
      // This is an uploaded file, use the upload endpoint
      return `${API_BASE}/upload/file/${selectedFile.file_id}`;
    }
    
    // For dataset files (including files selected from embeddings)
    // Use original dataset if available and it's a real dataset
    const datasetToUse = originalDataset && originalDataset !== "custom" ? originalDataset : dataset;
    
    if (datasetToUse && datasetToUse !== "custom") {
      // This is a dataset file from built-in or custom datasets
      const filename = encodeURIComponent(selectedFile.filename);
      
      // Handle custom datasets vs built-in datasets
      if (datasetToUse.startsWith('custom:')) {
        // Custom dataset: use the original route /{dataset}/file/{filename}
        // The backend handles the custom dataset format properly
        return `${API_BASE}/${encodeURIComponent(datasetToUse)}/file/${filename}`;
      } else {
        // Built-in dataset: use /{dataset}/file/{filename}
        return `${API_BASE}/${encodeURIComponent(datasetToUse)}/file/${filename}`;
      }
    } else {
      // Fallback to upload endpoint when dataset is "custom" (generic case)
      return `${API_BASE}/upload/file/${selectedFile.file_id}`;
    }
  })();

  // Get current file info (original or perturbed) with better data handling
  const currentFileInfo = (() => {
    if (showPerturbed && perturbationResult?.success) {
      return {
        filename: perturbationResult.filename,
        duration: perturbationResult.duration_ms / 1000,
        sample_rate: perturbationResult.sample_rate,
        size: undefined
      };
    }
    
    // For original file, try to get the most accurate data
    if (selectedFile) {
      return {
        filename: selectedFile.filename,
        duration: selectedFile.duration || undefined,
        sample_rate: selectedFile.sample_rate || undefined,
        size: selectedFile.size || undefined
      };
    }
    
    return null;
  })();

  // Add a state to track audio metadata from wavesurfer
  const [audioMetadata, setAudioMetadata] = useState<{
    duration?: number;
    sampleRate?: number;
  }>({});

  // Debug logging for selectedFile and audioUrl
  useEffect(() => {
    console.log('DatapointEditorPanel - selectedFile changed:', selectedFile);
    console.log('DatapointEditorPanel - dataset:', dataset);
    console.log('DatapointEditorPanel - originalDataset:', originalDataset);
    console.log('DatapointEditorPanel - audioUrl:', audioUrl);
  }, [selectedFile, audioUrl, dataset, originalDataset]);

  // Reset playback when file changes or when switching between original/perturbed
  useEffect(() => {
    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(0);
    setAudioMetadata({}); // Reset metadata when file changes
    
    // Reset wavesurfer instance if it exists
    if (wavesurferRef.current) {
      wavesurferRef.current.stop();
    }
  }, [selectedFile?.file_id, dataset, showPerturbed, perturbationResult?.filename]);
  
  return (
    <div className="h-full bg-gray-50 border-l border-gray-200 flex flex-col">
      <div className="bg-gray-100 p-3 border-b border-gray-200">
        <h3 className="font-medium text-sm text-gray-800">Datapoint Editor</h3>
      </div>
      
      <div className="flex-1 p-3 overflow-auto space-y-4">
        {/* Sample Info - Top */}
        <Card className="border-gray-200 bg-white">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm text-gray-800">Sample Info</CardTitle>
              {perturbationResult?.success && (
                <div className="flex items-center gap-1 p-1 bg-gray-100 rounded-lg">
                  <Button
                    variant={!showPerturbed ? "default" : "ghost"}
                    size="sm"
                    onClick={() => setShowPerturbed(false)}
                    className={`text-xs h-7 px-3 transition-all duration-200 ${
                      !showPerturbed 
                        ? 'bg-blue-600 hover:bg-blue-700 text-white shadow-sm border-0 scale-[1.02]' 
                        : 'text-gray-600 hover:bg-gray-200 hover:text-gray-800'
                    }`}
                  >
                    Original
                  </Button>
                  <Button
                    variant={showPerturbed ? "default" : "ghost"}
                    size="sm"
                    onClick={() => setShowPerturbed(true)}
                    className={`text-xs h-7 px-3 transition-all duration-200 ${
                      showPerturbed 
                        ? 'bg-blue-600 hover:bg-blue-700 text-white shadow-sm border-0 scale-[1.02]' 
                        : 'text-gray-600 hover:bg-gray-200 hover:text-gray-800'
                    }`}
                  >
                    Perturbed
                  </Button>
                </div>
              )}
            </div>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="text-xs">
              <span className="text-gray-600">File:</span>
              <span className="ml-2 font-mono text-gray-800">{currentFileInfo?.filename || "No file selected"}</span>
              {showPerturbed && (
                <Badge variant="secondary" className="ml-2 text-[10px] bg-blue-100 text-blue-700 border-blue-200">P</Badge>
              )}
            </div>
            <div className="text-xs">
              <span className="text-gray-600">Duration:</span>
              <span className="ml-2 text-gray-800">
                {currentFileInfo?.duration 
                  ? `${currentFileInfo.duration.toFixed(1)}s` 
                  : audioMetadata.duration 
                  ? `${audioMetadata.duration.toFixed(1)}s` 
                  : "Loading..."}
              </span>
            </div>
            <div className="text-xs">
              <span className="text-gray-600">Sample Rate:</span>
              <span className="ml-2 text-gray-800">
                {currentFileInfo?.sample_rate 
                  ? `${(currentFileInfo.sample_rate / 1000).toFixed(1)}kHz` 
                  : audioMetadata.sampleRate 
                  ? `${(audioMetadata.sampleRate / 1000).toFixed(1)}kHz` 
                  : "Loading..."}
              </span>
            </div>
            {currentFileInfo?.size && (
              <div className="text-xs">
                <span className="text-gray-600">Size:</span>
                <span className="ml-2 text-gray-800">{(currentFileInfo.size / 1024 / 1024).toFixed(2)} MB</span>
              </div>
            )}
            {showPerturbed && perturbationResult?.applied_perturbations && (
              <div className="text-xs">
                <span className="text-gray-600">Applied:</span>
                <div className="ml-2 mt-1 space-y-1">
                  {perturbationResult.applied_perturbations.map((pert, idx) => (
                    <Badge key={idx} variant="outline" className="text-[10px] mr-1 border-blue-300 text-blue-700">
                      {pert.type.replace('_', ' ')}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
            {showPerturbed && perturbationResult?.filename && predictionMap && (
              <div className="text-xs mt-2">
                <span className="text-gray-600">Perturbed Prediction:</span>
                <div className="ml-2 mt-1">
                  <Badge variant="secondary" className="text-[10px] bg-blue-100 text-blue-700 border-blue-200">
                    {predictionMap[perturbationResult.filename] || "Loading..."}
                  </Badge>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Predictions Section - Middle */}
        <PredictionDisplay
          selectedFile={selectedFile}
          selectedEmbeddingFile={selectedEmbeddingFile}
          model={model}
          wav2vecPrediction={wav2vecPrediction}
          whisperPrediction={whisperPrediction}
          perturbedPredictions={perturbedPredictions}
          isLoading={isLoadingPredictions}
          isLoadingPerturbed={isLoadingPerturbed}
          error={predictionError}
        />

        {/* Audio Player & Waveform - Bottom */}
        <Card className="border-gray-200 bg-white">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-gray-800">Audio Playback</CardTitle>
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
                
                // Update metadata state for file info display
                setAudioMetadata({
                  duration: duration,
                  sampleRate: wavesurfer.getDecodedData()?.sampleRate || undefined
                });
              }}
              onProgress={(time, dur) => {
                setCurrentTime(time);
                setDuration(dur);
                
                // Update duration in metadata if not already set
                if (!audioMetadata.duration && dur > 0) {
                  setAudioMetadata(prev => ({ ...prev, duration: dur }));
                }
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