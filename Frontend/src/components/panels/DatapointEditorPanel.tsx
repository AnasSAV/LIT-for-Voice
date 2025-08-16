import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { AudioPlayer } from "../audio/AudioPlayer";
import { WaveformViewer } from "../audio/WaveformViewer";
import { Play, Pause, RotateCcw, Trash2, Plus, Brain, Loader2 } from "lucide-react";
import { useAudio } from "@/contexts/AudioContext";
import { toast } from "sonner";

export const DatapointEditorPanel = () => {
  const [selectedLabel, setSelectedLabel] = useState<string>("neutral");
  const { 
    currentAudio, 
    runPrediction, 
    isLoadingPrediction 
  } = useAudio();

  const handleRunPrediction = async (model: string) => {
    if (!currentAudio) {
      toast.error("Please select an audio file first");
      return;
    }

    try {
      await runPrediction(currentAudio.id, model);
      toast.success(`${model} prediction completed`);
    } catch (error) {
      toast.error(`Failed to run ${model} prediction`);
      console.error('Prediction error:', error);
    }
  };

  const formatFileSize = (bytes: number) => {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    
    return `${size.toFixed(1)} ${units[unitIndex]}`;
  };

  const formatDuration = (duration: number | undefined) => {
    if (!duration) return "0:00";
    const mins = Math.floor(duration / 60);
    const secs = Math.floor(duration % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getLatestPrediction = (type: 'transcription' | 'emotion') => {
    if (!currentAudio?.predictions) return null;
    
    for (const pred of currentAudio.predictions) {
      if (type === 'transcription' && pred.transcription) {
        return { text: pred.transcription, model: pred.model };
      }
      if (type === 'emotion' && pred.emotion) {
        return { text: pred.emotion, model: pred.model };
      }
    }
    return null;
  };
  
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
            {currentAudio ? (
              <>
                <div className="text-xs">
                  <span className="text-muted-foreground">File:</span>
                  <span className="ml-2 font-mono" title={currentAudio.name}>
                    {currentAudio.name.length > 25 
                      ? `${currentAudio.name.substring(0, 25)}...` 
                      : currentAudio.name}
                  </span>
                </div>
                <div className="text-xs">
                  <span className="text-muted-foreground">Duration:</span>
                  <span className="ml-2">{formatDuration(currentAudio.duration)}</span>
                </div>
                <div className="text-xs">
                  <span className="text-muted-foreground">Size:</span>
                  <span className="ml-2">{formatFileSize(currentAudio.file.size)}</span>
                </div>
                <div className="text-xs">
                  <span className="text-muted-foreground">Type:</span>
                  <span className="ml-2">{currentAudio.file.type}</span>
                </div>
              </>
            ) : (
              <div className="text-xs text-muted-foreground text-center py-2">
                No audio file selected
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
            <WaveformViewer />
            <AudioPlayer />
          </CardContent>
        </Card>

        {/* Model Predictions */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Model Predictions</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid grid-cols-1 gap-2">
              <Button
                size="sm"
                variant="outline"
                className="w-full h-8 justify-start"
                onClick={() => handleRunPrediction('whisper-base')}
                disabled={!currentAudio || isLoadingPrediction}
              >
                {isLoadingPrediction ? (
                  <Loader2 className="h-3 w-3 mr-2 animate-spin" />
                ) : (
                  <Brain className="h-3 w-3 mr-2" />
                )}
                Whisper Base (ASR)
              </Button>
              
              <Button
                size="sm"
                variant="outline"
                className="w-full h-8 justify-start"
                onClick={() => handleRunPrediction('whisper-large')}
                disabled={!currentAudio || isLoadingPrediction}
              >
                {isLoadingPrediction ? (
                  <Loader2 className="h-3 w-3 mr-2 animate-spin" />
                ) : (
                  <Brain className="h-3 w-3 mr-2" />
                )}
                Whisper Large (ASR)
              </Button>
              
              <Button
                size="sm"
                variant="outline"
                className="w-full h-8 justify-start"
                onClick={() => handleRunPrediction('wav2vec2')}
                disabled={!currentAudio || isLoadingPrediction}
              >
                {isLoadingPrediction ? (
                  <Loader2 className="h-3 w-3 mr-2 animate-spin" />
                ) : (
                  <Brain className="h-3 w-3 mr-2" />
                )}
                Wav2Vec2 (Emotion)
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Transcription Results */}
        {getLatestPrediction('transcription') && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">
                Transcription 
                <Badge variant="outline" className="ml-2 text-xs">
                  {getLatestPrediction('transcription')?.model}
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-xs">
                <p className="p-2 bg-muted rounded text-foreground">
                  {getLatestPrediction('transcription')?.text}
                </p>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Emotion Results */}
        {getLatestPrediction('emotion') && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">
                Emotion Recognition
                <Badge variant="outline" className="ml-2 text-xs">
                  {getLatestPrediction('emotion')?.model}
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-xs">
                <Badge variant="default" className="text-xs">
                  {getLatestPrediction('emotion')?.text}
                </Badge>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Label Editor */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Label Editor</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
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
              <Button size="sm" className="w-full h-7" disabled={!currentAudio}>
                <Plus className="h-3 w-3 mr-1" />
                Save Label
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
      </div>
    </div>
  );
};