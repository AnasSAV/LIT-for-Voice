import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { AudioPlayer } from "../audio/AudioPlayer";
import { WaveformViewer } from "../audio/WaveformViewer";
import { Play, Pause, RotateCcw, Trash2, Plus } from "lucide-react";

export const DatapointEditorPanel = () => {
  const [selectedLabel, setSelectedLabel] = useState<string>("neutral");
  const [isPlaying, setIsPlaying] = useState(false);
  
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
              <span className="ml-2 font-mono">audio_sample_001.wav</span>
            </div>
            <div className="text-xs">
              <span className="text-muted-foreground">Duration:</span>
              <span className="ml-2">3.2s</span>
            </div>
            <div className="text-xs">
              <span className="text-muted-foreground">Sample Rate:</span>
              <span className="ml-2">16kHz</span>
            </div>
          </CardContent>
        </Card>

        {/* Audio Player & Waveform */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Audio Playback</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <WaveformViewer />
            <AudioPlayer 
              isPlaying={isPlaying}
              onPlayPause={() => setIsPlaying(!isPlaying)}
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
                neutral (0.87)
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