import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { CheckCircle, XCircle, PlayCircle } from "lucide-react";

interface PredictionResult {
  filename: string;
  prediction_type: "emotion" | "transcript";
  emotion_prediction?: string;
  transcript?: string;
  ground_truth_emotion?: string;
  ground_truth_transcript?: string;
  metadata: Record<string, string | number>;
}

interface PredictionResultsProps {
  results: PredictionResult[];
  model: string;
  dataset: string;
  onPlayFile?: (filename: string) => void;
}

export const PredictionResults = ({ results, model, dataset, onPlayFile }: PredictionResultsProps) => {
  const [selectedResult, setSelectedResult] = useState<PredictionResult | null>(null);

  const calculateAccuracy = () => {
    if (results.length === 0) return 0;
    
    const correct = results.filter(result => {
      if (result.prediction_type === "emotion") {
        return result.emotion_prediction?.toLowerCase() === result.ground_truth_emotion?.toLowerCase();
      } else {
        // For transcripts, you might want a more sophisticated comparison
        return result.transcript?.toLowerCase().trim() === result.ground_truth_transcript?.toLowerCase().trim();
      }
    }).length;
    
    return (correct / results.length) * 100;
  };

  const getEmotionDistribution = () => {
    const distribution: Record<string, number> = {};
    results.forEach(result => {
      if (result.prediction_type === "emotion" && result.emotion_prediction) {
        distribution[result.emotion_prediction] = (distribution[result.emotion_prediction] || 0) + 1;
      }
    });
    return distribution;
  };

  const accuracy = calculateAccuracy();
  const emotionDist = getEmotionDistribution();

  return (
    <div className="h-full flex flex-col space-y-4 p-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Prediction Results</h3>
        <Badge variant="outline">
          {results.length} files processed
        </Badge>
      </div>

      {/* Overall Statistics */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Overall Performance</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm">Accuracy</span>
            <span className="text-sm font-mono">{accuracy.toFixed(1)}%</span>
          </div>
          <Progress value={accuracy} className="h-2" />
          
          <div className="grid grid-cols-2 gap-4 pt-2">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{results.filter(r => r.prediction_type === "emotion" ? r.emotion_prediction?.toLowerCase() === r.ground_truth_emotion?.toLowerCase() : r.transcript?.toLowerCase().trim() === r.ground_truth_transcript?.toLowerCase().trim()).length}</div>
              <div className="text-xs text-muted-foreground">Correct</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600">{results.length - results.filter(r => r.prediction_type === "emotion" ? r.emotion_prediction?.toLowerCase() === r.ground_truth_emotion?.toLowerCase() : r.transcript?.toLowerCase().trim() === r.ground_truth_transcript?.toLowerCase().trim()).length}</div>
              <div className="text-xs text-muted-foreground">Incorrect</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Emotion Distribution (for emotion prediction) */}
      {Object.keys(emotionDist).length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Predicted Emotion Distribution</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1">
            {Object.entries(emotionDist).map(([emotion, count]) => (
              <div key={emotion} className="flex items-center justify-between text-sm">
                <Badge variant="secondary">{emotion}</Badge>
                <span>{count} files</span>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Results List */}
      <Card className="flex-1">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Individual Results</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <ScrollArea className="h-[300px]">
            <div className="space-y-1 p-3">
              {results.map((result, index) => {
                const isCorrect = result.prediction_type === "emotion" 
                  ? result.emotion_prediction?.toLowerCase() === result.ground_truth_emotion?.toLowerCase()
                  : result.transcript?.toLowerCase().trim() === result.ground_truth_transcript?.toLowerCase().trim();
                
                return (
                  <div
                    key={index}
                    className={`p-2 border rounded-md cursor-pointer hover:bg-muted/50 ${
                      selectedResult?.filename === result.filename ? 'bg-muted' : ''
                    }`}
                    onClick={() => setSelectedResult(result)}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        {isCorrect ? (
                          <CheckCircle className="h-4 w-4 text-green-600" />
                        ) : (
                          <XCircle className="h-4 w-4 text-red-600" />
                        )}
                        <span className="font-mono text-xs">{result.filename}</span>
                        {onPlayFile && (
                          <Button
                            size="sm"
                            variant="ghost"
                            className="h-6 w-6 p-0"
                            onClick={(e) => {
                              e.stopPropagation();
                              onPlayFile(result.filename);
                            }}
                          >
                            <PlayCircle className="h-3 w-3" />
                          </Button>
                        )}
                      </div>
                    </div>
                    
                    {result.prediction_type === "emotion" ? (
                      <div className="mt-1 space-y-1">
                        <div className="flex items-center gap-2 text-xs">
                          <span>Predicted:</span>
                          <Badge variant="default" className="text-xs">{result.emotion_prediction}</Badge>
                        </div>
                        <div className="flex items-center gap-2 text-xs">
                          <span>Ground Truth:</span>
                          <Badge variant="outline" className="text-xs">{result.ground_truth_emotion}</Badge>
                        </div>
                        {result.transcript && (
                          <div className="text-xs text-muted-foreground">
                            <span>Transcript: </span>
                            <span className="italic">"{result.transcript}"</span>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="mt-1 space-y-1">
                        <div className="text-xs">
                          <span>Predicted: </span>
                          <span className="italic">"{result.transcript}"</span>
                        </div>
                        <div className="text-xs text-muted-foreground">
                          <span>Ground Truth: </span>
                          <span className="italic">"{result.ground_truth_transcript}"</span>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
};
