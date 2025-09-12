import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { SaliencyVisualization } from "../visualization/SaliencyVisualization";
import { PerturbationTools } from "../analysis/PerturbationTools";
import { AttentionVisualization } from "../visualization/AttentionVisualization";
import { ScalersVisualization } from "../visualization/ScalersVisualization";
import { useState, useEffect } from "react";

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
}

interface PredictionPanelProps {
  selectedFile?: UploadedFile | null;
  selectedEmbeddingFile?: string | null;
  model?: string;
  dataset?: string;
}

export const PredictionPanel = ({ selectedFile, selectedEmbeddingFile, model, dataset }: PredictionPanelProps) => {
  const [wav2vecPrediction, setWav2vecPrediction] = useState<Wav2Vec2Prediction | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch wav2vec2 prediction when model is wav2vec2 and file is selected
  useEffect(() => {
    const fetchWav2vecPrediction = async () => {
      if (model !== "wav2vec2" || (!selectedFile && !selectedEmbeddingFile)) {
        setWav2vecPrediction(null);
        setError(null);
        return;
      }

      setIsLoading(true);
      setError(null);

      try {
        let requestBody: any = {};
        
        if (selectedFile) {
          // Use selectedFile if available
          if (dataset === "custom") {
            requestBody.file_path = selectedFile.file_path;
          } else {
            requestBody.dataset = dataset;
            requestBody.dataset_file = selectedFile.filename;
          }
        } else if (selectedEmbeddingFile && dataset) {
          // Use embedding file selection
          requestBody.dataset = dataset;
          requestBody.dataset_file = selectedEmbeddingFile;
        }

        const response = await fetch("http://localhost:8000/inferences/wav2vec2-detailed", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch prediction: ${response.status}`);
        }

        const prediction = await response.json();
        setWav2vecPrediction(prediction);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "Unknown error";
        setError(errorMessage);
        console.error("Error fetching wav2vec2 prediction:", err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchWav2vecPrediction();
  }, [selectedFile, selectedEmbeddingFile, model, dataset]);
  return (
    <div className="h-full panel-background border-t panel-border">
      <Tabs defaultValue="predictions" className="h-full">
        <div className="panel-header border-b panel-border px-3 py-2">
          <TabsList className="h-7 grid grid-cols-5 w-full">
            <TabsTrigger value="predictions" className="text-xs">Predictions</TabsTrigger>
            <TabsTrigger value="scalers" className="text-xs">Scalers</TabsTrigger>
            <TabsTrigger value="saliency" className="text-xs">Saliency</TabsTrigger>
            <TabsTrigger value="attention" className="text-xs">Attention</TabsTrigger>
            <TabsTrigger value="perturbation" className="text-xs">Perturbation</TabsTrigger>
          </TabsList>
        </div>
        
        <div className="h-[calc(100%-2.5rem)] overflow-auto">
          <TabsContent value="predictions" className="m-0 h-full">
            <div className="p-3 space-y-3">
              {/* Selected File Information */}
              {(selectedFile || selectedEmbeddingFile) && (
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">Selected File</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <div className="text-xs">
                      <div className="font-medium text-blue-800 mb-2">
                        {selectedFile ? selectedFile.filename : selectedEmbeddingFile}
                      </div>
                      {selectedFile && (
                        <div className="space-y-1 text-gray-600">
                          <div><span className="font-medium">File ID:</span> {selectedFile.file_id}</div>
                          <div><span className="font-medium">Path:</span> {selectedFile.file_path}</div>
                          {selectedFile.size && (
                            <div><span className="font-medium">Size:</span> {(selectedFile.size / 1024).toFixed(1)} KB</div>
                          )}
                          {selectedFile.duration && (
                            <div><span className="font-medium">Duration:</span> {selectedFile.duration.toFixed(2)}s</div>
                          )}
                          {selectedFile.sample_rate && (
                            <div><span className="font-medium">Sample Rate:</span> {selectedFile.sample_rate} Hz</div>
                          )}
                        </div>
                      )}
                      {model === "wav2vec2" && wav2vecPrediction && !isLoading && (
                        <div className="mt-3 p-2 bg-blue-50 rounded border border-blue-200">
                          <div className="text-xs">
                            <div className="font-medium text-blue-800 mb-1">Predicted Emotion</div>
                            <div className="flex items-center gap-2">
                              <Badge variant="default" className="text-xs capitalize">
                                {wav2vecPrediction.predicted_emotion}
                              </Badge>
                              <span className="text-blue-700 font-medium">
                                {(wav2vecPrediction.confidence * 100).toFixed(1)}% confidence
                              </span>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              )}

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">
                    Classification Results
                    {model === "wav2vec2" && (
                      <Badge variant="secondary" className="ml-2 text-[10px]">
                        Wav2Vec2 Emotion
                      </Badge>
                    )}
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {isLoading && (
                    <div className="text-xs text-gray-500 flex items-center gap-2">
                      <div className="w-3 h-3 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                      Loading prediction...
                    </div>
                  )}
                  
                  {error && (
                    <div className="text-xs text-red-500 p-2 bg-red-50 rounded border">
                      Error: {error}
                    </div>
                  )}
                  
                  {model === "wav2vec2" && wav2vecPrediction && !isLoading ? (
                    // Display wav2vec2 emotion predictions
                    Object.entries(wav2vecPrediction.probabilities)
                      .sort(([,a], [,b]) => b - a) // Sort by probability descending
                      .map(([emotion, probability]) => {
                        const isPredicted = emotion === wav2vecPrediction.predicted_emotion;
                        return (
                          <div key={emotion} className="flex items-center justify-between text-xs">
                            <div className="flex items-center gap-2">
                              <span className="capitalize">{emotion}</span>
                              {isPredicted && <Badge variant="default" className="text-[10px] px-1">P</Badge>}
                            </div>
                            <div className="flex items-center gap-2 flex-1 max-w-[120px]">
                              <Progress value={probability * 100} className="h-2" />
                              <span className="text-muted-foreground min-w-[2rem]">
                                {(probability * 100).toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        );
                      })
                  ) : model !== "wav2vec2" ? (
                    // Display placeholder/mock data for other models
                    [
                      { label: "Neutral", probability: 0.87, isTrue: true, isPredicted: true },
                      { label: "Happy", probability: 0.08, isTrue: false, isPredicted: false },
                      { label: "Sad", probability: 0.03, isTrue: false, isPredicted: false },
                      { label: "Angry", probability: 0.02, isTrue: false, isPredicted: false },
                    ].map((item, idx) => (
                      <div key={idx} className="flex items-center justify-between text-xs">
                        <div className="flex items-center gap-2">
                          <span>{item.label}</span>
                          {item.isPredicted && <Badge variant="default" className="text-[10px] px-1">P</Badge>}
                          {item.isTrue && <Badge variant="outline" className="text-[10px] px-1">T</Badge>}
                        </div>
                        <div className="flex items-center gap-2 flex-1 max-w-[120px]">
                          <Progress value={item.probability * 100} className="h-2" />
                          <span className="text-muted-foreground min-w-[2rem]">
                            {(item.probability * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                    ))
                  ) : null}
                </CardContent>
              </Card>
            </div>
          </TabsContent>
          
          <TabsContent value="scalers" className="m-0 h-full">
            <div className="p-3 h-full">
              <ScalersVisualization 
                model={model}
                dataset={dataset}
              />
            </div>
          </TabsContent>
          
          <TabsContent value="saliency" className="m-0 h-full">
            <div className="p-3">
              <SaliencyVisualization />
            </div>
          </TabsContent>
          
          <TabsContent value="attention" className="m-0 h-full">
            <div className="p-3">
              <AttentionVisualization />
            </div>
          </TabsContent>
          
          <TabsContent value="perturbation" className="m-0 h-full">
            <div className="p-3">
              <PerturbationTools />
            </div>
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
};