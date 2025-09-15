
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { SaliencyVisualization } from "../visualization/SaliencyVisualization";
import { PerturbationTools } from "../analysis/PerturbationTools";
import { AttentionVisualization } from "../visualization/AttentionVisualization";
import { ScalersVisualization } from "../visualization/ScalersVisualization";
import React, { useState, useEffect } from "react";


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

interface WhisperPrediction {
  predicted_transcript: string;
  ground_truth: string;
  accuracy_percentage: number;
  word_error_rate: number;
  character_error_rate: number;
  levenshtein_distance: number;
  exact_match: number;
  character_similarity: number;
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

interface PredictionPanelProps {
  selectedFile?: UploadedFile | null;
  selectedEmbeddingFile?: string | null;
  model?: string;
  dataset?: string;
  originalDataset?: string;
  onPerturbationComplete?: (result: PerturbationResult) => void;
  onPredictionRefresh?: (file: UploadedFile, prediction: string) => void;
  onPredictionUpdate?: (fileId: string, prediction: string) => void;
}

export const PredictionPanel = ({ selectedFile, selectedEmbeddingFile, model, dataset, originalDataset, onPerturbationComplete, onPredictionRefresh, onPredictionUpdate }: PredictionPanelProps) => {
  const [wav2vecPrediction, setWav2vecPrediction] = useState<Wav2Vec2Prediction | null>(null);
  const [whisperPrediction, setWhisperPrediction] = useState<WhisperPrediction | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [perturbedPredictions, setPerturbedPredictions] = useState<Wav2Vec2Prediction | WhisperPrediction | null>(null);
  const [originalFile, setOriginalFile] = useState<UploadedFile | null>(selectedFile || null);
  const [perturbedFile, setPerturbedFile] = useState<UploadedFile | null>(null);
  const [isLoadingPerturbed, setIsLoadingPerturbed] = useState(false);


  // Handle perturbation completion
  const handlePerturbationComplete = async (result: PerturbationResult) => {
    if (!result.success) {
      console.error("Perturbation failed:", result.error);
      return;
    }

    // Create a file-like object for the perturbed audio
    const perturbedFileObj: UploadedFile = {
      file_id: result.filename,
      filename: result.filename,
      file_path: result.perturbed_file,
      message: "Perturbed audio",
      duration: result.duration_ms / 1000,
      sample_rate: result.sample_rate
    };
    
    setPerturbedFile(perturbedFileObj);
    
    // Notify parent component about perturbation completion
    if (onPerturbationComplete) {
      onPerturbationComplete(result);
    }
    
    // Run inference on the perturbed audio
    await runInferenceOnPerturbed(perturbedFileObj);
  };

  // Run inference on perturbed audio
  const runInferenceOnPerturbed = async (perturbedFile: UploadedFile) => {
    if (!model) return;

    setIsLoadingPerturbed(true);
    setError(null);

    try {
      console.log("DEBUG: Running inference on perturbed file:", perturbedFile);
      
      let response;
      let prediction;
      
      if (model === "wav2vec2") {
        const requestBody = {
          file_path: perturbedFile.file_path
        };

        response = await fetch("http://localhost:8000/inferences/wav2vec2-detailed", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch perturbed wav2vec2 prediction: ${response.status}`);
        }

        prediction = await response.json();
      } else if (model?.includes("whisper")) {
        const requestBody = {
          model: model,
          file_path: perturbedFile.file_path
        };

        response = await fetch("http://localhost:8000/inferences/whisper-accuracy", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch perturbed whisper prediction: ${response.status}`);
        }

        prediction = await response.json();
      }

      console.log("DEBUG: Perturbed prediction result:", prediction);
      setPerturbedPredictions(prediction);
      
      // Extract prediction text and notify parent component
      let predictionText = "";
      if (model?.includes("whisper")) {
        // For whisper, extract the transcription text
        predictionText = prediction?.transcript || prediction?.prediction || "";
      } else if (model === "wav2vec2") {
        // For wav2vec2, extract the emotion prediction
        predictionText = prediction?.emotion || prediction?.prediction || "";
      }
      
      if (predictionText && onPredictionRefresh) {
        console.log("DEBUG: Calling onPredictionRefresh for perturbed file:", perturbedFile.filename, predictionText);
        onPredictionRefresh(perturbedFile, predictionText);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Unknown error";
      setError(errorMessage);
      console.error("Error fetching perturbed prediction:", err);
    } finally {
      setIsLoadingPerturbed(false);
    }
  };

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
          // Check if this is an uploaded file - more precise detection
          const isUploadedFile = selectedFile.file_path && (
            selectedFile.file_path.includes('uploads/') || 
            selectedFile.file_path.startsWith('uploads/') ||
            selectedFile.message === "Perturbed file" ||
            selectedFile.message === "File uploaded successfully" ||
            selectedFile.message === "File uploaded and processed successfully"
          ) && selectedFile.message !== "Selected from dataset";
          
          if (isUploadedFile) {
            // This is an uploaded file, use file_path
            requestBody.file_path = selectedFile.file_path;
          } else {
            // This is a dataset file, use originalDataset and dataset_file
            requestBody.dataset = originalDataset || dataset;
            requestBody.dataset_file = selectedFile.filename;
          }
        } else if (selectedEmbeddingFile && dataset) {
          // Use embedding file selection
          requestBody.dataset = originalDataset || dataset;
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
        
        // Update predictionMap for uploaded files (same as dataset files)
        if (selectedFile && onPredictionUpdate) {
          const isUploadedFile = selectedFile.file_path && (
            selectedFile.file_path.includes('uploads/') || 
            selectedFile.file_path.startsWith('uploads/') ||
            selectedFile.message === "Perturbed file" ||
            selectedFile.message === "File uploaded successfully" ||
            selectedFile.message === "File uploaded and processed successfully"
          ) && selectedFile.message !== "Selected from embeddings" && selectedFile.message !== "Selected from dataset";
          
          if (isUploadedFile) {
            const predictionText = typeof prediction === 'string' ? prediction : 
              prediction?.predicted_emotion || prediction?.prediction || prediction?.emotion || JSON.stringify(prediction);
            console.log("DEBUG: PredictionPanel - Calling onPredictionUpdate for Wav2Vec2:", selectedFile.file_id, predictionText);
            onPredictionUpdate(selectedFile.file_id, predictionText);
          }
        }
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

  // Fetch whisper prediction when model includes whisper and file is selected
  useEffect(() => {
    const fetchWhisperPrediction = async () => {
      if (!model?.includes("whisper") || (!selectedFile && !selectedEmbeddingFile)) {
        setWhisperPrediction(null);
        setError(null);
        return;
      }

      setIsLoading(true);
      setError(null);

      try {
        let requestBody: any = {
          model: model
        };
        
        let isUploadedFile = false;
        
        if (selectedFile) {
          // Check if this is an uploaded file - more precise detection
          isUploadedFile = selectedFile.file_path && (
            selectedFile.file_path.includes('uploads/') || 
            selectedFile.file_path.startsWith('uploads/') ||
            selectedFile.message === "Perturbed file" ||
            selectedFile.message === "File uploaded successfully" ||
            selectedFile.message === "File uploaded and processed successfully"
          ) && selectedFile.message !== "Selected from dataset";
          
          if (isUploadedFile) {
            // This is an uploaded file, use file_path
            requestBody.file_path = selectedFile.file_path;
          } else {
            // This is a dataset file, use originalDataset and dataset_file
            requestBody.dataset = originalDataset || dataset;
            requestBody.dataset_file = selectedFile.filename;
          }
        } else if (selectedEmbeddingFile && dataset) {
          // Use embedding file selection - this is a dataset file
          requestBody.dataset = originalDataset || dataset;
          requestBody.dataset_file = selectedEmbeddingFile;
          isUploadedFile = false;
        }

        // Choose the correct endpoint based on file type
        let endpoint: string;
        if (isUploadedFile) {
          // For uploaded files, use basic inference endpoint (no ground truth available)
          endpoint = "http://localhost:8000/inferences/run";
        } else {
          // For dataset files, use accuracy endpoint to get ground truth and metrics
          endpoint = "http://localhost:8000/inferences/whisper-accuracy";
        }

        const response = await fetch(endpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch whisper prediction: ${response.status}`);
        }

        const prediction = await response.json();
        
        let whisperPrediction: WhisperPrediction;
        
        if (isUploadedFile) {
          // For uploaded files, convert basic prediction to expected format
          whisperPrediction = {
            predicted_transcript: typeof prediction === 'string' ? prediction : prediction?.text || JSON.stringify(prediction),
            ground_truth: "",
            accuracy_percentage: 0,
            word_error_rate: 0,
            character_error_rate: 0,
            levenshtein_distance: 0,
            exact_match: 0,
            character_similarity: 0,
            word_count_predicted: 0,
            word_count_truth: 0
          };
        } else {
          // For dataset files, the accuracy endpoint returns all the metrics
          whisperPrediction = {
            predicted_transcript: prediction.predicted_transcript || "",
            ground_truth: prediction.ground_truth || "",
            accuracy_percentage: prediction.accuracy_percentage || 0,
            word_error_rate: prediction.word_error_rate || 0,
            character_error_rate: prediction.character_error_rate || 0,
            levenshtein_distance: prediction.levenshtein_distance || 0,
            exact_match: prediction.exact_match || 0,
            character_similarity: prediction.character_similarity || 0,
            word_count_predicted: prediction.word_count_predicted || 0,
            word_count_truth: prediction.word_count_truth || 0
          };
        }
        
        setWhisperPrediction(whisperPrediction);
        
        // Update predictionMap for uploaded files (same as dataset files)
        if (selectedFile && onPredictionUpdate && isUploadedFile) {
          console.log("DEBUG: PredictionPanel - Calling onPredictionUpdate for Whisper:", selectedFile.file_id, whisperPrediction.predicted_transcript);
          onPredictionUpdate(selectedFile.file_id, whisperPrediction.predicted_transcript);
        }
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "Unknown error";
        setError(errorMessage);
        console.error("Error fetching whisper prediction:", err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchWhisperPrediction();
  }, [selectedFile, selectedEmbeddingFile, model, dataset, originalDataset]);

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

              <Card className="border-gray-200 bg-white">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">
                    {model === "wav2vec2" ? "Classification Results" : model?.includes("whisper") ? "Transcription Results" : "Prediction Results"}
                    {model === "wav2vec2" && (
                      <Badge variant="secondary" className="ml-2 text-[10px]">
                        Wav2Vec2 Emotion
                      </Badge>
                    )}
                    {model?.includes("whisper") && (
                      <Badge variant="secondary" className="ml-2 text-[10px]">
                        {model.includes("large") ? "Whisper Large" : "Whisper Base"}
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
                    // Display wav2vec2 emotion predictions with comparison
                    <div className="space-y-3">
                      {/* Original Prediction */}
                      <div className="space-y-2">
                        <div className="text-xs font-medium flex items-center gap-2">
                          Original Audio
                          <Badge variant="outline" className="text-[10px]">O</Badge>
                        </div>
                        {Object.entries(wav2vecPrediction.probabilities)
                          .sort(([,a], [,b]) => b - a)
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
                          })}
                      </div>

                      {/* Perturbed Prediction */}
                      {perturbedPredictions && (
                        <div className="space-y-2 border-t pt-3">
                          <div className="text-xs font-medium flex items-center gap-2">
                            Perturbed Audio
                            <Badge variant="secondary" className="text-[10px]">P</Badge>
                            {isLoadingPerturbed && (
                              <div className="w-3 h-3 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                            )}
                          </div>
                          {!isLoadingPerturbed && Object.entries((perturbedPredictions as Wav2Vec2Prediction).probabilities)
                            .sort(([,a], [,b]) => b - a)
                            .map(([emotion, probability]) => {
                              const isPredicted = emotion === (perturbedPredictions as Wav2Vec2Prediction).predicted_emotion;
                              const originalProb = wav2vecPrediction.probabilities[emotion] || 0;
                              const change = (probability - originalProb) * 100;
                              return (
                                <div key={emotion} className="flex items-center justify-between text-xs">
                                  <div className="flex items-center gap-2">
                                    <span className="capitalize">{emotion}</span>
                                    {isPredicted && <Badge variant="secondary" className="text-[10px] px-1">P</Badge>}
                                  </div>
                                  <div className="flex items-center gap-2 flex-1 max-w-[120px]">
                                    <Progress value={probability * 100} className="h-2" />
                                    <span className="text-muted-foreground min-w-[2rem]">
                                      {(probability * 100).toFixed(1)}%
                                    </span>
                                    <span className={`text-[10px] min-w-[2rem] ${
                                      change > 0 ? "text-green-600" : change < 0 ? "text-red-600" : "text-muted-foreground"
                                    }`}>
                                      {change > 0 ? "+" : ""}{change.toFixed(1)}%
                                    </span>
                                  </div>
                                </div>
                              );
                            })}
                        </div>
                      )}
                    </div>
                  ) : model?.includes("whisper") && whisperPrediction && !isLoading ? (
                    // Display whisper transcription accuracy with comparison
                    <div className="space-y-3">
                      {/* Original Prediction */}
                      <div className="space-y-2">
                        <div className="text-xs font-medium flex items-center gap-2">
                          Original Audio
                          <Badge variant="outline" className="text-[10px]">O</Badge>
                        </div>
                        
                        {/* Accuracy Metrics */}
                        {whisperPrediction.ground_truth && (
                          <div className="grid grid-cols-2 gap-2 text-xs text-gray-600 mb-2">
                            <div>WER: {whisperPrediction.word_error_rate.toFixed(3)}</div>
                            <div>CER: {whisperPrediction.character_error_rate.toFixed(3)}</div>
                            <div>Words P: {whisperPrediction.word_count_predicted}</div>
                            <div>Words T: {whisperPrediction.word_count_truth}</div>
                            <div>Accuracy: {whisperPrediction.accuracy_percentage.toFixed(1)}%</div>
                            <div>Levenshtein: {whisperPrediction.levenshtein_distance}</div>
                          </div>
                        )}

                        {/* Predicted Transcript */}
                        <div className="space-y-1">
                          <div className="text-xs font-medium flex items-center gap-2">
                            Predicted
                            <Badge variant="outline" className="text-[10px] px-1">P</Badge>
                          </div>
                          <div className="text-xs p-2 bg-blue-50 rounded border font-mono">
                            "{whisperPrediction.predicted_transcript}"
                          </div>
                        </div>

                        {/* Ground Truth */}
                        {whisperPrediction.ground_truth && (
                          <div className="space-y-1">
                            <div className="text-xs font-medium flex items-center gap-2">
                              Ground Truth
                              <Badge variant="outline" className="text-[10px] px-1">T</Badge>
                            </div>
                            <div className="text-xs p-2 bg-green-50 rounded border font-mono">
                              "{whisperPrediction.ground_truth}"
                            </div>
                          </div>
                        )}
                      </div>

                      {/* Perturbed Prediction */}
                      {perturbedPredictions && (
                        <div className="space-y-2 border-t pt-3">
                          <div className="text-xs font-medium flex items-center gap-2">
                            Perturbed Audio
                            <Badge variant="secondary" className="text-[10px]">P</Badge>
                            {isLoadingPerturbed && (
                              <div className="w-3 h-3 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                            )}
                          </div>
                          
                          {!isLoadingPerturbed && (
                            <>
                              {/* Perturbed Accuracy Metrics */}
                              {typeof perturbedPredictions === 'object' && perturbedPredictions !== null ? (
                                <div className="grid grid-cols-2 gap-2 text-xs text-gray-600">
                                  <div>WER: {(perturbedPredictions as WhisperPrediction).word_error_rate?.toFixed(3) || 'N/A'}</div>
                                  <div>CER: {(perturbedPredictions as WhisperPrediction).character_error_rate?.toFixed(3) || 'N/A'}</div>
                                  <div>Words P: {(perturbedPredictions as WhisperPrediction).word_count_predicted || 'N/A'}</div>
                                  <div>Words T: {(perturbedPredictions as WhisperPrediction).word_count_truth || 'N/A'}</div>
                                  <div>Accuracy: {(perturbedPredictions as WhisperPrediction).accuracy_percentage?.toFixed(1) || 'N/A'}%</div>
                                  <div>Levenshtein: {(perturbedPredictions as WhisperPrediction).levenshtein_distance || 'N/A'}</div>
                                </div>
                              ) : (
                                <div className="text-xs text-gray-600">
                                  <div>Prediction: {typeof perturbedPredictions === 'string' ? perturbedPredictions : JSON.stringify(perturbedPredictions)}</div>
                                </div>
                              )}

                              {/* Perturbed Predicted Transcript */}
                              <div className="space-y-1">
                                <div className="text-xs font-medium flex items-center gap-2">
                                  Predicted
                                  <Badge variant="secondary" className="text-[10px] px-1">P</Badge>
                                </div>
                                <div className="text-xs p-2 bg-purple-50 rounded border font-mono">
                                  "{(perturbedPredictions as WhisperPrediction).predicted_transcript}"
                                </div>
                              </div>
                            </>
                          )}
                        </div>
                      )}
                    </div>
                  ) : !model?.includes("whisper") && model !== "wav2vec2" ? (
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
              <PerturbationTools 
                selectedFile={selectedFile} 
                onPerturbationComplete={handlePerturbationComplete}
                onPredictionRefresh={onPredictionRefresh}
                model={model}
                dataset={dataset}
                originalDataset={originalDataset}
              />
            </div>
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
};