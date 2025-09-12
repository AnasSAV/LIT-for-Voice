import React, { useState, useEffect, useCallback } from 'react';
import Plot from 'react-plotly.js';
import { useEmbedding } from '@/contexts/EmbeddingContext';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

interface ScalersVisualizationProps {
  model?: string;
  dataset?: string;
}

interface Wav2Vec2BatchPrediction {
  aggregated_probabilities: Record<string, number>;
  individual_predictions: Array<{
    filename: string;
    predicted_emotion: string;
    probabilities: Record<string, number>;
    confidence: number;
  }>;
  summary: {
    total_files: number;
    dominant_emotion: string;
    dominant_confidence: number;
  };
}

export const ScalersVisualization = ({ model, dataset }: ScalersVisualizationProps) => {
  const { embeddingData, isLoading, error } = useEmbedding();
  const [selectedPoints, setSelectedPoints] = useState<string[]>([]);
  const [batchPrediction, setBatchPrediction] = useState<Wav2Vec2BatchPrediction | null>(null);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [reductionMethod, setReductionMethod] = useState("pca");
  const [selectionMode, setSelectionMode] = useState<"box" | "lasso">("box");

  // Get the 2D coordinates from the embedding data
  const get2DCoordinates = () => {
    if (!embeddingData || !embeddingData.reduced_embeddings) return { x: [], y: [], text: [] };
    
    const coordinates = embeddingData.reduced_embeddings;
    if (!coordinates || coordinates.length === 0) return { x: [], y: [], text: [] };
    
    return {
      x: coordinates.map(point => point.coordinates[0]),
      y: coordinates.map(point => point.coordinates[1]),
      text: coordinates.map(point => point.filename)
    };
  };

  // Handle point selection (box select or lasso)
  const handleSelection = useCallback((event: any) => {
    if (event.points && event.points.length > 0) {
      const selectedFiles = event.points.map((point: any) => point.text);
      setSelectedPoints(selectedFiles);
    }
  }, []);

  // Clear selection
  const clearSelection = () => {
    setSelectedPoints([]);
    setBatchPrediction(null);
  };

  // Fetch batch predictions for selected points
  const fetchBatchPredictions = async () => {
    if (selectedPoints.length === 0 || model !== 'wav2vec2') return;

    setPredictionLoading(true);
    setPredictionError(null);
    try {
      const requestBody: any = {
        filenames: selectedPoints,
      };

      if (dataset) {
        requestBody.dataset = dataset;
      }

      console.log('Fetching batch predictions for:', requestBody);

      const response = await fetch("http://localhost:8000/inferences/wav2vec2-batch", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Batch prediction error:', response.status, errorText);
        throw new Error(`Failed to fetch batch predictions: ${response.status} - ${errorText}`);
      }

      const prediction = await response.json();
      setBatchPrediction(prediction);
    } catch (error) {
      console.error("Error fetching batch predictions:", error);
      setPredictionError(error instanceof Error ? error.message : 'Unknown error occurred');
      setBatchPrediction(null);
    } finally {
      setPredictionLoading(false);
    }
  };

  // Auto-fetch predictions when selection changes
  useEffect(() => {
    if (selectedPoints.length > 0) {
      fetchBatchPredictions();
    }
  }, [selectedPoints, model, dataset]);

  const { x, y, text } = get2DCoordinates();

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <div className="text-sm text-gray-600">Loading embedding data...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center text-red-600">
          <div className="text-sm">Error loading embeddings</div>
          <div className="text-xs mt-1">{error}</div>
        </div>
      </div>
    );
  }

  if (!embeddingData) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center text-gray-600">
          <div className="text-sm">No embedding data available</div>
          <div className="text-xs mt-1">Generate embeddings first in the main panel</div>
        </div>
      </div>
    );
  }

  // Create colors for selected vs unselected points
  const colors = text.map(filename => 
    selectedPoints.includes(filename) ? '#ef4444' : '#3b82f6'
  );

  const trace = {
    x,
    y,
    mode: 'markers' as const,
    type: 'scatter' as const,
    text,
    hovertemplate: '%{text}<extra></extra>',
    marker: {
      size: text.map(filename => selectedPoints.includes(filename) ? 10 : 6),
      color: colors,
      opacity: 0.7,
      line: { width: 1, color: 'white' }
    },
    name: 'Audio Files'
  };

  const layout = {
    autosize: true,
    margin: { l: 40, r: 40, t: 40, b: 40 },
    plot_bgcolor: 'white',
    paper_bgcolor: 'white',
    showlegend: false,
    font: { size: 10, color: '#374151' },
    xaxis: { 
      title: 'Component 1',
      gridcolor: '#e5e7eb',
      showgrid: true
    },
    yaxis: { 
      title: 'Component 2',
      gridcolor: '#e5e7eb',
      showgrid: true
    },
    dragmode: selectionMode === 'box' ? 'select' : 'lasso',
    selectdirection: 'any'
  };

  const config = {
    displayModeBar: true,
    modeBarButtonsToRemove: [
      'zoom2d', 'pan2d', 'autoScale2d', 'resetScale2d',
      'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleHover'
    ],
    displaylogo: false,
    responsive: true
  };

  return (
    <div className="h-full flex flex-col space-y-3">
      {/* Controls */}
      <div className="flex flex-wrap gap-2 items-center text-xs">
        <Select value={reductionMethod} onValueChange={setReductionMethod}>
          <SelectTrigger className="w-24 h-7">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="pca">PCA</SelectItem>
            <SelectItem value="umap">UMAP</SelectItem>
            <SelectItem value="tsne">t-SNE</SelectItem>
          </SelectContent>
        </Select>
        
        <Select value={selectionMode} onValueChange={(value: "box" | "lasso") => setSelectionMode(value)}>
          <SelectTrigger className="w-20 h-7">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="box">Box</SelectItem>
            <SelectItem value="lasso">Lasso</SelectItem>
          </SelectContent>
        </Select>

        <Button 
          variant="outline" 
          size="sm" 
          onClick={clearSelection}
          disabled={selectedPoints.length === 0}
          className="h-7 text-xs"
        >
          Clear ({selectedPoints.length})
        </Button>
      </div>

      <div className="flex-1 flex gap-3">
        {/* 2D Plot */}
        <div className="flex-1 border border-gray-200 rounded-lg bg-white">
          <Plot
            data={[trace]}
            layout={layout}
            config={config}
            style={{ width: '100%', height: '100%' }}
            onSelected={handleSelection}
            onDeselect={() => setSelectedPoints([])}
          />
        </div>

        {/* Prediction Results */}
        <div className="w-80 space-y-3">
          {model === 'wav2vec2' && selectedPoints.length > 0 && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">
                  Selected Files Analysis ({selectedPoints.length})
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {predictionLoading ? (
                  <div className="text-xs text-gray-600">Loading predictions...</div>
                ) : predictionError ? (
                  <div className="text-xs text-red-600">
                    <div className="font-medium">Error loading predictions:</div>
                    <div className="mt-1">{predictionError}</div>
                    <div className="mt-2 text-gray-600">
                      Make sure the backend server is running and the wav2vec2-batch endpoint is available.
                    </div>
                  </div>
                ) : batchPrediction ? (
                  <>
                    {/* Summary */}
                    <div className="space-y-2">
                      <div className="text-xs font-medium">Dominant Emotion</div>
                      <div className="flex items-center gap-2">
                        <Badge variant="default" className="text-xs">
                          {batchPrediction.summary.dominant_emotion}
                        </Badge>
                        <span className="text-xs text-gray-600">
                          {(batchPrediction.summary.dominant_confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>

                    {/* Aggregated Probabilities */}
                    <div className="space-y-2">
                      <div className="text-xs font-medium">Combined Probabilities</div>
                      <div className="space-y-1">
                        {Object.entries(batchPrediction.aggregated_probabilities)
                          .sort(([,a], [,b]) => b - a)
                          .map(([emotion, probability]) => (
                            <div key={emotion} className="space-y-1">
                              <div className="flex justify-between text-xs">
                                <span className="capitalize">{emotion}</span>
                                <span>{(probability * 100).toFixed(1)}%</span>
                              </div>
                              <Progress 
                                value={probability * 100} 
                                className="h-1"
                              />
                            </div>
                          ))}
                      </div>
                    </div>

                    {/* Individual Files */}
                    <div className="space-y-2">
                      <div className="text-xs font-medium">Individual Files</div>
                      <div className="max-h-32 overflow-y-auto space-y-1">
                        {batchPrediction.individual_predictions.map((pred, index) => (
                          <div key={index} className="text-xs p-2 bg-gray-50 rounded border">
                            <div className="font-mono text-blue-700 truncate">
                              {pred.filename}
                            </div>
                            <div className="flex items-center gap-2 mt-1">
                              <Badge variant="outline" className="text-xs">
                                {pred.predicted_emotion}
                              </Badge>
                              <span className="text-gray-600">
                                {(pred.confidence * 100).toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </>
                ) : null}
              </CardContent>
            </Card>
          )}

          {model !== 'wav2vec2' && selectedPoints.length > 0 && (
            <Card>
              <CardContent className="pt-6">
                <div className="text-xs text-gray-600 text-center">
                  Batch analysis is currently only available for wav2vec2 model.
                  <div className="mt-2">
                    Selected files: {selectedPoints.length}
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};