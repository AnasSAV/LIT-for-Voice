import React, { useState, useEffect, useCallback } from 'react';
import Plot from 'react-plotly.js';
import { useEmbedding } from '@/contexts/EmbeddingContext';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { API_BASE } from '@/lib/api';

interface ScalersVisualizationProps {
  model?: string;
  dataset?: string;
}

interface Wav2Vec2BatchPrediction {
  emotion_distribution: Record<string, number>;  // Percentage of files predicted as each emotion
  emotion_counts: Record<string, number>;        // Raw counts for each emotion
  individual_predictions: Array<{
    filename: string;
    predicted_emotion: string;
    probabilities: Record<string, number>;
    confidence: number;
  }>;
  summary: {
    total_files: number;
    dominant_emotion: string;
    dominant_count: number;
    dominant_percentage: number;
  };
  cache_info: {
    cached_count: number;
    missing_count: number;
    cache_hit_rate: number;
  };
}

interface WhisperBatchAnalysis {
  common_terms: Array<{
    term: string;
    count: number;
    percentage: number;
  }>;
  individual_transcripts: Array<{
    filename: string;
    transcript: string;
    word_count: number;
  }>;
  summary: {
    total_files: number;
    total_words: number;
    unique_words: number;
    avg_words_per_file: number;
  };
  cache_info: {
    cached_count: number;
    missing_count: number;
    cache_hit_rate: number;
  };
}

interface AudioFrequencyAnalysis {
  model_context: string;
  individual_analyses: Array<{
    filename: string;
    features: Record<string, number>;
  }>;
  aggregate_statistics: Record<string, {
    mean: number;
    std: number;
    min: number;
    max: number;
    median: number;
  }>;
  feature_distributions: Record<string, {
    histogram: number[];
    bins: number[];
  }>;
  most_common_features: Array<{
    feature: string;
    normalized_mean: number;
    stability_score: number;
    prevalence_score: number;
    mean: number;
    std: number;
  }>;
  feature_categories: Record<string, string[]>;
  summary: {
    total_files: number;
    total_features_extracted: number;
    avg_duration: number;
    avg_tempo: number;
  };
  cache_info: {
    cached_count: number;
    missing_count: number;
    cache_hit_rate: number;
  };
}

export const ScalersVisualization = ({ model, dataset }: ScalersVisualizationProps) => {
  const { embeddingData, isLoading, error } = useEmbedding();
  const [selectedPoints, setSelectedPoints] = useState<string[]>([]);
  const [batchPrediction, setBatchPrediction] = useState<Wav2Vec2BatchPrediction | null>(null);
  const [whisperAnalysis, setWhisperAnalysis] = useState<WhisperBatchAnalysis | null>(null);
  const [audioFrequencyAnalysis, setAudioFrequencyAnalysis] = useState<AudioFrequencyAnalysis | null>(null);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [reductionMethod, setReductionMethod] = useState("pca");
  const [selectionMode, setSelectionMode] = useState<"box" | "lasso">("box");
  const [analysisType, setAnalysisType] = useState<"default" | "frequency">("default");

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
    setWhisperAnalysis(null);
    setAudioFrequencyAnalysis(null);
  };

  // Fetch audio frequency analysis for selected points
  const fetchAudioFrequencyAnalysis = async () => {
    if (selectedPoints.length === 0) return;

    setPredictionLoading(true);
    setPredictionError(null);
    try {
      const requestBody: any = {
        filenames: selectedPoints,
        model: model,
      };

      if (dataset) {
        requestBody.dataset = dataset;
      }

      console.log('Fetching audio frequency analysis for:', requestBody);

      const response = await fetch("http://localhost:8000/inferences/audio-frequency-batch", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Audio frequency analysis error:', response.status, errorText);
        throw new Error(`Failed to fetch audio frequency analysis: ${response.status} - ${errorText}`);
      }

      const analysis = await response.json();
      setAudioFrequencyAnalysis(analysis);
    } catch (error) {
      console.error("Error fetching audio frequency analysis:", error);
      setPredictionError(error instanceof Error ? error.message : 'Unknown error occurred');
      setAudioFrequencyAnalysis(null);
    } finally {
      setPredictionLoading(false);
    }
  };

  // Fetch whisper batch analysis for selected points
  const fetchWhisperAnalysis = async () => {
    if (selectedPoints.length === 0 || !model?.includes('whisper')) return;

    setPredictionLoading(true);
    setPredictionError(null);
    try {
      const requestBody: any = {
        filenames: selectedPoints,
        model: model,
      };

      if (dataset) {
        requestBody.dataset = dataset;
      }

      console.log('Fetching whisper analysis for:', requestBody);

      const response = await fetch(`${API_BASE}/inferences/whisper-batch`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: 'include',
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Whisper analysis error:', response.status, errorText);
        throw new Error(`Failed to fetch whisper analysis: ${response.status} - ${errorText}`);
      }

      const analysis = await response.json();
      setWhisperAnalysis(analysis);
    } catch (error) {
      console.error("Error fetching whisper analysis:", error);
      setPredictionError(error instanceof Error ? error.message : 'Unknown error occurred');
      setWhisperAnalysis(null);
    } finally {
      setPredictionLoading(false);
    }
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

      const response = await fetch(`${API_BASE}/inferences/wav2vec2-batch`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: 'include',
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
      if (analysisType === "frequency") {
        fetchAudioFrequencyAnalysis();
      } else if (model === 'wav2vec2') {
        fetchBatchPredictions();
      } else if (model?.includes('whisper')) {
        fetchWhisperAnalysis();
      }
    }
  }, [selectedPoints, model, dataset, analysisType]);

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
        
        <Select value={selectionMode} onValueChange={(value: "box" | "lasso") => setSelectionMode(value)}>
          <SelectTrigger className="w-20 h-7">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="box">Box</SelectItem>
            <SelectItem value="lasso">Lasso</SelectItem>
          </SelectContent>
        </Select>

        <Select value={analysisType} onValueChange={(value: "default" | "frequency") => setAnalysisType(value)}>
          <SelectTrigger className="w-28 h-7">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="default">
              {model === 'wav2vec2' ? 'Predictions' : 'Common Terms'}
            </SelectItem>
            <SelectItem value="frequency">Audio Features</SelectItem>
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
        {/* 2D Plot - Fixed Height */}
        <div className="flex-1 border border-gray-200 rounded-lg bg-white" style={{ height: '400px' }}>
          <Plot
            data={[trace]}
            layout={layout}
            config={config}
            style={{ width: '100%', height: '100%' }}
            onSelected={handleSelection}
            onDeselect={() => setSelectedPoints([])}
          />
        </div>

        {/* Analysis Results - Single Consolidated Card */}
        <div className="w-80">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">
                {selectedPoints.length === 0 ? "Analysis Results" : (
                  analysisType === "frequency" ? "Audio Frequency Analysis" : 
                  model === 'wav2vec2' ? "Emotion Predictions" : "Transcript Analysis"
                )} {selectedPoints.length > 0 && `(${selectedPoints.length})`}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {selectedPoints.length === 0 ? (
                <div className="text-xs text-gray-600 text-center">
                  Select points on the plot to see analysis results
                </div>
              ) : predictionLoading ? (
                <div className="text-xs text-gray-600">
                  Loading {analysisType === "frequency" ? "audio features" : 
                          model === 'wav2vec2' ? "predictions" : "analysis"}...
                </div>
              ) : predictionError ? (
                <div className="text-xs text-red-600">
                  <div className="font-medium">Error loading analysis:</div>
                  <div className="mt-1">{predictionError}</div>
                  <div className="mt-2 text-gray-600">
                    Make sure the backend server is running.
                  </div>
                </div>
              ) : analysisType === "frequency" && audioFrequencyAnalysis ? (
                <>
                  {/* Audio Frequency Analysis Results */}
                  {/* Cache Info */}
                  {audioFrequencyAnalysis.cache_info && (
                    <div className="space-y-2">
                      <div className="text-xs font-medium">Cache Performance</div>
                      <div className="text-xs text-blue-600 bg-blue-50 p-2 rounded">
                        <div className="flex justify-between">
                          <span>Cache hits:</span>
                          <span>{audioFrequencyAnalysis.cache_info.cached_count}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>New extractions:</span>
                          <span>{audioFrequencyAnalysis.cache_info.missing_count}</span>
                        </div>
                        <div className="flex justify-between font-medium">
                          <span>Hit rate:</span>
                          <span>{(audioFrequencyAnalysis.cache_info.cache_hit_rate * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Summary */}
                  <div className="space-y-2">
                    <div className="text-xs font-medium">Summary</div>
                    <div className="grid grid-cols-2 gap-2 text-xs text-gray-600">
                      <div>Files: {audioFrequencyAnalysis.summary.total_files}</div>
                      <div>Features: {audioFrequencyAnalysis.summary.total_features_extracted}</div>
                      <div>Avg Duration: {audioFrequencyAnalysis.summary.avg_duration.toFixed(1)}s</div>
                      <div>Avg Tempo: {audioFrequencyAnalysis.summary.avg_tempo.toFixed(0)} BPM</div>
                    </div>
                  </div>

                  {/* Most Common Features */}
                  <div className="space-y-2">
                    <div className="text-xs font-medium">Top 5 Most Common Features</div>
                    <div className="space-y-1">
                      {audioFrequencyAnalysis.most_common_features.slice(0, 5).map((feature, index) => (
                        <div key={index} className="space-y-1">
                          <div className="flex justify-between text-xs">
                            <span className="font-mono text-blue-700 truncate">{feature.feature.replace(/_/g, ' ')}</span>
                            <span>Score: {feature.prevalence_score.toFixed(2)}</span>
                          </div>
                          <div className="text-xs text-gray-500">
                            μ={feature.mean.toFixed(2)}, σ={feature.std.toFixed(2)}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Feature Categories */}
                  <div className="space-y-2">
                    <div className="text-xs font-medium">Feature Categories</div>
                    <div className="space-y-1">
                      {Object.entries(audioFrequencyAnalysis.feature_categories)
                        .filter(([_, features]) => features.length > 0)
                        .map(([category, features]) => (
                          <div key={category} className="text-xs">
                            <div className="flex items-center gap-2">
                              <Badge variant="outline" className="text-xs capitalize">
                                {category}
                              </Badge>
                              <span className="text-gray-600">{features.length} features</span>
                            </div>
                          </div>
                        ))}
                    </div>
                  </div>

                  {/* Individual Files */}
                  <div className="space-y-2">
                    <div className="text-xs font-medium">Individual Files</div>
                    <div className="max-h-32 overflow-y-auto space-y-1">
                      {audioFrequencyAnalysis.individual_analyses.slice(0, 10).map((analysis, index) => (
                        <div key={index} className="text-xs p-2 bg-gray-50 rounded border">
                          <div className="font-mono text-blue-700 truncate">
                            {analysis.filename}
                          </div>
                          <div className="text-gray-600 mt-1 flex justify-between">
                            <span>{analysis.features.duration?.toFixed(1)}s</span>
                            <span>{analysis.features.tempo?.toFixed(0)} BPM</span>
                          </div>
                        </div>
                      ))}
                      {audioFrequencyAnalysis.individual_analyses.length > 10 && (
                        <div className="text-xs text-gray-500 text-center pt-1">
                          ... and {audioFrequencyAnalysis.individual_analyses.length - 10} more files
                        </div>
                      )}
                    </div>
                  </div>
                </>
              ) : analysisType === "default" && model === 'wav2vec2' && batchPrediction ? (
                <>
                  {/* Wav2Vec2 Emotion Analysis */}
                  {/* Cache Info */}
                  {batchPrediction.cache_info && (
                    <div className="space-y-2">
                      <div className="text-xs font-medium">Cache Performance</div>
                      <div className="text-xs text-blue-600 bg-blue-50 p-2 rounded">
                        <div className="flex justify-between">
                          <span>Cache hits:</span>
                          <span>{batchPrediction.cache_info.cached_count}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>New predictions:</span>
                          <span>{batchPrediction.cache_info.missing_count}</span>
                        </div>
                        <div className="flex justify-between font-medium">
                          <span>Hit rate:</span>
                          <span>{(batchPrediction.cache_info.cache_hit_rate * 100).toFixed(1)}%</span>
                        </div>
                        {batchPrediction.cache_info.missing_count === 0 && (
                          <div className="text-green-700 mt-1 text-center">
                            ✓ All predictions loaded from cache
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Summary */}
                  <div className="space-y-2">
                    <div className="text-xs font-medium">Dominant Emotion</div>
                    <div className="flex items-center gap-2">
                      <Badge variant="default" className="text-xs">
                        {batchPrediction.summary.dominant_emotion}
                      </Badge>
                      <span className="text-xs text-gray-600">
                        {(batchPrediction.summary.dominant_percentage * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  {/* Emotion Distribution */}
                  <div className="space-y-2">
                    <div className="text-xs font-medium">Emotion Distribution</div>
                    <div className="space-y-1">
                      {Object.entries(batchPrediction.emotion_distribution)
                        .sort(([,a], [,b]) => b - a)
                        .map(([emotion, percentage]) => (
                          <div key={emotion} className="space-y-1">
                            <div className="flex justify-between text-xs">
                              <span className="capitalize">{emotion}</span>
                              <span>{(percentage * 100).toFixed(1)}% ({batchPrediction.emotion_counts[emotion]} files)</span>
                            </div>
                            <Progress 
                              value={percentage * 100} 
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
              ) : analysisType === "default" && model?.includes('whisper') && whisperAnalysis ? (
                <>
                  {/* Whisper Transcript Analysis */}
                  {/* Summary */}
                  <div className="space-y-2">
                    <div className="text-xs font-medium">Summary</div>
                    <div className="grid grid-cols-2 gap-2 text-xs text-gray-600">
                      <div>Total Words: {whisperAnalysis.summary.total_words}</div>
                      <div>Unique Words: {whisperAnalysis.summary.unique_words}</div>
                      <div>Avg/File: {whisperAnalysis.summary.avg_words_per_file.toFixed(1)}</div>
                      <div>Files: {whisperAnalysis.summary.total_files}</div>
                    </div>
                    {whisperAnalysis.cache_info && (
                      <div className="text-xs text-blue-600 bg-blue-50 p-2 rounded">
                        Cache: {whisperAnalysis.cache_info.cached_count} found, {whisperAnalysis.cache_info.missing_count} missing
                        {whisperAnalysis.cache_info.missing_count > 0 && (
                          <div className="text-blue-700 mt-1">
                            Run inference first for missing files to get complete analysis
                          </div>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Top Common Terms */}
                  <div className="space-y-2">
                    <div className="text-xs font-medium">Top 5 Common Terms</div>
                    <div className="space-y-1">
                      {whisperAnalysis.common_terms.slice(0, 5).map((term, index) => (
                        <div key={index} className="space-y-1">
                          <div className="flex justify-between text-xs">
                            <span className="font-mono text-blue-700">"{term.term}"</span>
                            <span>{term.percentage.toFixed(1)}% ({term.count}x)</span>
                          </div>
                          <Progress 
                            value={term.percentage} 
                            className="h-1"
                          />
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Individual Transcripts */}
                  <div className="space-y-2">
                    <div className="text-xs font-medium">Individual Transcripts</div>
                    <div className="max-h-32 overflow-y-auto space-y-1">
                      {whisperAnalysis.individual_transcripts.map((transcript, index) => (
                        <div key={index} className="text-xs p-2 bg-gray-50 rounded border">
                          <div className="font-mono text-blue-700 truncate">
                            {transcript.filename}
                          </div>
                          <div className="text-gray-600 mt-1 text-xs">
                            {transcript.word_count} words
                          </div>
                          <div className="text-gray-800 mt-1 text-xs italic line-clamp-2">
                            "{transcript.transcript}"
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              ) : selectedPoints.length > 0 && !model?.includes('whisper') && model !== 'wav2vec2' ? (
                <div className="text-xs text-gray-600 text-center">
                  {analysisType === "frequency" ? (
                    <div>
                      Audio frequency analysis is available for all models.
                      <div className="mt-2">
                        Selected files: {selectedPoints.length}
                      </div>
                    </div>
                  ) : (
                    <div>
                      Batch analysis is available for wav2vec2 (emotions) and whisper (transcripts) models.
                      <div className="mt-2">
                        Selected files: {selectedPoints.length}
                      </div>
                    </div>
                  )}
                </div>
              ) : null}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};