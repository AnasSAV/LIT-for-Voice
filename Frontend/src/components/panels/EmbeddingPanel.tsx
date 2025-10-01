import { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip";
import { EmbeddingPlot } from "../visualization/EmbeddingPlot";
import { ScalarPlot } from "../visualization/ScalarPlot";
import { useEmbedding } from "../../contexts/EmbeddingContext";
import { RefreshCw, Eye, Box, Square, BarChart3, HelpCircle } from "lucide-react";
import { getFeatureExplanation } from "@/lib/audioFeatures";

interface EmbeddingPanelProps {
  model?: string;
  dataset?: string;
  availableFiles?: string[];
  selectedFile?: string | null;
  onFileSelect?: (filename: string) => void;
}

// Audio Frequency Analysis interface (reusing from ScalersVisualization)
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

export const EmbeddingPanel = ({ model = "whisper-base", dataset = "common-voice", availableFiles = [], selectedFile, onFileSelect }: EmbeddingPanelProps) => {
  const [reductionMethod, setReductionMethod] = useState("pca");
  const [is3D, setIs3D] = useState(false);
  const [selectedByAngle, setSelectedByAngle] = useState<string[]>([]);
  const [audioFrequencyAnalysis, setAudioFrequencyAnalysis] = useState<AudioFrequencyAnalysis | null>(null);
  const [isLoadingFrequency, setIsLoadingFrequency] = useState(false);
  const [frequencyError, setFrequencyError] = useState<string | null>(null);
  const debounceRef = useRef<NodeJS.Timeout | null>(null);
  const { embeddingData, isLoading, error, fetchEmbeddings, clearEmbeddings } = useEmbedding();

  // Auto-fetch embeddings when model, dataset, or reduction method changes
  useEffect(() => {
    if (availableFiles.length > 0 && model && dataset) {
      const filesToProcess = availableFiles;
      const nComponents = is3D ? 3 : 2;
      fetchEmbeddings(model, dataset, filesToProcess, reductionMethod, nComponents);
    }
  }, [model, dataset, availableFiles, reductionMethod, fetchEmbeddings]);

  // Cleanup debounce timer on unmount
  useEffect(() => {
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, []);

  const handleFetchEmbeddings = () => {
    if (availableFiles.length > 0) {
      // Use entire dataset for better visualization
      const filesToProcess = availableFiles;
      const nComponents = is3D ? 3 : 2;
      fetchEmbeddings(model, dataset, filesToProcess, reductionMethod, nComponents);
    }
  };

  const handleReductionMethodChange = (method: string) => {
    setReductionMethod(method);
    // The useEffect will handle re-fetching when method changes
  };

  const handle3DToggle = (checked: boolean) => {
    setIs3D(checked);
    // Re-fetch with new dimensionality using entire dataset
    if (embeddingData && availableFiles.length > 0) {
      const filesToProcess = availableFiles;
      const nComponents = checked ? 3 : 2;
      fetchEmbeddings(model, dataset, filesToProcess, reductionMethod, nComponents);
    }
  };

  const handlePointSelect = (filename: string, coordinates: number[]) => {
    if (onFileSelect) {
      onFileSelect(filename);
    }
  };

  const handleAngleRangeSelect = (selectedFiles: string[]) => {
    // Only update if the selection has actually changed
    const currentSelection = selectedByAngle.sort().join(',');
    const newSelection = selectedFiles.sort().join(',');
    
    if (currentSelection !== newSelection) {
      setSelectedByAngle(selectedFiles);
      
      // Clear any existing debounce timer
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
      
      // Debounce the frequency analysis fetch to prevent rapid successive calls
      debounceRef.current = setTimeout(() => {
        if (selectedFiles.length > 0) {
          fetchFrequencyAnalysis(selectedFiles);
        } else {
          setAudioFrequencyAnalysis(null);
        }
      }, 300); // 300ms debounce
    }
  };

  // Fetch audio frequency analysis for selected files
  const fetchFrequencyAnalysis = async (filenames: string[]) => {
    if (filenames.length === 0) return;

    setIsLoadingFrequency(true);
    setFrequencyError(null);
    
    try {
      const requestBody: any = {
        filenames: filenames,
        model: model,
      };

      if (dataset) {
        requestBody.dataset = dataset;
      }

      const response = await fetch("http://localhost:8000/inferences/audio-frequency-batch", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to fetch audio frequency analysis: ${response.status} - ${errorText}`);
      }

      const analysis = await response.json();
      setAudioFrequencyAnalysis(analysis);
    } catch (error) {
      console.error("Error fetching audio frequency analysis:", error);
      setFrequencyError(error instanceof Error ? error.message : 'Unknown error occurred');
      setAudioFrequencyAnalysis(null);
    } finally {
      setIsLoadingFrequency(false);
    }
  };

  return (
    <TooltipProvider>
      <div className="h-full bg-white border-r border-gray-200 flex flex-col">
        <div className="panel-header p-4 border-b border-gray-200">
          <h3 className="font-medium text-sm text-gray-800 flex items-center gap-2">
            Audio Embeddings
            <Tooltip>
              <TooltipTrigger>
                <HelpCircle className="h-4 w-4 text-muted-foreground" />
              </TooltipTrigger>
              <TooltipContent>
                Visualize high-dimensional audio features in 2D/3D space
              </TooltipContent>
            </Tooltip>
          </h3>
        </div>
      
      <div className="flex-1 p-4 bg-white overflow-auto">
        <div className="space-y-4">
          {/* Controls Section */}
          <div className="flex-shrink-0 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {embeddingData && (
                  <Badge variant="default" className="text-[10px] bg-blue-100 text-blue-800 border-blue-200">
                    {embeddingData.model.toUpperCase()}
                  </Badge>
                )}
              </div>
              <div className="flex items-center gap-2">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div>
                      <Select value={reductionMethod} onValueChange={handleReductionMethodChange}>
                        <SelectTrigger className="w-20 h-6 text-xs border-gray-300 hover:border-gray-400 transition-colors">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="pca">PCA</SelectItem>
                          <SelectItem value="umap">UMAP</SelectItem>
                          <SelectItem value="tsne">t-SNE</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent>
                    Choose dimensionality reduction method for visualization
                  </TooltipContent>
                </Tooltip>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={handleFetchEmbeddings}
                      disabled={isLoading || availableFiles.length === 0}
                      className="h-6 w-6 p-0 border-gray-300 hover:border-blue-400 hover:bg-blue-50 transition-all duration-200"
                    >
                      <RefreshCw className={`h-3 w-3 ${isLoading ? 'animate-spin text-blue-600' : 'text-gray-600'}`} />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    Refresh embeddings visualization
                  </TooltipContent>
                </Tooltip>
              </div>
            </div>
            
            {/* 3D Toggle */}
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border border-gray-200">
              <div className="flex items-center space-x-2">
                <Switch
                  id="3d-mode"
                  checked={is3D}
                  onCheckedChange={handle3DToggle}
                  disabled={isLoading}
                />
                <Label htmlFor="3d-mode" className="text-xs flex items-center gap-1 font-medium text-gray-700">
                  {is3D ? <Box className="h-3 w-3 text-blue-600" /> : <Square className="h-3 w-3 text-gray-500" />}
                  <span className={is3D ? "text-blue-600" : "text-gray-500"}>
                    {is3D ? '3D View' : '2D View'}
                  </span>
                </Label>
              </div>
            </div>

            {/* Status Messages */}
            {availableFiles.length === 0 && (
              <div className="text-xs text-gray-500 flex items-center gap-2 p-2 bg-yellow-50 rounded border border-yellow-200">
                <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                No files available for embedding extraction
              </div>
            )}
            {availableFiles.length > 0 && !embeddingData && !isLoading && (
              <div className="text-xs text-gray-600 flex items-center gap-2 p-2 bg-blue-50 rounded border border-blue-200">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                Click <RefreshCw className="inline h-3 w-3 mx-1" /> to extract embeddings from all {availableFiles.length} files
              </div>
            )}
            {isLoading && (
              <div className="text-xs text-blue-600 flex items-center gap-2 p-2 bg-blue-50 rounded border border-blue-200">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-ping"></div>
                Processing {availableFiles.length} files... This may take a few moments.
              </div>
            )}
            {error && (
              <div className="text-xs text-red-600 flex items-center gap-2 p-2 bg-red-50 rounded border border-red-200">
                <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                {error}
              </div>
            )}
          </div>

          {/* Embedding Plot */}
          <div className="h-[500px] border border-gray-200 rounded-lg bg-white p-2 overflow-hidden">
            <EmbeddingPlot 
              selectedMethod={reductionMethod} 
              is3D={is3D}
              onPointSelect={handlePointSelect}
              onAngleRangeSelect={handleAngleRangeSelect}
              selectedFile={selectedFile}
            />
          </div>

          {/* Frequency Analysis Panel - Only show when files are selected by angle range */}
          {selectedByAngle.length > 0 && (
            <div className="border border-gray-200 rounded-lg bg-white">
              <Tabs defaultValue="frequency" className="w-full">
                <div className="border-b border-gray-200 px-4 py-2 bg-gray-50">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <BarChart3 className="h-4 w-4 text-blue-600" />
                      <span className="text-sm font-medium">Frequency Analysis</span>
                      <Badge variant="outline" className="text-xs">
                        {selectedByAngle.length} files
                      </Badge>
                    </div>
                  </div>
                </div>
                
                <TabsContent value="frequency" className="mt-0">
                  <div className="p-4 max-h-96 overflow-y-auto">
                    {isLoadingFrequency ? (
                      <div className="text-xs-tight text-gray-600 flex items-center gap-2">
                        <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                        Loading frequency analysis...
                      </div>
                    ) : frequencyError ? (
                      <div className="text-xs-tight text-red-600">
                        <div className="font-medium">Error loading analysis:</div>
                        <div className="mt-1">{frequencyError}</div>
                      </div>
                    ) : audioFrequencyAnalysis ? (
                      <div className="space-y-4">
                        {/* Cache Info
                        {audioFrequencyAnalysis.cache_info && (
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
                        )} */}

                        {/* Summary */}
                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-1">
                            <div className="text-sm-tight font-medium">Summary</div>
                            <div className="space-y-1">
                              <div className="text-xs-tight text-gray-500">
                                <span className="text-gray-700 font-medium">Files:</span> {audioFrequencyAnalysis.summary.total_files}
                              </div>
                              <div className="text-xs-tight text-gray-500">
                                <span className="text-gray-700 font-medium">Features:</span> {audioFrequencyAnalysis.summary.total_features_extracted}
                              </div>
                            </div>
                          </div>
                          <div className="space-y-1">
                            <div className="text-sm-tight font-medium">Audio Metrics</div>
                            <div className="space-y-1">
                              <div className="text-xs-tight text-gray-500">
                                <span className="text-gray-700 font-medium">Avg Duration:</span> {audioFrequencyAnalysis.summary.avg_duration.toFixed(1)}s
                              </div>
                              <div className="text-xs-tight text-gray-500">
                                <span className="text-gray-700 font-medium">Avg Tempo:</span> {audioFrequencyAnalysis.summary.avg_tempo.toFixed(0)} BPM
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* Top Features */}
                        <div className="space-y-2">
                          <div className="text-sm-tight font-medium flex items-center gap-2">
                            Top 5 Most Common Features
                            <Tooltip>
                              <TooltipTrigger>
                                <HelpCircle className="h-3 w-3 text-muted-foreground" />
                              </TooltipTrigger>
                              <TooltipContent className="max-w-xs">
                                Features ranked by prevalence and stability across selected audio files
                              </TooltipContent>
                            </Tooltip>
                          </div>
                          <div className="space-y-2">
                            {audioFrequencyAnalysis.most_common_features.slice(0, 5).map((feature, index) => (
                              <div key={index} className="p-2 bg-gray-50 rounded border">
                                <div className="flex justify-between items-start">
                                  <div className="flex items-center gap-2 flex-1">
                                    <span className="font-mono text-blue-700 text-xs-tight font-medium">
                                      {feature.feature.replace(/_/g, ' ').toUpperCase()}
                                    </span>
                                    <Tooltip>
                                      <TooltipTrigger>
                                        <HelpCircle className="h-3 w-3 text-gray-400 hover:text-gray-600" />
                                      </TooltipTrigger>
                                      <TooltipContent className="max-w-sm">
                                        <div className="space-y-1">
                                          <div className="font-medium text-xs">{feature.feature.replace(/_/g, ' ')}</div>
                                          <div className="text-xs">{getFeatureExplanation(feature.feature)}</div>
                                        </div>
                                      </TooltipContent>
                                    </Tooltip>
                                  </div>
                                  <span className="text-xs-tight text-gray-600">Score: {feature.prevalence_score.toFixed(2)}</span>
                                </div>
                                <Progress 
                                  value={feature.prevalence_score * 100} 
                                  className="h-1 my-1"
                                />
                                <div className="text-xs-tight text-gray-500">
                                  Mean: {feature.mean.toFixed(3)} • Std: {feature.std.toFixed(3)} • Stability: {feature.stability_score.toFixed(2)}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Feature Categories */}
                        <div className="space-y-2">
                          <div className="text-sm-tight font-medium flex items-center gap-2">
                            Feature Categories
                            <Tooltip>
                              <TooltipTrigger>
                                <HelpCircle className="h-3 w-3 text-muted-foreground" />
                              </TooltipTrigger>
                              <TooltipContent className="max-w-xs">
                                Audio features grouped by type: spectral (frequency-based), temporal (time-based), and harmonic (pitch-based)
                              </TooltipContent>
                            </Tooltip>
                          </div>
                          <div className="grid grid-cols-2 gap-2">
                            {Object.entries(audioFrequencyAnalysis.feature_categories)
                              .filter(([_, features]) => features.length > 0)
                              .map(([category, features]) => (
                                <div key={category} className="p-2 bg-gray-50 rounded border">
                                  <div className="flex items-center justify-between">
                                    <Badge variant="outline" className="text-xs-tight capitalize">
                                      {category}
                                    </Badge>
                                    <span className="text-xs-tight text-gray-600">{features.length}</span>
                                  </div>
                                  {features.length <= 3 && (
                                    <div className="mt-1 space-y-1">
                                      {features.map((feature, idx) => (
                                        <div key={idx} className="text-xs-tight text-gray-500 truncate">
                                          {feature.replace(/_/g, ' ')}
                                        </div>
                                      ))}
                                    </div>
                                  )}
                                </div>
                              ))}
                          </div>
                        </div>

                        {/* Selected Files Preview */}
                        <div className="space-y-2">
                          <div className="text-sm-tight font-medium">Selected Files ({selectedByAngle.length} total)</div>
                          <div className="max-h-24 overflow-y-auto space-y-1">
                            {selectedByAngle.slice(0, 5).map((filename, index) => (
                              <div key={index} className="text-xs-tight font-mono text-blue-700 truncate bg-gray-50 px-2 py-1 rounded border">
                                {filename}
                              </div>
                            ))}
                            {selectedByAngle.length > 5 && (
                              <div className="text-xs-tight text-gray-500 text-center">
                                ... and {selectedByAngle.length - 5} more files
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="text-xs-tight text-gray-600 text-center">
                        No frequency analysis data available
                      </div>
                    )}
                  </div>
                </TabsContent>
              </Tabs>
            </div>
          )}

        </div>
      </div>
    </div>
    </TooltipProvider>
  );
};