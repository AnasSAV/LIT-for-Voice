import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useState, useEffect } from "react";
import { API_BASE } from "@/lib/api";

interface AttentionPair {
  from_word: string;
  to_word: string;
  from_time: [number, number];
  to_time: [number, number];
  attention_weight: number;
  from_index: number;
  to_index: number;
}

interface TimestampAttention {
  time: number;
  attention: number;
  max_outgoing?: number;
  avg_incoming?: number;
  self_attention?: number;
  attention_entropy?: number;
  frame_index?: number;
}

interface AttentionPairsVisualizationProps {
  selectedFile: any;
  model: string;
  dataset?: string;
}

export const AttentionPairsVisualization = ({ selectedFile, model, dataset }: AttentionPairsVisualizationProps) => {
  const [selectedLayer, setSelectedLayer] = useState(6);  // Middle layer for better semantic attention patterns
  const [selectedHead, setSelectedHead] = useState(0);
  const [attentionData, setAttentionData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchAttentionData = async () => {
    console.log("AttentionPairsVisualization - fetchAttentionData called:", {
      selectedFile,
      model,
      dataset,
      hasWhisper: model?.includes('whisper')
    });

    if (!selectedFile || !model || !model.includes('whisper')) {
      console.log("AttentionPairsVisualization - Skipping fetch due to conditions");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const requestBody: any = {
        model: model,
        layer: selectedLayer,
        head: selectedHead
      };

      // Handle file path resolution following your patterns
      if (typeof selectedFile === 'string') {
        // Dataset file
        if (dataset) {
          requestBody.dataset = dataset;
          requestBody.dataset_file = selectedFile;
        } else {
          throw new Error("Dataset required for dataset file selection");
        }
      } else if (selectedFile?.file_path) {
        // Check if we have a dataset - if so, this is a dataset file
        if (dataset) {
          // This is a dataset file (either custom or standard dataset)
          requestBody.dataset = dataset;
          requestBody.dataset_file = selectedFile.file_path;
        } else {
          // Regular uploaded file
          requestBody.file_path = selectedFile.file_path;
        }
      } else {
        throw new Error("No valid file selected");
      }

      console.log("AttentionPairsVisualization - Request body:", requestBody);
      console.log("AttentionPairsVisualization - API URL:", `${API_BASE}/inferences/attention-pairs`);

      const response = await fetch(`${API_BASE}/inferences/attention-pairs`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: 'include',
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`${response.status} - ${errorText}`);
      }

      const data = await response.json();
      console.log("AttentionPairsVisualization - Response data:", data);
      setAttentionData(data);

    } catch (err: any) {
      console.error("AttentionPairsVisualization - Error:", err);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchAttentionData();
  }, [selectedFile, model, dataset, selectedLayer, selectedHead]);

  const renderWordPairsMatrix = () => {
    if (!attentionData?.attention_pairs) return null;

    const pairs = attentionData.attention_pairs as AttentionPair[];
    const words = [...new Set(pairs.map(p => p.from_word))];
    
    // Create attention matrix based on actual sequence length, not just unique words
    const maxIndex = Math.max(
      ...pairs.map(p => Math.max(p.from_index, p.to_index)),
      words.length - 1
    );
    const matrixSize = maxIndex + 1;
    
    // Create attention matrix with proper size
    const matrix: number[][] = Array(matrixSize).fill(null).map(() => Array(matrixSize).fill(0));
    
    // Safely populate matrix with bounds checking
    pairs.forEach(pair => {
      const fromIdx = pair.from_index;
      const toIdx = pair.to_index;
      
      // Bounds check to prevent array access errors
      if (fromIdx >= 0 && fromIdx < matrixSize && toIdx >= 0 && toIdx < matrixSize) {
        matrix[fromIdx][toIdx] = pair.attention_weight;
      } else {
        console.warn(`Index out of bounds: from=${fromIdx}, to=${toIdx}, matrixSize=${matrixSize}`);
      }
    });

    // Calculate color scale excluding self-attention for better contrast
    const nonSelfAttentionValues = pairs
      .filter(p => p.from_index !== p.to_index) // Exclude self-attention
      .map(p => p.attention_weight)
      .filter(val => val > 0);
    
    const minNonSelfAttention = Math.min(...nonSelfAttentionValues, 0);
    const maxNonSelfAttention = Math.max(...nonSelfAttentionValues, 1);
    
    // Function to normalize attention values for coloring
    const normalizeAttention = (value: number, isSelfAttention: boolean) => {
      if (isSelfAttention) {
        // Self-attention gets a distinct color (gold/yellow)
        return { intensity: 0.8, isSelf: true };
      } else {
        // Non-self attention uses relative scale
        const normalized = (value - minNonSelfAttention) / (maxNonSelfAttention - minNonSelfAttention);
        return { intensity: Math.max(normalized, 0.1), isSelf: false };
      }
    };

    // Get top pairs for display (exclude self-attention pairs)
    const topPairs = pairs
      .filter(pair => pair.from_index !== pair.to_index) // Exclude self-attention
      .sort((a, b) => b.attention_weight - a.attention_weight)
      .slice(0, 10);

    return (
      <div className="space-y-4">
        {/* Attention Matrix */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Word-to-Word Attention Matrix</CardTitle>
          </CardHeader>
          <CardContent>
            {/* Limit matrix display size for performance and readability */}
            {words.length > 50 ? (
              <div className="text-sm text-muted-foreground p-4">
                Attention matrix too large to display ({words.length}x{words.length}). 
                Showing top attention pairs below instead.
              </div>
            ) : (
              <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${words.length + 1}, minmax(0, 1fr))` }}>
                <div></div>
                {words.map((word, i) => (
                  <div key={i} className="text-xs p-1 text-center font-medium truncate" title={word}>
                    {word.length > 6 ? word.substring(0, 6) + '...' : word}
                  </div>
                ))}
                
                {words.map((fromWord, i) => (
                <>
                  <div key={`row-${i}`} className="text-xs p-1 font-medium truncate" title={fromWord}>
                    {fromWord.length > 8 ? fromWord.substring(0, 8) + '...' : fromWord}
                  </div>
                  {words.map((toWord, j) => {
                    // Safe matrix access with bounds checking
                    const attentionValue = (i < matrix.length && j < matrix[0]?.length) ? matrix[i][j] : 0;
                    const isSelfAttention = i === j;
                    
                    if (isSelfAttention) {
                      // Block out diagonal cells (self-attention)
                      return (
                        <div
                          key={`cell-${i}-${j}`}
                          className="aspect-square border border-gray-300 flex items-center justify-center"
                          style={{
                            backgroundColor: 'rgba(0, 0, 0, 0.1)', // Light black/gray
                            minWidth: '24px',
                            minHeight: '24px'
                          }}
                          title="Self-attention (blocked)"
                        >
                          {/* No value displayed */}
                        </div>
                      );
                    }
                    
                    // For non-diagonal cells, use relative coloring
                    const colorInfo = normalizeAttention(attentionValue, false);
                    
                    return (
                      <div
                        key={`cell-${i}-${j}`}
                        className="aspect-square border border-gray-200 text-xs flex items-center justify-center cursor-pointer hover:border-gray-400 transition-all"
                        style={{
                          backgroundColor: `rgba(59, 130, 246, ${colorInfo.intensity})`,
                          color: colorInfo.intensity > 0.6 ? 'white' : 'black',
                          minWidth: '24px',
                          minHeight: '24px'
                        }}
                        title={`${fromWord} → ${toWord}: ${(attentionValue * 100).toFixed(2)}%`}
                      >
                        {attentionValue > 0.01 ? (attentionValue * 100).toFixed(0) : '0'}
                      </div>
                    );
                  })}
                </>
              ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Top Attention Pairs */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Strongest Attention Relationships</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {topPairs.map((pair, i) => (
                <div key={i} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                  <div className="flex items-center space-x-2">
                    <Badge variant="outline">{pair.from_word}</Badge>
                    <span className="text-muted-foreground">→</span>
                    <Badge variant="outline">{pair.to_word}</Badge>
                  </div>
                  <div className="text-sm font-medium">
                    {(pair.attention_weight * 100).toFixed(1)}%
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    );
  };

  const renderTimelineView = () => {
    if (!attentionData?.timestamp_attention) return null;

    const timestamps = attentionData.timestamp_attention as TimestampAttention[];
    if (timestamps.length === 0) return null;

    try {
      const attentionValues = timestamps.map(t => t.attention || 0);
      
      // Check for valid data
      if (attentionValues.length === 0 || attentionValues.some(val => isNaN(val))) {
        return <div className="text-red-500 p-4">Invalid attention data</div>;
      }
      
      const minVal = Math.min(...attentionValues);
      const maxVal = Math.max(...attentionValues);
      const range = maxVal - minVal;
      
      // Always use full 0-1 scale for maximum visual clarity
      const normalizedValues = attentionValues.map(val => (val - minVal) / (range || 1));
    
    const width = 800;
    const height = 250;
    const padding = 60;

    return (
      <Card>
        <CardHeader>
          <CardTitle>Attention Timeline</CardTitle>
          <p className="text-sm text-muted-foreground">
            Showing attention values over time - scaled to highlight differences
          </p>
        </CardHeader>
        <CardContent>
          <div className="flex justify-center">
            <svg width={width} height={height} className="border rounded bg-white">
              {/* Grid lines for reference */}
              {[0, 0.2, 0.4, 0.6, 0.8, 1].map(ratio => {
                const y = padding + (1 - ratio) * (height - 2 * padding);
                const actualValue = minVal + ratio * range;
                return (
                  <g key={`grid-${ratio}`}>
                    <line
                      x1={padding}
                      y1={y}
                      x2={width - padding}
                      y2={y}
                      stroke={ratio === 0 || ratio === 1 ? "#666" : "#e5e5e5"}
                      strokeWidth={ratio === 0 || ratio === 1 ? "2" : "1"}
                      strokeDasharray={ratio === 0 || ratio === 1 ? "none" : "2,2"}
                    />
                    <text x={padding - 10} y={y + 4} textAnchor="end" className="text-xs fill-gray-600">
                      {(actualValue * 100).toFixed(2)}%
                    </text>
                  </g>
                );
              })}
              
              {/* Smooth attention curve */}
              <polyline
                points={normalizedValues.map((val, i) => {
                  const x = padding + (i / (normalizedValues.length - 1)) * (width - 2 * padding);
                  const y = padding + (1 - val) * (height - 2 * padding);
                  return `${x},${y}`;
                }).join(' ')}
                fill="none"
                stroke="#2563eb"
                strokeWidth="2"
              />
              
              {/* Data points */}
              {normalizedValues.map((val, i) => {
                const x = padding + (i / (normalizedValues.length - 1)) * (width - 2 * padding);
                const y = padding + (1 - val) * (height - 2 * padding);
                const originalVal = attentionValues[i];
                
                return (
                  <circle
                    key={`point-${i}`}
                    cx={x}
                    cy={y}
                    r="3"
                    fill="#2563eb"
                    stroke="white"
                    strokeWidth="1"
                    className="hover:r-4 transition-all"
                  >
                    <title>
                      Time: {timestamps[i].time.toFixed(2)}s
                      Attention: {(originalVal * 100).toFixed(4)}%
                    </title>
                  </circle>
                );
              })}
              
              {/* Time axis */}
              {[0, 0.25, 0.5, 0.75, 1].map(ratio => {
                const x = padding + ratio * (width - 2 * padding);
                const timeIndex = Math.floor(ratio * (timestamps.length - 1));
                const time = timestamps[timeIndex]?.time || 0;
                return (
                  <g key={`time-${ratio}`}>
                    <line x1={x} y1={height - padding} x2={x} y2={height - padding + 5} stroke="#666" />
                    <text x={x} y={height - padding + 18} textAnchor="middle" className="text-xs fill-gray-600">
                      {time.toFixed(1)}s
                    </text>
                  </g>
                );
              })}
              
              {/* Axis labels */}
              <text x={width / 2} y={height - 15} textAnchor="middle" className="text-sm font-medium fill-gray-700">
                Time (seconds)
              </text>
              <text x={15} y={height / 2} textAnchor="middle" className="text-sm font-medium fill-gray-700" transform={`rotate(-90, 15, ${height / 2})`}>
                Attention Values
              </text>
            </svg>
          </div>
          
          {/* Simple stats */}
          <div className="mt-4 flex justify-center gap-8 text-sm">
            <div><span className="font-medium">Min:</span> {(minVal * 100).toFixed(4)}%</div>
            <div><span className="font-medium">Max:</span> {(maxVal * 100).toFixed(4)}%</div>
            <div><span className="font-medium">Range:</span> {(range * 100).toFixed(4)}%</div>
            <div><span className="font-medium">Points:</span> {attentionValues.length}</div>
          </div>
          
          <div className="mt-2 text-xs text-center text-gray-500">
            Hover over points for exact values • Scale adjusted to show all differences
          </div>

          <div className="mt-3 text-xs text-muted-foreground space-y-1">
            <div>
              Duration: {attentionData.total_duration?.toFixed(1)}s | 
              Points: {timestamps.length} | 
              Resolution: {((attentionData.total_duration || 0) / timestamps.length * 1000).toFixed(0)}ms
            </div>
          </div>
        </CardContent>
      </Card>
    );
    } catch (error) {
      console.error('Error rendering timeline:', error);
      return (
        <Card>
          <CardHeader>
            <CardTitle>Attention Timeline</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-red-500 p-4">
              Error displaying attention timeline. Please try again.
            </div>
          </CardContent>
        </Card>
      );
    }
  };

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Attention Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <div className="flex items-center space-x-2">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary"></div>
              <span className="text-muted-foreground">Extracting attention patterns...</span>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Attention Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <div className="text-red-500 text-sm">{error}</div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!selectedFile || !model?.includes('whisper')) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Attention Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-muted-foreground py-8">
            Select a Whisper model and audio file to analyze attention patterns
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {/* Controls */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Attention Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-xs font-medium">Layer</label>
              <Select value={selectedLayer.toString()} onValueChange={(v) => setSelectedLayer(parseInt(v))}>
                <SelectTrigger className="h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Array.from({ length: 12 }, (_, i) => (
                    <SelectItem key={i} value={i.toString()}>Layer {i}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <label className="text-xs font-medium">Head</label>
              <Select value={selectedHead.toString()} onValueChange={(v) => setSelectedHead(parseInt(v))}>
                <SelectTrigger className="h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Array.from({ length: 12 }, (_, i) => (
                    <SelectItem key={i} value={i.toString()}>Head {i}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          {attentionData && (
            <div className="text-xs text-muted-foreground">
              Model: {attentionData.model} | Words: {attentionData.word_chunks?.length || 0} | 
              Sequence Length: {attentionData.sequence_length}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Attention Views */}
      <Tabs defaultValue="pairs" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="pairs">Word Pairs</TabsTrigger>
          <TabsTrigger value="timeline">Timeline</TabsTrigger>
        </TabsList>
        
        <TabsContent value="pairs" className="mt-4">
          {renderWordPairsMatrix()}
        </TabsContent>
        
        <TabsContent value="timeline" className="mt-4">
          {renderTimelineView()}
        </TabsContent>
      </Tabs>
    </div>
  );
};