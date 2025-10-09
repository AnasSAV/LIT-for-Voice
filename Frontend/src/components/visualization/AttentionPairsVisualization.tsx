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
  frame_index: number;
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
        // Check if it's a dataset file path or upload path
        if (dataset && (selectedFile.file_path.includes('/') || selectedFile.file_path.includes('\\'))) {
          // Dataset file with path prefix
          const cleanFileName = selectedFile.file_path.split(/[\\\/]/).pop();
          requestBody.dataset = dataset;
          requestBody.dataset_file = cleanFileName;
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

    // Get top pairs for display
    const topPairs = pairs
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
                    {word.substring(0, 3)}
                  </div>
                ))}
                
                {words.map((fromWord, i) => (
                <>
                  <div key={`row-${i}`} className="text-xs p-1 font-medium truncate" title={fromWord}>
                    {fromWord.substring(0, 3)}
                  </div>
                  {words.map((toWord, j) => {
                    // Safe matrix access with bounds checking
                    const attentionValue = (i < matrix.length && j < matrix[0]?.length) ? matrix[i][j] : 0;
                    return (
                      <div
                        key={`cell-${i}-${j}`}
                        className="aspect-square border border-gray-200 text-xs flex items-center justify-center"
                        style={{
                          backgroundColor: `rgba(59, 130, 246, ${attentionValue * 100})`
                        }}
                        title={`${fromWord} → ${toWord}: ${(attentionValue * 100).toFixed(2)}%`}
                      >
                        {(attentionValue * 100).toFixed(0)}
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
    const maxAttention = Math.max(...timestamps.map(t => t.attention));

    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Attention Over Time</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-32 relative border rounded">
            <svg className="w-full h-full">
              {/* Attention line */}
              <polyline
                fill="none"
                stroke="rgb(59, 130, 246)"
                strokeWidth="2"
                points={timestamps.map((t, i) => 
                  `${(t.time / (attentionData.total_duration || 1)) * 100},${100 - (t.attention / maxAttention) * 80}`
                ).join(' ')}
              />
              
              {/* Data points */}
              {timestamps.map((t, i) => (
                <circle
                  key={i}
                  cx={`${(t.time / (attentionData.total_duration || 1)) * 100}%`}
                  cy={`${100 - (t.attention / maxAttention) * 80}%`}
                  r="2"
                  fill="rgb(59, 130, 246)"
                />
              ))}
            </svg>
          </div>
          
          {/* Timeline info */}
          <div className="mt-2 text-xs text-muted-foreground">
            Duration: {attentionData.total_duration?.toFixed(1)}s | 
            Max Attention: {(maxAttention * 100).toFixed(1)}% |
            Points: {timestamps.length}
          </div>
        </CardContent>
      </Card>
    );
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