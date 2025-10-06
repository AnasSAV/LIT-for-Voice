import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AlertCircle, RefreshCw } from "lucide-react";
import { API_BASE } from "@/lib/api";

interface AttentionPair {
  from_word: string;
  to_word: string;
  from_time: [number, number];
  to_time: [number, number];
  attention_weight: number;  // Raw attention value (0.001-0.008)
  attention_normalized?: number;  // 0-100 percentage of maximum
  attention_relative?: number;  // Relative to average
  from_index: number;
  to_index: number;
}

interface TimestampAttention {
  time: number;
  attention: number;
  frame_index: number;
}

interface AttentionData {
  model: string;
  layer: number;
  head: number;
  attention_pairs: AttentionPair[];
  timestamp_attention: TimestampAttention[];
  total_duration: number;
  sequence_length: number;
  word_chunks: any[];
}

interface Props {
  selectedFile?: any;
  model?: string;
  dataset?: string;
}

export const AttentionVisualization: React.FC<Props> = ({ selectedFile, model, dataset }) => {
  const [attentionData, setAttentionData] = useState<AttentionData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedLayer, setSelectedLayer] = useState(6);
  const [selectedHead, setSelectedHead] = useState(0);
  const [viewMode, setViewMode] = useState<"pairs" | "timeline">("pairs");

  const fetchAttentionData = async () => {
    console.log("AttentionVisualization - fetchAttentionData called:", {
      selectedFile,
      selectedFileType: typeof selectedFile,
      selectedFileValue: selectedFile,
      model,
      dataset,
      modelIncludesWhisper: model?.includes('whisper')
    });

    if (!selectedFile || !model || !model.includes('whisper')) {
      console.log("AttentionVisualization - Skipping fetch due to conditions");
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

      // Handle different file selection scenarios
      if (typeof selectedFile === 'string') {
        // Dataset file (filename as string)
        if (dataset) {
          requestBody.dataset = dataset;
          // Extract just the filename if it includes dataset prefix
          const cleanFileName = selectedFile.includes('\\') || selectedFile.includes('/') 
            ? selectedFile.split(/[\\\/]/).pop() 
            : selectedFile;
          console.log("AttentionVisualization - File processing:", {
            originalSelectedFile: selectedFile,
            cleanFileName: cleanFileName,
            dataset: dataset
          });
          requestBody.dataset_file = cleanFileName;
        } else {
          throw new Error("Dataset required for dataset file selection");
        }
      } else if (typeof selectedFile === 'object' && selectedFile) {
        // Check if it's a dataset file (file_path contains dataset prefix)
        if (selectedFile.file_path && dataset && 
            (selectedFile.file_path.includes('/') || selectedFile.file_path.includes('\\'))) {
          // Dataset file with path like "cv-valid-dev/sample-000775.mp3"
          const cleanFileName = selectedFile.file_path.split(/[\\\/]/).pop();
          console.log("AttentionVisualization - Dataset object processing:", {
            originalFilePath: selectedFile.file_path,
            cleanFileName: cleanFileName,
            dataset: dataset
          });
          requestBody.dataset = dataset;
          requestBody.dataset_file = cleanFileName;
        } else if (selectedFile.file_path) {
          // Regular uploaded file
          requestBody.file_path = selectedFile.file_path;
        } else if (selectedFile.file_id) {
          // If only file_id is available, construct path
          requestBody.file_path = selectedFile.file_path || `/uploads/${selectedFile.file_id}`;
        } else {
          throw new Error("Selected upload has no file_path or file_id");
        }
      } else {
        throw new Error("No valid file selected");
      }

      console.log("AttentionVisualization - Request body:", requestBody);
      console.log("AttentionVisualization - API URL:", `${API_BASE}/inferences/attention-pairs`);

      const response = await fetch(`${API_BASE}/inferences/attention-pairs`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: 'include',
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorText = await response.text().catch(() => 'Unknown server error');
        throw new Error(`Failed to fetch attention data: ${response.status} - ${errorText}`);
      }

      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }

      console.log("AttentionVisualization - Success:", data);
      setAttentionData(data);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Unknown error";
      setError(errorMessage);
      console.error("Error fetching attention data:", err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchAttentionData();
  }, [selectedFile, model, dataset, selectedLayer, selectedHead]);

  const getAttentionMatrix = (): number[][] => {
    if (!attentionData || !attentionData.word_chunks) return [];
    
    const numWords = attentionData.word_chunks.length;
    const matrix: number[][] = Array(numWords).fill(null).map(() => Array(numWords).fill(0));
    
    attentionData.attention_pairs.forEach(pair => {
      if (pair.from_index < numWords && pair.to_index < numWords) {
        matrix[pair.from_index][pair.to_index] = pair.attention_weight;
      }
    });
    
    return matrix;
  };

  const getTopAttentionPairs = (limit: number = 5): AttentionPair[] => {
    if (!attentionData) return [];
    
    return attentionData.attention_pairs
      .filter(pair => pair.from_index !== pair.to_index) // Exclude self-attention
      .sort((a, b) => b.attention_weight - a.attention_weight)
      .slice(0, limit);
  };

  if (!selectedFile || !model || !model.includes('whisper')) {
    return (
      <div className="space-y-4">
        <Card>
          <CardContent className="p-6 text-center text-muted-foreground">
            <AlertCircle className="w-8 h-8 mx-auto mb-2" />
            <p className="text-sm">Select a Whisper model and audio file to view attention patterns</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm">Attention Analysis</CardTitle>
            <div className="flex items-center gap-2">
              <Select value={selectedLayer.toString()} onValueChange={(value) => setSelectedLayer(parseInt(value))}>
                <SelectTrigger className="w-20 h-6 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Array.from({length: 12}, (_, i) => (
                    <SelectItem key={i} value={i.toString()}>L{i}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Select value={selectedHead.toString()} onValueChange={(value) => setSelectedHead(parseInt(value))}>
                <SelectTrigger className="w-16 h-6 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Array.from({length: 12}, (_, i) => (
                    <SelectItem key={i} value={i.toString()}>H{i}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button
                variant="outline"
                size="sm" 
                className="h-6 w-6 p-0"
                onClick={fetchAttentionData}
                disabled={isLoading}
              >
                <RefreshCw className={`w-3 h-3 ${isLoading ? 'animate-spin' : ''}`} />
              </Button>
            </div>
          </div>
        </CardHeader>
        
        <CardContent className="space-y-3">
          {isLoading && (
            <div className="text-center py-4">
              <div className="text-sm text-muted-foreground">Loading attention data...</div>
            </div>
          )}
          
          {error && (
            <div className="text-center py-4">
              <div className="text-sm text-red-600">Error: {error}</div>
            </div>
          )}
          
          {attentionData && !isLoading && (
            <Tabs defaultValue="pairs" className="w-full">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="pairs" className="text-xs">Word Pairs</TabsTrigger>
                <TabsTrigger value="timeline" className="text-xs">Timeline</TabsTrigger>
              </TabsList>
              
              <TabsContent value="pairs" className="space-y-3">
                {/* Attention Matrix */}
                <div className="overflow-auto">
                  <div className="text-xs font-medium mb-2">Word-to-Word Attention Matrix:</div>
                  <div className={`grid gap-1 text-xs min-w-fit`} 
                       style={{gridTemplateColumns: `auto repeat(${attentionData.word_chunks.length}, 1fr)`}}>
                    <div></div>
                    {attentionData.word_chunks.map((chunk, idx) => (
                      <div key={idx} className="text-center font-mono text-[10px] p-1 max-w-16 truncate">
                        {chunk.text?.trim() || `W${idx}`}
                      </div>
                    ))}
                    
                    {getAttentionMatrix().map((row, i) => (
                      <React.Fragment key={i}>
                        <div className="text-right font-mono text-[10px] p-1 max-w-16 truncate">
                          {attentionData.word_chunks[i]?.text?.trim() || `W${i}`}
                        </div>
                        {row.map((attention, j) => (
                          <div
                            key={j}
                            className="w-6 h-6 border border-border/20 flex items-center justify-center cursor-pointer rounded-sm"
                            style={{
                              backgroundColor: `hsl(220 70% ${50 + attention * 30}% / ${0.3 + attention * 0.7})`,
                            }}
                            title={`${attentionData.word_chunks[i]?.text?.trim() || `W${i}`} → ${attentionData.word_chunks[j]?.text?.trim() || `W${j}`}: ${(attention * 100).toFixed(1)}%`}
                          >
                            <span className="text-[8px] text-white font-bold">
                              {attention > 0.6 ? '●' : attention > 0.3 ? '·' : ''}
                            </span>
                          </div>
                        ))}
                      </React.Fragment>
                    ))}
                  </div>
                </div>

                {/* Top Attention Pairs */}
                <div className="text-xs space-y-2">
                  <div className="font-medium">Highest Attention Pairs:</div>
                  <div className="space-y-1">
                    {getTopAttentionPairs().map((pair, idx) => (
                      <div key={idx} className="flex items-center justify-between p-2 bg-muted/50 rounded">
                        <div className="flex items-center gap-2">
                          <span className="font-mono font-medium">{pair.from_word}</span>
                          <span className="text-muted-foreground">→</span>
                          <span className="font-mono font-medium">{pair.to_word}</span>
                        </div>
                        <Badge variant="outline" className="text-[10px]">
                          {pair.attention_normalized?.toFixed(1) || (pair.attention_weight * 100).toFixed(1)}%
                        </Badge>
                      </div>
                    ))}
                  </div>
                </div>
              </TabsContent>
              
              <TabsContent value="timeline" className="space-y-3">
                {/* Timestamp-level Attention */}
                <div className="text-xs font-medium mb-2">Attention Over Time:</div>
                <div className="space-y-2">
                  {/* Timeline visualization */}
                  <div className="relative h-20 bg-muted/20 rounded border">
                    <div className="absolute inset-0 flex items-end px-2 py-1">
                      {attentionData.timestamp_attention.map((point, idx) => {
                        const x = (point.time / attentionData.total_duration) * 100;
                        const height = point.attention * 60; // Max 60px height
                        return (
                          <div
                            key={idx}
                            className="absolute bg-primary/70"
                            style={{
                              left: `${x}%`,
                              bottom: '4px',
                              width: '2px',
                              height: `${height}px`
                            }}
                            title={`Time: ${point.time.toFixed(2)}s, Attention: ${(point.attention * 100).toFixed(1)}%`}
                          />
                        );
                      })}
                    </div>
                    
                    {/* Word overlays */}
                    {attentionData.word_chunks.map((chunk, idx) => {
                      const startX = (chunk.timestamp?.[0] || 0) / attentionData.total_duration * 100;
                      const endX = (chunk.timestamp?.[1] || 0) / attentionData.total_duration * 100;
                      const width = endX - startX;
                      
                      return (
                        <div
                          key={idx}
                          className="absolute top-1 text-[8px] font-mono bg-background/80 px-1 rounded border"
                          style={{
                            left: `${startX}%`,
                            width: `${width}%`,
                            minWidth: '20px'
                          }}
                          title={`${chunk.text?.trim()}: ${chunk.timestamp?.[0]?.toFixed(2)}s - ${chunk.timestamp?.[1]?.toFixed(2)}s`}
                        >
                          {chunk.text?.trim()}
                        </div>
                      );
                    })}
                  </div>
                  
                  {/* Timeline stats */}
                  <div className="grid grid-cols-3 gap-4 text-xs">
                    <div className="p-2 bg-muted/30 rounded">
                      <div className="text-muted-foreground">Duration</div>
                      <div className="font-medium">{attentionData.total_duration.toFixed(2)}s</div>
                    </div>
                    <div className="p-2 bg-muted/30 rounded">
                      <div className="text-muted-foreground">Avg Attention</div>
                      <div className="font-medium">
                        {(attentionData.timestamp_attention.reduce((sum, p) => sum + p.attention, 0) / attentionData.timestamp_attention.length * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="p-2 bg-muted/30 rounded">
                      <div className="text-muted-foreground">Words</div>
                      <div className="font-medium">{attentionData.word_chunks.length}</div>
                    </div>
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          )}
        </CardContent>
      </Card>
    </div>
  );
};