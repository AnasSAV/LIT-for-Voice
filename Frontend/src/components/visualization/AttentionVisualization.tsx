import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { useState } from "react";

interface AttentionVisualizationProps {
  attention: number[][][] | null;
  transcript?: string | null;
  isLoading?: boolean;
}

export const AttentionVisualization = ({ attention, transcript, isLoading }: AttentionVisualizationProps) => {
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [selectedHead, setSelectedHead] = useState(0);

  // Show loading state while attention data is being fetched
  if (isLoading) {
    return (
      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Attention Visualization</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-center py-8">
              <div className="flex items-center space-x-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary"></div>
                <span className="text-muted-foreground">Loading attention data...</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // If no real attention data, show placeholder
  if (!attention || attention.length === 0) {
    return (
      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Attention Visualization</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-center text-muted-foreground py-8">
              {!attention ? 
                "Attention extraction not available for this model/audio combination. Only authentic attention data is shown - no synthetic data is generated." :
                `Attention data received but empty (${attention.length} layers). This model may not support attention extraction for this audio file.`
              }
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Get the selected attention matrix
  const currentMatrix = attention[selectedLayer] || [];
  const currentHeadMatrix = currentMatrix[selectedHead] || [];

  // Generate appropriate tokens for visualization based on attention matrix size
  const generateTokens = (size: number) => {
    // If we have a transcript, use real words
    if (transcript && typeof transcript === 'string') {
      const words = transcript.trim().split(/\s+/);
      const tokens = [];
      
      // Add start token
      tokens.push("<start>");
      
      // Add actual words, but limit to the size we need
      const availableSlots = Math.max(0, size - 2); // Reserve slots for start and end tokens
      for (let i = 0; i < availableSlots && i < words.length; i++) {
        tokens.push(words[i]);
      }
      
      // Fill remaining slots with generic tokens if needed
      while (tokens.length < size - 1) {
        tokens.push(`t${tokens.length}`);
      }
      
      // Add end token
      if (size > 1) {
        tokens.push("<end>");
      }
      
      return tokens.slice(0, size); // Ensure exact size
    }
    
    // Fallback to generic tokens
    const tokens = [];
    for (let i = 0; i < size; i++) {
      if (i === 0) tokens.push("<start>");
      else if (i === size - 1) tokens.push("<end>");
      else tokens.push(`t${i}`);
    }
    return tokens;
  };

  const tokens = generateTokens(currentHeadMatrix.length);
  
  // Find high attention pairs from the current matrix
  const getHighAttentionPairs = () => {
    const pairs = [];
    if (currentHeadMatrix && Array.isArray(currentHeadMatrix)) {
      for (let i = 0; i < currentHeadMatrix.length && i < tokens.length; i++) {
        const row = currentHeadMatrix[i];
        if (Array.isArray(row)) {
          for (let j = 0; j < row.length && j < tokens.length; j++) {
            const score = row[j];
            if (typeof score === 'number' && score > 0.5) {  // Threshold for "high" attention
              pairs.push({
                from: tokens[i],
                to: tokens[j],
                score: score,
                fromIdx: i,
                toIdx: j
              });
            }
          }
        }
      }
    }
    return pairs.sort((a, b) => b.score - a.score).slice(0, 5); // Top 5 pairs
  };

  const highAttentionPairs = getHighAttentionPairs();

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-sm">Attention Patterns</CardTitle>
              <div className="text-xs text-muted-foreground mt-1">
                {attention.length} layers, {currentMatrix.length} heads, {tokens.length}×{tokens.length} matrix
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Select value={selectedLayer.toString()} onValueChange={(value) => setSelectedLayer(parseInt(value))}>
                <SelectTrigger className="w-20 h-6 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {attention && attention.map((_, index) => (
                    <SelectItem key={index} value={index.toString()}>L{index + 1}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Select value={selectedHead.toString()} onValueChange={(value) => setSelectedHead(parseInt(value))}>
                <SelectTrigger className="w-16 h-6 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {currentMatrix && currentMatrix.map && currentMatrix.map((_, index) => (
                    <SelectItem key={index} value={index.toString()}>H{index}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          {/* Attention Matrix */}
          <div className="overflow-auto">
            <div className="grid gap-1 text-xs min-w-fit" style={{ gridTemplateColumns: `auto repeat(${tokens.length}, 1fr)` }}>
              <div></div>
              {tokens.map((token, idx) => (
                <div key={idx} className="text-center font-mono text-[10px] p-1">
                  {token}
                </div>
              ))}
              
              {currentHeadMatrix.map && currentHeadMatrix.map((row, i) => (
                <div key={i} className="contents">
                  <div className="text-right font-mono text-[10px] p-1">
                    {tokens[i]}
                  </div>
                  {Array.isArray(row) && row.map((attentionValue, j) => (
                    <div
                      key={j}
                      className="w-6 h-6 border border-border/20 flex items-center justify-center cursor-pointer"
                      style={{
                        backgroundColor: `hsl(var(--primary) / ${attentionValue * 0.8})`,
                      }}
                      title={`${tokens[i]} → ${tokens[j]}: ${(attentionValue * 100).toFixed(0)}%`}
                    >
                      <span className="text-[8px] text-white font-bold">
                        {attentionValue > 0.5 ? '●' : ''}
                      </span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>

          {/* Attention Summary */}
          <div className="text-xs space-y-2">
            <div className="font-medium">High Attention Pairs:</div>
            {highAttentionPairs.length > 0 ? (
              highAttentionPairs.map((item, idx) => (
                <div key={idx} className="flex items-center justify-between p-2 bg-muted/50 rounded">
                  <div className="flex items-center gap-2">
                    <span className="font-mono">{item.from}</span>
                    <span className="text-muted-foreground">→</span>
                    <span className="font-mono">{item.to}</span>
                  </div>
                  <Badge variant="outline" className="text-[10px]">
                    {(item.score * 100).toFixed(0)}%
                  </Badge>
                </div>
              ))
            ) : (
              <div className="text-muted-foreground text-center py-2">
                No high attention pairs found (threshold &gt; 50%)
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};