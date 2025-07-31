import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";

export const AttentionVisualization = () => {
  // Generate mock attention matrix
  const generateAttentionMatrix = () => {
    const tokens = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy"];
    const matrix = [];
    
    for (let i = 0; i < tokens.length; i++) {
      const row = [];
      for (let j = 0; j < tokens.length; j++) {
        const attention = Math.random();
        row.push(attention);
      }
      matrix.push(row);
    }
    
    return { tokens, matrix };
  };

  const { tokens, matrix } = generateAttentionMatrix();

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm">Attention Patterns</CardTitle>
            <div className="flex items-center gap-2">
              <Select defaultValue="layer-6">
                <SelectTrigger className="w-20 h-6 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="layer-1">L1</SelectItem>
                  <SelectItem value="layer-2">L2</SelectItem>
                  <SelectItem value="layer-3">L3</SelectItem>
                  <SelectItem value="layer-6">L6</SelectItem>
                </SelectContent>
              </Select>
              <Select defaultValue="head-0">
                <SelectTrigger className="w-16 h-6 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="head-0">H0</SelectItem>
                  <SelectItem value="head-1">H1</SelectItem>
                  <SelectItem value="head-2">H2</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          {/* Attention Matrix */}
          <div className="overflow-auto">
            <div className="grid grid-cols-9 gap-1 text-xs min-w-fit">
              <div></div>
              {tokens.map((token, idx) => (
                <div key={idx} className="text-center font-mono text-[10px] p-1">
                  {token}
                </div>
              ))}
              
              {matrix.map((row, i) => (
                <div key={i} className="contents">
                  <div className="text-right font-mono text-[10px] p-1">
                    {tokens[i]}
                  </div>
                  {row.map((attention, j) => (
                    <div
                      key={j}
                      className="w-6 h-6 border border-border/20 flex items-center justify-center cursor-pointer"
                      style={{
                        backgroundColor: `hsl(var(--primary) / ${attention * 0.8})`,
                      }}
                      title={`${tokens[i]} → ${tokens[j]}: ${(attention * 100).toFixed(0)}%`}
                    >
                      <span className="text-[8px] text-white font-bold">
                        {attention > 0.5 ? '●' : ''}
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
            {[
              { from: "fox", to: "jumps", score: 0.91 },
              { from: "quick", to: "brown", score: 0.84 },
              { from: "over", to: "lazy", score: 0.78 }
            ].map((item, idx) => (
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
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};