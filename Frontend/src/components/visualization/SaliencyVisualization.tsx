import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Eye, Download } from "lucide-react";

export const SaliencyVisualization = () => {
  // Generate mock saliency data
  const generateSaliencyBars = () => {
    const timeSegments = [];
    for (let i = 0; i < 32; i++) {
      const intensity = Math.random();
      let colorClass = "saliency-low";
      
      if (intensity > 0.7) colorClass = "saliency-high";
      else if (intensity > 0.4) colorClass = "saliency-medium";
      
      timeSegments.push(
        <div
          key={i}
          className={`h-8 ${colorClass} border-r border-background/20`}
          style={{
            opacity: 0.3 + intensity * 0.7,
            width: '100%'
          }}
          title={`Time: ${(i * 0.1).toFixed(1)}s, Intensity: ${(intensity * 100).toFixed(0)}%`}
        />
      );
    }
    return timeSegments;
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm">Saliency Overlay</CardTitle>
            <div className="flex items-center gap-2">
              <Select defaultValue="grad-cam">
                <SelectTrigger className="w-24 h-6 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="grad-cam">GradCAM</SelectItem>
                  <SelectItem value="lime">LIME</SelectItem>
                  <SelectItem value="shap">SHAP</SelectItem>
                </SelectContent>
              </Select>
              <Button size="sm" variant="outline" className="h-6">
                <Download className="h-3 w-3" />
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          {/* Waveform with saliency overlay */}
          <div className="relative">
            <div className="h-16 bg-muted/30 rounded mb-2 flex items-center justify-center">
              <span className="text-xs text-muted-foreground">Waveform Background</span>
            </div>
            
            {/* Saliency overlay */}
            <div className="absolute inset-0 grid grid-cols-32 gap-0 rounded">
              {generateSaliencyBars()}
            </div>
          </div>

          {/* Legend */}
          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 saliency-low rounded"></div>
              <span>Low</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 saliency-medium rounded"></div>
              <span>Medium</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 saliency-high rounded"></div>
              <span>High</span>
            </div>
          </div>

          {/* Time segments */}
          <div className="text-xs space-y-2">
            <div className="font-medium">Top Salient Segments:</div>
            {[
              { time: "1.2-1.8s", score: 0.87, segment: "brown fox" },
              { time: "2.1-2.4s", score: 0.72, segment: "jumps" },
              { time: "0.8-1.1s", score: 0.68, segment: "quick" }
            ].map((item, idx) => (
              <div key={idx} className="flex items-center justify-between p-2 bg-muted/50 rounded">
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className="text-[10px]">{item.time}</Badge>
                  <span className="font-mono">{item.segment}</span>
                </div>
                <span className="text-muted-foreground">{(item.score * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};