import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { SaliencyVisualization } from "../visualization/SaliencyVisualization";
import { PerturbationTools } from "../analysis/PerturbationTools";
import { AttentionVisualization } from "../visualization/AttentionVisualization";
import { useDatasetPredictions } from "@/hooks/useDatasetPredictions";
import { useMemo } from "react";
import { DatasetFile } from "@/lib/api/datasets";
import { PredictionResult } from "@/lib/api/predictions";

// Temporary mock for useDataset until it's properly implemented
const useDataset = () => {
  return {
    selectedFile: null as DatasetFile | null,
    files: [] as DatasetFile[],
  };
};

interface ClassLabel {
  label: string;
  probability: number;
  isPredicted: boolean;
  isTrue?: boolean;
}

// Extend the PredictionResult type to include our expected fields
type ExtendedPredictionResult = PredictionResult & {
  probabilities?: Record<string, number>;
};

export const PredictionPanel = () => {
  const { selectedFile } = useDataset();
  const { getPrediction, isLoading } = useDatasetPredictions({
    files: selectedFile ? [selectedFile] : [],
    model: "default",
    enabled: !!selectedFile,
  });

  const prediction = (selectedFile ? getPrediction(selectedFile) : null) as ExtendedPredictionResult | null;

  const classLabels = useMemo<ClassLabel[]>(() => {
    if (!prediction?.probabilities) return [];
    
    const probs = prediction.probabilities;
    const trueLabel = selectedFile?.label;
    const values = Object.values(probs);
    const maxProbability = values.length > 0 ? Math.max(...values) : 0;
    
    return Object.entries(probs)
      .map(([label, probability]) => ({
        label,
        probability: Number(probability),
        isPredicted: Math.abs(Number(probability) - maxProbability) < 0.0001,
        isTrue: label === trueLabel,
      }))
      .sort((a, b) => b.probability - a.probability);
  }, [prediction, selectedFile]);

  return (
    <div className="h-full panel-background border-t panel-border">
      <Tabs defaultValue="predictions" className="h-full">
        <div className="panel-header border-b panel-border px-3 py-2">
          <TabsList className="h-7 grid grid-cols-4 w-full">
            <TabsTrigger value="predictions" className="text-xs">Predictions</TabsTrigger>
            <TabsTrigger value="saliency" className="text-xs">Saliency</TabsTrigger>
            <TabsTrigger value="attention" className="text-xs">Attention</TabsTrigger>
            <TabsTrigger value="perturbation" className="text-xs">Perturbation</TabsTrigger>
          </TabsList>
        </div>
        
        <div className="h-[calc(100%-2.5rem)] overflow-auto">
          <TabsContent value="predictions" className="m-0 h-full">
            <div className="p-3 space-y-3">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Classification Results</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {[
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
                  ))}
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Token-level Predictions (ASR)</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-xs space-y-1">
                    {["The", "quick", "brown", "fox", "jumps"].map((token, idx) => (
                      <div key={idx} className="flex items-center justify-between">
                        <span className="font-mono">{token}</span>
                        <span className="text-muted-foreground">
                          {(0.95 - idx * 0.02).toFixed(2)}
                        </span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
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
              <PerturbationTools />
            </div>
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
};