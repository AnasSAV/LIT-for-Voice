import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { SaliencyVisualization } from "../visualization/SaliencyVisualization";
import { PerturbationTools } from "../analysis/PerturbationTools";
import { AttentionVisualization } from "../visualization/AttentionVisualization";
import { PredictionResults } from "../analysis/PredictionResults";

interface PredictionPanelProps {
  selectedFile?: any;
  predictionResults?: any[];
  model?: string;
  dataset?: string;
}

export const PredictionPanel = ({ selectedFile, predictionResults = [], model = "", dataset = "" }: PredictionPanelProps) => {
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
            {predictionResults.length > 0 ? (
              <PredictionResults 
                results={predictionResults}
                model={model}
                dataset={dataset}
                onPlayFile={(filename) => {
                  // Handle file playback
                  console.log('Play file:', filename);
                }}
              />
            ) : (
              <div className="p-3 space-y-3">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">No Predictions Yet</CardTitle>
                  </CardHeader>
                  <CardContent className="text-center py-8">
                    <p className="text-sm text-muted-foreground">
                      {model === "wav2vec2" && dataset === "ravdess" 
                        ? "Select wav2vec2 model and ravdess dataset, then click 'Run Predictions' to see emotion predictions and transcripts."
                        : "Upload files or select a dataset and model to see predictions here."}
                    </p>
                  </CardContent>
                </Card>
              </div>
            )}
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