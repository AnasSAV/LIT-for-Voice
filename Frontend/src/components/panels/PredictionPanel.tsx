import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { SaliencyVisualization } from "../visualization/SaliencyVisualization";
import { PerturbationTools } from "../analysis/PerturbationTools";
import { AttentionVisualization } from "../visualization/AttentionVisualization";

interface UploadedFile {
  file_id: string;
  filename: string;
  file_path: string;
  message: string;
  size?: number;
  duration?: number;
  sample_rate?: number;
}

interface PredictionPanelProps {
  selectedFile?: UploadedFile | null;
  selectedEmbeddingFile?: string | null;
}

export const PredictionPanel = ({ selectedFile, selectedEmbeddingFile }: PredictionPanelProps) => {
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
              {/* Selected File Information */}
              {(selectedFile || selectedEmbeddingFile) && (
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">Selected File</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <div className="text-xs">
                      <div className="font-medium text-blue-800 mb-2">
                        {selectedFile ? selectedFile.filename : selectedEmbeddingFile}
                      </div>
                      {selectedFile && (
                        <div className="space-y-1 text-gray-600">
                          <div><span className="font-medium">File ID:</span> {selectedFile.file_id}</div>
                          <div><span className="font-medium">Path:</span> {selectedFile.file_path}</div>
                          {selectedFile.size && (
                            <div><span className="font-medium">Size:</span> {(selectedFile.size / 1024).toFixed(1)} KB</div>
                          )}
                          {selectedFile.duration && (
                            <div><span className="font-medium">Duration:</span> {selectedFile.duration.toFixed(2)}s</div>
                          )}
                          {selectedFile.sample_rate && (
                            <div><span className="font-medium">Sample Rate:</span> {selectedFile.sample_rate} Hz</div>
                          )}
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              )}

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