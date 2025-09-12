"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Volume2, Scissors, Plus, Play, Zap } from "lucide-react"

interface UploadedFile {
  file_id: string;
  filename: string;
  file_path: string;
  message: string;
  size?: number;
  duration?: number;
  sample_rate?: number;
}

type PerturbationToolsProps = {
  selectedFile: UploadedFile | null;  // adjust type depending on what you store
};

export const PerturbationTools: React.FC<PerturbationToolsProps> = ({
  selectedFile,
}) => {
  const [noiseLevel, setNoiseLevel] = useState([10])
  const [maskStart, setMaskStart] = useState([20])
  const [maskEnd, setMaskEnd] = useState([40])

const handleAddPerturbations = async () => {
  console.log("Adding perturbations...");

  try {
    // TODO: Replace with your actual session/audio IDs

    
    // Slider states are arrays, so take the first element
    const noiseValue = noiseLevel[0] / 100.0; // map 10 → 0.1
    const start = maskStart[0];
    const end = maskEnd[0];

    const reqBody = {
      file_path: selectedFile.file_path, // must be provided
      perturbations: [
        {
          type: "noise",
          noise_level: noiseValue,
          mask_start: start,
          mask_end: end,
        },
      ],
    };

    const response = await fetch('http://localhost:8000/perturb', {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(reqBody),
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();
    console.log("Perturbation response:", data);
  } catch (err) {
    console.error("Error adding perturbations:", err);
  }
};


  return (
    <div className="space-y-4">
      <Card className="border-primary/20 bg-gradient-to-r from-primary/5 to-primary/10">
        <CardContent className="pt-4">
          <Button
            onClick={handleAddPerturbations}
            className="w-full h-10 bg-primary hover:bg-primary/90 text-primary-foreground font-medium"
            size="lg"
          >
            <Zap className="h-4 w-4 mr-2" />
            Add Perturbations
          </Button>
          <p className="text-xs text-muted-foreground mt-2 text-center">
            Apply current perturbation settings to analyze all slides in your presentation
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Perturbation Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="noise" className="w-full">
            <TabsList className="grid w-full grid-cols-3 h-7">
              <TabsTrigger value="noise" className="text-xs">
                Noise
              </TabsTrigger>
              <TabsTrigger value="masking" className="text-xs">
                Masking
              </TabsTrigger>
              <TabsTrigger value="slicing" className="text-xs">
                Slicing
              </TabsTrigger>
            </TabsList>

            <TabsContent value="noise" className="mt-3 space-y-3">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs">Noise Level</span>
                  <span className="text-xs text-muted-foreground">{noiseLevel[0]}%</span>
                </div>
                <Slider value={noiseLevel} onValueChange={setNoiseLevel} max={50} step={1} className="w-full" />
              </div>

              <div className="flex items-center gap-2">
                <Button size="sm" className="h-7 flex-1">
                  <Volume2 className="h-3 w-3 mr-1" />
                  Add Noise
                </Button>
                <Button size="sm" variant="outline" className="h-7 bg-transparent">
                  <Play className="h-3 w-3" />
                </Button>
              </div>
            </TabsContent>

            <TabsContent value="masking" className="mt-3 space-y-3">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs">Mask Region</span>
                  <span className="text-xs text-muted-foreground">
                    {maskStart[0]}% - {maskEnd[0]}%
                  </span>
                </div>
                <div className="space-y-1">
                  <Slider value={maskStart} onValueChange={setMaskStart} max={100} step={1} className="w-full" />
                  <Slider value={maskEnd} onValueChange={setMaskEnd} max={100} step={1} className="w-full" />
                </div>
              </div>

              <div className="flex items-center gap-2">
                <Button size="sm" className="h-7 flex-1">
                  <Scissors className="h-3 w-3 mr-1" />
                  Apply Mask
                </Button>
                <Button size="sm" variant="outline" className="h-7 bg-transparent">
                  <Play className="h-3 w-3" />
                </Button>
              </div>
            </TabsContent>

            <TabsContent value="slicing" className="mt-3 space-y-3">
              <div className="text-xs text-muted-foreground mb-2">Create audio segments for analysis</div>

              <div className="space-y-2">
                {[
                  { name: "Segment 1", range: "0.0-1.5s", confidence: 0.89 },
                  { name: "Segment 2", range: "1.5-3.0s", confidence: 0.76 },
                ].map((segment, idx) => (
                  <div key={idx} className="flex items-center justify-between p-2 bg-muted/50 rounded">
                    <div className="flex items-center gap-2">
                      <span className="text-xs">{segment.name}</span>
                      <Badge variant="outline" className="text-[10px]">
                        {segment.range}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-1">
                      <span className="text-xs text-muted-foreground">{(segment.confidence * 100).toFixed(0)}%</span>
                      <Button size="sm" variant="ghost" className="h-5 w-5 p-0">
                        <Play className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>

              <Button size="sm" variant="outline" className="h-7 w-full bg-transparent">
                <Plus className="h-3 w-3 mr-1" />
                Add Segment
              </Button>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Results */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Perturbation Results</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-xs space-y-2">
            <div className="font-medium">Prediction Changes:</div>
            {[
              { perturbation: "Noise +10%", original: "neutral (87%)", perturbed: "neutral (72%)", change: -15 },
              { perturbation: "Mask 20-40%", original: "neutral (87%)", perturbed: "uncertain (45%)", change: -42 },
            ].map((result, idx) => (
              <div key={idx} className="p-2 bg-muted/50 rounded space-y-1">
                <div className="flex items-center justify-between">
                  <Badge variant="outline" className="text-[10px]">
                    {result.perturbation}
                  </Badge>
                  <span className={`text-[10px] ${result.change < -20 ? "text-destructive" : "text-muted-foreground"}`}>
                    {result.change > 0 ? "+" : ""}
                    {result.change}%
                  </span>
                </div>
                <div className="text-[10px] text-muted-foreground">
                  {result.original} → {result.perturbed}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
