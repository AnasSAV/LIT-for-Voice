import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import { Toolbar } from "./Toolbar";
import { EmbeddingPanel } from "../panels/EmbeddingPanel";
import { AudioDatasetPanel } from "../panels/AudioDatasetPanel";
import { DatapointEditorPanel } from "../panels/DatapointEditorPanel";
import { PredictionPanel } from "../panels/PredictionPanel";
import React, { useState } from "react";

export const MainLayout = () => {
  const [apiData, setApiData] = useState(null); // NEW
  return (
    <div className="h-screen flex flex-col bg-background">
      {/* Top Navigation Bar */}
      <Toolbar
      apiData={apiData}
      setApiData={setApiData} 
      />
      
      {/* Main Content Area */}
      <div className="flex-1 overflow-hidden">
        <PanelGroup direction="horizontal" className="h-full">
          {/* Left Panel: Embeddings & Scalar Plots */}
          <Panel defaultSize={25} minSize={20}>
            <EmbeddingPanel />
          </Panel>
          
          <PanelResizeHandle className="w-1 bg-border hover:bg-border/80 transition-colors" />
          
          {/* Center Panel: Audio Dataset Table */}
          <Panel defaultSize={50} minSize={30}>
            <PanelGroup direction="vertical">
              <Panel defaultSize={70} minSize={40}>
                <AudioDatasetPanel
                apiData = {apiData}
                />
              </Panel>
              
              <PanelResizeHandle className="h-1 bg-border hover:bg-border/80 transition-colors" />
              
              {/* Bottom Panel: Predictions */}
              <Panel defaultSize={30} minSize={20}>
                <PredictionPanel />
              </Panel>
            </PanelGroup>
          </Panel>
          
          <PanelResizeHandle className="w-1 bg-border hover:bg-border/80 transition-colors" />
          
          {/* Right Panel: Audio Player & Label Editor */}
          <Panel defaultSize={25} minSize={20}>
            <DatapointEditorPanel />
          </Panel>
        </PanelGroup>
      </div>
    </div>
  );
};