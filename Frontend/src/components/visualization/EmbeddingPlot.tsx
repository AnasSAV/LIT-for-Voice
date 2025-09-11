import { useEffect, useRef } from "react";
import Plot from "react-plotly.js";
import { useEmbedding } from "../../contexts/EmbeddingContext";

interface EmbeddingPlotProps {
  selectedMethod?: string;
}

export const EmbeddingPlot = ({ selectedMethod = "pca" }: EmbeddingPlotProps) => {
  const { embeddingData, isLoading, error } = useEmbedding();

  // Generate mock data as fallback
  const generateMockData = () => {
    const n = 50;
    const x = [];
    const y = [];
    const colors = [];
    const text = [];
    
    for (let i = 0; i < n; i++) {
      x.push(Math.random() * 20 - 10);
      y.push(Math.random() * 20 - 10);
      colors.push(['neutral', 'happy', 'sad', 'angry'][Math.floor(Math.random() * 4)]);
      text.push(`Sample ${i + 1}`);
    }
    
    return { x, y, colors, text };
  };

  // Use real embedding data if available, otherwise fall back to mock data
  const getPlotData = () => {
    if (embeddingData && embeddingData.reduced_embeddings && embeddingData.reduced_embeddings.length > 0) {
      const x = embeddingData.reduced_embeddings.map(point => point.coordinates[0]);
      const y = embeddingData.reduced_embeddings.map(point => point.coordinates[1]);
      const text = embeddingData.reduced_embeddings.map(point => point.filename);
      
      // Generate colors based on filename patterns or use default
      const colors = embeddingData.reduced_embeddings.map(point => {
        const filename = point.filename.toLowerCase();
        if (filename.includes('neutral')) return 'neutral';
        if (filename.includes('happy') || filename.includes('joy')) return 'happy';
        if (filename.includes('sad') || filename.includes('sadness')) return 'sad';
        if (filename.includes('angry') || filename.includes('anger')) return 'angry';
        if (filename.includes('fear') || filename.includes('afraid')) return 'fear';
        if (filename.includes('disgust')) return 'disgust';
        if (filename.includes('surprise')) return 'surprise';
        return 'unknown';
      });
      
      return { x, y, colors, text };
    }
    
    return generateMockData();
  };

  const { x, y, colors, text } = getPlotData();

  // Create color mapping for categorical colors
  const colorMap = {
    'neutral': '#94a3b8',
    'happy': '#fbbf24',
    'sad': '#3b82f6',
    'angry': '#ef4444',
    'fear': '#8b5cf6',
    'disgust': '#10b981',
    'surprise': '#f97316',
    'unknown': '#6b7280'
  };

  const numericColors = colors.map(color => {
    const colorKeys = Object.keys(colorMap);
    return colorKeys.indexOf(color) >= 0 ? colorKeys.indexOf(color) : colorKeys.indexOf('unknown');
  });

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-xs text-muted-foreground flex items-center gap-2">
          <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
          Loading embeddings...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full flex items-center justify-center p-4">
        <div className="text-xs text-red-500 text-center">
          <div className="font-medium">⚠️ Error loading embeddings</div>
          <div className="mt-1">{error}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full">
      <Plot
        data={[
          {
            x: x,
            y: y,
            mode: 'markers',
            type: 'scatter',
            marker: {
              size: 8,
              color: numericColors,
              colorscale: Object.entries(colorMap).map(([key, value], index) => [
                index / (Object.keys(colorMap).length - 1),
                value
              ]),
              showscale: false,
              line: {
                width: 1,
                color: '#ffffff'
              }
            },
            text: text,
            hovertemplate: '<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Emotion: ' + 
              colors.map(c => c).join(',').replace(/,/g, '<br>Emotion: ') + '<extra></extra>',
            customdata: colors,
          }
        ]}
        layout={{
          autosize: true,
          margin: { l: 30, r: 30, t: 30, b: 30 },
          xaxis: {
            showgrid: true,
            gridcolor: 'hsl(var(--border))',
            showticklabels: false,
            title: embeddingData?.reduction_method?.toUpperCase() + ' 1' || 'Component 1'
          },
          yaxis: {
            showgrid: true,
            gridcolor: 'hsl(var(--border))',
            showticklabels: false,
            title: embeddingData?.reduction_method?.toUpperCase() + ' 2' || 'Component 2'
          },
          plot_bgcolor: 'transparent',
          paper_bgcolor: 'transparent',
          showlegend: false,
          font: {
            size: 10,
            color: 'hsl(var(--foreground))'
          },
          annotations: embeddingData ? [{
            text: `${embeddingData.total_files} files • ${embeddingData.original_dimension}D → 2D`,
            xref: 'paper',
            yref: 'paper',
            x: 0.02,
            y: 0.98,
            xanchor: 'left',
            yanchor: 'top',
            font: { size: 9, color: 'hsl(var(--muted-foreground))' },
            showarrow: false,
            bgcolor: 'rgba(0,0,0,0)',
          }] : []
        }}
        config={{
          displayModeBar: false,
          responsive: true
        }}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
};