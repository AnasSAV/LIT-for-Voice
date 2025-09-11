import { useEffect, useRef, useState, useCallback } from "react";
import Plot from "react-plotly.js";
import { useEmbedding } from "../../contexts/EmbeddingContext";

interface EmbeddingPlotProps {
  selectedMethod?: string;
  is3D?: boolean;
  onPointSelect?: (filename: string, coordinates: number[]) => void;
}

export const EmbeddingPlot = ({ selectedMethod = "pca", is3D = false, onPointSelect }: EmbeddingPlotProps) => {
  const { embeddingData, isLoading, error } = useEmbedding();
  const [selectedPoint, setSelectedPoint] = useState<string | null>(null);

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

  // Handle point selection
  const handlePointClick = useCallback((event: any) => {
    if (event.points && event.points.length > 0) {
      const point = event.points[0];
      const filename = point.text;
      const coordinates = is3D ? [point.x, point.y, point.z] : [point.x, point.y];
      
      setSelectedPoint(filename);
      if (onPointSelect) {
        onPointSelect(filename, coordinates);
      }
    }
  }, [onPointSelect, is3D]);

  // Use real embedding data if available, otherwise fall back to mock data
  const getPlotData = () => {
    if (embeddingData && embeddingData.reduced_embeddings && embeddingData.reduced_embeddings.length > 0) {
      const x = embeddingData.reduced_embeddings.map(point => point.coordinates[0]);
      const y = embeddingData.reduced_embeddings.map(point => point.coordinates[1]);
      const z = is3D && embeddingData.reduced_embeddings[0].coordinates.length > 2 
        ? embeddingData.reduced_embeddings.map(point => point.coordinates[2]) 
        : undefined;
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
      
      return { x, y, z, colors, text };
    }
    
    const mockData = generateMockData();
    if (is3D) {
      // Generate mock Z coordinates
      const z = mockData.x.map(() => Math.random() * 20 - 10);
      return { ...mockData, z };
    }
    return mockData;
  };

  const plotData = getPlotData();
  const { x, y, colors, text } = plotData;
  const z = 'z' in plotData ? plotData.z : undefined;

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

  // Create marker sizes based on selection
  const markerSizes = text.map(filename => 
    selectedPoint === filename ? 12 : 8
  );

  // Create trace data
  const traceData: any = {
    x: x,
    y: y,
    mode: 'markers',
    type: is3D ? 'scatter3d' : 'scatter',
    marker: {
      size: markerSizes,
      color: numericColors,
      colorscale: Object.entries(colorMap).map(([key, value], index) => [
        index / (Object.keys(colorMap).length - 1),
        value
      ]),
      showscale: false,
      line: {
        width: 1,
        color: '#ffffff'
      },
      opacity: 0.8
    },
    text: text,
    customdata: colors,
  };

  // Add Z coordinate for 3D plots
  if (is3D && z) {
    traceData.z = z;
    traceData.hovertemplate = '<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<br>Category: %{customdata}<extra></extra>';
  } else {
    traceData.hovertemplate = '<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Category: %{customdata}<extra></extra>';
  }

  // Layout configuration
  const layout: any = {
    autosize: true,
    margin: { l: 30, r: 30, t: 30, b: 30 },
    plot_bgcolor: 'transparent',
    paper_bgcolor: 'transparent',
    showlegend: false,
    font: {
      size: 10,
      color: 'hsl(var(--foreground))'
    }
  };

  if (is3D) {
    // 3D scene configuration
    layout.scene = {
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
      zaxis: {
        showgrid: true,
        gridcolor: 'hsl(var(--border))',
        showticklabels: false,
        title: embeddingData?.reduction_method?.toUpperCase() + ' 3' || 'Component 3'
      },
      bgcolor: 'transparent',
      camera: {
        eye: { x: 1.5, y: 1.5, z: 1.5 }
      }
    };
  } else {
    // 2D axis configuration
    layout.xaxis = {
      showgrid: true,
      gridcolor: 'hsl(var(--border))',
      showticklabels: false,
      title: embeddingData?.reduction_method?.toUpperCase() + ' 1' || 'Component 1'
    };
    layout.yaxis = {
      showgrid: true,
      gridcolor: 'hsl(var(--border))',
      showticklabels: false,
      title: embeddingData?.reduction_method?.toUpperCase() + ' 2' || 'Component 2'
    };
  }

  // Add annotation
  if (embeddingData) {
    layout.annotations = [{
      text: `${embeddingData.total_files} files • ${embeddingData.original_dimension}D → ${is3D ? '3D' : '2D'}${selectedPoint ? ` • Selected: ${selectedPoint}` : ''}`,
      xref: 'paper',
      yref: 'paper',
      x: 0.02,
      y: 0.98,
      xanchor: 'left',
      yanchor: 'top',
      font: { size: 9, color: 'hsl(var(--muted-foreground))' },
      showarrow: false,
      bgcolor: 'rgba(0,0,0,0)',
    }];
  }

  return (
    <div className="h-full">
      <Plot
        data={[traceData]}
        layout={layout}
        onClick={handlePointClick}
        config={{
          displayModeBar: true,
          modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d', 'autoScale2d'],
          displaylogo: false,
          responsive: true
        }}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
};