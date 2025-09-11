import { useEffect, useRef, useState, useCallback } from "react";
import Plot from "react-plotly.js";
import { useEmbedding } from "../../contexts/EmbeddingContext";
import { Button } from "@/components/ui/button";
import { ZoomIn, ZoomOut, RotateCcw, Maximize2 } from "lucide-react";

interface EmbeddingPlotProps {
  selectedMethod?: string;
  is3D?: boolean;
  onPointSelect?: (filename: string, coordinates: number[]) => void;
  selectedFile?: string | null;
}

export const EmbeddingPlot = ({ selectedMethod = "pca", is3D = false, onPointSelect, selectedFile }: EmbeddingPlotProps) => {
  const { embeddingData, isLoading, error } = useEmbedding();
  const plotRef = useRef<any>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

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
      
      if (onPointSelect) {
        onPointSelect(filename, coordinates);
      }
    }
  }, [onPointSelect, is3D]);

  // Zoom control functions
  const handleZoomIn = useCallback(() => {
    if (plotRef.current) {
      const plot = plotRef.current;
      if (is3D) {
        // For 3D plots, adjust camera distance
        const currentEye = plot.layout.scene?.camera?.eye || { x: 1.5, y: 1.5, z: 1.5 };
        const newEye = {
          x: currentEye.x * 0.8,
          y: currentEye.y * 0.8,
          z: currentEye.z * 0.8
        };
        plot.relayout({ 'scene.camera.eye': newEye });
      } else {
        // For 2D plots, use Plotly's zoom functionality
        plot.relayout({
          'xaxis.range': plot._fullLayout.xaxis.range ? 
            [plot._fullLayout.xaxis.range[0] * 0.8, plot._fullLayout.xaxis.range[1] * 0.8] : undefined,
          'yaxis.range': plot._fullLayout.yaxis.range ?
            [plot._fullLayout.yaxis.range[0] * 0.8, plot._fullLayout.yaxis.range[1] * 0.8] : undefined
        });
      }
    }
  }, [is3D]);

  const handleZoomOut = useCallback(() => {
    if (plotRef.current) {
      const plot = plotRef.current;
      if (is3D) {
        // For 3D plots, adjust camera distance
        const currentEye = plot.layout.scene?.camera?.eye || { x: 1.5, y: 1.5, z: 1.5 };
        const newEye = {
          x: currentEye.x * 1.25,
          y: currentEye.y * 1.25,
          z: currentEye.z * 1.25
        };
        plot.relayout({ 'scene.camera.eye': newEye });
      } else {
        // For 2D plots, use Plotly's zoom functionality
        plot.relayout({
          'xaxis.range': plot._fullLayout.xaxis.range ? 
            [plot._fullLayout.xaxis.range[0] * 1.25, plot._fullLayout.xaxis.range[1] * 1.25] : undefined,
          'yaxis.range': plot._fullLayout.yaxis.range ?
            [plot._fullLayout.yaxis.range[0] * 1.25, plot._fullLayout.yaxis.range[1] * 1.25] : undefined
        });
      }
    }
  }, [is3D]);

  const handleResetZoom = useCallback(() => {
    if (plotRef.current) {
      const plot = plotRef.current;
      if (is3D) {
        plot.relayout({ 
          'scene.camera.eye': { x: 1.5, y: 1.5, z: 1.5 },
          'scene.camera.center': { x: 0, y: 0, z: 0 }
        });
      } else {
        plot.relayout({
          'xaxis.autorange': true,
          'yaxis.autorange': true
        });
      }
    }
  }, [is3D]);

  const toggleFullscreen = useCallback(() => {
    setIsFullscreen(!isFullscreen);
  }, [isFullscreen]);

  // Use real embedding data if available, otherwise fall back to mock data
  const getPlotData = () => {
    if (embeddingData && embeddingData.reduced_embeddings && embeddingData.reduced_embeddings.length > 0) {
      const x = embeddingData.reduced_embeddings.map(point => point.coordinates[0]);
      const y = embeddingData.reduced_embeddings.map(point => point.coordinates[1]);
      const z = is3D && embeddingData.reduced_embeddings[0].coordinates.length > 2 
        ? embeddingData.reduced_embeddings.map(point => point.coordinates[2]) 
        : undefined;
      const text = embeddingData.reduced_embeddings.map(point => point.filename);
      
      // Enhanced color mapping with spatial clustering
      const colors = embeddingData.reduced_embeddings.map((point, index) => {
        const filename = point.filename.toLowerCase();
        
        // First try emotion-based coloring from RAVDESS dataset
        if (filename.includes('01-01') || filename.includes('neutral')) return 'neutral';
        if (filename.includes('01-03') || filename.includes('happy') || filename.includes('joy')) return 'happy';
        if (filename.includes('01-04') || filename.includes('sad') || filename.includes('sadness')) return 'sad';
        if (filename.includes('01-05') || filename.includes('angry') || filename.includes('anger')) return 'angry';
        if (filename.includes('01-06') || filename.includes('fear') || filename.includes('afraid')) return 'fear';
        if (filename.includes('01-07') || filename.includes('disgust')) return 'disgust';
        if (filename.includes('01-08') || filename.includes('surprise')) return 'surprise';
        if (filename.includes('01-02') || filename.includes('calm')) return 'calm';
        
        // For Common Voice or other datasets, use spatial clustering
        const coords = point.coordinates;
        if (coords.length >= 2) {
          const [px, py] = coords;
          
          // Calculate quartiles for better spatial distribution
          const sortedX = x.slice().sort((a, b) => a - b);
          const sortedY = y.slice().sort((a, b) => a - b);
          const q1X = sortedX[Math.floor(sortedX.length * 0.25)];
          const q3X = sortedX[Math.floor(sortedX.length * 0.75)];
          const q1Y = sortedY[Math.floor(sortedY.length * 0.25)];
          const q3Y = sortedY[Math.floor(sortedY.length * 0.75)];
          
          // Assign colors based on spatial regions
          if (px > q3X && py > q3Y) return 'region1'; // Top-right
          if (px < q1X && py > q3Y) return 'region2'; // Top-left
          if (px < q1X && py < q1Y) return 'region3'; // Bottom-left
          if (px > q3X && py < q1Y) return 'region4'; // Bottom-right
          if (px >= q1X && px <= q3X && py >= q1Y && py <= q3Y) return 'center'; // Center
          if (px >= q1X && px <= q3X) return 'mid_vertical'; // Middle band
          if (py >= q1Y && py <= q3Y) return 'mid_horizontal'; // Middle band
        }
        
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

  // Enhanced color mapping for emotions and spatial regions
  const colorMap = {
    // Emotion-based colors (vibrant and distinct)
    'neutral': '#94a3b8',      // Slate gray
    'happy': '#fbbf24',        // Amber
    'sad': '#3b82f6',          // Blue
    'angry': '#ef4444',        // Red
    'fear': '#8b5cf6',         // Purple
    'disgust': '#10b981',      // Emerald
    'surprise': '#f97316',     // Orange
    'calm': '#06b6d4',         // Cyan
    
    // Spatial region colors (cooler palette for non-emotion data)
    'region1': '#ec4899',      // Pink - Top-right
    'region2': '#8b5cf6',      // Purple - Top-left
    'region3': '#06b6d4',      // Cyan - Bottom-left
    'region4': '#10b981',      // Emerald - Bottom-right
    'center': '#f59e0b',       // Amber - Center
    'mid_vertical': '#6366f1',  // Indigo - Middle vertical
    'mid_horizontal': '#84cc16', // Lime - Middle horizontal
    'unknown': '#6b7280'       // Gray
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
    selectedFile === filename ? 12 : 8
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
    traceData.hovertemplate = '<b>%{text}</b><extra></extra>';
  } else {
    traceData.hovertemplate = '<b>%{text}</b><extra></extra>';
  }

  // Layout configuration
  const layout: any = {
    autosize: true,
    margin: { l: 35, r: 35, t: 35, b: 35 },
    plot_bgcolor: 'white',
    paper_bgcolor: 'white',
    showlegend: false,
    font: {
      size: 11,
      color: '#374151'
    },
    dragmode: is3D ? 'orbit' : 'zoom',
    hovermode: 'closest',
    uirevision: true // Maintains UI state on data updates
  };

  if (is3D) {
    // 3D scene configuration
    layout.scene = {
      xaxis: {
        showgrid: true,
        gridcolor: '#e5e7eb',
        showticklabels: false,
        title: { text: 'X', font: { size: 10 } },
        backgroundcolor: 'white',
        showspikes: false
      },
      yaxis: {
        showgrid: true,
        gridcolor: '#e5e7eb',
        showticklabels: false,
        title: { text: 'Y', font: { size: 10 } },
        backgroundcolor: 'white',
        showspikes: false
      },
      zaxis: {
        showgrid: true,
        gridcolor: '#e5e7eb',
        showticklabels: false,
        title: { text: 'Z', font: { size: 10 } },
        backgroundcolor: 'white',
        showspikes: false
      },
      bgcolor: 'white',
      camera: {
        eye: { x: 1.5, y: 1.5, z: 1.5 },
        center: { x: 0, y: 0, z: 0 },
        up: { x: 0, y: 0, z: 1 }
      },
      aspectmode: 'cube',
      dragmode: 'orbit'
    };
  } else {
    // 2D axis configuration with enhanced zoom support
    layout.xaxis = {
      showgrid: true,
      gridcolor: '#e5e7eb',
      showticklabels: false,
      title: { text: 'X', font: { size: 10 } },
      zeroline: true,
      zerolinecolor: '#d1d5db',
      zerolinewidth: 1,
      fixedrange: false // Allow zoom
    };
    layout.yaxis = {
      showgrid: true,
      gridcolor: '#e5e7eb',
      showticklabels: false,
      title: { text: 'Y', font: { size: 10 } },
      zeroline: true,
      zerolinecolor: '#d1d5db',
      zerolinewidth: 1,
      fixedrange: false // Allow zoom
    };
  }

  // Add compact annotation
  if (embeddingData) {
    layout.annotations = [{
      text: `${embeddingData.total_files} files • ${is3D ? '3D' : '2D'}`,
      xref: 'paper',
      yref: 'paper',
      x: 0.02,
      y: 0.98,
      xanchor: 'left',
      yanchor: 'top',
      font: { size: 9, color: '#6b7280' },
      showarrow: false,
      bgcolor: 'rgba(255,255,255,0.8)',
      bordercolor: '#e5e7eb',
      borderwidth: 1,
      borderpad: 2
    }];
  }

  return (
    <div className="w-full h-full min-h-0">
      <Plot
        data={[traceData]}
        layout={layout}
        onClick={handlePointClick}
        config={{
          displayModeBar: true,
          modeBarButtonsToRemove: is3D 
            ? ['pan2d', 'lasso2d', 'select2d'] 
            : [],
          displaylogo: false,
          responsive: true,
          autosizable: true,
          scrollZoom: true,
          doubleClick: 'reset+autosize',
          showTips: true,
          toImageButtonOptions: {
            format: 'png',
            filename: `embeddings_${selectedMethod}_${is3D ? '3D' : '2D'}`,
            height: 800,
            width: 800,
            scale: 2
          }
        }}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler={true}
      />
    </div>
  );
};