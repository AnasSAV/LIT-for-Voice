import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import Plot from "react-plotly.js";
import { useEmbedding } from "../../contexts/EmbeddingContext";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ZoomIn, ZoomOut, RotateCcw, Maximize2, Layers3, Target } from "lucide-react";

interface EmbeddingPlotProps {
  selectedMethod?: string;
  is3D?: boolean;
  onPointSelect?: (filename: string, coordinates: number[]) => void;
  onAngleRangeSelect?: (selectedFiles: string[]) => void;
  selectedFile?: string | null;
}

type PlaneType = 'none' | 'xy' | 'xz' | 'yz';

export const EmbeddingPlot = ({ selectedMethod = "pca", is3D = false, onPointSelect, onAngleRangeSelect, selectedFile }: EmbeddingPlotProps) => {
  const { embeddingData, isLoading, error } = useEmbedding();
  const plotRef = useRef<any>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [selectedPlane, setSelectedPlane] = useState<PlaneType>('none');
  const [angleMin, setAngleMin] = useState<number>(40);
  const [angleMax, setAngleMax] = useState<number>(50);
  const [selectedByAngle, setSelectedByAngle] = useState<string[]>([]);

  // Reset plane selection when switching to 2D
  useEffect(() => {
    if (!is3D) {
      setSelectedPlane('none');
      setSelectedByAngle([]);
    }
  }, [is3D]);

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

  // Create transparent plane surfaces for 3D visualization
  const createPlane = (planeType: PlaneType, bounds: { x: [number, number], y: [number, number], z: [number, number] }) => {
    if (!is3D || planeType === 'none') return null;

    const [xMin, xMax] = bounds.x;
    const [yMin, yMax] = bounds.y;
    const [zMin, zMax] = bounds.z;

    const planeAlpha = 0.2; // 80% transparency (20% opacity)
    
    switch (planeType) {
      case 'xy': // X-Y plane through origin (Z = 0)
        return {
          type: 'surface' as const,
          x: [[xMin, xMax], [xMin, xMax]],
          y: [[yMin, yMin], [yMax, yMax]],
          z: [[0, 0], [0, 0]], // Always pass through Z = 0 (origin)
          opacity: planeAlpha,
          colorscale: [[0, 'rgba(59, 130, 246, 0.3)'], [1, 'rgba(59, 130, 246, 0.3)']], // Blue
          showscale: false,
          hoverinfo: 'skip',
          name: 'X-Y Plane (Z=0)'
        };
      case 'xz': // X-Z plane through origin (Y = 0)
        return {
          type: 'surface' as const,
          x: [[xMin, xMax], [xMin, xMax]],
          y: [[0, 0], [0, 0]], // Always pass through Y = 0 (origin)
          z: [[zMin, zMin], [zMax, zMax]],
          opacity: planeAlpha,
          colorscale: [[0, 'rgba(16, 185, 129, 0.3)'], [1, 'rgba(16, 185, 129, 0.3)']], // Green
          showscale: false,
          hoverinfo: 'skip',
          name: 'X-Z Plane (Y=0)'
        };
      case 'yz': // Y-Z plane through origin (X = 0)
        return {
          type: 'surface' as const,
          x: [[0, 0], [0, 0]], // Always pass through X = 0 (origin)
          y: [[yMin, yMax], [yMin, yMax]],
          z: [[zMin, zMin], [zMax, zMax]],
          opacity: planeAlpha,
          colorscale: [[0, 'rgba(239, 68, 68, 0.3)'], [1, 'rgba(239, 68, 68, 0.3)']], // Red
          showscale: false,
          hoverinfo: 'skip',
          name: 'Y-Z Plane (X=0)'
        };
      default:
        return null;
    }
  };

  // Calculate angle between point and selected plane relative to origin (0,0,0)
  const calculateAngleToPlane = (x: number, y: number, z: number, plane: PlaneType): number => {
    if (plane === 'none') return 0;
    
    const point = [x, y, z];
    const origin = [0, 0, 0];
    
    // Calculate vector from origin to point
    const vector = [x - origin[0], y - origin[1], z - origin[2]];
    const vectorMagnitude = Math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2);
    
    if (vectorMagnitude === 0) return 0; // Point at origin
    
    // Define plane normal vectors
    let planeNormal: number[];
    switch (plane) {
      case 'xy': planeNormal = [0, 0, 1]; break; // Z axis (normal to XY plane)
      case 'xz': planeNormal = [0, 1, 0]; break; // Y axis (normal to XZ plane)  
      case 'yz': planeNormal = [1, 0, 0]; break; // X axis (normal to YZ plane)
      default: planeNormal = [0, 0, 1]; break;
    }
    
    // Calculate dot product
    const dotProduct = vector[0] * planeNormal[0] + vector[1] * planeNormal[1] + vector[2] * planeNormal[2];
    
    // Calculate angle between vector and plane normal (0° = perpendicular to plane, 90° = in plane)
    const angleToNormal = Math.acos(Math.abs(dotProduct) / vectorMagnitude) * (180 / Math.PI);
    
    // Convert to angle from plane (90° - angle to normal)
    return 90 - angleToNormal;
  };

  // Select points based on angle range - memoized to prevent unnecessary recalculations
  const selectedFiles = useMemo(() => {
    if (!is3D || selectedPlane === 'none' || !embeddingData?.reduced_embeddings) {
      return [];
    }

    return embeddingData.reduced_embeddings
      .filter(point => {
        if (point.coordinates.length < 3) return false;
        
        const [x, y, z] = point.coordinates;
        const angle = calculateAngleToPlane(x, y, z, selectedPlane);
        
        return angle >= angleMin && angle <= angleMax;
      })
      .map(point => point.filename);
  }, [is3D, selectedPlane, embeddingData, angleMin, angleMax]);

  // Update selected points when calculated files change
  useEffect(() => {
    setSelectedByAngle(selectedFiles);
    
    // Notify parent component only if selection actually changed
    if (onAngleRangeSelect && selectedFiles.join(',') !== selectedByAngle.join(',')) {
      onAngleRangeSelect(selectedFiles);
    }
  }, [selectedFiles, onAngleRangeSelect]); // Remove selectedByAngle from dependencies to prevent loops

  // Select points based on angle range
  const selectPointsByAngleRange = useCallback(() => {
    if (!is3D || selectedPlane === 'none' || !embeddingData?.reduced_embeddings) {
      setSelectedByAngle([]);
      return;
    }

    const selectedFiles = embeddingData.reduced_embeddings
      .filter(point => {
        if (point.coordinates.length < 3) return false;
        
        const [x, y, z] = point.coordinates;
        const angle = calculateAngleToPlane(x, y, z, selectedPlane);
        
        return angle >= angleMin && angle <= angleMax;
      })
      .map(point => point.filename);

    setSelectedByAngle(selectedFiles);
    
    // Notify parent component
    if (onAngleRangeSelect) {
      onAngleRangeSelect(selectedFiles);
    }
  }, [is3D, selectedPlane, angleMin, angleMax, embeddingData, onAngleRangeSelect]);

  // Auto-update selection when parameters change
  useEffect(() => {
    selectPointsByAngleRange();
  }, [selectPointsByAngleRange]);

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

  // Calculate bounds for plane creation
  const bounds = x.length > 0 ? {
    x: [Math.min(...x) * 1.1, Math.max(...x) * 1.1] as [number, number],
    y: [Math.min(...y) * 1.1, Math.max(...y) * 1.1] as [number, number],
    z: z && z.length > 0 ? [Math.min(...z) * 1.1, Math.max(...z) * 1.1] as [number, number] : [0, 0] as [number, number]
  } : { x: [0, 0] as [number, number], y: [0, 0] as [number, number], z: [0, 0] as [number, number] };

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
  const markerSizes = text.map(filename => {
    if (selectedFile === filename) return 12; // Currently selected file (medium-large)
    if (selectedByAngle.includes(filename)) return 8; // Angle range selected (medium)
    return 6; // Default (smaller)
  });

  // Create marker colors based on selection
  const markerColors = text.map(filename => {
    if (selectedFile === filename) return '#fbbf24'; // Amber for selected file
    if (selectedByAngle.includes(filename)) return '#ef4444'; // Red for angle selected
    return numericColors[text.indexOf(filename)]; // Default color mapping
  });

  // Create marker opacities based on selection
  const hasSelection = selectedFile || selectedByAngle.length > 0;
  const markerOpacities = text.map(filename => {
    if (!hasSelection) return 0.8; // Default opacity when no selection
    if (selectedFile === filename) return 1.0; // Full opacity for selected file
    if (selectedByAngle.includes(filename)) return 0.9; // High opacity for angle selected
    return 0.15; // Very low opacity (85% transparent) for non-selected when there's a selection
  });

  // Create traces array - start with main scatter plot
  const traces: any[] = [];

  // Create hover text with angle information
  const hoverText = text.map((filename, index) => {
    let baseText = `<b>${filename}</b>`;
    
    // Add angle information if this point is selected by angle range and in 3D mode
    if (is3D && selectedPlane !== 'none' && selectedByAngle.includes(filename) && z) {
      const [px, py, pz] = [x[index], y[index], z[index]];
      const angle = calculateAngleToPlane(px, py, pz, selectedPlane);
      baseText += `<br>Angle: ${angle.toFixed(1)}°`;
      baseText += `<br>Plane: ${selectedPlane.toUpperCase()}`;
    }
    
    return baseText;
  });

  // Create main trace data
  const traceData: any = {
    x: x,
    y: y,
    mode: 'markers',
    type: is3D ? 'scatter3d' : 'scatter',
    marker: {
      size: markerSizes,
      color: markerColors,
      colorscale: Object.entries(colorMap).map(([key, value], index) => [
        index / (Object.keys(colorMap).length - 1),
        value
      ]),
      showscale: false,
      line: {
        width: 0, // Remove marker outlines
        color: 'transparent'
      },
      opacity: markerOpacities // Use dynamic opacity array
    },
    text: hoverText,
    customdata: colors,
  };

  // Add Z coordinate for 3D plots
  if (is3D && z) {
    traceData.z = z;
    traceData.hovertemplate = '%{text}<extra></extra>';
  } else {
    traceData.hovertemplate = '%{text}<extra></extra>';
  }

  traces.push(traceData);

  // Add origin point (0,0,0) highlight for 3D plots or (0,0) for 2D plots
  const originTrace: any = {
    x: [0],
    y: [0],
    mode: 'markers',
    type: is3D ? 'scatter3d' : 'scatter',
    marker: {
      size: is3D ? 5 : 4, // Slightly smaller to match the new scale
      color: '#000000', // Black for origin
      symbol: 'diamond',
      line: {
        width: 1, // Thinner outline
        color: '#ffffff' // White outline for visibility
      },
      opacity: 0.8 // Slightly transparent
    },
    text: [is3D ? 'Origin (0,0,0)' : 'Origin (0,0)'],
    hovertemplate: is3D ? '<b>Origin (0,0,0)</b><extra></extra>' : '<b>Origin (0,0)</b><extra></extra>',
    name: 'Origin',
    showlegend: false
  };

  if (is3D) {
    originTrace.z = [0];
  }

  traces.push(originTrace);

  // Add plane if selected and in 3D mode
  if (is3D && selectedPlane !== 'none') {
    const planeTrace = createPlane(selectedPlane, bounds);
    if (planeTrace) {
      traces.push(planeTrace);
    }
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
        showgrid: false, // Remove grid lines
        gridcolor: '#e5e7eb',
        showticklabels: false,
        title: { text: 'X', font: { size: 10 } },
        backgroundcolor: 'white',
        showspikes: false,
        zeroline: false, // Remove zero line
        showline: false  // Remove axis line
      },
      yaxis: {
        showgrid: false, // Remove grid lines
        gridcolor: '#e5e7eb',
        showticklabels: false,
        title: { text: 'Y', font: { size: 10 } },
        backgroundcolor: 'white',
        showspikes: false,
        zeroline: false, // Remove zero line
        showline: false  // Remove axis line
      },
      zaxis: {
        showgrid: false, // Remove grid lines
        gridcolor: '#e5e7eb',
        showticklabels: false,
        title: { text: 'Z', font: { size: 10 } },
        backgroundcolor: 'white',
        showspikes: false,
        zeroline: false, // Remove zero line
        showline: false  // Remove axis line
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
    <div className="w-full h-full min-h-0 relative">
      {/* Plane Selection Controls - Only show in 3D mode */}
      {is3D && (
        <div className="absolute top-2 right-2 z-10 bg-white/95 backdrop-blur-sm border border-gray-200 rounded-md p-2 shadow-sm">
          {/* Plane Selection */}
          <div className="flex items-center gap-2 mb-2">
            <Layers3 className="h-3 w-3 text-gray-600" />
            <span className="text-xs text-gray-600 font-medium">Plane:</span>
            <Select
              value={selectedPlane}
              onValueChange={(value: PlaneType) => setSelectedPlane(value)}
            >
              <SelectTrigger className="w-16 h-6 text-xs border-gray-300 hover:border-gray-400 transition-colors">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">None</SelectItem>
                <SelectItem value="xy">X-Y</SelectItem>
                <SelectItem value="xz">X-Z</SelectItem>
                <SelectItem value="yz">Y-Z</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          {/* Angle Range Selector */}
          {selectedPlane !== 'none' && (
            <div className="space-y-2 pt-2 border-t border-gray-200">
              <div className="flex items-center gap-1">
                <Target className="h-3 w-3 text-gray-600" />
                <span className="text-xs text-gray-600 font-medium">Angle Range:</span>
              </div>
              
              <div className="flex items-center gap-1">
                <Input
                  type="number"
                  min="0"
                  max="90"
                  step="1"
                  value={angleMin}
                  onChange={(e) => setAngleMin(Number(e.target.value))}
                  className="w-14 h-6 text-xs text-center px-1"
                />
                <span className="text-xs text-gray-500">-</span>
                <Input
                  type="number"
                  min="0"
                  max="90"
                  step="1"
                  value={angleMax}
                  onChange={(e) => setAngleMax(Number(e.target.value))}
                  className="w-14 h-6 text-xs text-center px-1"
                />
                <span className="text-xs text-gray-500">°</span>
              </div>
              
              {selectedByAngle.length > 0 && (
                <div className="text-[10px] text-red-600 bg-red-50 px-2 py-1 rounded">
                  🔴 {selectedByAngle.length} points selected
                </div>
              )}
            </div>
          )}
          
          {selectedPlane !== 'none' && (
            <div className="text-[10px] text-gray-500 mt-1">
              {selectedPlane === 'xy' && '🔵 Blue plane: X-Y (Z=0)'}
              {selectedPlane === 'xz' && '🟢 Green plane: X-Z (Y=0)'}
              {selectedPlane === 'yz' && '🔴 Red plane: Y-Z (X=0)'}
            </div>
          )}
        </div>
      )}
      
      <Plot
        data={traces}
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
            filename: `embeddings_${selectedMethod}_${is3D ? '3D' : '2D'}${selectedPlane !== 'none' ? `_${selectedPlane}` : ''}`,
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