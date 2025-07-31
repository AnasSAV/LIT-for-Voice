import { useEffect, useRef } from "react";
import Plot from "react-plotly.js";

export const EmbeddingPlot = () => {
  // Generate mock embedding data
  const generateMockData = () => {
    const n = 200;
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

  const { x, y, colors, text } = generateMockData();

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
              size: 6,
              color: colors,
              colorscale: [
                [0, 'hsl(var(--saliency-low))'],
                [0.33, 'hsl(var(--saliency-medium))'],
                [0.66, 'hsl(var(--saliency-high))'],
                [1, 'hsl(var(--primary))']
              ],
              line: {
                width: 1,
                color: 'hsl(var(--border))'
              }
            },
            text: text,
            hovertemplate: '<b>%{text}</b><br>X: %{x}<br>Y: %{y}<extra></extra>'
          }
        ]}
        layout={{
          autosize: true,
          margin: { l: 30, r: 30, t: 30, b: 30 },
          xaxis: {
            showgrid: true,
            gridcolor: 'hsl(var(--border))',
            showticklabels: false
          },
          yaxis: {
            showgrid: true,
            gridcolor: 'hsl(var(--border))',
            showticklabels: false
          },
          plot_bgcolor: 'transparent',
          paper_bgcolor: 'transparent',
          showlegend: false,
          font: {
            size: 10,
            color: 'hsl(var(--foreground))'
          }
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