import Plot from "react-plotly.js";

interface ScalarPlotProps {
  type: "confidence" | "attention";
}

export const ScalarPlot = ({ type }: ScalarPlotProps) => {
  const generateData = () => {
    const n = 50;
    const values = [];
    
    for (let i = 0; i < n; i++) {
      if (type === "confidence") {
        values.push(Math.random() * 0.4 + 0.6); // 0.6 to 1.0
      } else {
        values.push(Math.random()); // 0 to 1.0
      }
    }
    
    return values;
  };

  const data = generateData();

  return (
    <div className="h-full">
      <Plot
        data={[
          {
            y: data,
            type: 'box',
            boxpoints: 'outliers',
            marker: {
              color: 'hsl(var(--primary))',
              size: 3
            },
            line: {
              color: 'hsl(var(--primary))'
            },
            fillcolor: 'hsl(var(--primary) / 0.3)',
            name: type === "confidence" ? "Confidence" : "Attention"
          }
        ]}
        layout={{
          autosize: true,
          margin: { l: 40, r: 20, t: 10, b: 30 },
          yaxis: {
            showgrid: true,
            gridcolor: 'hsl(var(--border))',
            range: [0, 1],
            tickfont: { size: 10 }
          },
          xaxis: {
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