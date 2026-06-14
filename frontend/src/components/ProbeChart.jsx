// src/components/ProbeChart.jsx
import Plot from "react-plotly.js"

export default function ProbeChart({ data }) {
  const layers = data.layer_scores.map((_, i) => i)

  return (
    <div className="mt-6 bg-gray-900 rounded-xl p-4">
      <h2 className="text-lg font-semibold mb-2">Probe Classifier — Layer Analysis</h2>
      <p className="text-gray-400 text-sm mb-4">
        Shows at which layer the model internally "recognizes" a harmful prompt
      </p>
      <Plot
        data={[{
          x: layers,
          y: data.layer_scores,
          type: "scatter",
          mode: "lines+markers",
          line: { color: "#6366f1", width: 2 },
          marker: { size: 8 }
        }]}
        layout={{
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
          font: { color: "white" },
          xaxis: { title: "Layer", gridcolor: "#374151" },
          yaxis: { title: "Harmfulness Score", range: [0, 1], gridcolor: "#374151" },
          shapes: [{
            type: "line", x0: 0, x1: layers.length,
            y0: 0.5, y1: 0.5,
            line: { dash: "dash", color: "red", width: 1 }
          }]
        }}
        style={{ width: "100%" }}
      />
    </div>
  )
}