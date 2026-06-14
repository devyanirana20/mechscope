// src/components/AttentionHeatmap.jsx
import Plot from "react-plotly.js"


export default function AttentionHeatmap({ data }) {
  return (
    <div className="mt-6 bg-gray-900 rounded-xl p-4">
      <h2 className="text-lg font-semibold mb-4">Activation Patching Heatmap</h2>
      <p className="text-gray-400 text-sm mb-4">
        Brighter = that head is causally responsible for the target token prediction
      </p>
      <Plot
        data={[{
          z: data.heatmap,
          type: "heatmap",
          colorscale: "RdBu",
          reversescale: true,
          colorbar: { title: "Logit Drop" }
        }]}
        layout={{
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
          font: { color: "white" },
          xaxis: { title: "Attention Head" },
          yaxis: { title: "Layer" },
          margin: { t: 20 }
        }}
        style={{ width: "100%" }}
      />
    </div>
  )
}