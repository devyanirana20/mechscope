// src/App.jsx
import { useState } from "react"
import PromptInput from "./components/PromptInput"
import AttentionHeatmap from "./components/AttentionHeatmap"
import ProbeChart from "./components/ProbeChart"
import AttentionViewer from "./components/AttentionViewer"

const TABS = ["Activation Patching", "Probe Classifier", "Attention Patterns"]

export default function App() {
  const [activeTab, setActiveTab] = useState(0)
  const [results, setResults] = useState<any>(null)
  const [loading, setLoading] = useState<boolean>(false)
  return (
    <div className="min-h-screen bg-gray-950 text-white p-6">
      <h1 className="text-3xl font-bold text-center mb-2">MechScope</h1>
      <p className="text-center text-gray-400 mb-8">
        Mechanistic Interpretability Toolkit for Open LLMs
      </p>

      {/* Tab Nav */}
      <div className="flex gap-2 justify-center mb-8">
        {TABS.map((tab, i) => (
          <button
            key={i}
            onClick={() => { setActiveTab(i); setResults(null) }}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition
              ${activeTab === i
                ? "bg-indigo-600 text-white"
                : "bg-gray-800 text-gray-400 hover:bg-gray-700"}`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 0 && (
        <>
          <PromptInput mode="patch" onResult={setResults} setLoading={setLoading} />
          {loading && <p className="text-center text-gray-400 mt-4">Running patching... (~20s)</p>}
          {results && <AttentionHeatmap data={results} />}
        </>
      )}
      {activeTab === 1 && (
        <>
          <PromptInput mode="probe" onResult={setResults} setLoading={setLoading} />
          {loading && <p className="text-center text-gray-400 mt-4">Probing layers...</p>}
          {results && <ProbeChart data={results} />}
        </>
      )}
      {activeTab === 2 && (
        <>
          <PromptInput mode="attention" onResult={setResults} setLoading={setLoading} />
          {results && <AttentionViewer data={results} />}
        </>
      )}
    </div>
  )
}