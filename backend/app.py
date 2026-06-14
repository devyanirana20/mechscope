# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from model_wrapper import MechScopeModel
import os

app = Flask(__name__)
CORS(app)

print("Loading model... (takes ~30s on CPU)")
scope = MechScopeModel()
print("Model ready.")

@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/api/patch", methods=["POST"])
def patch():
    data = request.json
    clean    = data.get("clean_prompt", "")
    corrupt  = data.get("corrupted_prompt", "")
    target   = data.get("target_token", "")

    if not all([clean, corrupt, target]):
        return jsonify({"error": "Missing fields"}), 400

    results = scope.activation_patch(clean, corrupt, target)
    return jsonify({
        "heatmap": results.tolist(),
        "n_layers": results.shape[0],
        "n_heads":  results.shape[1]
    })

@app.route("/api/probe", methods=["POST"])
def probe():
    data = request.json
    prompt = data.get("prompt", "")
    scores = scope.probe_all_layers(prompt)
    return jsonify({"layer_scores": scores})

@app.route("/api/attention", methods=["POST"])
def attention():
    data = request.json
    prompt = data.get("prompt", "")
    patterns = scope.get_attention_patterns(prompt)
    return jsonify({"patterns": patterns})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)