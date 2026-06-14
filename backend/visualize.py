# backend/visualize.py
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

results = np.load("patching_results.npy")

fig = go.Figure(data=go.Heatmap(
    z=results,
    colorscale="RdBu",
    reversescale=True,
    colorbar=dict(title="Logit Drop"),
    hovertemplate="Layer %{y}, Head %{x}<br>Logit Drop: %{z:.3f}<extra></extra>"
))

fig.update_layout(
    title=dict(
        text="Activation Patching: Which Heads Control 'Paris' Prediction",
        font=dict(size=16)
    ),
    xaxis=dict(title="Attention Head", tickmode="linear"),
    yaxis=dict(title="Layer", tickmode="linear"),
    width=700,
    height=500
)

# Save as HTML (viewable in Codespaces browser preview)
fig.write_html("patching_heatmap.html")
print("Heatmap saved to patching_heatmap.html")

# Print top 5 most important heads
flat = results.flatten()
top5 = flat.argsort()[-5:][::-1]
print("\nTop 5 most causally important heads:")
for idx in top5:
    l, h = divmod(idx, results.shape[1])
    print(f"  Layer {l:2d}, Head {h:2d} → logit drop: {results[l, h]:.4f}")