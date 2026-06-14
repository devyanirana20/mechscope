# backend/patching.py
from transformer_lens import HookedTransformer
import torch
import numpy as np

# ── Load model ──────────────────────────────────────────────
model = HookedTransformer.from_pretrained("gpt2")
model.eval()

# ── Define prompt pair ───────────────────────────────────────
clean_prompt     = "The capital of France is"
corrupted_prompt = "The capital of Germany is"
target_token     = " Paris"   # note the space — GPT-2 tokenizes " Paris" not "Paris"

# ── Tokenize ─────────────────────────────────────────────────
clean_tokens = model.to_tokens(clean_prompt)
corr_tokens  = model.to_tokens(corrupted_prompt)

# ── Get target token ID ──────────────────────────────────────
target_id = model.to_single_token(target_token)
print(f"Target token '{target_token}' has ID: {target_id}")

# ── Cache both runs ──────────────────────────────────────────
clean_logits, clean_cache = model.run_with_cache(clean_tokens)
_,            corr_cache  = model.run_with_cache(corr_tokens)

# ── Baseline score ───────────────────────────────────────────
baseline_score = clean_logits[0, -1, target_id].item()
print(f"Baseline logit for '{target_token}': {baseline_score:.4f}")

# # ── Patch a single head first (layer 9, head 9) ─────────────
# layer_to_test = 9
# head_to_test  = 9

# def single_patch_hook(value, hook):
#     # Replace head 9's output with the corrupted run's version
#     value[:, :, head_to_test, :] = corr_cache[hook.name][:, :, head_to_test, :]
#     return value

# hook_name = f"blocks.{layer_to_test}.attn.hook_z"

# patched_logits = model.run_with_hooks(
#     clean_tokens,
#     fwd_hooks=[(hook_name, single_patch_hook)]
# )

# patched_score = patched_logits[0, -1, target_id].item()
# logit_drop    = baseline_score - patched_score

# print(f"\nPatching Layer {layer_to_test}, Head {head_to_test}:")
# print(f"  Patched logit : {patched_score:.4f}")
# print(f"  Logit drop    : {logit_drop:.4f}")
# print(f"  {'HIGH IMPACT ⚠️' if logit_drop > 1.0 else 'low impact'}")

# ── Full patching loop ───────────────────────────────────────
n_layers = model.cfg.n_layers  # 12
n_heads  = model.cfg.n_heads   # 12

results = np.zeros((n_layers, n_heads))

print(f"\nRunning patching loop: {n_layers} layers × {n_heads} heads = {n_layers * n_heads} patches")
print("This takes ~5-8 minutes on CPU. Progress:")

for layer in range(n_layers):
    for head in range(n_heads):

        # Need corr_cache captured OUTSIDE the hook
        # so we capture it in a closure
        def make_hook(l, h):
            def patch_hook(value, hook):
                value[:, :, h, :] = corr_cache[f"blocks.{l}.attn.hook_z"][:, :, h, :]
                return value
            return patch_hook

        hook_name = f"blocks.{layer}.attn.hook_z"

        patched_logits = model.run_with_hooks(
            clean_tokens,
            fwd_hooks=[(hook_name, make_hook(layer, head))]
        )

        patched_score         = patched_logits[0, -1, target_id].item()
        results[layer, head]  = baseline_score - patched_score

    # Progress indicator
    print(f"  Layer {layer:2d} done | max drop this layer: {results[layer].max():.3f}")

print("\nPatching complete!")
print(f"Most impactful head: Layer {results.max(axis=1).argmax()}, Head {results.argmax() % n_heads}")


# ── Save results ─────────────────────────────────────────────
np.save("patching_results.npy", results)
print("Results saved to patching_results.npy")