# backend/test_load.py
#basic test file for testing whether our GPT model is getting loaded correctly 

from transformer_lens import HookedTransformer
import torch

print("Loading GPT-2 Small...")
model = HookedTransformer.from_pretrained("gpt2")
model.eval()

print(f"Layers : {model.cfg.n_layers}")   # 12
print(f"Heads  : {model.cfg.n_heads}")    # 12
print(f"d_model: {model.cfg.d_model}")    # 768
print("Model loaded successfully!")

# Test a basic forward pass
prompt = "The capital of France is"
tokens = model.to_tokens(prompt)
print(f"\nTokens: {model.to_str_tokens(prompt)}")

logits = model(tokens)
next_token_id = logits[0, -1, :].argmax()
print(f"Predicted next token: '{model.to_string(next_token_id)}'")
# Should print: ' Paris'

# This is the core call you'll use everywhere
logits, cache = model.run_with_cache(tokens)

# Inspect what's cached
print("\nCached activations:")
for key in list(cache.keys())[:10]:
    print(f"  {key}: {cache[key].shape}")