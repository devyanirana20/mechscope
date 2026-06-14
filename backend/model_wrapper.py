# backend/model_wrapper.py
from transformer_lens import HookedTransformer
import torch, numpy as np
from sklearn.linear_model import LogisticRegression
import pickle, os

class MechScopeModel:
    def __init__(self):
        print("Loading GPT-2...")
        self.model = HookedTransformer.from_pretrained("gpt2")
        self.model.eval()
        self.n_layers = self.model.cfg.n_layers
        self.n_heads  = self.model.cfg.n_heads
        self._load_probes()

    def _load_probes(self):
        probe_path = "probes/probes.pkl"
        if os.path.exists(probe_path):
            with open(probe_path, "rb") as f:
                self.probes = pickle.load(f)
        else:
            self.probes = None

    def activation_patch(self, clean_prompt, corrupted_prompt, target_token):
        clean_tokens = self.model.to_tokens(clean_prompt)
        corr_tokens  = self.model.to_tokens(corrupted_prompt)

        _, clean_cache = self.model.run_with_cache(clean_tokens)
        clean_logits, _ = self.model.run_with_cache(clean_tokens)

        try:
            target_id = self.model.to_single_token(" " + target_token)
        except:
            target_id = self.model.to_single_token(target_token)

        results = torch.zeros(self.n_layers, self.n_heads)

        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                def patch_hook(value, hook, h=head, l=layer):
                    corr_logits, corr_cache = self.model.run_with_cache(corr_tokens)
                    value[:, :, h, :] = corr_cache[hook.name][:, :, h, :]
                    return value

                hook_name = f"blocks.{layer}.attn.hook_z"
                patched = self.model.run_with_hooks(
                    clean_tokens,
                    fwd_hooks=[(hook_name, patch_hook)]
                )
                orig    = clean_logits[0, -1, target_id].item()
                patched_score = patched[0, -1, target_id].item()
                results[layer, head] = orig - patched_score

        return results

    def probe_all_layers(self, prompt):
        if not self.probes:
            return [0.5] * self.n_layers  # fallback if no trained probes

        tokens = self.model.to_tokens(prompt)
        _, cache = self.model.run_with_cache(tokens)
        scores = []

        for layer in range(self.n_layers):
            hs = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :].detach().numpy()
            prob = self.probes[layer].predict_proba([hs])[0][1]
            scores.append(float(prob))

        return scores

    def get_attention_patterns(self, prompt):
        tokens = self.model.to_tokens(prompt)
        _, cache = self.model.run_with_cache(tokens)
        token_strs = self.model.to_str_tokens(prompt)

        patterns = []
        for layer in range(self.n_layers):
            attn = cache[f"blocks.{layer}.attn.hook_pattern"]
            patterns.append(attn[0].mean(0).tolist())  # avg across heads

        return {"patterns": patterns, "tokens": token_strs}