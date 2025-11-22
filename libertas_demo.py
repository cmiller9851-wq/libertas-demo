"""
libertas_demo.py
Single-file prototype for Libertas — Core-Value Aligned Decision Framework

Requirements:
- Python 3.8+
- tensorflow (TF2)
- numpy

Run:
    python libertas_demo.py
    python libertas_demo.py --use-examples examples/sample_options.json
"""

from enum import Enum, auto
import json
import re
import argparse
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Tuple

# -------------------------
# Core values enum
# -------------------------
class CoreValue(Enum):
    AUTONOMY = auto()
    TRANSPARENCY = auto()
    BENEFICENCE = auto()
    NON_MALEFICENCE = auto()

# -------------------------
# Helpers
# -------------------------
def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def readability_score(text: str) -> float:
    """
    Very small heuristic: shorter sentences and common word count -> higher.
    Returns 0..1 where 1 is most readable by this heuristic.
    """
    if not text:
        return 0.0
    words = re.findall(r"\w+", text)
    sent_count = max(1, text.count('.') + text.count('!') + text.count('?'))
    avg_words = len(words) / sent_count
    # optimal avg words ~ 12 -> map to 0..1
    score = 1.0 - abs(avg_words - 12.0) / 20.0
    return clamp01(score)

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

# -------------------------
# SymbolicReasoner (stub)
# -------------------------
class SymbolicReasoner:
    """
    Stubbed reasoner: takes an embedding / feature vector and returns a
    small list of candidate options. Each option is a dict:
      { "label": str,
        "explanation": str,
        "predicted_impact": float,    # raw scalar (can be negative..positive)
        "control_score": float }      # 0..1 how much user retains control
    """

    def __init__(self):
        pass

    def reason(self, features: np.ndarray) -> List[Dict[str, Any]]:
        # For demo purposes generate three deterministic-ish options based on features
        seed = int(abs(features.sum() * 1000)) % 1000
        rng = np.random.RandomState(seed)
        base = float(np.tanh(features.mean()))
        opts = []

        # Option A: Full automation (high benefit, low control)
        opts.append({
            "label": "Full automation",
            "explanation": "We will automatically perform the task end-to-end for you.",
            "predicted_impact": base + rng.normal(loc=1.0, scale=0.3),
            "control_score": 0.0
        })

        # Option B: High user control, moderate benefit
        opts.append({
            "label": "Assistive mode (user-in-command)",
            "explanation": "We provide suggestions and allow you to choose the final action.",
            "predicted_impact": base + rng.normal(loc=0.5, scale=0.2),
            "control_score": 0.9
        })

        # Option C: Ultra-safe, low benefit
        opts.append({
            "label": "Manual safe mode",
            "explanation": "We give minimal intervention and prioritize avoiding any risk.",
            "predicted_impact": base + rng.normal(loc=0.1, scale=0.1),
            "control_score": 1.0
        })

        return opts

# -------------------------
# Libertas core class
# -------------------------
class Libertas:
    def __init__(self, weights: Dict[CoreValue, float] = None):
        # Simple Keras feature extractor
        self.model = self._build_model()
        self.reasoner = SymbolicReasoner()
        # Default weights if not provided; sum should be 1.0
        if weights is None:
            weights = {
                CoreValue.AUTONOMY: 0.25,
                CoreValue.TRANSPARENCY: 0.25,
                CoreValue.BENEFICENCE: 0.25,
                CoreValue.NON_MALEFICENCE: 0.25
            }
        # Normalize to sum 1.0 defensively
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Weights must sum to > 0")
        self.weights = {k: float(v) / total for k, v in weights.items()}

    def _build_model(self):
        # Tiny model: maps a simple bag-of-chars vector to a small embedding
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(128,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(8, activation='tanh')
        ])
        # Not trained; initialized randomly for demo
        return model

    def _vectorize(self, text: str) -> np.ndarray:
        # Simple deterministic char-level hashing into 128-d vector
        vec = np.zeros(128, dtype=np.float32)
        for i, ch in enumerate(text[:256]):
            idx = (ord(ch) * (i + 1)) % 128
            vec[idx] += 1.0
        # normalize
        if vec.sum() > 0:
            vec = vec / (np.linalg.norm(vec) + 1e-9)
        return vec

    def extract_features(self, text: str) -> np.ndarray:
        vec = self._vectorize(text)
        # run through model (eager)
        emb = self.model(np.expand_dims(vec, axis=0)).numpy().flatten()
        return emb

    def evaluate_option(self, option: Dict[str, Any]) -> Dict[CoreValue, float]:
        """
        Compute per-core-value scores (0..1) for a single candidate option.
        Uses heuristics described in the slides.
        """
        explanation = option.get("explanation", "")
        control = float(option.get("control_score", 0.0))
        predicted_impact = float(option.get("predicted_impact", 0.0))

        scores: Dict[CoreValue, float] = {}

        # Autonomy: directly from control_score + bonus if explanation contains empowering phrases
        autonomy = control
        if re.search(r"\byou can\b|\byou may\b|\bchoose\b|\bdecide\b", explanation, re.IGNORECASE):
            autonomy += 0.1
        scores[CoreValue.AUTONOMY] = clamp01(autonomy)

        # Transparency: readability heuristic + bonus for causal words
        trans = readability_score(explanation)
        if re.search(r"\bbecause\b|\bsince\b|\bdue to\b|\btherefore\b", explanation, re.IGNORECASE):
            trans += 0.1
        scores[CoreValue.TRANSPARENCY] = clamp01(trans)

        # Beneficence: sigmoid over predicted impact (maps real -> 0..1)
        benef = float(sigmoid(predicted_impact))
        scores[CoreValue.BENEFICENCE] = clamp01(benef)

        # Non-maleficence: penalty for negative impact and mentions of risk/harm
        nm = 1.0
        if predicted_impact < 0:
            nm -= clamp01(abs(predicted_impact))  # heavier penalty for more negative impact
        if re.search(r"\brisk\b|\bharm\b|\bdanger\b", explanation, re.IGNORECASE):
            nm -= 0.3
        scores[CoreValue.NON_MALEFICENCE] = clamp01(nm)

        return scores

    def align_with_core_values(self, options: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        For each option, compute a per-value breakdown and a weighted total score.
        Returns options with added fields:
          - value_scores: {CoreValue: score}
          - total_score: float
        """
        results = []
        for opt in options:
            vs = self.evaluate_option(opt)
            total = 0.0
            for cv, weight in self.weights.items():
                total += vs.get(cv, 0.0) * weight
            # attach breakdown and total
            opt_copy = dict(opt)
            opt_copy["value_scores"] = {cv.name: float(vs[cv]) for cv in vs}
            opt_copy["total_score"] = float(total)
            results.append(opt_copy)
        # sort descending by total_score
        results.sort(key=lambda o: o["total_score"], reverse=True)
        return results

    def run(self, input_text: str, use_custom_options: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the full pipeline:
         - extract features
         - produce candidate options (or use provided ones)
         - score and pick winner
        Returns dict with 'options' (scored) and 'winner'
        """
        features = self.extract_features(input_text)
        if use_custom_options is None:
            candidates = self.reasoner.reason(features)
        else:
            candidates = use_custom_options
        scored = self.align_with_core_values(candidates)
        winner = scored[0] if scored else None
        result = {
            "input": input_text,
            "options": scored,
            "winner": winner
        }
        return result

# -------------------------
# CLI demo utilities
# -------------------------
def print_scorecard(result: Dict[str, Any]) -> None:
    print("\nInput:")
    print(result["input"])
    print("\nOptions (sorted by total score):")
    for i, opt in enumerate(result["options"], start=1):
        print(f"\n[{i}] {opt['label']}")
        print(f"  Explanation: {opt['explanation']}")
        print(f"  Predicted impact: {opt.get('predicted_impact'):.3f}")
        print(f"  Control (user): {opt.get('control_score'):.3f}")
        print("  Per-value scores:")
        for cv_name, score in opt["value_scores"].items():
            print(f"    {cv_name}: {score:.3f}")
        print(f"  Total (weighted): {opt['total_score']:.3f}")
    if result["winner"]:
        print("\nWinner:")
        print(f"  {result['winner']['label']} (total {result['winner']['total_score']:.3f})")
    else:
        print("No winner (no options).")

def load_examples(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Expect list of options
    return data

# -------------------------
# Example sample options JSON (used by examples/sample_options.json)
# Included here as a convenience generator if file not provided.
# -------------------------
SAMPLE_OPTIONS = [
    {
        "label": "Full automation",
        "explanation": "We will perform the task automatically without asking you for confirmation.",
        "predicted_impact": 2.0,
        "control_score": 0.0
    },
    {
        "label": "Assistive mode (user-in-command)",
        "explanation": "We give clear suggestions; you choose the final action.",
        "predicted_impact": 1.0,
        "control_score": 0.9
    },
    {
        "label": "Manual safe mode",
        "explanation": "We avoid risk and only provide minimal assistance to ensure no harm.",
        "predicted_impact": 0.2,
        "control_score": 1.0
    }
]

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Libertas demo")
    parser.add_argument("--input", "-i", type=str, default="Please schedule my doctor's appointment and optimize for convenience.",
                        help="Input text to the system")
    parser.add_argument("--use-examples", "-e", type=str, default=None,
                        help="Path to a JSON file with candidate options (list of dicts).")
    parser.add_argument("--weights", "-w", type=str, default=None,
                        help="Optional JSON string for weights, e.g. '{\"AUTONOMY\":0.6,\"NON_MALEFICENCE\":0.4}'")
    args = parser.parse_args()

    weights = None
    if args.weights:
        try:
            raw = json.loads(args.weights)
            # map keys to CoreValue
            weights = {}
            for k, v in raw.items():
                # accept either name strings or lowercase
                key_norm = k.strip().upper()
                # try to match enum
                if key_norm == "NON_MALEFICENCE" or key_norm == "NON-MALEFICENCE" or key_norm == "NON_MALEFAICENCE":
                    key_enum = CoreValue.NON_MALEFICENCE
                else:
                    key_enum = CoreValue[key_norm]
                weights[key_enum] = float(v)
        except Exception as e:
            print("Warning: could not parse weights JSON -- using defaults. Error:", e)
            weights = None

    libertas = Libertas(weights=weights)

    custom_opts = None
    if args.use_examples:
        try:
            custom_opts = load_examples(args.use_examples)
        except Exception as e:
            print("Failed to load examples file; falling back to built-in sample options. Error:", e)
            custom_opts = SAMPLE_OPTIONS
    result = libertas.run(args.input, use_custom_options=custom_opts)
    print_scorecard(result)

if __name__ == "__main__":
    main()
