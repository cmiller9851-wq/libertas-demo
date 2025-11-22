Libertas — architecture notes

Overview
- Single-file prototype: libertas_demo.py
- Three main components:
  1. Tiny Keras feature extractor (untrained) — maps input text → embedding
  2. SymbolicReasoner (stub) — produces labeled candidate options with explanation, predicted_impact, control_score
  3. Alignment scorer — evaluate each candidate on AUTONOMY, TRANSPARENCY, BENEFICENCE, NON_MALEFICENCE and compute weighted total

Where to plug real components
- Feature extractor
  - Replace the tiny Keras model with a pretrained encoder (e.g., SentenceTransformers, an LLM embedding endpoint, or a fine-tuned TF model).
  - Ensure output is a fixed-size vector compatible with the reasoner/adapter.

- SymbolicReasoner
  - Swap stub for adapters:
    - Rules engine (Drools, custom rule set)
    - Retrieval + LLM summarizer (retrieve candidate actions, ask LLM to generate explanations and predicted impact)
    - LLMReasoner adapter (e.g., Llama/Claude) that returns the required fields
  - Keep outputs explicit and structured to preserve auditability.

- Predicted impact model
  - Replace the placeholder scalar with a calibrated outcome model (regression or probabilistic estimator).
  - Consider returning distributional outcomes and summarize (e.g., expected utility) prior to mapping via sigmoid.

- Auditing & logging
  - Add persistent JSONL audit log with timestamp, input, options, per-value scores, weights, and model/version metadata.
  - Sign or hash logs for tamper-evidence if needed.

- Human-in-the-loop
  - Add gating layer for high-risk decisions: threshold on Non-maleficence or a separate risk classifier that requires explicit human approval.

Weights and governance
- Keep core-value weights in a single, versioned config file or database.
- Record weight changes in audit logs and require stakeholder approvals for policy-sensitive updates.

Testing & CI
- Unit tests for evaluators with deterministic fixtures.
- Integration tests for end-to-end behavior with mocked reasoner/model outputs.
- CI to run linting, tests, and optionally small smoke runs of the demo.

Security & privacy
- Sanitize inputs and avoid logging sensitive PII in cleartext in audit logs.
- Prefer on-prem or VPC-hosted models for regulated domains.

Scaling notes
- For higher throughput, separate feature extraction and reasoning into microservices, use batching for embeddings, and cache common inputs.

Suggested next tech tasks (short)
- Implement LLMReasoner adapter
- Add JSONL audit log and CLI flag to enable it
- Replace predicted_impact with a calibrated model and unit tests
- Add config file for weights and a simple web UI to adjust/view them
