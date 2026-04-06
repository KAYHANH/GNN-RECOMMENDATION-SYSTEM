# Explainability Path

The original notebook uses a graph explainability flow to attribute a recommendation back to influential edges in a customer's historical subgraph.

In this upgraded scaffold:

- The API currently exposes lightweight explanation text through heuristic history matching.
- The data pipeline now preserves the original heterogeneous graph structure through `training/build_hetero_graph.py`.
- This keeps the project compatible with a future `CaptumExplainer` or `GNNExplainer` stage once the full PyTorch Geometric explainability stack and trained heterogeneous model artifacts are available.

Suggested next step after dataset ingestion:

1. Rebuild the heterogeneous graph with recent transactions.
2. Train a heterogeneous explainability-compatible baseline alongside the LightGCN candidate generator.
3. Store per-node mappings and prediction targets for edge attribution.
4. Replace the heuristic `/explain` implementation with graph-attribution-backed reasons.

