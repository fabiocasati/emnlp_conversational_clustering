# Conversational Clustering — Supplementary Analyses

Analysis scripts, configurations, and result artifacts for the supplementary
experiments reported in *Rethinking Clustering as a Conversational Supervision
Problem* (EMNLP 2026): the non-conversational reference points (paper
Appendix C), the robustness analysis at two intent granularities (App. B.1.4),
the benchmark orthogonality statistics, and the open-weight judge check
(App. B.1.6). The main framework, reference implementation, and benchmark
generator are released separately; this repository contains everything needed
to verify and re-derive the numbers of these analyses.

## Layout

| Directory | Paper section | Contents |
|---|---|---|
| `data/` | §4.2 | The released 1,000-ticket benchmark CSV and its generation manifest |
| `oneshot_baselines/` | App. C | One-shot pipelines (K-means, Ward, HDBSCAN × TF-IDF+SVD, Qwen3-Embedding-8B, stated-POV prefixed) scored against all 24 reference partitions; full 65-configuration grid in `results/` |
| `robustness/` | App. B.1.4 | Two-granularity pair analysis (resolved theme vs. stated intent), permutation tests, invariance/responsiveness sweep; the scored pair records are in `inputs/` |
| `orthogonality/` | §4.2 / App. C | Pairwise ARI between the 24 reference labelings |
| `openweight_judge/` | App. B.1.6 | Re-judging configuration (Qwen3-235B-A22B-Instruct, temperature 0), per-metric agreement analysis, and both judges' detail records |

## Reproducing

All analyses run offline from the artifacts in this repository (Python 3.11,
`numpy`, `scipy`, `scikit-learn`, `pandas`); only optional embedding
regeneration (`oneshot_baselines/gen_embeddings.py`) calls the OpenRouter API.

- **One-shot baselines** (App. C, Tables): `python oneshot_baselines/exp_a_baselines.py --dataset data/enterprise_ticket_dataset_1000.csv --out-dir oneshot_baselines/results` (see script headers for the embedding options).
- **Robustness two-granularity analysis** (App. B.1.4): `python robustness/exp_b_from_pairs.py --pairs-jsonl robustness/inputs/robustness_expectations.jsonl --out-dir robustness/results`.
- **Orthogonality** : `python orthogonality/orthogonality.py --dataset data/enterprise_ticket_dataset_1000.csv --out-dir orthogonality/results`.
- **Open-weight judge agreement** (App. B.1.6): `python openweight_judge/judge_agreement.py` — compares the closed-judge details (`openweight_judge/inputs/closed_judge_details/`) with the open-weight re-judging (`openweight_judge/results/`) and writes `judge_agreement.json`. The re-judging itself was produced by running `openweight_judge/final_clustering_quality_qwen.toml` with the released framework's evaluation runner on the saved run artifacts.

Result files checked in under each `results/` directory are the exact
artifacts behind the numbers reported in the paper.

## License

License to be finalized upon publication.
