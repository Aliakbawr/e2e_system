# Analysis path relocation record

No experimental observation or statistical value was changed during this reorganization. Files were grouped by evaluation version and research role. Historical manifests are retained byte-for-byte under `historical_pre_restructure/`; their embedded paths describe the former layout.

Frozen metadata JSON files likewise retain their original absolute creation-time paths. Those fields are provenance records, not current path configuration; runnable script defaults now point to the reorganized locations.

| Former location | Current location |
|---|---|
| `results/fleurs_asr_qa_eval_input_v1.csv` | `analysis/inputs/fleurs_asr_qa_eval_input_v1.csv` |
| `results/vosk_fleurs_qa_eval_input_v1.*` | `analysis/inputs/vosk_fleurs_qa_eval_input_v1.*` |
| Top-level answer-span result files | `analysis/results/answer_span/original_asr_v1/` |
| `results/gemma2_9b_propagation_v1/` | `analysis/results/initial_evaluation_v1/original_asr/` |
| `results/gemma2_9b_vosk_propagation_v1/` | `analysis/results/initial_evaluation_v1/vosk/` |
| `results/gemma2_9b_vosk_comparison_v1/` | `analysis/results/initial_evaluation_v1/comparisons/asr_conditions/` |
| `results/paired_delta_auc_v1/` | `analysis/results/initial_evaluation_v1/comparisons/paired_delta_auc/` |
| `results/nemo_paper_normalized_metrics_v1/` | `analysis/results/nemo_paper_v1/asr_metrics/` |
| `results/gemma2_9b_original_nemo_paper_propagation_v1/` | `analysis/results/nemo_paper_v1/original_asr/` |
| `results/gemma2_9b_vosk_nemo_paper_propagation_v1/` | `analysis/results/nemo_paper_v1/vosk/` |
| `results/gemma2_9b_nemo_paper_asr_comparison_v1/` | `analysis/results/nemo_paper_v1/comparisons/asr_conditions/` |
| `results/paired_delta_auc_nemo_paper_v1/` | `analysis/results/nemo_paper_v1/comparisons/paired_delta_auc/` |
| `reports/` | `analysis/reports/`, grouped by evaluation version |
| Root and nested freeze manifests | `analysis/manifests/historical_pre_restructure/` |

The current paths and current script hashes are verified by `analysis/manifests/current/RESEARCH_ARTIFACTS_V1.sha256`.
