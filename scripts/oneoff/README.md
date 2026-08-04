# One-off investigations

Scripts written to answer a specific question at a specific time. They are kept
because the analysis behind them is occasionally worth re-reading or re-running,
not because anything in the pipeline calls them — nothing does.

Expect them to encode assumptions from when they were written: dataset names,
column layouts, and thresholds that were current then. **Read one before running
it.** Several write into a dataset directory in place.

| Script | Question it answered | Written |
|---|---|---|
| `clean_cop_below_vgrf_threshold.py` | Zero COP samples recorded below a vertical-GRF threshold, and rewrite the affected trials. | 2026-05-20 |
| `Clean_COP_experiment.py` | Compare COP cleaning variants before one was adopted into `ProcessData.py`. | 2026-05-20 |
| `find_short_grf_stances.py` | Which trials contain stance phases too short to be real, and how many. | 2026-05-21 |
| `restore_untrimmed.py` | Roll trials back to their pre-trim arrays from the recorded processing information. | 2026-05-08 |
| `count_dataset_stats.py` | Trial, subject and frame counts across a dataset. | 2026-04-20 |

## Paths

These resolve a relative `--dataset` against the **repository root**, not against
this directory. They used `Path(__file__).resolve().parent`, which was the repo
root only because they used to live there; Stage 7 moved them here and switched
them to `paths.REPO_ROOT` so the behaviour is unchanged and no longer depends on
where the file sits. Run them from anywhere.
