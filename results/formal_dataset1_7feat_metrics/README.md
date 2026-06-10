# Formal 7-Feature Metric Artifacts

These small JSON files preserve the model-comparison metrics behind the README results table.
They were copied from the corresponding local checkpoint run directories so the reported
numbers remain auditable from a fresh clone even though `checkpoints/` is ignored by Git.

| Target | Source checkpoint run | Tracked metric JSON |
| --- | --- | --- |
| THM4 | `checkpoints/formal_dataset1_7feat_cl2d_contact_time_thm4_avg/20260331T214748Z/metrics/model_comparison.json` | `thm4/model_comparison.json` |
| BDCM | `checkpoints/formal_dataset1_7feat_cl2d_contact_time_bdcm_avg/20260331T214748Z/metrics/model_comparison.json` | `bdcm/model_comparison.json` |
| DBCM | `checkpoints/formal_dataset1_7feat_cl2d_contact_time_dbcm_avg/20260331T225500Z/metrics/model_comparison.json` | `dbcm/model_comparison.json` |
