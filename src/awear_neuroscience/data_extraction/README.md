# Data Extraction

## Time conversion
| Rule | Condition                               | Assumption / Action                                                      |
| ---- | --------------------------------------- | ------------------------------------------------------------------------ |
| 1    | `duration_minutes == 0.5`               | Common quick-label → **trust timestamp as end**                          |
| 2    | `empirical_duration ≠ duration_minutes` | User likely made a mistake or edited manually → **use timestamp as end** |
| 3    | `end_time > timestamp`                  | Illogical future end time → **use timestamp as end**                     |
| 4    | Otherwise                               | Trust user's `start_time` and `end_time` as reliable                     |
