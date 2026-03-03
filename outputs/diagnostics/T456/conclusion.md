# Diagnostic Conclusion: T456

- Overall status: **red**
- Persons: 5, reps: 597

## Risk checklist
- coverage_insufficient: red (value=0.47257383966244726, threshold=0.6) | target_n=56, others_median=118.5
- distribution_shift: green (value=0, threshold=3) | |z|>2.0 metrics=0
- signal_quality_risk: green (value=0.4366747688756508, threshold=2.0) | quality_z={'nan_ratio': 0.0, 'constant_ratio': 0.4366747688756508, 'spike_ratio': -0.23000753163548962}
- boundary_sensitive: green (value=0.39285714285714285, threshold=0.5) | ratio of samples at RPE 6/7

## Top shifted features
- jerk_score: z=1.486, d=1.416, delta=25.255848
- gyro_norm_std: z=0.939, d=0.973, delta=9.195782
- constant_ratio: z=0.717, d=0.716, delta=0.003706
- acc_norm_mean: z=-0.638, d=-0.663, delta=-0.012065
- gyro_norm_mean: z=0.516, d=0.519, delta=13.153099
- cheating_index: z=-0.504, d=-0.528, delta=-1.412096
- duration_s: z=-0.420, d=-0.435, delta=-0.187722
- emg_p2p: z=-0.370, d=-0.352, delta=-1.032730

## Suggested next action
- Prioritize data-level fix first (coverage / quality / distribution alignment).
- Re-run LOO after cleaning before touching model architecture.