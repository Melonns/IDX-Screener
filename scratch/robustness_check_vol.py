import sys, os
sys.path.insert(0, 'src')
import optuna
import config as app_config

optuna.logging.set_verbosity(optuna.logging.WARNING)
storage = f'sqlite:///{os.path.join(app_config.DATA_DIR, "optuna_vol_accum.db")}'

studies = optuna.get_all_study_names(storage)
study = optuna.load_study(study_name=studies[-1], storage=storage)
best = study.best_trial

near_threshold = best.value - abs(best.value * 0.1)
near_trials = [t for t in study.trials if t.value is not None and t.value >= near_threshold]

print('=== TAHAP 2a ROBUSTNESS CHECK: vol_accum_5d ===')
print(f'Best value      : {best.value:.5f}')
print(f'Best params     : rank={best.params["rank_threshold"]:.1f}%, target={best.params["target_return_pct"]:.2f}%')
print(f'Near threshold  : {near_threshold:.5f} (dalam 10% dari best)')
print(f'Trial di sekitar: {len(near_trials)} dari {len(study.trials)}')
verdict = 'PLATEAU (robust)' if len(near_trials) >= 5 else 'SPIKE (mencurigakan)'
print(f'Verdict         : {verdict}')

vals = sorted([t.value for t in study.trials if t.value is not None and t.value > -999], reverse=True)
print(f'\nTop 10 objective values:')
for i, v in enumerate(vals[:10]):
    print(f'  {i+1:2}. {v:+.5f}')

print(f'\nParameter cluster:')
for t in sorted(near_trials, key=lambda x: x.value, reverse=True)[:10]:
    print(f'  rank={t.params["rank_threshold"]:.1f}%  target={t.params["target_return_pct"]:.2f}%  obj={t.value:+.5f}')

print(f'\n=== PERBANDINGAN LANGSUNG: Tahap 1 vs Tahap 2a ===')
print(f'Tahap 1 (rel_strength_5d): best obj = -0.004998')
print(f'Tahap 2a (vol_accum_5d)  : best obj = {best.value:+.6f}')
diff = -0.004998 - best.value
print(f'Selisih                  : {diff:+.6f} ({"2a lebih baik" if diff < 0 else "1 lebih baik"})')
