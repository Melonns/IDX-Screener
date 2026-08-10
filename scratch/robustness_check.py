import sys, os
sys.path.insert(0, 'src')
import optuna
import config as app_config

optuna.logging.set_verbosity(optuna.logging.WARNING)
storage = f'sqlite:///{os.path.join(app_config.DATA_DIR, "optuna_contextual.db")}'

studies = optuna.get_all_study_names(storage)
print(f'Studies di storage: {studies}')

study = optuna.load_study(study_name=studies[-1], storage=storage)
best = study.best_trial
print(f'Best value: {best.value:.6f}')
print(f'Best params: {best.params}')

# Robustness check: semua trial yg nilainya tidak lebih dari 10% lebih buruk dari best
# (nilai negatif: best=-0.005, 10% worse = -0.0055)
near_threshold = best.value - abs(best.value * 0.1)
near_trials = [t for t in study.trials if t.value is not None and t.value >= near_threshold]

print('')
print('=== ROBUSTNESS CHECK ===')
print(f'Best objective  : {best.value:.5f}')
print(f'Near threshold  : {near_threshold:.5f} (dalam 10% dari best)')
print(f'Trial di sekitar best: {len(near_trials)} dari {len(study.trials)}')

if len(near_trials) >= 5:
    verdict = 'PLATEAU (robust - banyak kombinasi parameter yang hasilnya mirip)'
else:
    verdict = 'SPIKE (mencurigakan - cuma 1-2 trial yang bagus, sisanya jelek)'
print(f'Verdict: {verdict}')

vals = sorted([t.value for t in study.trials if t.value is not None and t.value > -999], reverse=True)
print('')
print('Top 10 objective values:')
for i, v in enumerate(vals[:10]):
    print(f'  {i+1:2}. {v:+.5f}')

print('')
print('Parameter cluster di sekitar best:')
for t in sorted(near_trials, key=lambda x: x.value, reverse=True)[:10]:
    print(f'  rank={t.params["rank_threshold"]:.1f}%  target={t.params["target_return_pct"]:.2f}%  obj={t.value:+.5f}')
