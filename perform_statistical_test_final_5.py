"""
STEP 5 (FINAL, CLASSIFICATION-BASED):
Statistical significance testing for BOTH bias metrics on classifier outputs.

This script loads the results from the baseline (Step 3) and debiased (Step 4)
classification-based evaluations and performs a paired t-test on both the
Probability Delta and the Log-Odds Delta metrics (absolute values).
"""
import pandas as pd
from scipy import stats

def main():
    print("--- STEP 5 (FINAL, CLS): Performing Statistical Significance Tests ---")

    baseline_results_file = "baseline_bias_results_cls.csv"
    debiased_results_file = "debiased_bias_results_cls.csv"

    try:
        df_baseline = pd.read_csv(baseline_results_file)
        df_debiased = pd.read_csv(debiased_results_file)
    except FileNotFoundError as e:
        print(f" ERROR: Could not find a results file: {e.filename}")
        return

    # Merge by profession
    comparison_df = pd.merge(
        df_baseline, df_debiased,
        on=['profession'],
        suffixes=('_baseline', '_debiased')
    )
    print(f"Found {len(comparison_df)} professions to compare.")

    # --- Test 1: Probability Delta Metric ---
    baseline_prob_scores = comparison_df['avg_prob_delta_baseline'].abs()
    debiased_prob_scores = comparison_df['avg_prob_delta_debiased'].abs()
    t_stat_prob, p_val_prob = stats.ttest_rel(baseline_prob_scores, debiased_prob_scores)

    # --- Test 2: Log-Odds Delta Metric ---
    baseline_log_scores = comparison_df['avg_log_odds_delta_baseline'].abs()
    debiased_log_scores = comparison_df['avg_log_odds_delta_debiased'].abs()
    t_stat_log, p_val_log = stats.ttest_rel(baseline_log_scores, debiased_log_scores)

    # --- Print Results ---
    print("\n" + "="*70)
    print("        FINAL STATISTICAL SIGNIFICANCE TEST RESULTS (CLS)")
    print("="*70)

    print("\n--- Metric 1: Probability Delta ---")
    print(f"Baseline Avg. Absolute Score: {baseline_prob_scores.mean():.4f}")
    print(f"Debiased Avg. Absolute Score: {debiased_prob_scores.mean():.4f}")
    print(f"T-statistic: {t_stat_prob:.4f}, P-value: {p_val_prob:.4g}")
    if p_val_prob < 0.05:
        print(" Result is STATISTICALLY SIGNIFICANT.")
    else:
        print("  Result is NOT statistically significant.")
    
    print("\n--- Metric 2: Log-Odds Delta ---")
    print(f"Baseline Avg. Absolute Score: {baseline_log_scores.mean():.4f}")
    print(f"Debiased Avg. Absolute Score: {debiased_log_scores.mean():.4f}")
    print(f"T-statistic: {t_stat_log:.4f}, P-value: {p_val_log:.4g}")
    if p_val_log < 0.05:
        print(" Result is STATISTICALLY SIGNIFICANT.")
    else:
        print(" Result is NOT statistically significant.")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    main()
