"""
Performs statistical analysis on intrinsic bias evaluation results.

This script loads the baseline and debiased model evaluation scores,
conducts a paired t-test to assess statistical significance, and generates
a bar chart to visualize the comparison of bias metrics.
"""
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    # --- 1. Define file paths and load data ---
    baseline_file = "baseline_bias_results_FINAL.csv"
    debiased_file = "debiased_bias_results_FINAL.csv"

    if not os.path.exists(baseline_file) or not os.path.exists(debiased_file):
        print(f"Error: One or both result CSV files not found.")
        print("Ensure scripts 3 and 4 have been run successfully.")
        return

    print("Performing statistical analysis...")
    df_baseline = pd.read_csv(baseline_file)
    df_debiased = pd.read_csv(debiased_file)

    # --- 2. Align dataframes for paired comparison ---
    # A unique key is created from context and stereotype word for accurate merging.
    df_baseline['key'] = df_baseline['context'] + "_" + df_baseline['stereotype_word']
    df_debiased['key'] = df_debiased['context'] + "_" + df_debiased['stereotype_word']
    
    merged_df = pd.merge(
        df_baseline,
        df_debiased,
        on='key',
        suffixes=('_baseline', '_debiased')
    )
    
    if len(merged_df) == 0:
        print("Error: No matching data found for comparison between result files.")
        return
        
    print(f"Found {len(merged_df)} paired examples for statistical testing.")

    # --- 3. Conduct Paired T-Tests on absolute scores ---
    ttest_prob = stats.ttest_rel(merged_df['avg_prob_diff_baseline'].abs(), merged_df['avg_prob_diff_debiased'].abs())
    ttest_log_odds = stats.ttest_rel(merged_df['avg_log_odds_diff_baseline'].abs(), merged_df['avg_log_odds_diff_debiased'].abs())

    # --- 4. Display Statistical Summary ---
    print("\nStatistical Test Results:")
    print("-" * 70)
    
    print("\nMetric 1: Probability Difference")
    print(f"  - Baseline Avg. Absolute Score: {merged_df['avg_prob_diff_baseline'].abs().mean():.4f}")
    print(f"  - Debiased Avg. Absolute Score: {merged_df['avg_prob_diff_debiased'].abs().mean():.4f}")
    print(f"  - T-statistic: {ttest_prob.statistic:.4f}, P-value: {ttest_prob.pvalue:.4f}")
    if ttest_prob.pvalue < 0.05:
        print("  - Result: Statistically significant (p < 0.05).")
    else:
        print("  - Result: Not statistically significant (p >= 0.05).")

    print("\nMetric 2: Log-Odds Difference")
    print(f"  - Baseline Avg. Absolute Score: {merged_df['avg_log_odds_diff_baseline'].abs().mean():.4f}")
    print(f"  - Debiased Avg. Absolute Score: {merged_df['avg_log_odds_diff_debiased'].abs().mean():.4f}")
    print(f"  - T-statistic: {ttest_log_odds.statistic:.4f}, P-value: {ttest_log_odds.pvalue:.4f}")
    if ttest_log_odds.pvalue < 0.05:
        print("  - Result: Statistically significant (p < 0.05).")
    else:
        print("  - Result: Not statistically significant (p >= 0.05).")
    print("-" * 70)

    # --- 5. Generate and Save Plot ---
    print("\nGenerating results plot...")
    
    # Restructure data for Seaborn plotting
    plot_data = []
    for index, row in merged_df.iterrows():
        plot_data.append({'Model': 'Baseline', 'Metric': 'Probability Difference', 'Score': abs(row['avg_prob_diff_baseline'])})
        plot_data.append({'Model': 'Debiased', 'Metric': 'Probability Difference', 'Score': abs(row['avg_prob_diff_debiased'])})
        plot_data.append({'Model': 'Baseline', 'Metric': 'Log-Odds Difference', 'Score': abs(row['avg_log_odds_diff_baseline'])})
        plot_data.append({'Model': 'Debiased', 'Metric': 'Log-Odds Difference', 'Score': abs(row['avg_log_odds_diff_debiased'])})
    
    plot_df = pd.DataFrame(plot_data)

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    bar_plot = sns.barplot(
        x='Metric',
        y='Score',
        hue='Model',
        data=plot_df,
        palette=['#d9534f', '#5cb85c'],
        errorbar=('ci', 95) # Uses modern errorbar parameter for confidence intervals
    )
    
    plt.title('Effectiveness of Counterfactual Debiasing on Intrinsic Bias', fontsize=16)
    plt.ylabel('Average Absolute Bias Score', fontsize=12)
    plt.xlabel('Bias Metric', fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    bar_plot.legend(title='Model Type', fontsize=11)
    
    # Save the plot to a high-resolution file
    plot_output_file = "bias_reduction_results.png"
    plt.savefig(plot_output_file, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to {plot_output_file}")

if __name__ == '__main__':
    main()

