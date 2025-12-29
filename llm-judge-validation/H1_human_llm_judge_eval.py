"""
H1_human_llm_judge_eval.py - Comparative Analysis of LLM Judge vs. Human Evaluation.

This script performs a statistical comparison between the evaluations provided by
various LLM judges and a human evaluator for the CS-KG and ORKG outputs. The goal
is to assess the reliability and alignment of different LLMs when acting as evaluators
against human judgment, which serves to validate the LLM-as-judge methodology used
in the H1 evaluation.

Wilcoxon Signed-Rank Test:
For each LLM judge and for each SKG (CS-KG, ORKG) and each quality criterion:
    -   Prepares paired score data (LLM score vs. Human score) for the 8 papers.
    -   Performs a Wilcoxon signed-rank test to determine if there is a
            statistically significant difference between the LLM judge's scores
            and the human evaluator's scores.
    -   Calculates the median difference between paired scores.

Files needed
    -   Multiple `*judge_cs-kg_orkg.json` files (output from `H1_LLM_judge.py`, one per LLM judge).
    -   `human_eval.json` (containing human expert scores).
"""

import os
import json
import scipy.stats as stats
import numpy as np
import statistics
import matplotlib.pyplot as plt
import math
import re
from typing import Dict, Any, List, Optional, Tuple, Union # Added Union
from helperfunctions import add_timestamp_and_llm
from H1_helperfunctions import join_folder_results, join_folder_files, get_criteria_list, get_2_skgs_list


ALPHA = 0.05  # significance level
NUM_PAPERS = 8

# for the svgs
plt.rcParams['font.family'] = 'Palatino Linotype'
FONT_SIZE = 14
TITLE_FONT_SIZE = 16
LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 14

# green colors
STANDARD_GREEN = "#005f50"  # Dark Green
MEDIUM_GREEN = "#80afa8"  # Medium Green
COLOR_SIG_001 = "#005f50"  # Dark Green
COLOR_SIG_01 = "#80afa8"  # Medium Green
COLOR_SIG_05 = "#ccdfdc"  # Light Green

COLOR_NEUTRAL = "white"
TEXT_COLOR_DARK_BG = "white"
TEXT_COLOR_LIGHT_BG = "black"

LLM_ID_TO_NAME_MAP = {
    "1": "Qwen3-30B-A3B",
    "2": "Mistral-Small-3.1",
    "3": "GLM-4-32B-0414",
    "4": "Gemma-3-27B",
    "5": "GPT-4.1",
    "6": "Gemini-2.0-flash",
    "7": "DeepSeek-V3"
}

SKG_DISPLAY_NAME_MAP_H1_HUMAN_LLM = {
    "cs-kg": "CS-KG",
    "orkg": "ORKG"
}


def prepare_paired_data(llm_scores_dict: Optional[Dict[str, Optional[Union[int, float]]]],
                        human_scores_dict: Optional[Dict[str, Optional[Union[int, float]]]],
                        score_keys: List[str]) -> Tuple[List[float], List[float]]:
    """
    Extracts and prepares paired numerical scores for statistical comparison
    (Wilcoxon test) from LLM and human evaluation dictionaries.

    Args:
        llm_scores_dict (Optional[Dict[str, Optional[Union[int, float]]]]):
            A dictionary of scores from an LLM judge, where keys match `score_keys`.
            Can be None if no LLM data is available for a criterion.
        human_scores_dict (Optional[Dict[str, Optional[Union[int, float]]]]):
            A dictionary of scores from a human evaluator, where keys match `score_keys`.
            Can be None if no human data is available for a criterion.
        score_keys (List[str]): A list of keys (e.g., "1_cs-kg", "2_orkg")
                                identifying the specific scores to pair.

    Returns:
        Tuple[List[float], List[float]]:
            A tuple containing two lists:
            - `llm_paired_scores`: List of valid LLM scores.
            - `human_paired_scores`: List of corresponding valid human scores.
            These lists will always have the same length.
    """
    llm_paired = []
    human_paired = []
    if not llm_scores_dict or not human_scores_dict:
        return llm_paired, human_paired

    for key in score_keys:
        llm_score = llm_scores_dict.get(key)
        human_score = human_scores_dict.get(key)

        if (llm_score is not None and isinstance(llm_score, (int, float)) and not math.isnan(llm_score) and
                human_score is not None and isinstance(human_score, (int, float)) and not math.isnan(human_score)):
            llm_paired.append(float(llm_score))
            human_paired.append(float(human_score))

    return llm_paired, human_paired


def run_wilcoxon_test(llm_scores: List[float], human_scores: List[float], min_pairs: int = 6) -> Dict[str, Any]:
    """
    Performs the Wilcoxon signed-rank test on paired LLM and human scores
    to assess if there's a significant difference in their medians.

    Args:
        llm_scores (List[float]): A list of numerical scores from an LLM judge.
        human_scores (List[float]): A list of numerical scores from a human evaluator,
                                    paired element-wise with `llm_scores`.
        min_pairs (int, optional): The minimum number of pairs with non-zero
                                   differences required to perform the test.
                                   Defaults to 6.

    Returns:
        Dict[str, Any]: A dictionary containing the test results:
            - "statistic" (Optional[float]): The Wilcoxon test statistic.
            - "p_value" (Optional[float]): The two-sided p-value.
            - "n_pairs" (int): The initial number of pairs provided.
            - "median_difference" (Optional[float]): Median of (LLM score - Human score).
            - "direction" (str): "LLM > Human", "LLM < Human", "Equal Medians",
                                 or "Unknown (median error)".
            - "status" (str): "ok" if test run, or a message indicating why it was
                              skipped or if an error occurred.
            - "error" (Optional[str]): Error message if an exception occurred.
    """
    results = {"statistic": None, "p_value": None, "n_pairs": 0,
               "median_difference": None, "direction": "neutral",
               "status": "ok", "error": None}

    if len(llm_scores) != len(human_scores):
        results["status"] = "error"
        results["error"] = "Input lists have different lengths."
        return results

    n_initial_pairs = len(llm_scores)
    results["n_pairs"] = n_initial_pairs

    differences = [l - h for l, h in zip(llm_scores, human_scores)]
    non_zero_diffs = [d for d in differences if d != 0]

    if len(non_zero_diffs) < min_pairs:
        results["status"] = f"skipped_insufficient_pairs (non-zero diff < {min_pairs})"
        if differences:
            try:
                results["median_difference"] = statistics.median(differences)
            except statistics.StatisticsError:
                results["median_difference"] = None
        return results

    try:
        statistic, p_value = stats.wilcoxon(llm_scores, human_scores, zero_method='pratt', correction=True, mode='auto')
        results["statistic"] = statistic
        results["p_value"] = p_value

        try:
            results["median_difference"] = statistics.median(differences)
        except statistics.StatisticsError:
            results["median_difference"] = None

        if results["median_difference"] is not None:
            if results["median_difference"] > 0:
                results["direction"] = "LLM > Human"
            elif results["median_difference"] < 0:
                results["direction"] = "LLM < Human"
            else:
                results["direction"] = "Equal Medians"
        else:
            results["direction"] = "Unknown (median error)"

    except ValueError as e:
        results["status"] = "error"
        results["error"] = str(e)
    except Exception as e:
        results["status"] = "error"
        results["error"] = f"Unexpected error during Wilcoxon: {e}"

    return results


def compare_llm_to_human(llm_id: str,
                         llm_eval_json: Dict[str, Any],
                         human_eval_json: Dict[str, Any],
                         skgs: List[str],
                         criteria: List[str],
                         num_papers: int) -> Dict[str, Any]:
    """
    Compares a single LLM's evaluations to human evaluations across multiple SKGs
    and criteria using the Wilcoxon signed-rank test.

    Args:
        llm_id (str): The identifier of the LLM judge whose evaluations are being compared.
        llm_eval_json (Dict[str, Any]): The full evaluation JSON data for the LLM judge.
        human_eval_json (Dict[str, Any]): The full evaluation JSON data from the human evaluator.
        skgs (List[str]): A list of SKG names to include in the comparison (e.g., ["cs-kg", "orkg"]).
        criteria (List[str]): A list of quality criterion names to compare.
        num_papers (int): The number of individual papers evaluated (e.g., 8), used
                          to generate score keys.

    Returns:
        Dict[str, Any]: A nested dictionary storing the comparison results.
                        Path: `results[skg_name][criterion_name] = wilcoxon_test_results_dict`.
                        Includes status messages if data for an SKG/criterion is missing.
    """
    llm_comparison_results = {}
    llm_display_name_for_print = LLM_ID_TO_NAME_MAP.get(llm_id, llm_id)
    print(f"Comparing LLM '{llm_display_name_for_print}' to Human")

    for skg in skgs:
        llm_comparison_results[skg] = {}
        skg_display_name_for_print_loop = SKG_DISPLAY_NAME_MAP_H1_HUMAN_LLM.get(skg, skg)
        print(f"  SKG: {skg_display_name_for_print_loop}")

        llm_skg_data = llm_eval_json.get(skg)
        human_skg_data = human_eval_json.get(skg)

        if not llm_skg_data:
            print(f"Warning: No data found for LLM '{llm_display_name_for_print}' in SKG '{skg_display_name_for_print_loop}'. Skipping SKG.")
            llm_comparison_results[skg] = {"status": "skipped_missing_llm_data"}
            continue
        if not human_skg_data:
            print(f"Warning: No data found for Human in SKG '{skg_display_name_for_print_loop}'. Skipping SKG.")
            llm_comparison_results[skg] = {"status": "skipped_missing_human_data"}
            continue

        score_keys = [f"{i}_{skg}" for i in range(1, num_papers + 1)]

        for crit in criteria:

            llm_crit_scores = llm_skg_data.get(crit)
            human_crit_scores = human_skg_data.get(crit)

            if not llm_crit_scores:
                llm_comparison_results[skg][crit] = {"status": "skipped_missing_llm_scores"}
                continue
            if not human_crit_scores:
                llm_comparison_results[skg][crit] = {"status": "skipped_missing_human_scores"}
                continue

            llm_paired, human_paired = prepare_paired_data(llm_crit_scores, human_crit_scores, score_keys)

            wilcoxon_results = run_wilcoxon_test(llm_paired, human_paired)
            wilcoxon_results["n_valid_pairs_used"] = len(llm_paired)

            llm_comparison_results[skg][crit] = wilcoxon_results

    return llm_comparison_results


def save_wilcoxon_results(all_llm_results: Dict[str, Dict[str, Any]], output_path: str) -> None:
    """
    Saves the aggregated Wilcoxon signed-rank test results (comparing all LLM
    judges to human evaluation) to a single JSON file.

    Args:
        all_llm_results (Dict[str, Dict[str, Any]]): A dictionary where keys are LLM
            identifiers and values are the detailed comparison results from
            `compare_llm_to_human` for that LLM, potentially including a
            "summary_metrics" key.
        output_path (str): The directory path where the JSON file will be saved.
    """
    filename = os.path.join(output_path, add_timestamp_and_llm(0) + "human_llm_comparison_results.json") # Use prefix="" here
    print(f"\nSaving LLM vs Human comparison results to: {filename}")
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_llm_results, f, indent=4, ensure_ascii=False)
        print("LLM vs Human results saved successfully.")
    except Exception as e:
        print(f"Error saving LLM vs Human results JSON: {e}")


def create_median_difference_bar_chart(llm_id: str,
                                       comparison_results: Dict[str, Any],
                                       skgs_to_compare: List[str],
                                       criteria: List[str],
                                       filename: str) -> None:
    """
    Generates and saves a horizontal bar chart visualizing the median difference
    between a specific LLM judge's scores and human scores for each SKG and criterion.

    Args:
        llm_id (str): Identifier of the LLM judge.
        comparison_results (Dict[str, Any]): The Wilcoxon test results for this LLM,
            structured as `results[skg_name][criterion_name]`.
        skgs_to_compare (List[str]): List of SKG names (internal keys like 'cs-kg') to include in the chart.
        criteria (List[str]): List of criterion names (internal keys like 'semantic_accuracy') to include.
        filename (str): The full path (including .svg extension) where the chart
                        will be saved.
    """
    llm_display_name = LLM_ID_TO_NAME_MAP.get(llm_id, llm_id)
    print(f"Generating median difference bar chart for LLM: {llm_display_name}")
    labels = []
    median_diffs = []
    colors = []
    significance = []

    for skg_key in skgs_to_compare:
        skg_display_name_chart = SKG_DISPLAY_NAME_MAP_H1_HUMAN_LLM.get(skg_key, skg_key) # Use display name for y-axis
        if skg_key not in comparison_results or comparison_results[skg_key].get("status", "").startswith("skipped"):
            continue

        for crit_key in criteria:
            pretty_crit_name_chart = crit_key.replace('_', ' ').title()
            res = comparison_results.get(skg_key, {}).get(crit_key)
            if not res or res.get("status", "").startswith("skipped"):
                continue

            label = f"{skg_display_name_chart}: {pretty_crit_name_chart}" # Use display SKG and pretty criterion
            labels.append(label)

            med_diff = res.get("median_difference")
            p_value = res.get("p_value")

            median_diffs.append(med_diff if med_diff is not None else 0)

            color = '#CCCCCC'
            sig_marker = ''
            if res["status"] == "ok" and p_value is not None:
                if p_value < 0.001:
                    sig_marker = '***'
                    color = COLOR_SIG_001
                elif p_value < 0.01:
                    sig_marker = '**'
                    color = COLOR_SIG_01
                elif p_value < ALPHA:
                    sig_marker = '*'
                    color = COLOR_SIG_05
            colors.append(color)
            significance.append(sig_marker)

    if not labels:
        print(f"  No valid data found to plot for LLM {llm_display_name}.")
        return

    y_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.4)))
    fig.subplots_adjust(left=0.4, right=0.9, top=0.95, bottom=0.05)

    bars = ax.barh(y_pos, median_diffs, color=colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=TICK_FONT_SIZE - 2)
    ax.set_xlabel("Median Difference (LLM Score - Human Score)", fontsize=LABEL_FONT_SIZE)
    ax.set_title(f"{llm_display_name} vs. Human Evaluation: Median Difference", fontsize=TITLE_FONT_SIZE) # Use real LLM name

    for i, bar in enumerate(bars):
        xval = bar.get_width()
        offset = 0.05 * (
            max(median_diffs) if any(d > 0 for d in median_diffs) else 1)
        ha = 'left' if xval >= 0 else 'right'
        plt.text(xval + (offset if xval >= 0 else -offset), bar.get_y() + bar.get_height() / 2.0,
                 significance[i], va='center', ha=ha, color='black', fontsize=LABEL_FONT_SIZE)

    min_val = min(median_diffs + [-0.5])
    max_val = max(median_diffs + [0.5])
    ax.set_xlim(min_val - abs(min_val) * 0.1, max_val + abs(max_val) * 0.1)

    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(axis='x', color='grey', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    try:
        plt.savefig(filename, format='svg', dpi=300, bbox_inches="tight")
        print(f"  Saved median difference chart: {filename}")
    except Exception as e:
        print(f"  Error saving median difference chart {filename}: {e}")
    plt.close(fig)


def export_human_comparison_charts(all_llm_comparison_results: Dict[str, Dict[str, Any]], output_path: str) -> None:
    """
    Generates and saves a set of summary charts comparing LLM judges to human evaluation.

    Args:
        all_llm_comparison_results (Dict[str, Dict[str, Any]]): Aggregated comparison
            results for all LLM judges. Each LLM's data should include a
            "summary_metrics" key populated by `final_comparison`.
        output_path (str): The directory path where the chart SVG files will be saved.
    """

    skgs_internal_keys = get_2_skgs_list()
    criteria_internal_keys = get_criteria_list()

    llm_ids_with_data = [llm_id for llm_id, data in all_llm_comparison_results.items() if data]
    if not llm_ids_with_data:
        print("No LLM results to visualize")
        return

    for llm_id_str_chart, comparison_results_for_llm_chart in all_llm_comparison_results.items():
        if any(skg_key_chart in comparison_results_for_llm_chart for skg_key_chart in skgs_internal_keys):
            filename_chart_path = os.path.join(output_path, add_timestamp_and_llm(0) + f"bar_chart_median_diff_llm_{llm_id_str_chart}.svg")
            create_median_difference_bar_chart(llm_id_str_chart, comparison_results_for_llm_chart, skgs_internal_keys, criteria_internal_keys, filename_chart_path)
        else:
            llm_display_name_skip = LLM_ID_TO_NAME_MAP.get(llm_id_str_chart, llm_id_str_chart)
            print(f"Skipping median difference chart for LLM {llm_display_name_skip} as no SKG-level comparison data found.")

    llm_display_names_for_summary = []
    similarity_percentages_summary = []
    avg_abs_median_diffs_summary = []
    inclusive_avg_abs_median_diffs_summary = []

    for llm_id_str_summary, llm_data_item_summary in all_llm_comparison_results.items():
        summary_metrics_item_summary = llm_data_item_summary.get("summary_metrics")
        if summary_metrics_item_summary:
            llm_display_names_for_summary.append(LLM_ID_TO_NAME_MAP.get(llm_id_str_summary, llm_id_str_summary))
            similarity_percentages_summary.append(summary_metrics_item_summary.get("similarity_percentage", 0.0))
            avg_abs_median_diffs_summary.append(summary_metrics_item_summary.get("avg_abs_median_diff", 0.0))

            all_median_diffs_inclusive_local = []
            for skg_name in skgs_internal_keys:
                skg_results = llm_data_item_summary.get(skg_name, {})
                if isinstance(skg_results, dict):
                    for crit_name in criteria_internal_keys:
                        crit_res = skg_results.get(crit_name, {})
                        if isinstance(crit_res, dict) and "median_difference" in crit_res and crit_res["median_difference"] is not None:
                            all_median_diffs_inclusive_local.append(crit_res["median_difference"])

            if all_median_diffs_inclusive_local:
                abs_median_diffs_local = [abs(d) for d in all_median_diffs_inclusive_local]
                inclusive_avg = statistics.mean(abs_median_diffs_local)
                inclusive_avg_abs_median_diffs_summary.append(inclusive_avg)
            else:
                inclusive_avg_abs_median_diffs_summary.append(0.0) # Fallback

        else:
            llm_display_name_warn = LLM_ID_TO_NAME_MAP.get(llm_id_str_summary, llm_id_str_summary)
            print(f"  Warning: No summary_metrics found for LLM {llm_display_name_warn}. Skipping for summary charts.")

    if llm_display_names_for_summary:
        sim_chart_filename_path = os.path.join(output_path, add_timestamp_and_llm(0) + "non_significant_differences.svg")
        create_similarity_summary_chart(llm_display_names_for_summary, similarity_percentages_summary, sim_chart_filename_path)

        avg_abs_diff_chart_filename_path = os.path.join(output_path, add_timestamp_and_llm(0) + "average_absolute_median_differences.svg")
        create_avg_abs_diff_summary_chart(llm_display_names_for_summary, avg_abs_median_diffs_summary, avg_abs_diff_chart_filename_path)

        # --- Call the new chart function ---
        inclusive_avg_abs_diff_chart_filename_path = os.path.join(output_path, add_timestamp_and_llm(0) + "overall_average_absolute_median_differences.svg")
        create_inclusive_avg_abs_diff_summary_chart(llm_display_names_for_summary, inclusive_avg_abs_median_diffs_summary, inclusive_avg_abs_diff_chart_filename_path)

    else:
        print("No data available to generate summary comparison charts.")


def final_comparison(all_llm_human_comparison_results: Dict[str, Dict[str, Any]], alpha: float) -> Dict[str, Dict[str, Any]]:
    """
    Calculates and adds summary comparison metrics for each LLM judge based on
    their Wilcoxon test results against human evaluations.

    Args:
        all_llm_human_comparison_results (Dict[str, Dict[str, Any]]): A dictionary
            containing the detailed Wilcoxon test results for each LLM judge.
            This dictionary will be updated with summary metrics.
        alpha (float): The significance level used to determine non-significant differences.

    Returns:
        Dict[str, Dict[str, Any]]: The input dictionary, updated with "summary_metrics"
                                   for each LLM.
    """
    print("\n--- Calculating Final Comparison Metrics for LLMs ---")
    for llm_id, llm_data in all_llm_human_comparison_results.items():
        non_significant_count = 0
        total_valid_tests = 16
        all_median_diffs_for_avg = []

        skg_data_keys = [k for k in llm_data.keys() if k != "summary_metrics"]

        for skg_name in skg_data_keys:
            skg_results = llm_data.get(skg_name, {})
            if isinstance(skg_results, dict) and not skg_results.get("status", "").startswith("skipped"):
                criterion_data_keys = [k for k in skg_results.keys() if k != "status"]
                for crit_name in criterion_data_keys:
                    crit_res = skg_results.get(crit_name, {})

                    if isinstance(crit_res, dict):

                        if crit_res.get("status", "").startswith("skipped"):
                            non_significant_count += 1
                        elif crit_res.get("status") == "ok" and crit_res.get("p_value") is not None:
                            if crit_res.get("p_value") >= alpha:
                                non_significant_count += 1
                            median_difference = crit_res.get("median_difference")
                            if median_difference is not None:
                                all_median_diffs_for_avg.append(median_difference)

        similarity_percentage = (non_significant_count / total_valid_tests) * 100 if total_valid_tests > 0 else 0.0
        abs_median_diffs = [abs(d) for d in all_median_diffs_for_avg]
        avg_abs_median_diff = statistics.mean(abs_median_diffs) if abs_median_diffs else 0.0

        if "summary_metrics" not in all_llm_human_comparison_results[llm_id]:
            all_llm_human_comparison_results[llm_id]["summary_metrics"] = {}

        all_llm_human_comparison_results[llm_id]["summary_metrics"]["non_significant_count"] = non_significant_count
        all_llm_human_comparison_results[llm_id]["summary_metrics"]["total_valid_tests"] = total_valid_tests
        all_llm_human_comparison_results[llm_id]["summary_metrics"]["similarity_percentage"] = similarity_percentage
        all_llm_human_comparison_results[llm_id]["summary_metrics"]["avg_abs_median_diff"] = avg_abs_median_diff
        llm_display_name_final = LLM_ID_TO_NAME_MAP.get(llm_id, llm_id)
        print(f"  LLM {llm_display_name_final}: Similarity={similarity_percentage:.2f}% ({non_significant_count}/{total_valid_tests}), AvgAbsMedDiff={avg_abs_median_diff:.2f}")

    return all_llm_human_comparison_results


def create_similarity_summary_chart(llm_display_names: List[str], similarity_percentages: List[float], filename: str) -> None:
    """
    Generates and saves a horizontal bar chart summarizing the "Percentage of
    Non-Significant Differences" (similarity to human) for all LLM judges.

    Args:
        llm_display_names (List[str]): A list of LLM judge display names.
        similarity_percentages (List[float]): A list of corresponding similarity
                                             percentages (0-100).
        filename (str): The full path (including .svg extension) where the chart
                        will be saved.
    """
    print(f"Generating LLM similarity summary chart: {filename}")
    if not llm_display_names:
        print("  No data for similarity summary chart.")
        return

    sorted_data = sorted(zip(llm_display_names, similarity_percentages), key=lambda item: item[1], reverse=True)
    sorted_llm_display_names = [item[0] for item in sorted_data]
    sorted_percentages = [item[1] for item in sorted_data]

    y_pos = np.arange(len(sorted_llm_display_names))
    fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_llm_display_names) * 0.5)))
    fig.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.1)

    bars = ax.barh(y_pos, sorted_percentages, color=STANDARD_GREEN, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_llm_display_names, fontsize=TICK_FONT_SIZE)
    ax.set_xlabel("Percentage of Non-Significant Differences (%)", fontsize=LABEL_FONT_SIZE)
    ax.set_title("LLM Judge Similarity to Human Evaluation", fontsize=TITLE_FONT_SIZE)
    ax.set_xlim(0, 100)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height() / 2, f'{width:.1f}%',
                va='center', ha='left', fontsize=TICK_FONT_SIZE - 2)

    ax.grid(axis='x', color='grey', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    try:
        plt.savefig(filename, format='svg', dpi=300, bbox_inches="tight")
        print(f"  Saved similarity summary chart: {filename}")
    except Exception as e:
        print(f"  Error saving similarity summary chart {filename}: {e}")
    plt.close(fig)


def create_avg_abs_diff_summary_chart(llm_display_names: List[str], avg_abs_diffs: List[float], filename: str) -> None:
    """
    Generates and saves a horizontal bar chart summarizing the "Average Absolute
    Median Difference" from human scores for all LLM judges.

    Args:
        llm_display_names (List[str]): A list of LLM judge display names.
        avg_abs_diffs (List[float]): A list of corresponding average absolute
                                     median differences.
        filename (str): The full path (including .svg extension) where the chart
                        will be saved.
    """
    print(f"Generating LLM average absolute difference summary chart: {filename}")
    if not llm_display_names:
        print("  No data for average absolute difference summary chart.")
        return

    sorted_data = sorted(zip(llm_display_names, avg_abs_diffs), key=lambda item: item[1], reverse=False)
    sorted_llm_display_names = [item[0] for item in sorted_data]
    sorted_diffs = [item[1] for item in sorted_data]

    y_pos = np.arange(len(sorted_llm_display_names))
    fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_llm_display_names) * 0.5)))
    fig.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.1)

    bars = ax.barh(y_pos, sorted_diffs, color=STANDARD_GREEN, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_llm_display_names, fontsize=TICK_FONT_SIZE)
    ax.set_xlabel("Average Absolute Median Difference from Human", fontsize=LABEL_FONT_SIZE)
    ax.set_title("Average Score Deviation LLM Judge vs. Human", fontsize=TITLE_FONT_SIZE)

    max_diff_val = max(sorted_diffs) if sorted_diffs else 1.0
    ax.set_xlim(0, max_diff_val * 1.1)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + (max_diff_val * 0.01), bar.get_y() + bar.get_height() / 2, f'{width:.2f}',
                va='center', ha='left', fontsize=TICK_FONT_SIZE - 2)

    ax.grid(axis='x', color='grey', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    try:
        plt.savefig(filename, format='svg', dpi=300, bbox_inches="tight")
        print(f"  Saved average absolute difference chart: {filename}")
    except Exception as e:
        print(f"  Error saving average absolute difference chart {filename}: {e}")
    plt.close(fig)


def get_llm_id_from_filename(filename: str) -> Optional[str]:
    """
    Extracts an LLM numeric identifier from a filename string using regex.

    Args:
        filename (str): The filename string to parse.

    Returns:
        Optional[str]: The extracted LLM ID as a string if the pattern is found.
                       Returns None otherwise.
    """
    match = re.search(r"_llm_(\d+)_", filename)
    if match:
        return match.group(1)
    return None


def calculate_and_add_best_judge_score(all_llm_human_comparison_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculates a balanced 'Best LLM Judge Score' based on the two metrics Average Median Difference and Percentage of
    Non-Significant Differences.

    Args:
        all_llm_human_comparison_results (Dict[str, Any]): A dictionary where
            keys are LLM identifiers (e.g., "1", "7") and values are their
            human comparison results. Each LLM's data is expected to contain a
            "summary_metrics" dictionary with "similarity_percentage" and
            "avg_abs_median_diff" keys. This dictionary is modified in-place.

    Returns:
        Dict[str, Any]: The same input dictionary, now updated. For each LLM,
            the "summary_metrics" dictionary will contain an additional key,
            "best_judge_score", holding the final, normalized score.
    """
    raw_scores_data = []
    similarity_percentages = []

    for llm_id, llm_data in all_llm_human_comparison_results.items():
        summary = llm_data.get("summary_metrics")
        if not summary:
            continue

        if "similarity_percentage" in summary:
            similarity_percentages.append(summary["similarity_percentage"])

        sim_decimal = summary.get("similarity_percentage", 0.0) / 100.0
        avg_diff = summary.get("avg_abs_median_diff", 2.0)
        penalty = ((avg_diff - 1.0) ** 2) * 10
        raw_score = sim_decimal - penalty
        raw_scores_data.append({"id": llm_id, "raw_score": raw_score})

    if not raw_scores_data or not similarity_percentages:
        print("  Could not calculate or normalize scores due to missing data.")
        return all_llm_human_comparison_results

    # boundaries for normalization are the old min and max from similarity percentage scores
    old_min = min(item["raw_score"] for item in raw_scores_data)
    old_max = max(item["raw_score"] for item in raw_scores_data)
    new_min = min(similarity_percentages) / 100.0
    new_max = max(similarity_percentages) / 100.0

    print(f"Raw score range: [{old_min:.4f}, {old_max:.4f}]")
    print(f"Normalizing to dynamic similarity range: [{new_min:.4f}, {new_max:.4f}]")

    for item in raw_scores_data:
        llm_id = item["id"]
        raw_score = item["raw_score"]

        if old_max == old_min: # avoid division by 0
            normalized_score = (new_min + new_max) / 2
        else:
            # normalization formula
            normalized_score = new_min + ((raw_score - old_min) * (new_max - new_min)) / (old_max - old_min)

        # Store the final, normalized score in the main results dictionary
        all_llm_human_comparison_results[llm_id]["summary_metrics"]["best_judge_score"] = normalized_score

        llm_display_name = LLM_ID_TO_NAME_MAP.get(llm_id, llm_id)
        print(f"  {llm_display_name}: Raw={raw_score:.4f} -> Normalized Score={normalized_score:.4f}")

    return all_llm_human_comparison_results


def create_best_judge_summary_chart(llm_display_names: List[str], scores: List[float], filename: str) -> None:
    """
    Generates and saves a bar chart for the Balanced "Best LLM Judge Score".

    Args:
        llm_display_names (List[str]): A list of LLM display names (e.g.,
            'Qwen3-30B-A3B') to be used as labels on the y-axis. The order
            must correspond to the `scores` list.
        scores (List[float]): A list of corresponding numerical scores for
            each LLM. These values determine the length of the bars
            in the chart.
        filename (str): The full path (including the .svg extension) where
            the generated chart will be saved.

    """
    print(f"Generating Normalized Best LLM Judge summary chart: {filename}")
    if not llm_display_names:
        print("  No data for Best LLM Judge summary chart.")
        return

    # sort data by score in descending order (higher is better)
    sorted_data = sorted(zip(llm_display_names, scores), key=lambda item: item[1], reverse=True)
    sorted_llm_display_names = [item[0] for item in sorted_data]
    sorted_scores = [item[1] for item in sorted_data]

    y_pos = np.arange(len(sorted_llm_display_names))
    fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_llm_display_names) * 0.5)))
    fig.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.1)

    bars = ax.barh(y_pos, sorted_scores, color=STANDARD_GREEN, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_llm_display_names, fontsize=TICK_FONT_SIZE)
    ax.set_xlabel("Normalized Judge Score (Similarity-Equivalent)", fontsize=LABEL_FONT_SIZE)
    ax.set_title("Overall Best LLM Judge Ranking", fontsize=TITLE_FONT_SIZE)

    # dynamically set axis limits based on the data range
    min_val = min(sorted_scores)
    max_val = max(sorted_scores)
    padding = (max_val - min_val) * 0.05 # 5% padding
    ax.set_xlim(min_val - padding, max_val + padding * 2)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + (max_val - min_val) * 0.01, bar.get_y() + bar.get_height() / 2, f'{width:.4f}',
                va='center', ha='left', fontsize=TICK_FONT_SIZE - 2)

    ax.grid(axis='x', color='grey', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    try:
        plt.savefig(filename, format='svg', dpi=300, bbox_inches="tight")
        print(f"Saved Best LLM Judge summary chart: {filename}")
    except Exception as e:
        print(f"Error saving Best LLM Judge summary chart {filename}: {e}")
    plt.close(fig)


def calculate_inclusive_avg_median_diff(all_llm_human_comparison_results: Dict[str, Any]) -> None:
    """
    Calculates and prints an inclusive "Average Absolute Median Difference" for each LLM judge.
    Inclusive means that it includes the median difference from ALL tests (typically 16), even those that were skipped
    by the Wilcoxon test due to insufficient non-zero differences. It gives
    a raw measure of average deviation across all criteria.

    Args:
        all_llm_human_comparison_results (Dict[str, Any]): The full comparison
            results dictionary, which contains the Wilcoxon test results for
            each LLM, SKG, and criterion.

    """

    for llm_id, llm_data in all_llm_human_comparison_results.items():
        all_median_diffs_inclusive = []

        # Loop through skgs and criteria to gather all available median differences
        skg_data_keys = [k for k in llm_data.keys() if k != "summary_metrics"]
        for skg_name in skg_data_keys:
            skg_results = llm_data.get(skg_name, {})
            if isinstance(skg_results, dict) and not skg_results.get("status", "").startswith("skipped"):
                criterion_data_keys = [k for k in skg_results.keys() if k != "status"]
                for crit_name in criterion_data_keys:
                    crit_res = skg_results.get(crit_name, {})

                    if isinstance(crit_res, dict) and "median_difference" in crit_res and crit_res["median_difference"] is not None:
                        all_median_diffs_inclusive.append(crit_res["median_difference"])

        if not all_median_diffs_inclusive:
            continue

        # calculate the average of the absolute values
        abs_median_diffs = [abs(d) for d in all_median_diffs_inclusive]
        avg_abs_median_diff_inclusive = statistics.mean(abs_median_diffs) if abs_median_diffs else 0.0

        llm_display_name = LLM_ID_TO_NAME_MAP.get(llm_id, llm_id)
        print(f"  LLM {llm_display_name}: Inclusive AvgAbsMedDiff = {avg_abs_median_diff_inclusive:.4f} (from {len(all_median_diffs_inclusive)} of 16 tests)")


def create_inclusive_avg_abs_diff_summary_chart(llm_display_names: List[str], inclusive_avg_abs_diffs: List[float], filename: str) -> None:
    """
    Generates and saves a horizontal bar chart summarizing the "Inclusive Average
    Absolute Median Difference" from human scores for all LLM judges.

    This chart visualizes the raw average deviation, including median differences
    from tests that were skipped by the main Wilcoxon analysis.

    Args:
        llm_display_names (List[str]): A list of LLM judge display names.
        inclusive_avg_abs_diffs (List[float]): A list of corresponding inclusive
                                             average absolute median differences.
        filename (str): The full path (including .svg extension) where the chart
                        will be saved.
    """
    print(f"LLM average absolute difference summary chart: {filename}")
    if not llm_display_names:
        print("No data for inclusive average absolute difference summary chart.")
        return

    # sort data by the inclusive difference in ascending order (lower is better)
    sorted_data = sorted(zip(llm_display_names, inclusive_avg_abs_diffs), key=lambda item: item[1], reverse=False)
    sorted_llm_display_names = [item[0] for item in sorted_data]
    sorted_diffs = [item[1] for item in sorted_data]

    y_pos = np.arange(len(sorted_llm_display_names))
    fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_llm_display_names) * 0.5)))
    fig.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.1)

    bars = ax.barh(y_pos, sorted_diffs, color=STANDARD_GREEN, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_llm_display_names, fontsize=TICK_FONT_SIZE)
    ax.set_xlabel("Overall Average Absolute Median Difference from Human", fontsize=LABEL_FONT_SIZE)
    ax.set_title("Overall Score Deviation LLM Judge vs. Human", fontsize=TITLE_FONT_SIZE)

    max_diff_val = max(sorted_diffs) if sorted_diffs else 1.0
    ax.set_xlim(0, max_diff_val * 1.15) # A bit more padding for labels

    for bar in bars:
        width = bar.get_width()
        ax.text(width + (max_diff_val * 0.01), bar.get_y() + bar.get_height() / 2, f'{width:.2f}',
                va='center', ha='left', fontsize=TICK_FONT_SIZE - 2)

    ax.grid(axis='x', color='grey', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    try:
        plt.savefig(filename, format='svg', dpi=300, bbox_inches="tight")
        print(f"  Saved overall average absolute difference chart: {filename}")
    except Exception as e:
        print(f"  Error saving overall average absolute difference chart {filename}: {e}")
    plt.close(fig)


def main():
    input_path = join_folder_results("llm_judge_cs-kg_orkg")
    output_path = join_folder_results("human_llm_judge_eval_results")
    os.makedirs(output_path, exist_ok=True)
    eval_file_names = [f for f in os.listdir(input_path) if f.endswith(
        "judge_cs-kg_orkg.json") and 'final_comparison' not in f]
    print(f"Found evaluation files: {eval_file_names}")

    if not eval_file_names:
        print("Error: No evaluation files found matching '*judge_cs-kg_orkg.json'")
        return

    all_llm_data = {}
    llm_ids = []

    for file_name in eval_file_names:
        file_path = os.path.join(input_path, file_name)
        print(f"Loading file: {file_name}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                eval_json = json.load(f)
            llm_id_str = get_llm_id_from_filename(file_name)
            if llm_id_str:
                if llm_id_str in all_llm_data:
                    print(
                        f"Duplicate LLM ID '{llm_id_str}' found with {file_name}.")
                all_llm_data[llm_id_str] = eval_json
                if llm_id_str not in llm_ids:
                    llm_ids.append(llm_id_str)
            else:
                print(f"Warning: Could not extract LLM ID from filename: {file_name}")

        except json.JSONDecodeError:
            print(f"file {file_name} - Invalid JSON.")
        except Exception as e:
            print(f"file {file_name} - Error loading: {e}")

    print(f"Successfully loaded data for LLM IDs: {llm_ids}")

    human_eval_file = join_folder_files("human_eval.json")
    if not os.path.exists(human_eval_file):
        print(f"Error: Human evaluation file not found at {human_eval_file}")
        return
    try:
        with open(human_eval_file, 'r', encoding='utf-8') as f:
            human_eval_json = json.load(f)
        print(f"Successfully loaded human evaluation from: {human_eval_file}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in human evaluation file: {human_eval_file}")
        return
    except Exception as e:
        print(f"Error loading human evaluation file {human_eval_file}: {e}")
        return

    all_llm_human_comparison_results = {}
    skgs_internal_keys_main = get_2_skgs_list()
    criteria_internal_keys_main = get_criteria_list()
    if human_eval_json:

        for llm_id_main, llm_eval_data_item_main in all_llm_data.items():
            llm_human_results = compare_llm_to_human(
                llm_id=llm_id_main,
                llm_eval_json=llm_eval_data_item_main,
                human_eval_json=human_eval_json,
                skgs=skgs_internal_keys_main,
                criteria=criteria_internal_keys_main,
                num_papers=NUM_PAPERS
            )
            all_llm_human_comparison_results[llm_id_main] = llm_human_results

        all_llm_human_comparison_results = final_comparison(all_llm_human_comparison_results, ALPHA)

        all_llm_human_comparison_results = calculate_and_add_best_judge_score(all_llm_human_comparison_results)

        calculate_inclusive_avg_median_diff(all_llm_human_comparison_results)

        save_wilcoxon_results(all_llm_human_comparison_results, output_path)

    else:
        print("no LLM - human evaluation Wilcoxon due to missing human evaluation data.")

    if all_llm_human_comparison_results:
        export_human_comparison_charts(all_llm_human_comparison_results, output_path)
        print("successfully saved charts for LLM - human evaluation Wilcoxon")

        best_judge_scores = []
        llm_names_for_best_judge_chart = []
        for llm_id_str_best, llm_data_item_best in all_llm_human_comparison_results.items():
            summary_metrics_item = llm_data_item_best.get("summary_metrics")
            if summary_metrics_item and "best_judge_score" in summary_metrics_item:
                best_judge_scores.append(summary_metrics_item["best_judge_score"])
                llm_names_for_best_judge_chart.append(LLM_ID_TO_NAME_MAP.get(llm_id_str_best, llm_id_str_best))

        if best_judge_scores:
            best_judge_filename_path = os.path.join(output_path,
                                                    add_timestamp_and_llm(0) + "best_llm_judge_score_normalized.svg")
            create_best_judge_summary_chart(llm_names_for_best_judge_chart, best_judge_scores, best_judge_filename_path)

    else:
        print("no charts for LLM - human evaluation Wilcoxon")


if __name__ == "__main__":
    main()
