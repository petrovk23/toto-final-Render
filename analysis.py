# analysis.py
import pandas as pd
import itertools
import gc
import time
from math import comb
import sqlite3
from database import get_db_connection
import heapq
from config import Config

def check_common_subsets(combo, k, draws, start_idx):
    """
    Check if combo has any common k-number subsets with draws starting from start_idx.
    Returns (draws_until_match, remaining_draws) tuple. If no match found,
    draws_until_match will be None and remaining_draws will contain count of checked draws.
    """
    combo_subsets = set(tuple(sorted(s)) for s in itertools.combinations(combo, k))

    remaining_draws = len(draws) - start_idx
    for idx, draw in enumerate(draws[start_idx:], 1):
        draw_nums = [x for x in draw if x is not None]
        draw_subsets = set(tuple(sorted(s)) for s in itertools.combinations(draw_nums, k))

        if combo_subsets & draw_subsets:  # if there's any intersection
            return idx, None

    return None, remaining_draws

def run_analysis_chain(game_type, j, k, m, last_offset, progress_callback=None, should_stop=lambda: False):
    """
    Runs a chain of analyses, starting from initial offset and continuing until no more draws to check.
    """
    chain_results = []
    current_offset = last_offset
    total_draws = None
    rows = None

    while True:
        if should_stop():
            break

        if total_draws is None:
            conn = get_db_connection(game_type)
            c = conn.cursor()
            total_draws = c.execute("SELECT COUNT(*) as cnt FROM draws").fetchone()["cnt"]
            rows = c.execute(
                "SELECT number1, number2, number3, number4, number5, number6 FROM draws "
                "ORDER BY sort_order"
            ).fetchall()
            conn.close()

        if current_offset >= total_draws:
            break

        if current_offset < 0:
            current_offset = 0

        use_count = total_draws - current_offset
        subset_occurrence_dict = {}

        for idx in reversed(range(use_count)):
            if should_stop():
                return chain_results
            row = rows[idx]
            row_list = [x for x in row if x is not None]
            weight = (use_count - 1) - idx
            for subset in itertools.combinations(row_list, k):
                s = tuple(sorted(subset))
                if s not in subset_occurrence_dict:
                    subset_occurrence_dict[s] = weight

        max_number = Config.GAMES[game_type]['max_number']
        total_combos = comb(max_number, j)
        count_subsets_in_combo = comb(j, k)
        best_combo = None
        best_sort_field = float("-inf")
        best_metrics = None
        best_subsets = None
        processed = 0

        for combo in itertools.combinations(range(1, max_number + 1), j):
            if should_stop():
                return chain_results

            processed += 1
            if progress_callback and processed % 50000 == 0:
                progress_callback(processed, total_combos)

            sum_occurrences = 0
            min_val = float("inf")
            subsets_with_counts = []

            for subset in itertools.combinations(combo, k):
                s = tuple(sorted(subset))
                occurrence = subset_occurrence_dict.get(s, 0)
                subsets_with_counts.append((s, occurrence))
                sum_occurrences += occurrence
                if occurrence < min_val:
                    min_val = occurrence

            avg_rank = sum_occurrences / count_subsets_in_combo
            sort_field = avg_rank if m == 'avg' else min_val

            if sort_field > best_sort_field:
                best_sort_field = sort_field
                best_combo = combo
                best_metrics = (avg_rank, min_val)
                best_subsets = subsets_with_counts

        if best_combo is None:
            break

        draws_until_match, remaining_draws = check_common_subsets(best_combo, k, rows, len(rows) - current_offset)

        chain_results.append({
            'Offset': current_offset,
            'Combination': str(best_combo),
            'Average Rank': best_metrics[0],
            'MinValue': best_metrics[1],
            'Subsets': str(best_subsets),
            'Draws Until Common Subset': str(draws_until_match - 1) if draws_until_match is not None else str(remaining_draws),
            'Analysis Start Draw': total_draws - current_offset,
            'Draw Count': current_offset,
        })

        if draws_until_match is None:
            break

        current_offset = current_offset - draws_until_match

    return chain_results

def run_analysis(game_type='6_42', j=6, k=3, m='min', l=1, n=0,
                 last_offset=0,
                 progress_callback=None,
                 should_stop=lambda: False):
    """
    Main analysis function that handles both regular and chain analysis.
    Chain analysis is triggered when l=-1.
    """
    start_time = time.time()

    if l == -1:  # Chain analysis when l=-1, regardless of n value
        chain_results = run_analysis_chain(game_type, j, k, m, last_offset, progress_callback, should_stop)
        if not chain_results:
            return None, None, 0
        top_df = pd.DataFrame(chain_results)
        selected_df = None
    else:
        conn = get_db_connection(game_type)
        c = conn.cursor()

        row_count = c.execute("SELECT COUNT(*) as cnt FROM draws").fetchone()["cnt"]
        if last_offset < 0:
            last_offset = 0
        if last_offset > row_count:
            last_offset = row_count

        use_count = row_count - last_offset
        if use_count < 1:
            conn.close()
            return None, None, 0

        rows = c.execute(
            "SELECT number1, number2, number3, number4, number5, number6 FROM draws "
            "ORDER BY sort_order LIMIT ?",
            (use_count,)
        ).fetchall()
        conn.close()

        if should_stop():
            return None, None, 0

        toto_draws = len(rows)

        subset_occurrence_dict = {}
        for idx in reversed(range(toto_draws)):
            if should_stop():
                return None, None, 0
            row = rows[idx]
            row_list = [x for x in row if x is not None]
            weight = (toto_draws - 1) - idx
            for subset in itertools.combinations(row_list, k):
                s = tuple(sorted(subset))
                if s not in subset_occurrence_dict:
                    subset_occurrence_dict[s] = weight
        gc.collect()

        max_number = Config.GAMES[game_type]['max_number']
        total_combos = comb(max_number, j)
        count_subsets_in_combo = comb(j, k)
        top_heap = []
        processed = 0

        for combo in itertools.combinations(range(1, max_number + 1), j):
            if should_stop():
                return None, None, 0
            processed += 1

            if progress_callback and processed % 50000 == 0:
                progress_callback(processed, total_combos)

            sum_occurrences = 0
            min_val = float("inf")
            subsets_with_counts = []

            for subset in itertools.combinations(combo, k):
                s = tuple(sorted(subset))
                occurrence = subset_occurrence_dict.get(s, 0)
                subsets_with_counts.append((s, occurrence))
                sum_occurrences += occurrence
                if occurrence < min_val:
                    min_val = occurrence

            avg_rank = sum_occurrences / count_subsets_in_combo
            sort_field = avg_rank if m == 'avg' else min_val

            if len(top_heap) < l:
                heapq.heappush(top_heap, (sort_field, combo, (avg_rank, min_val), subsets_with_counts))
            else:
                if sort_field > top_heap[0][0]:
                    heapq.heapreplace(top_heap, (sort_field, combo, (avg_rank, min_val), subsets_with_counts))

        if progress_callback:
            progress_callback(total_combos, total_combos)

        top_list = list(top_heap)
        top_list.sort(key=lambda x: x[0], reverse=True)
        sorted_combinations = [(item[1], item[2], item[3]) for item in top_list]

        top_data = []
        for cmb, vals, subs in sorted_combinations:
            top_data.append({
                'Combination': str(cmb),
                'Average Rank': vals[0],
                'MinValue': vals[1],
                'Subsets': str(subs)
            })
        top_df = pd.DataFrame(top_data)

        if n == 0:
            selected_df = None
        else:
            selected_data = []
            seen_subsets = set()
            for cmb, vals, subs in sorted_combinations:
                subset_list = [(s[0], s[1]) for s in eval(str(subs))]
                subset_tuples = [s[0] for s in subset_list]

                if not any(s in seen_subsets for s in subset_tuples):
                    selected_data.append({
                        'Combination': str(cmb),
                        'Average Rank': vals[0],
                        'MinValue': vals[1],
                        'Subsets': str(subs)
                    })
                    seen_subsets.update(subset_tuples)

                    if len(selected_data) >= n:
                        break

            selected_df = pd.DataFrame(selected_data)

    elapsed = round(time.time() - start_time)
    return selected_df, top_df, elapsed
