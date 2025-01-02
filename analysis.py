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

def run_analysis(game_type='6_42', j=6, k=3, m='min', l=1, n=0,
                 last_offset=0,
                 progress_callback=None,
                 should_stop=lambda: False):
    """
    Run analysis, ignoring the last 'last_offset' draws from the DB.
    If last_offset=0, we use all draws as before.
    If last_offset=5, e.g. we skip the last 5 draws (by sort_order).
    """

    start_time = time.time()
    conn = get_db_connection(game_type)
    c = conn.cursor()

    # 1) Count how many total draws
    row_count = c.execute("SELECT COUNT(*) as cnt FROM draws").fetchone()["cnt"]
    # clamp offset if bigger than total
    if last_offset < 0:
        last_offset = 0
    if last_offset > row_count:
        last_offset = row_count

    # number of draws to use
    use_count = row_count - last_offset
    if use_count < 1:
        # If offset >= row_count, no draws are used => no analysis
        conn.close()
        return None, None, 0

    # 2) Retrieve the first 'use_count' draws by sort_order
    rows = c.execute(
        "SELECT number1, number2, number3, number4, number5, number6 FROM draws "
        "ORDER BY sort_order LIMIT ?",
        (use_count,)
    ).fetchall()
    conn.close()

    if should_stop():
        return None, None, 0

    toto_draws = len(rows)

    # 3) Build subset occurrence dictionary
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

    # 4) Evaluate combos
    max_number = Config.GAMES[game_type]['max_number']
    total_combos = comb(max_number, j)
    count_subsets_in_combo = comb(j, k)
    top_heap = []
    processed = 0

    def all_combos(n, r):
        return itertools.combinations(range(1, n + 1), r)

    for combo in all_combos(max_number, j):
        if should_stop():
            return None, None, 0
        processed += 1

        if progress_callback and processed % 50000 == 0:
            progress_callback(processed, total_combos)
            if should_stop():
                return None, None, 0

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
            top_heap.append((sort_field, combo, (avg_rank, min_val), subsets_with_counts))
        else:
            if sort_field > top_heap[0][0]:
                top_heap[0] = (sort_field, combo, (avg_rank, min_val), subsets_with_counts)
                import heapq
                heapq.heapify(top_heap)

    if progress_callback:
        progress_callback(total_combos, total_combos)

    top_list = list(top_heap)
    top_list.sort(key=lambda x: x[0], reverse=True)
    sorted_combinations = [(item[1], item[2], item[3]) for item in top_list]

    # 5) Build top_df
    top_data = []
    for cmb, vals, subs in sorted_combinations:
        top_data.append({
            'Combination': str(cmb),
            'Average Rank': vals[0],
            'MinValue': vals[1],
            'Subsets': str(subs)
        })
    top_df = pd.DataFrame(top_data)

    # 6) Overlap logic if n != 0
    if n == 0:
        selected_df = None
    else:
        selected_data = []
        seen_subsets = set()
        number = 0
        for combo, ranking, subsets_with_counts in sorted_combinations:
            if any(tuple(sorted(s)) in seen_subsets for s, _ in subsets_with_counts):
                continue
            number += 1
            selected_data.append({
                'Number': number,
                'Combination': str(combo),
                'Average Rank': ranking[0],
                'MinValue': ranking[1],
                'Subsets': str(subsets_with_counts)
            })
            for s, _ in subsets_with_counts:
                seen_subsets.add(tuple(sorted(s)))
            if number >= n:
                break
        selected_df = pd.DataFrame(selected_data)

    elapsed = round(time.time() - start_time)
    return selected_df, top_df, elapsed
