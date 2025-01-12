# analysis.py
import ctypes
import os
import sys
import math
import time
import tempfile

import sqlite3
from database import get_db_connection
from config import Config

# Load the compiled C library at import time
# We'll assume the compiled library is named 'libanalysis_engine.so' (Linux),
# or 'analysis_engine.dll' (Windows), or 'libanalysis_engine.dylib' (macOS).
# For simplicity, let's assume Codespaces => Linux => .so
LIB_PATH = os.path.join(
    os.path.dirname(__file__),
    "C_engine", "src", "libanalysis_engine.so"
)

analysis_lib = ctypes.CDLL(LIB_PATH)

# We must define a python equivalent for AnalysisResultItem
class AnalysisResultItem(ctypes.Structure):
    _fields_ = [
        ("combination", ctypes.c_char * 256),
        ("avg_rank", ctypes.c_double),
        ("min_value", ctypes.c_double),
        ("subsets", ctypes.c_char * 512),
        ("draw_offset", ctypes.c_int),
        ("draws_until_common", ctypes.c_int),
        ("analysis_start_draw", ctypes.c_int),
        ("is_chain_result", ctypes.c_int),
    ]

# Provide function signatures
analysis_lib.run_analysis_c.argtypes = [
    ctypes.c_char_p,        # const char* game_type
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)), # draws
    ctypes.c_int,           # draws_count
    ctypes.c_int,           # j
    ctypes.c_int,           # k
    ctypes.c_char_p,        # m
    ctypes.c_int,           # l
    ctypes.c_int,           # n
    ctypes.c_int,           # last_offset
    ctypes.POINTER(ctypes.c_int) # out_len
]
analysis_lib.run_analysis_c.restype = ctypes.POINTER(AnalysisResultItem)

analysis_lib.free_analysis_results.argtypes = [ctypes.POINTER(AnalysisResultItem)]
analysis_lib.free_analysis_results.restype = None

def run_analysis(game_type='6_42', j=6, k=3, m='min', l=1, n=0,
                 last_offset=0,
                 progress_callback=None,
                 should_stop=lambda: False):
    """
    Re-implemented to call the C library for all the logic.
    Returns (selected_df, top_df, elapsed) to keep the rest of your Flask code working the same.
    We do NOT do partial progress here, so 'progress_callback' is ignored.
    'should_stop' is also not used.
    """
    start_time = time.time()

    # 1) get the draws from the DB
    conn = get_db_connection(game_type)
    c = conn.cursor()
    row_count = c.execute("SELECT COUNT(*) as cnt FROM draws").fetchone()["cnt"]
    if last_offset < 0:
        last_offset = 0
    if last_offset > row_count:
        last_offset = row_count

    draws = c.execute(
        "SELECT number1, number2, number3, number4, number5, number6 FROM draws "
        "ORDER BY sort_order"
    ).fetchall()
    conn.close()

    # Prepare the C array-of-arrays
    draws_count = len(draws)
    # create an array (of pointers to int[6])
    draws_c = (ctypes.POINTER(ctypes.c_int) * draws_count)()
    # each subarray
    for i, row in enumerate(draws):
        row_arr = (ctypes.c_int * 6)()
        for jx in range(6):
            row_arr[jx] = row[jx] if row[jx] is not None else 1
        draws_c[i] = row_arr

    out_len = ctypes.c_int(0)

    results_ptr = analysis_lib.run_analysis_c(
        game_type.encode('utf-8'),    # game_type
        draws_c,                     # draws
        draws_count,                 # draws_count
        j,                           # j
        k,                           # k
        m.encode('utf-8'),           # m
        l,                           # l
        n,                           # n
        last_offset,                 # last_offset
        ctypes.byref(out_len)        # out_len
    )

    res_count = out_len.value
    # Convert them into python lists
    python_results = []
    for i in range(res_count):
        item = results_ptr[i]
        # parse it
        # is_chain_result = 1 => chain analysis row
        # is_chain_result = 0 => normal or top combos
        combination = item.combination.decode('utf-8')
        avg_rank = item.avg_rank
        min_val = item.min_value
        subsets = item.subsets.decode('utf-8')
        draw_offset = item.draw_offset
        draws_until_common = item.draws_until_common
        analysis_start_draw = item.analysis_start_draw
        is_chain_result = item.is_chain_result

        python_results.append({
            "combination": combination,
            "avg_rank": avg_rank,
            "min_value": min_val,
            "subsets": subsets,
            "draw_offset": draw_offset,
            "draws_until_common": draws_until_common,
            "analysis_start_draw": analysis_start_draw,
            "is_chain_result": is_chain_result
        })

    # free memory
    analysis_lib.free_analysis_results(results_ptr)

    elapsed = round(time.time() - start_time)

    if l == -1:
        # chain analysis => top_df is the chain results
        import pandas as pd
        chain_data = []
        for row in python_results:
            chain_data.append({
                'Offset': row['draw_offset'],
                'Combination': row['combination'],
                'Average Rank': row['avg_rank'],
                'MinValue': row['min_value'],
                'Subsets': row['subsets'],
                'Draws Until Common Subset': row['draws_until_common'],
                'Analysis Start Draw': row['analysis_start_draw'],
                'Draw Count': row['draw_offset']
            })
        if len(chain_data) == 0:
            return None, None, 0
        top_df = pd.DataFrame(chain_data)
        selected_df = None
        return selected_df, top_df, elapsed
    else:
        # normal => we have top combos in the front, optional "selected combos" appended
        # your python code was returning top_df and selected_df as two separate DataFrames
        # so let's do the same approach
        import pandas as pd
        # we must figure out how many are top combos and how many are "selected combos"
        # we decided to put them all in one array: top combos first, selected combos second
        # Let's guess the split: if n>0, the second half are "selected combos"
        # we can't know the exact split from the C code, but we do the same logic we used
        # "top-l" combos => first l items (or fewer if fewer combos)
        # "selected combos" => second n items (or fewer if not enough combos)
        # if the total was l + n
        top_data = []
        selected_data = []
        if res_count > 0:
            # figure out how many are "top combos" portion
            top_count = min(l, res_count)
            for i in range(top_count):
                r = python_results[i]
                top_data.append({
                    'Combination': r['combination'],
                    'Average Rank': r['avg_rank'],
                    'MinValue': r['min_value'],
                    'Subsets': r['subsets']
                })
            # the selected portion
            if n > 0 and (res_count > top_count):
                selected_count = min(n, res_count - top_count)
                for i in range(top_count, top_count+selected_count):
                    r = python_results[i]
                    selected_data.append({
                        'Combination': r['combination'],
                        'Average Rank': r['avg_rank'],
                        'MinValue': r['min_value'],
                        'Subsets': r['subsets']
                    })

        top_df = None
        selected_df = None
        if len(top_data) > 0:
            top_df = pd.DataFrame(top_data)
        if len(selected_data) > 0:
            selected_df = pd.DataFrame(selected_data)

        return selected_df, top_df, elapsed
