# analysis.py
"""
analysis.py
-----------
This module orchestrates the analysis by calling a compiled C library (the "C_engine"),
thereby performing the same logic that existed in the Python-only approach but far more
quickly in C. The C library is loaded via ctypes.

Major points about this file:
1) We load the C library (libanalysis_engine.so on Linux) from the 'C_engine/src' folder.
2) We define a ctypes.Structure (AnalysisResultItem) mirroring the C struct that holds
   the analysis results.
3) We declare function signatures for run_analysis_c(...) and free_analysis_results(...)
   so that Python knows how to call into the C code.
4) In run_analysis(...), we read the draws from the database, prepare them in the format
   the C code expects, then invoke run_analysis_c(...).
5) We parse the returned pointers into Python objects and free the allocated results
   by calling free_analysis_results(...).
6) This approach preserves the original logic for "chain analysis" (l == -1) and "normal"
   top-l analysis (l >= 1), so that the final outcome matches your original Python results.
7) The only difference from the Python-only version is performance: this C-based code runs
   orders of magnitude faster, especially on larger sets of combos.

Important considerations:
- We do no partial progress reporting from the C code (the callback is effectively ignored).
- For chain analysis (l == -1), we return (selected_df=None, top_df=chain_results, elapsed).
- For normal analysis (l >= 1), we return (selected_df, top_df, elapsed).
- The "selected_df" portion is the "non-overlapping subsets" portion in the Python code;
  the C code lumps them into the second portion of the returned array, but we split them
  accordingly.
"""

import ctypes
import os
import time
import sqlite3
from database import get_db_connection
from config import Config

# ------------------------------------------------------------------------------
# 1) Load the compiled C library.
#    We assume the library is named 'libanalysis_engine.so' and is placed
#    in a subfolder "C_engine/src" next to this 'analysis.py'.
# ------------------------------------------------------------------------------
LIB_PATH = os.path.join(
    os.path.dirname(__file__),
    "C_engine", "src", "libanalysis_engine.so"
)
analysis_lib = ctypes.CDLL(LIB_PATH)

# ------------------------------------------------------------------------------
# 2) Define the AnalysisResultItem structure in Python
#    This must match the fields in analysis_engine.h
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# 3) Tell ctypes about the run_analysis_c(...) function signature
# ------------------------------------------------------------------------------
analysis_lib.run_analysis_c.argtypes = [
    ctypes.c_char_p,                               # const char* game_type
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),  # int** draws
    ctypes.c_int,                                  # draws_count
    ctypes.c_int,                                  # j
    ctypes.c_int,                                  # k
    ctypes.c_char_p,                               # m
    ctypes.c_int,                                  # l
    ctypes.c_int,                                  # n
    ctypes.c_int,                                  # last_offset
    ctypes.POINTER(ctypes.c_int)                   # out_len (pointer)
]
analysis_lib.run_analysis_c.restype = ctypes.POINTER(AnalysisResultItem)

# ------------------------------------------------------------------------------
# 4) Tell ctypes about the free_analysis_results(...) function signature
# ------------------------------------------------------------------------------
analysis_lib.free_analysis_results.argtypes = [ctypes.POINTER(AnalysisResultItem)]
analysis_lib.free_analysis_results.restype = None


# ------------------------------------------------------------------------------
# 5) The main function called by app.py to run the analysis
# ------------------------------------------------------------------------------
def run_analysis(game_type='6_42', j=6, k=3, m='min', l=1, n=0,
                 last_offset=0,
                 progress_callback=None,
                 should_stop=lambda: False):
    """
    This function replicates your Python "analysis" logic but uses the C engine
    for all heavy computations.

    Parameters:
    -----------
    game_type : str  (e.g. '6_42')
    j         : int  (the 'n-number subsets' or 'j' in the original code)
    k         : int  (the 'k-number subsets of n')
    m         : str  (sorting option, 'avg' or 'min')
    l         : int  (number of top-ranked combos, or -1 for chain analysis)
    n         : int  (number of top-ranked combos w/o overlapping subsets)
    last_offset : int (offset from the last draw, used in both chain & normal)
    progress_callback : function (ignored here, we do not do partial updates in C)
    should_stop : function (ignored as well for simplicity in C)

    Returns:
    --------
    (selected_df, top_df, elapsed_seconds)

    Where:
      - For chain analysis (l == -1), top_df is actually the chain results,
        selected_df is None.
      - For normal analysis (l >= 1), top_df is the top-l combos, selected_df
        is the additional "non-overlapping" portion (up to n combos).
      - elapsed_seconds is an int giving total time spent.

    Raises:
    -------
    OSError if the .so library isn't found or can't be loaded.
    """
    start_time = time.time()

    # 1) Load draws from the DB
    conn = get_db_connection(game_type)
    c = conn.cursor()
    row_count = c.execute("SELECT COUNT(*) as cnt FROM draws").fetchone()["cnt"]

    # Constrain last_offset to [0, row_count]
    if last_offset < 0:
        last_offset = 0
    if last_offset > row_count:
        last_offset = row_count

    # Actually fetch the draws
    draws = c.execute(
        "SELECT number1, number2, number3, number4, number5, number6 FROM draws ORDER BY sort_order"
    ).fetchall()
    conn.close()

    draws_count = len(draws)

    # 2) Convert these draws into an array-of-pointers for the C library
    draws_c = (ctypes.POINTER(ctypes.c_int) * draws_count)()
    for i, row in enumerate(draws):
        row_arr = (ctypes.c_int * 6)()
        # If any row element is None, we clamp it to 1 or some fallback
        for jx in range(6):
            row_arr[jx] = row[jx] if row[jx] is not None else 1
        draws_c[i] = row_arr

    # 3) Prepare out_len
    out_len = ctypes.c_int(0)

    # 4) Call the C function
    results_ptr = analysis_lib.run_analysis_c(
        game_type.encode('utf-8'),  # const char*
        draws_c,                    # int** draws
        draws_count,                # draws_count
        j,                          # j
        k,                          # k
        m.encode('utf-8'),         # m
        l,                          # l
        n,                          # n
        last_offset,                # last_offset
        ctypes.byref(out_len)       # out_len
    )

    res_count = out_len.value

    # 5) Convert the returned results into Python objects
    if not results_ptr or res_count <= 0:
        # No results or error
        elapsed = round(time.time() - start_time)
        return None, None, elapsed

    python_results = []
    for i in range(res_count):
        item = results_ptr[i]
        python_results.append({
            "combination": item.combination.decode('utf-8'),
            "avg_rank": item.avg_rank,
            "min_value": item.min_value,
            "subsets": item.subsets.decode('utf-8'),
            "draw_offset": item.draw_offset,
            "draws_until_common": item.draws_until_common,
            "analysis_start_draw": item.analysis_start_draw,
            "is_chain_result": item.is_chain_result
        })

    # 6) Free the results in C
    analysis_lib.free_analysis_results(results_ptr)

    # 7) Build Pandas DataFrames in the same shape as the original code expects
    import pandas as pd
    elapsed = round(time.time() - start_time)

    # Chain analysis
    if l == -1:
        # If chain analysis, just return them all in top_df
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
            return None, None, elapsed
        top_df = pd.DataFrame(chain_data)
        selected_df = None
        return selected_df, top_df, elapsed

    # Normal analysis
    else:
        # The C code arranges the top-l combos first, then up to n "selected" combos next
        top_data = []
        selected_data = []
        # figure out the boundary
        # If we had "filled" combos for top-l in C, that count is min(l, actual).
        # Then "n" combos follow for the "selected" portion, if any.
        # So we do the same split here:
        top_count = min(l, res_count)
        for i in range(top_count):
            r = python_results[i]
            top_data.append({
                'Combination': r['combination'],
                'Average Rank': r['avg_rank'],
                'MinValue': r['min_value'],
                'Subsets': r['subsets']
            })
        if n > 0 and res_count > top_count:
            selected_count = min(n, res_count - top_count)
            for i in range(top_count, top_count + selected_count):
                r = python_results[i]
                selected_data.append({
                    'Combination': r['combination'],
                    'Average Rank': r['avg_rank'],
                    'MinValue': r['min_value'],
                    'Subsets': r['subsets']
                })

        top_df = pd.DataFrame(top_data) if top_data else None
        selected_df = pd.DataFrame(selected_data) if selected_data else None
        return selected_df, top_df, elapsed
