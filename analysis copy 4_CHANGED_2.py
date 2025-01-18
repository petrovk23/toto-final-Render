import ctypes
import os
import time
import sqlite3
import pandas as pd
from database import get_db_connection
from config import Config

# Load the compiled C library
LIB_PATH = os.path.join(
    os.path.dirname(__file__),
    "C_engine", "src", "libanalysis_engine.so"
)
analysis_lib = ctypes.CDLL(LIB_PATH)

# Define the AnalysisResultItem structure
class AnalysisResultItem(ctypes.Structure):
    _fields_ = [
        ("combination", ctypes.c_char * 256),
        ("avg_rank", ctypes.c_double),
        ("min_value", ctypes.c_double),
        ("subsets", ctypes.c_char_p),  # Changed from fixed array to pointer
        ("draw_offset", ctypes.c_int),
        ("draws_until_common", ctypes.c_int),
        ("analysis_start_draw", ctypes.c_int),
        ("is_chain_result", ctypes.c_int),
    ]

# Tell ctypes about the run_analysis_c function signature
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
    ctypes.POINTER(ctypes.c_int)                   # out_len
]
analysis_lib.run_analysis_c.restype = ctypes.POINTER(AnalysisResultItem)

# Tell ctypes about the free_analysis_results function signature
analysis_lib.free_analysis_results.argtypes = [
    ctypes.POINTER(AnalysisResultItem),
    ctypes.POINTER(ctypes.c_int)
]
analysis_lib.free_analysis_results.restype = None

CHUNK_SIZE = 1000

def get_draws(game_type='6_42', limit=100, offset=0):
    conn = get_db_connection(game_type)
    draws = []
    for chunk_offset in range(offset, offset + limit, CHUNK_SIZE):
        chunk_limit = min(CHUNK_SIZE, offset + limit - chunk_offset)
        chunk = conn.execute(
            "SELECT * FROM draws ORDER BY sort_order LIMIT ? OFFSET ?",
            (chunk_limit, chunk_offset)
        ).fetchall()
        draws.extend(chunk)
        if len(chunk) < chunk_limit:
            break
    conn.close()
    return draws

def run_analysis(game_type='6_42', j=6, k=3, m='min', l=1, n=0, last_offset=0):
    start_time = time.time()

    conn = get_db_connection(game_type)
    c = conn.cursor()
    row_count = c.execute("SELECT COUNT(*) as cnt FROM draws").fetchone()["cnt"]

    if last_offset < 0:
        last_offset = 0
    if last_offset > row_count:
        last_offset = row_count

    draws = get_draws(game_type)
    conn.close()

    draws_count = len(draws)
    draws_c = (ctypes.POINTER(ctypes.c_int) * draws_count)()

    for i, row in enumerate(draws):
        row_arr = (ctypes.c_int * 6)()
        for jx in range(6):
            value = None
            if jx == 0:
                value = row['number1']
            elif jx == 1:
                value = row['number2']
            elif jx == 2:
                value = row['number3']
            elif jx == 3:
                value = row['number4']
            elif jx == 4:
                value = row['number5']
            elif jx == 5:
                value = row['number6']
            row_arr[jx] = int(value) if value is not None else 1
        draws_c[i] = row_arr

    out_len = ctypes.c_int(0)

    results_ptr = analysis_lib.run_analysis_c(
        game_type.encode('utf-8'),
        draws_c,
        draws_count,
        j,
        k,
        m.encode('utf-8'),
        l,
        n,
        last_offset,
        ctypes.byref(out_len)
    )

    res_count = out_len.value

    if not results_ptr or res_count <= 0:
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

    analysis_lib.free_analysis_results(results_ptr, ctypes.byref(out_len))

    import pandas as pd
    elapsed = round(time.time() - start_time)

    if l == -1:
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

    else:
        top_count = min(l, res_count)
        top_data = []
        selected_data = []

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
