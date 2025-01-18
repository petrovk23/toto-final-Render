import os
import io
import pandas as pd
import threading
import time
import psutil
import ctypes

from flask import Flask, render_template, request, make_response, jsonify, redirect
from flask import url_for, session, Response
from flask_session import Session

from database import *
from analysis import run_analysis
from config import Config

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
Session(app)

CHUNK_SIZE = 1000

analysis_in_progress = False
analysis_selected_df = None
analysis_top_df = None
analysis_elapsed = None
analysis_thread = None
analysis_cancel_requested = False

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB

@app.context_processor
def utility_processor():
    return dict(config=Config)

@app.route('/')
def index():
    """
    The home page: lists all available games, e.g. 6/42, 6/49, etc.
    """
    return render_template('index.html', games=Config.GAMES)

@app.route('/select_game/<game_type>')
def select_game(game_type):
    """
    Store the selected game in the session, then show the game options:
    'View/Edit Combinations' and 'Run Analysis'.
    """
    if game_type not in Config.GAMES:
        return redirect(url_for('index'))
    session['game_type'] = game_type
    game_config = Config.GAMES[game_type]
    return render_template('game_options.html', game_type=game_type, game_config=game_config)

@app.route('/combos', methods=['GET'])
def combos():
    """
    Shows the combos page with Handsontable.
    We preserve the original logic:
      - If no 'offset' is provided, jump to the bottom (last page).
      - If 'offset' is explicitly set, we use that.
    """
    game_type = session.get('game_type', '6_42')
    if not game_type:
        return redirect(url_for('index'))

    limit = request.args.get('limit', 20, type=int)
    offset_param = request.args.get('offset', None)
    total_count = count_draws(game_type)

    if offset_param is None:
        # Jump to last page
        offset = max(total_count - limit, 0)
    else:
        offset = int(offset_param)

    return render_template(
        'combos.html',
        limit=limit,
        offset=offset,
        total_count=total_count,
        game_config=Config.GAMES[game_type]
    )

@app.route('/combos_data', methods=['GET'])
def combos_data():
    """
    Endpoint that returns JSON data for the combos in the specified limit/offset.
    The JavaScript in combos.html loads this data into Handsontable.
    """
    game_type = session.get('game_type', '6_42')
    limit = request.args.get('limit', 20, type=int)
    offset = request.args.get('offset', 0, type=int)
    draws = []

    # Chunked loading of draws
    for chunk_offset in range(offset, offset + limit, CHUNK_SIZE):
        chunk_limit = min(CHUNK_SIZE, offset + limit - chunk_offset)
        chunk = get_draws_chunk(game_type, chunk_limit, chunk_offset)
        draws.extend(chunk)
        if len(chunk) < chunk_limit:
            break

    data = []
    for d in draws:
        data.append([
            d['draw_number'],
            d['number1'],
            d['number2'],
            d['number3'],
            d['number4'],
            d['number5'],
            d['number6'],
            d['id']
        ])
    return jsonify(data)

@app.route('/update_combo_hot', methods=['POST'])
def update_combo_hot():
    """
    Called by the Handsontable afterChange hook to update a row.
    """
    game_type = session.get('game_type', '6_42')
    draw_id = request.form.get('id', type=int)
    nums = [
        request.form.get('num1', type=int),
        request.form.get('num2', type=int),
        request.form.get('num3', type=int),
        request.form.get('num4', type=int),
        request.form.get('num5', type=int),
        request.form.get('num6', type=int),
    ]
    update_draw(draw_id, nums, game_type)
    return "OK"

@app.route('/add_combo_hot', methods=['POST'])
def add_combo_hot():
    """
    Called by the Handsontable 'afterCreateRow' hook to insert a new row.
    We place it after a given ID if provided, then re-run renumbering.
    """
    game_type = session.get('game_type', '6_42')
    after_id = request.form.get('after_id', type=int)
    insert_draw([None, None, None, None, None, None], game_type, after_id=after_id)
    return "OK"

@app.route('/delete_combos_hot', methods=['POST'])
def delete_combos_hot():
    """
    Called when the user removes rows in Handsontable.
    We delete each ID from the DB, then renumber.
    """
    game_type = session.get('game_type', '6_42')
    ids = request.form.getlist('ids[]', type=int)
    if not ids:
        return "No IDs", 400
    delete_draws(ids, game_type)
    return "OK"

@app.route('/move_row_hot', methods=['POST'])
def move_row_hot():
    """
    Called after a user manually reorders rows in Handsontable.
    We reorder 'sort_order' accordingly, then renumber them so that
    draw_number fields remain consistent.
    """
    game_type = session.get('game_type', '6_42')
    new_order = request.form.getlist('new_order[]', type=int)
    conn = get_db_connection(game_type)
    c = conn.cursor()
    for i, id_val in enumerate(new_order, start=1):
        c.execute("UPDATE draws SET sort_order=? WHERE id=?", (i, id_val))
    conn.commit()
    conn.close()
    renumber_all(game_type)
    return "OK"

@app.route('/download_all_combos', methods=['GET'])
def download_all_combos():
    """
    Let users download all combos as CSV with chunked streaming.
    """
    game_type = session.get('game_type', '6_42')

    def generate():
        # Write header
        yield 'Draw,#1,#2,#3,#4,#5,#6\n'.encode('utf-8')

        # Stream data in chunks
        offset = 0
        while True:
            chunk = get_draws_chunk(game_type, CHUNK_SIZE, offset)
            if not chunk:
                break

            chunk_data = []
            for row in chunk:
                chunk_data.append([
                    row['draw_number'],
                    row['number1'],
                    row['number2'],
                    row['number3'],
                    row['number4'],
                    row['number5'],
                    row['number6']
                ])

            if chunk_data:
                df_chunk = pd.DataFrame(chunk_data,
                    columns=['Draw', '#1', '#2', '#3', '#4', '#5', '#6'])
                yield df_chunk.to_csv(index=False, header=False).encode('utf-8')

            offset += CHUNK_SIZE
            if len(chunk) < CHUNK_SIZE:
                break

    response = Response(generate(), mimetype='text/csv')
    response.headers["Content-Disposition"] = "attachment; filename=all_combos.csv"
    return response

@app.route('/analysis_start', methods=['GET'])
def analysis_start():
    """
    Renders the analysis page (results.html) with empty results, ready for user parameters.
    """
    game_type = session.get('game_type', '6_42')
    if not game_type:
        return redirect(url_for('index'))
    game_config = Config.GAMES[game_type]
    j_default = 6
    return render_template(
        'results.html',
        game_config=game_config,
        selected_df=None,
        top_df=None,
        elapsed=None,
        j=j_default,
        k=3,
        m='min',
        l=1,
        n=0,
        offset_last=0
    )

@app.route('/analysis')
def analysis_route():
    """
    Once the analysis thread finishes, it stores the results in analysis_selected_df
    and analysis_top_df, plus analysis_elapsed. We display them here.
    """
    game_type = session.get('game_type', '6_42')
    j = request.args.get('j', 6, type=int)
    k = request.args.get('k', 3, type=int)
    m = request.args.get('m', 'min')
    l = request.args.get('l', 1, type=int)
    n_val = request.args.get('n', 0, type=int)
    offset_last = request.args.get('offset_last', 0, type=int)

    global analysis_selected_df, analysis_top_df, analysis_elapsed
    return render_template(
        'results.html',
        j=j, k=k, m=m, l=l, n=n_val,
        offset_last=offset_last,
        selected_df=analysis_selected_df,
        top_df=analysis_top_df,
        elapsed=analysis_elapsed,
        game_config=Config.GAMES[game_type]
    )

@app.route('/analysis_run', methods=['POST'])
def analysis_run():
    """
    Starts the analysis in a separate thread. If an existing analysis is in progress,
    we cancel (join) it first.
    """
    global analysis_in_progress, analysis_processed, analysis_total
    global analysis_selected_df, analysis_top_df, analysis_elapsed
    global analysis_thread, analysis_cancel_requested

    game_type = session.get('game_type', '6_42')
    j = request.form.get('j', type=int, default=6)
    k = request.form.get('k', type=int, default=3)
    m = request.form.get('m', type=str, default='min')
    l = request.form.get('l', type=int, default=1)
    n_val = request.form.get('n', type=int, default=0)
    offset_val = request.form.get('offset_last', type=int, default=0)

    if analysis_in_progress:
        analysis_cancel_requested = True
        if analysis_thread and analysis_thread.is_alive():
            analysis_thread.join()
        analysis_cancel_requested = False
        analysis_in_progress = False

    analysis_in_progress = True
    analysis_selected_df = None
    analysis_top_df = None
    analysis_elapsed = None

    def worker():
        global analysis_in_progress, analysis_selected_df, analysis_top_df, analysis_elapsed
        print("Analysis starting...")
        try:
            sel_df, top_df, elapsed = run_analysis(
                game_type=game_type,
                j=j, k=k, m=m, l=l, n=n_val,
                last_offset=offset_val
            )
            print(f"Analysis completed in {elapsed} seconds")
            analysis_selected_df = sel_df
            analysis_top_df = top_df
            analysis_elapsed = elapsed
            analysis_in_progress = False
            print("Worker thread finished, in_progress=False")
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            analysis_in_progress = False
            raise

    analysis_thread = threading.Thread(target=worker)
    analysis_thread.start()
    return "OK"

@app.route('/analysis_progress', methods=['GET'])
def analysis_progress():
    """
    Simple endpoint to check if analysis is done, with memory usage info
    """
    global analysis_in_progress, analysis_elapsed
    resp = {
        'in_progress': analysis_in_progress,
        'done': (not analysis_in_progress) and (analysis_elapsed is not None),
        'elapsed': analysis_elapsed,
        'memory_mb': get_memory_usage()
    }
    return jsonify(resp)

@app.route('/download_top_csv', methods=['GET'])
def download_top_csv():
    """
    Let users download the top combos as a streamed CSV file.
    """
    global analysis_top_df
    if analysis_top_df is None:
        return "No analysis run yet", 400

    def generate():
        yield analysis_top_df.to_csv(index=False).encode('utf-8')

    response = Response(generate(), mimetype='text/csv')
    response.headers["Content-Disposition"] = "attachment; filename=top_combinations.csv"
    return response

@app.route('/download_selected_csv', methods=['GET'])
def download_selected_csv():
    """
    Let users download the selected combos as a streamed CSV file.
    """
    global analysis_selected_df
    if analysis_selected_df is None:
        return "No analysis run yet", 400

    def generate():
        yield analysis_selected_df.to_csv(index=False).encode('utf-8')

    response = Response(generate(), mimetype='text/csv')
    response.headers["Content-Disposition"] = "attachment; filename=selected_combinations.csv"
    return response

def get_draws_chunk(game_type, limit, offset):
    """
    Helper function to get a chunk of draws.
    """
    conn = get_db_connection(game_type)
    draws = conn.execute(
        "SELECT * FROM draws ORDER BY sort_order LIMIT ? OFFSET ?",
        (limit, offset)
    ).fetchall()
    conn.close()
    return draws

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
