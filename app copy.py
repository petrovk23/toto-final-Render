# app.py
"""
app.py
------
This is the Flask application entry point. It handles:
- Session setup
- Routing for pages (index, select_game, combos, analysis, etc.)
- Database I/O for combos
- The analysis run invocation
- A simple "processing" status while analysis runs (replacing the previous progress bar)
- The ability to cancel any ongoing analysis if a new one starts.

Key changes to support the new functionality:
- In /analysis_run, if an analysis is already in progress, we cancel (join) the existing
  thread before starting a new one. This way, the user doesn't need to wait for the
  previous one to finish if they made a mistake and want to run again.
- The progress bar has been replaced with a simple "Processing..." spinner/message in
  templates/results.html, and we do not display partial progress.
"""

import os
import io
import pandas as pd
import threading
import time

from flask import Flask, render_template, request, make_response, jsonify, redirect, url_for, session
from flask_session import Session

from database import *
from analysis import run_analysis
from config import Config

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
Session(app)

analysis_in_progress = False
analysis_selected_df = None
analysis_top_df = None
analysis_elapsed = None
analysis_thread = None
analysis_cancel_requested = False

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
    draws = get_draws(game_type, limit=limit, offset=offset)
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
    Let users download all combos as CSV.
    """
    game_type = session.get('game_type', '6_42')
    rows = get_all_draws(game_type)
    df = pd.DataFrame([
        [
            row['draw_number'],
            row['number1'],
            row['number2'],
            row['number3'],
            row['number4'],
            row['number5'],
            row['number6']
        ]
        for row in rows
    ], columns=['Draw', '#1', '#2', '#3', '#4', '#5', '#6'])
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=all_combos.csv"
    response.headers["Content-type"] = "text/csv"
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
    we cancel (join) it first, so the user doesn't have to wait for a mistake run to finish.
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

    # Reset all analysis state
    analysis_cancel_requested = True
    analysis_in_progress = False
    analysis_thread = None
    analysis_selected_df = None
    analysis_top_df = None
    analysis_elapsed = None

    # Wait a moment to ensure old state is cleared
    time.sleep(0.1)

    # Reset cancel flag before starting new analysis
    analysis_cancel_requested = False

    analysis_in_progress = True
    analysis_selected_df = None
    analysis_top_df = None
    analysis_elapsed = None

    def worker():
        global analysis_in_progress, analysis_selected_df, analysis_top_df, analysis_elapsed
        global analysis_cancel_requested
        print("Analysis starting...")
        try:
            if analysis_cancel_requested:
                analysis_in_progress = False
                return

            sel_df, top_df, elapsed = run_analysis(
                game_type=game_type,
                j=j, k=k, m=m, l=l, n=n_val,
                last_offset=offset_val
            )

            # For both regular and chain analysis, check cancellation after completion
            if analysis_cancel_requested:
                analysis_in_progress = False
                return

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

    analysis_thread = threading.Thread(target=worker, daemon=True)
    analysis_thread.start()
    return "OK"  # Added return statement

@app.route('/analysis_progress', methods=['GET'])
def analysis_progress():
    global analysis_in_progress, analysis_elapsed
    resp = {
        'in_progress': analysis_in_progress and not analysis_cancel_requested,
        'done': (not analysis_in_progress) and (analysis_elapsed is not None),
        'elapsed': analysis_elapsed
    }
    return jsonify(resp)

@app.route('/download_top_csv', methods=['GET'])
def download_top_csv():
    global analysis_top_df
    if analysis_top_df is None:
        return "No analysis run yet", 400

    # Create a copy to avoid modifying the original DataFrame
    df_to_save = analysis_top_df.copy()

    # For chain analysis (l=-1), rename columns and drop 'Draw Count'
    if 'Offset' in df_to_save.columns:  # This indicates it's chain analysis
        df_to_save = df_to_save.drop('Draw Count', axis=1)
        df_to_save = df_to_save.rename(columns={
            'Offset': 'Analysis #',
            'Average Rank': 'Avg Rank',
            'MinValue': 'Min Rank',
            'Draws Until Common Subset': 'Top-Ranked Duration',
            'Analysis Start Draw': 'For Draw'
        })
        # Reorder columns to put 'For Draw' as second column
        df_to_save = df_to_save.reindex(columns=[
            'Analysis #',
            'For Draw',
            'Combination',
            'Avg Rank',
            'Min Rank',
            'Top-Ranked Duration',
            'Subsets'
        ])
    else:
        df_to_save = df_to_save.rename(columns={
            'Average Rank': 'Avg Rank',
            'MinValue': 'Min Rank'
        })
    output = io.StringIO()
    df_to_save.to_csv(output, index=False)
    output.seek(0)
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=top_combinations.csv"
    response.headers["Content-type"] = "text/csv"
    return response

@app.route('/download_selected_csv', methods=['GET'])
def download_selected_csv():
    """
    Let users download the 'selected combos' (non-overlapping subsets) as CSV,
    matching the original code's functionality.
    """
    global analysis_selected_df
    if analysis_selected_df is None:
        return "No analysis run yet", 400

    # Create a copy to avoid modifying the original DataFrame
    df_to_save_sel = analysis_selected_df.copy()

    # For chain analysis (l=-1), rename columns and drop 'Draw Count'
    if 'Offset' not in df_to_save_sel.columns:  # This indicates it's chain analysis
        df_to_save_sel = df_to_save_sel.rename(columns={
            'Average Rank': 'Avg Rank',
            'MinValue': 'Min Rank'
        })

    output = io.StringIO()
    df_to_save_sel.to_csv(output, index=False)
    output.seek(0)
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=selected_combinations.csv"
    response.headers["Content-type"] = "text/csv"
    return response

if __name__ == '__main__':
    # Typically run with "python app.py" or "gunicorn app:app"
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
