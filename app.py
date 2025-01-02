# app.py
from flask import Flask, render_template, request, make_response, jsonify, redirect, url_for, session
from flask_session import Session
from database import *
from analysis import run_analysis
import io
import pandas as pd
import threading
from config import Config
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
Session(app)

analysis_in_progress = False
analysis_processed = 0
analysis_total = 0
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
    return render_template('index.html', games=Config.GAMES)

@app.route('/select_game/<game_type>')
def select_game(game_type):
    if game_type not in Config.GAMES:
        return redirect(url_for('index'))
    session['game_type'] = game_type
    game_config = Config.GAMES[game_type]
    return render_template('game_options.html', game_type=game_type, game_config=game_config)

@app.route('/combos', methods=['GET'])
def combos():
    game_type = session.get('game_type', '6_42')
    if not game_type:
        return redirect(url_for('index'))
    limit = request.args.get('limit', 20, type=int)
    offset = request.args.get('offset', 0, type=int)
    total_count = count_draws(game_type)
    return render_template('combos.html',
        limit=limit,
        offset=offset,
        total_count=total_count,
        game_config=Config.GAMES[game_type]
    )

@app.route('/analysis_start', methods=['GET'])
def analysis_start():
    game_type = session.get('game_type', '6_42')
    if not game_type:
        return redirect(url_for('index'))
    game_config = Config.GAMES[game_type]
    # We want n-number subsets default to 6, not 42
    j_default = 6

    return render_template(
        'results.html',
        game_config=game_config,
        selected_df=None,
        top_df=None,
        elapsed=None,
        j=j_default,  # changed default from 42 to 6
        k=3,
        m='min',
        l=1,
        n=0,
        offset_last=0  # default offset is 0
    )

@app.route('/combos_data', methods=['GET'])
def combos_data():
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
    game_type = session.get('game_type', '6_42')
    after_id = request.form.get('after_id', type=int)
    insert_draw([None, None, None, None, None, None], game_type, after_id=after_id)
    return "OK"

@app.route('/delete_combos_hot', methods=['POST'])
def delete_combos_hot():
    game_type = session.get('game_type', '6_42')
    ids = request.form.getlist('ids[]', type=int)
    if not ids:
        return "No IDs", 400
    delete_draws(ids, game_type)
    return "OK"

@app.route('/move_row_hot', methods=['POST'])
def move_row_hot():
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
    game_type = session.get('game_type', '6_42')
    rows = get_all_draws(game_type)
    df = pd.DataFrame([[
        row['draw_number'],
        row['number1'],
        row['number2'],
        row['number3'],
        row['number4'],
        row['number5'],
        row['number6']
    ] for row in rows], columns=['Draw', '#1', '#2', '#3', '#4', '#5', '#6'])
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=all_combos.csv"
    response.headers["Content-type"] = "text/csv"
    return response

@app.route('/analysis', methods=['GET'])
def analysis_route():
    game_type = session.get('game_type', '6_42')
    j = request.args.get('j', 6, type=int)
    k = request.args.get('k', 3, type=int)
    m = request.args.get('m', 'min')
    l = request.args.get('l', 1, type=int)
    n_val = request.args.get('n', 0, type=int)
    offset_last = request.args.get('offset_last', 0, type=int)  # keep offset visible
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
    global analysis_in_progress, analysis_processed, analysis_total
    global analysis_selected_df, analysis_top_df, analysis_elapsed
    global analysis_thread, analysis_cancel_requested

    game_type = session.get('game_type', '6_42')
    j = request.form.get('j', type=int, default=6)
    k = request.form.get('k', type=int, default=3)
    m = request.form.get('m', type=str, default='min')
    l = request.form.get('l', type=int, default=1)
    n_val = request.form.get('n', type=int, default=0)
    offset_val = request.form.get('offset_last', type=int, default=0)  # new offset param

    # If there's an existing analysis in progress, cancel it
    if analysis_in_progress:
        analysis_cancel_requested = True
        if analysis_thread and analysis_thread.is_alive():
            analysis_thread.join()
        analysis_cancel_requested = False
        analysis_in_progress = False

    analysis_in_progress = True
    analysis_processed = 0
    analysis_total = 0
    analysis_selected_df = None
    analysis_top_df = None
    analysis_elapsed = None

    def progress_callback(processed, total):
        global analysis_processed, analysis_total
        analysis_processed = processed
        analysis_total = total

    def should_stop():
        return analysis_cancel_requested

    def worker():
        global analysis_in_progress, analysis_selected_df, analysis_top_df, analysis_elapsed
        sel_df, top_df, elapsed = run_analysis(
            game_type=game_type,
            j=j, k=k, m=m, l=l, n=n_val,
            last_offset=offset_val,
            progress_callback=progress_callback,
            should_stop=should_stop
        )
        analysis_selected_df = sel_df
        analysis_top_df = top_df
        analysis_elapsed = elapsed
        analysis_in_progress = False

    analysis_thread = threading.Thread(target=worker)
    analysis_thread.start()
    return "OK"

@app.route('/analysis_progress', methods=['GET'])
def analysis_progress():
    global analysis_in_progress, analysis_processed, analysis_total, analysis_elapsed
    resp = {
        'in_progress': analysis_in_progress,
        'processed': analysis_processed,
        'total': analysis_total,
        'done': (not analysis_in_progress) and (analysis_elapsed is not None),
        'elapsed': analysis_elapsed
    }
    return jsonify(resp)

@app.route('/download_selected_csv', methods=['GET'])
def download_selected_csv():
    global analysis_selected_df
    if analysis_selected_df is None:
        return "No analysis run yet", 400
    output = io.StringIO()
    analysis_selected_df.to_csv(output, index=False)
    output.seek(0)
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=selected_combinations.csv"
    response.headers["Content-type"] = "text/csv"
    return response

@app.route('/download_top_csv', methods=['GET'])
def download_top_csv():
    global analysis_top_df
    if analysis_top_df is None:
        return "No analysis run yet", 400
    output = io.StringIO()
    analysis_top_df.to_csv(output, index=False)
    output.seek(0)
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=top_combinations.csv"
    response.headers["Content-type"] = "text/csv"
    return response

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)


# if __name__ == '__main__':
#     app.run(debug=True)  # for local development
