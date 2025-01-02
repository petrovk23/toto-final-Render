# database.py

import sqlite3
import pandas as pd
from config import Config

def get_db_connection(game_type='6_42'):
    db_path = Config.GAMES[game_type]['db_name']
    conn = sqlite3.connect(db_path, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn

def renumber_all(game_type='6_42'):
    conn = get_db_connection(game_type)
    c = conn.cursor()
    rows = c.execute("SELECT id FROM draws ORDER BY sort_order").fetchall()
    for i, row in enumerate(rows, start=1):
        draw_number = f"{i:04d}"
        c.execute("UPDATE draws SET sort_order=?, draw_number=? WHERE id=?",
                  (i, draw_number, row['id']))
    conn.commit()
    conn.close()

def get_draws(game_type='6_42', limit=100, offset=0):
    conn = get_db_connection(game_type)
    draws = conn.execute(
        "SELECT * FROM draws ORDER BY sort_order LIMIT ? OFFSET ?",
        (limit, offset)
    ).fetchall()
    conn.close()
    return draws

def count_draws(game_type='6_42'):
    conn = get_db_connection(game_type)
    count = conn.execute("SELECT COUNT(*) as cnt FROM draws").fetchone()["cnt"]
    conn.close()
    return count

def insert_draw(numbers, game_type='6_42', after_id=None):
    numbers = clamp_numbers(numbers, game_type)
    conn = get_db_connection(game_type)
    c = conn.cursor()

    if after_id:
        row = c.execute("SELECT sort_order FROM draws WHERE id=?", (after_id,)).fetchone()
        if row:
            after_sort = row["sort_order"]
            c.execute("UPDATE draws SET sort_order = sort_order + 1 WHERE sort_order > ?", (after_sort,))
            new_sort_order = after_sort + 1
        else:
            max_so = c.execute("SELECT MAX(sort_order) FROM draws").fetchone()[0]
            new_sort_order = (max_so or 0) + 1
    else:
        max_so = c.execute("SELECT MAX(sort_order) FROM draws").fetchone()[0]
        new_sort_order = (max_so or 0) + 1

    draw_number = "temp"
    c.execute('''
        INSERT INTO draws (draw_number, number1, number2, number3, number4, number5, number6, sort_order)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (draw_number, *numbers, new_sort_order))
    conn.commit()
    conn.close()

    renumber_all(game_type)

def delete_draw(draw_id, game_type='6_42'):
    conn = get_db_connection(game_type)
    conn.execute("DELETE FROM draws WHERE id = ?", (draw_id,))
    conn.commit()
    conn.close()
    renumber_all(game_type)

def delete_draws(ids, game_type='6_42'):
    conn = get_db_connection(game_type)
    q_marks = ",".join("?" for _ in ids)
    sql = f"DELETE FROM draws WHERE id IN ({q_marks})"
    conn.execute(sql, ids)
    conn.commit()
    conn.close()
    renumber_all(game_type)

def update_draw(draw_id, numbers, game_type='6_42'):
    numbers = clamp_numbers(numbers, game_type)
    conn = get_db_connection(game_type)
    conn.execute('''
        UPDATE draws
        SET number1=?, number2=?, number3=?, number4=?, number5=?, number6=?
        WHERE id=?
    ''', (*numbers, draw_id))
    conn.commit()
    conn.close()

def get_all_draws(game_type='6_42'):
    conn = get_db_connection(game_type)
    rows = conn.execute("SELECT * FROM draws ORDER BY sort_order").fetchall()
    conn.close()
    return rows

def swap_sort_order(id1, id2, game_type='6_42'):
    conn = get_db_connection(game_type)
    c = conn.cursor()
    so1 = c.execute("SELECT sort_order FROM draws WHERE id=?", (id1,)).fetchone()["sort_order"]
    so2 = c.execute("SELECT sort_order FROM draws WHERE id=?", (id2,)).fetchone()["sort_order"]
    c.execute("UPDATE draws SET sort_order=? WHERE id=?", (so2, id1))
    c.execute("UPDATE draws SET sort_order=? WHERE id=?", (so1, id2))
    conn.commit()
    conn.close()
    renumber_all(game_type)

def clamp_numbers(nums, game_type='6_42'):
    max_number = Config.GAMES[game_type]['max_number']
    cleaned = []
    for n in nums:
        if n is None:
            cleaned.append(None)
        else:
            if n < 1:
                n = 1
            if n > max_number:
                n = max_number
            cleaned.append(n)
    return cleaned
