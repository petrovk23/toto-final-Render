# init_database.py
import pandas as pd
import sqlite3
import sys
from config import Config

def init_db(game_type):
    game_config = Config.GAMES[game_type]
    max_number = game_config['max_number']
    conn = sqlite3.connect(game_config['db_name'])
    c = conn.cursor()

    c.execute('DROP TABLE IF EXISTS draws')

    max_number = game_config['max_number']
    c.execute(f'''
        CREATE TABLE draws (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            draw_number TEXT,
            number1 INTEGER CHECK (number1 IS NULL OR (number1 >= 1 AND number1 <= {max_number})),
            number2 INTEGER CHECK (number2 IS NULL OR (number2 >= 1 AND number2 <= {max_number})),
            number3 INTEGER CHECK (number3 IS NULL OR (number3 >= 1 AND number3 <= {max_number})),
            number4 INTEGER CHECK (number4 IS NULL OR (number4 >= 1 AND number4 <= {max_number})),
            number5 INTEGER CHECK (number5 IS NULL OR (number5 >= 1 AND number5 <= {max_number})),
            number6 INTEGER CHECK (number6 IS NULL OR (number6 >= 1 AND number6 <= {max_number})),
            sort_order INTEGER
        )
    ''')

    conn.commit()
    return conn

def load_csv_to_db(game_type):
    try:
        game_config = Config.GAMES[game_type]
        df = pd.read_csv(game_config['csv_file'], header=None)
        df = df.iloc[1:].reset_index(drop=True)
        df['draw_number'] = df.index.map(lambda x: f'{x+1:04d}')

        conn = init_db(game_type)
        cursor = conn.cursor()

        for index, row in df.iterrows():
            numbers = sorted(row.iloc[:6].astype(int).tolist())
            draw_number = row['draw_number']
            cursor.execute('''
                INSERT INTO draws (draw_number, number1, number2, number3, number4, number5, number6)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (draw_number, *numbers))

        cursor.execute("UPDATE draws SET sort_order = id")
        conn.commit()
        conn.close()

        print(f"Successfully loaded {len(df)} draws into {game_config['db_name']}")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    for game_type in Config.GAMES:
        print(f"\nInitializing database for {Config.GAMES[game_type]['name']}...")
        success = load_csv_to_db(game_type)
        if not success:
            sys.exit(1)
