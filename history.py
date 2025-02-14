import sqlite3

class ConversationHistory:
    def __init__(self, db_path: str = 'conversation_history.db'):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS history 
            (timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, query TEXT, response TEXT)
        ''')
        self.conn.commit()

    def save(self, query: str, response: str):
        self.cursor.execute(
            "INSERT INTO history (query, response) VALUES (?, ?)",
            (query, response)
        )
        self.conn.commit()

    def get_history(self, limit: int = 10):
        self.cursor.execute(
            "SELECT timestamp, query, response FROM history ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        return self.cursor.fetchall()

    def __del__(self):
        self.conn.close()
