from flask import Flask, jsonify, request
import psycopg2  # or any other DB connector for your DB
from dotenv import load_dotenv


# set API keys for OpenAI (the LLM we will use) and Tavily (the search tool we will use)
# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

def get_db_connection():
    conn = psycopg2.connect(
        host="localhost",
        database="your_database",
        user="your_username",
        password="your_password"
    )
    return conn

@app.route('/data', methods=['GET'])
def get_data():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM your_table;')
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
