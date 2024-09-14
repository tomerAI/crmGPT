from flask import Flask, jsonify, request
import psycopg2  # or any other DB connector for your DB
from dotenv import load_dotenv
import os


# set API keys for OpenAI (the LLM we will use) and Tavily (the search tool we will use)
# Load environment variables from .env file
load_dotenv()

# Retrieve MongoDB credentials from environment variables
db_host = os.getenv("db_host")
db_database = os.getenv("db_database")
db_user = os.getenv("db_user")
db_password = os.getenv("db_password")


app = Flask(__name__)

def get_db_connection():
    conn = psycopg2.connect(
        host=db_host,
        database=db_database,
        user=db_user,
        password=db_password
    )
    return conn

@app.route('/data', methods=['GET'])
def get_data():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM public.supplier LIMIT 10;')
    data = cursor.fetchall()
    # Print the selected data in the terminal
    print("Selected Data:")
    for row in data:
        print(row)
    cursor.close()
    conn.close()
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
