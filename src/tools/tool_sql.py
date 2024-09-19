import os
import yaml
from dotenv import load_dotenv
from langchain_core.tools import tool


load_dotenv()

# Retrieve DB credentials from environment variables
db_host = os.getenv("db_host")
db_database = os.getenv("db_database")
db_user = os.getenv("db_user")
db_password = os.getenv("db_password")

def get_db_connection():
    """Establishes and returns a connection to the PostgreSQL database."""
    conn = psycopg2.connect(
        host=db_host,
        database=db_database,
        user=db_user,
        password=db_password
    )
    return conn

@tool
def execute_sql_query(query: str) -> str:
    """Executes the given SQL query on the PostgreSQL database and returns the results as a YAML string."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        data = cursor.fetchall()
        cursor.close()
        conn.close()

        # Convert the data to YAML format
        yaml_data = yaml.dump(data, default_flow_style=False)
        return yaml_data
    
    except Exception as e:
        cursor.close()
        conn.close()
        return str(e)
