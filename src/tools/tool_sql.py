import os
import pickle
import yaml
from dotenv import load_dotenv
from langchain_core.tools import tool
import psycopg2

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
    """Executes the given SQL query on the PostgreSQL database, saves the results as a Pickle file, 
    and returns the results as a YAML string."""
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(query)
        data = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]  # Get column names
        cursor.close()
        conn.close()

        # Combine column names with data
        data_with_columns = {
            "columns": column_names,
            "data": data
        }

        # Save data to a Pickle file
        with open('src/temp/sql_output.pkl', 'wb') as file:
            pickle.dump(data_with_columns, file)
        
        # Convert the data to YAML format for returning as a string
        yaml_data = yaml.dump(data_with_columns, default_flow_style=False)
        return yaml_data
    
    except Exception as e:
        cursor.close()
        conn.close()
        return str(e)

