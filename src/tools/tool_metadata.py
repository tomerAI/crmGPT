import json
import yaml
from langchain_core.tools import tool


@tool
def fetch_metadata_as_yaml() -> str:
    """Reads the catalog.json file and returns its contents as a YAML string."""
    try:
        # Define the path to the JSON file
        json_file_path = 'src/utilities/catalog.json'

        # Load the JSON data
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

        # Convert JSON data to YAML
        yaml_data = yaml.dump(data, default_flow_style=False)

        return yaml_data
    except Exception as e:
        return str(e)
    

import psycopg2
import load_dotenv
import os

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

def fetch_metadata_as_json():
    """
    Connects to the PostgreSQL database, fetches metadata from the metadata_table,
    and returns it as a JSON string.

    Parameters:
    - db_config (dict): A dictionary containing database connection parameters:
        {
            'dbname': 'your_database_name',
            'user': 'your_username',
            'password': 'your_password',
            'host': 'your_host',
            'port': 'your_port'  # usually 5432
        }

    Returns:
    - str: A JSON-formatted string containing the metadata.
    """
    try:
        # Establish a connection to the PostgreSQL database
        conn = get_db_connection()
        cursor = conn.cursor()

        # Fetch metadata from the metadata_table
        query = """
        SELECT
            schema_name,
            table_name,
            column_name,
            data_type,
            column_description,
            constraint_name,
            constraint_type
        FROM public.metadata_table;
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        # Get column names from cursor description
        col_names = [desc[0] for desc in cursor.description]

        # Convert rows to list of dictionaries
        metadata_list = [dict(zip(col_names, row)) for row in rows]

        # Serialize the list of dictionaries to a JSON-formatted string
        metadata_json = json.dumps(metadata_list, indent=4)

        return metadata_json

    except Exception as e:
        print(f"Error fetching metadata: {e}")
        return None

    finally:
        # Clean up the database connection
        if cursor:
            cursor.close()
        if conn:
            conn.close()
