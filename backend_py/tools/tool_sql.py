import os
import psycopg2
import yaml
from dotenv import load_dotenv
from langchain_core.tools import tool

class SQLTool:
    """This tool executes SQL queries on a PostgreSQL database and returns the results as YAML."""

    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        # Retrieve DB credentials from environment variables
        self.db_host = os.getenv("db_host")
        self.db_database = os.getenv("db_database")
        self.db_user = os.getenv("db_user")
        self.db_password = os.getenv("db_password")

    def get_db_connection(self):
        """Establishes and returns a connection to the PostgreSQL database."""
        conn = psycopg2.connect(
            host=self.db_host,
            database=self.db_database,
            user=self.db_user,
            password=self.db_password
        )
        return conn

    @tool
    def execute_sql_query(self, query: str) -> str:
        """Executes the given SQL query on the PostgreSQL database and returns the results as a YAML string."""
        conn = self.get_db_connection()
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
