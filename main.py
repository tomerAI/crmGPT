from flask import Flask, jsonify, request
import psycopg2  # or any other DB connector for your DB
from dotenv import load_dotenv
import os
from backend_py.teams.team_sql import SQLTeam

# set API keys for OpenAI (the LLM we will use) and Tavily (the search tool we will use)
# Load environment variables from .env file
load_dotenv()

# Retrieve DB credentials from environment variables
db_host = os.getenv("db_host")
db_database = os.getenv("db_database")
db_user = os.getenv("db_user")
db_password = os.getenv("db_password")

# Access the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
hf_api_key = os.getenv("HF_API_KEY")

# Set the API keys 
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ['USER_AGENT'] = 'myagent'

# set API key for LangSmith tracing, which will give us best-in-class observability.
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "crmGPT"


def run_chain_sql():
    # Instantiate the team
    team = SQLTeam("gpt-4")
    metadata_node = team.metadata_agent()
    sql_creation_node = team.sql_creation_agent()
    sql_review_node = team.sql_review_agent()
    sql_execution_node = team.sql_execution_agent()
    output_node = team.output_agent()
    supervisor = team.supervisor_agent(["MetadataAgent", "SQLCreationAgent", "SQLReviewAgent", "SQLExecutionAgent", "OutputAgent"])


def main():
    run_chain_sql()