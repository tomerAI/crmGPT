from flask import Flask, jsonify, request
import psycopg2  # or any other DB connector for your DB
from dotenv import load_dotenv
import os

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


from graphs.graph import PostgreSQLChain

def run_chain_sql(query, model):
    chain_sql = PostgreSQLChain(model)

    members = ["SQLCreationAgent", "SQLReviewAgent", "SQLExecutionAgent", 
               "OutputFormattingAgent"]
    chain_sql.build_graph(members)

    compiled_chain = chain_sql.compile_chain()

    # Enter the chain with a query
    output = chain_sql.enter_chain(query, compiled_chain, members)

    return output


def main():
    model = "gpt-4-1106-preview"

    query = (
        "I want to retrieve the customer with the highest revenue from the database."
    )

    output = run_chain_sql(query, model)

    print(output)


if __name__ == "__main__":
    main()