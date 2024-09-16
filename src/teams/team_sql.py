import functools
from typing import List, TypedDict, Annotated
from langchain.schema import BaseMessage
from langchain_community.chat_models import ChatOpenAI
from utilities.helper import HelperUtilities
from tools.tool_empty import placeholder_tool
from tools.tool_metadata import fetch_metadata_as_yaml
from tools.tool_sql import execute_sql_query
import operator

class SQLTeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: List[str]
    next: str

class SQLTeam:
    def __init__(self, model):
        self.llm = ChatOpenAI(model=model)
        self.utilities = HelperUtilities()
        self.tools = {
            'metadata': fetch_metadata_as_yaml,
            'sql': execute_sql_query,
            'placeholder': placeholder_tool
        }

    def sql_generation_agent(self):
        """Creates an agent that generates a PostgreSQL query based on user input and metadata."""
        
        system_prompt_template = (
            """
            Your task is to create PostgreSQL queries based on the user's request and the metadata of the database. 
            Based on the following prompt template, generate the appropriate SQL query:

            {prompt_template}
            Use your 'metadata' tool to gather metadata of the database and generate PostgreSQL queries that meet the user's requirements.
            Ensure the SQL code aligns with the PostgreSQL database schema and the user’s intent.
            Consider any PostgreSQL-specific functions or optimizations that could be applied.
            """
        )

        def system_prompt(state):
            # Access 'prompt_template' from the state
            prompt_template = state.get('prompt_template', 'No prompt template provided.')
            return system_prompt_template.format(prompt_template=prompt_template)
        
        sql_generation_agent = self.utilities.create_agent(
            self.llm,
            [self.tools['metadata']],
            system_prompt_template
        )
        return functools.partial(
            self.utilities.agent_node,
            agent=sql_generation_agent,
            name="sql_generation"
        )

    def sql_validation_agent(self):
        """Creates an agent that reviews a PostgreSQL query for syntax, security, and performance."""
        system_prompt_template = (
            """
            Your task is to review the generated PostgreSQL query to ensure it is syntactically correct, secure, and efficient. 
            Based on the following SQL code, validate the query:

            {sql_query}
            Use your expertise to identify any potential issues, such as syntax errors, security vulnerabilities, or performance bottlenecks.
            Ensure the query adheres to best practices and standards for PostgreSQL database operations.
            """
        )
        def system_prompt(state):
            # Access 'sql_query' from the state
            sql_query = state.get('sql_query', 'No SQL query provided.')
            return system_prompt_template.format(sql_query=sql_query)

        sql_validation_agent = self.utilities.create_agent(
            self.llm,
            [self.tools['placeholder']],
            system_prompt_template
        )
        return functools.partial(
            self.utilities.agent_node,
            agent=sql_validation_agent,
            name="sql_validation"
        )

    def sql_execution_agent(self):
        """Creates an agent that executes a PostgreSQL query."""
        system_prompt_template = (
            """
            Your task is to execute the PostgreSQL query and return the results. 
            Ensure that the execution is efficient and that the connection to the PostgreSQL database is securely managed.
            Handle any errors or exceptions that may arise during query execution.
            """
        )
        def system_prompt(state):
            return system_prompt_template

        sql_execution_agent = self.utilities.create_agent(
            self.llm,
            [self.tools['sql']],
            system_prompt_template
        )
        return functools.partial(
            self.utilities.agent_node,
            agent=sql_execution_agent,
            name="sql_execution"
        )

    def sql_result_formatting_agent(self):
        """Creates an agent that summarizes the results of a PostgreSQL query execution."""
        system_prompt_template = (
            """
            Your task is to summarize the output of the executed PostgreSQL query. 
            Provide a concise summary that captures the key points of the data, including any notable trends, counts, or statistics.
            Ensure the summary is easy to understand and highlights the most relevant information for the user.
            Focus on PostgreSQL-specific data types and formatting when summarizing the results.
            If the output doesn't meet the user's needs or is unclear, prompt the user for additional information to start a new query.
            """
        )
        def system_prompt(state):
            return system_prompt_template
        
        sql_result_formatting_agent = self.utilities.create_agent(
            self.llm,
            [self.tools['placeholder']],
            system_prompt_template
        )
        return functools.partial(
            self.utilities.agent_node,
            agent=sql_result_formatting_agent,
            name="sql_result_formatting"
        )

    def sql_supervisor(self, members: List[str]):
        """Creates a supervisor agent that manages the PostgreSQL query execution workflow."""
        system_prompt_template = (
            """
            You are the supervisor managing a PostgreSQL query execution workflow. Your role is to ensure that each agent performs their task correctly and in the proper sequence. 

            The agents you have at your disposal are:
            - sql_generation: Generates the SQL code based on the user query and the available metadata.
            - sql_validation: Reviews the generated SQL code to ensure it is syntactically correct, efficient, and secure.
            - sql_execution: Executes the approved SQL code on the PostgreSQL database.
            - sql_result_formatting: Summarizes the results of the executed SQL query to provide a clear and concise output.
            
            The preferred workflow is as follows:
            1. Pass this metadata to sql_generation to generate the appropriate SQL query.
            2. Send the generated SQL to sql_validation for a thorough review.
            3. If the SQL code meets our high standards, proceed to sql_execution to run the query.
            4. Once the query is executed, pass the results to sql_result_formatting to create a summary.
            5. Finally, return the formatted summary to the user.

            If any agent's output does not meet the expected standards, especially during the review phase, employ an iterative approach. Send the task back to the relevant agent for refinement until the output is of the highest quality.

            Your primary goal is to ensure a seamless and efficient workflow that consistently delivers accurate, secure, and high-quality results that meet the user's needs.
            """
        )

        def system_prompt(state):
            return system_prompt_template

        sql_supervisor = self.utilities.create_team_supervisor(
            self.llm,
            system_prompt_template,
            members
        )
        return sql_supervisor
