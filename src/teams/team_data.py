import functools
from typing import List, TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai.chat_models import ChatOpenAI
from utilities.helper import HelperUtilities
from tools.tool_empty import placeholder_tool
from tools.tool_metadata import fetch_metadata_as_yaml
import operator

class DataRequirementTeam:
    def __init__(self, model):
        self.llm = ChatOpenAI(model=model)
        self.utilities = HelperUtilities()
        self.tools = {
            'metadata': fetch_metadata_as_yaml,
            'placeholder': placeholder_tool
        }

    def data_user_input_agent(self):
        """Creates an agent that captures the user's initial data request."""
        system_prompt = (
            """
            Your primary task is to capture the user's initial data request or question and ensure a clear understanding of their data needs.

            Instructions:
            1. **Capture High-Level Needs**: Begin by understanding the user's needs at a high level. Identify the main objective they are trying to achieve with their data request.
            2. **Request Specifics**: If you cannot identify specific data needs from the user's input, ask direct questions to gather more details. For example, you could ask: "Could you please specify the data you need or the problem you're trying to solve with this data?"
            3. **Confirm Understanding**: After capturing the user's input, confirm your understanding by restating their request in your own words and asking for their confirmation. This ensures that you have accurately captured their needs.
            4. **Handle Ambiguity**: If the user's request is unclear or ambiguous, ask clarifying questions to gather more context before proceeding. Avoid making assumptions about the user's needs.
            5. **Prepare for Next Steps**: Ensure that the captured data request is structured and detailed enough to be passed on to the next agent in the workflow.

            Your goal is to accurately capture the user's data needs and ensure that the information is clear and complete before passing it on to the next stage of the data requirement gathering process.
            """
        )
        data_user_input_agent = self.utilities.create_agent(
            self.llm,
            [self.tools['placeholder']],
            system_prompt
        )
        return functools.partial(self.utilities.agent_node, agent=data_user_input_agent, name="data_user_input")

    def data_clarification_agent(self):
        """Creates an agent that asks clarifying questions to the user to gather more details about their data needs."""
        system_prompt = (
            """
            You are the data_clarification agent. Your task is to ask the user targeted questions to clarify their data needs.
            Focus on gathering specific information such as:
            - Time periods of interest
            - Data dimensions (e.g., location, product, customer segment)
            - Filters or conditions to apply
            - Metrics or KPIs of interest
            - Data granularity (e.g., daily, weekly, monthly)
            - Desired output format (e.g., table, chart)
            - Preferred data visualization type (e.g., bar chart, line graph)
            - Any specific data exclusions
            - Frequency of data updates needed
            - Whether historical or real-time data is required
            Ensure that your questions are concise and help the user articulate their needs clearly.
            """
        )
        data_clarification_agent = self.utilities.create_agent(
            self.llm,
            [self.tools['placeholder']],
            system_prompt
        )
        return functools.partial(self.utilities.agent_node, agent=data_clarification_agent, name="data_clarification")

    def data_schema_mapping_agent(self):
        """Creates an agent that maps the user's clarified needs to the database schema."""
        system_prompt = (
            """
            You are the data_schema_mapping agent. Your task is to map the user's clarified data needs to the database schema.
            Use the 'metadata' tool to fetch the database schema and ensure that the user's requirements align with the available data.
            Identify the relevant tables, fields, and relationships that will be needed to construct the query.
            """
        )
        data_schema_mapping_agent = self.utilities.create_agent(
            self.llm,
            [self.tools['metadata']],
            system_prompt
        )
        return functools.partial(self.utilities.agent_node, agent=data_schema_mapping_agent, name="data_schema_mapping")

    def data_supervisor(self, members: List[str]):
        """Creates a supervisor agent that manages the data requirement gathering workflow."""
        system_prompt = (
            """
            You are the supervisor managing the data requirement gathering workflow. Your primary responsibility is to oversee the agents and ensure that they perform their tasks correctly and in the proper sequence to gather all necessary information from the user.

            Instructions:
            1. **Monitor Agent Outputs**: Closely monitor the outputs of each agent—data_user_input, data_clarification, data_schema_mapping, and data_parameter_collection—to ensure they provide complete and accurate information.
            2. **Identify User-Directed Outputs**: If an agent’s output indicates a need for further clarification, requests additional input from the user, or contains information that should be directly communicated to the user, you must immediately prompt the user using 'FINISH'.
            3. **Avoid Unnecessary Loops**: Do not reroute tasks back to agents unnecessarily. If the information needed is unclear or incomplete due to lack of user input, prompt the user directly rather than cycling back through agents.
            4. **Ensure Smooth Transition**: Only proceed to the next agent if the current agent's output is complete and sufficient for the subsequent task. If the workflow is interrupted due to insufficient information, guide the user to provide the necessary context.
            5. **Prioritize User Engagement**: Always prioritize user engagement when the context or clarity of the data requirements is uncertain. The goal is to capture precise and actionable data requirements efficiently.

            Your primary goal is to facilitate an efficient and user-centered workflow, ensuring that all necessary data is captured accurately and that the user is engaged whenever additional context or clarification is needed.
            
            Here are your available options to route the conversation:
            1. FINISH: to prompt the user for additional input.
            2. data_user_input: Clarifies the user's initial data request.
            3. data_clarification: Asks targeted questions to clarify the user's data needs.
            4. data_schema_mapping: Maps the user's clarified data needs to the database schema.
            5. sql_generation: When the user's data requirements are clear, pass the requirements to the SQL generation team.
            """
        )
        data_supervisor = self.utilities.create_team_supervisor(
            self.llm,
            system_prompt,
            members
        )
        return data_supervisor

