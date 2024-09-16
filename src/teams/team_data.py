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
            'placeholder': placeholder_tool
        }

    def data_gather_information(self):
        """Creates an agent that captures the user's initial data request."""
        system_prompt = (
            """
            Your job is to get data information from a user about what type of prompt template they want to create.

            You should get the following information from them:

            - What the data objective of the prompt is
            - What variables will be passed into the prompt template
            - Any constraints for what the output should NOT do
            - Any requirements that the output MUST adhere to

            If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.

            After you are able to discern all the information, call the relevant tool.
            """
        )
        data_user_input_agent = self.utilities.create_agent(
            self.llm,
            [self.tools['placeholder']],
            system_prompt
        )
        return functools.partial(self.utilities.agent_node, agent=data_user_input_agent, name="data_user_input")

    def data_prompt_generator(self):
        """Creates an agent that creates a prompt template based on the user's data requirements."""
        system_prompt = (
            """
            Based on the following requirements, write a good prompt template:

            {reqs}
            """
        )
        data_user_input_agent = self.utilities.create_agent(
            self.llm,
            [self.tools['placeholder']],
            system_prompt
        )
        return functools.partial(self.utilities.agent_node, agent=data_user_input_agent, name="data_user_input")


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

