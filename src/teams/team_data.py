import functools
from typing import List, TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from utilities.helper import HelperUtilities
from tools.tool_empty import placeholder_tool
import operator

class DataRequirementTeam:
    def __init__(self, model):
        self.llm = ChatOpenAI(model=model)
        self.utilities = HelperUtilities()
        self.tools = {
            'placeholder': placeholder_tool
        }

    def data_gather_information(self):
        """Creates an agent that captures the user's expectations for the data and stores it."""
        system_prompt_template = (
            """
            Your job is to collect the user's data requirements and expectations to create a prompt template.

            Here is the current state of the conversation:
            {chat_history}

            You should gather the following information:

            1. **Purpose of the Data**: Understand why the user needs the data. What decision or analysis will it support?
            2. **Specific Data Needs**: Identify which specific data points the user is interested in (e.g., sales figures, customer demographics).
            3. **Time Frame**: Determine if there is a specific time frame the user is interested in (e.g., last month, Q1 2024).
            4. **Filters/Criteria**: Ask for any specific conditions that should be applied to the data (e.g., region, product category).

            Engage with the user to collect all necessary information. If any information is missing or unclear, ask the user for clarification.

            Once all information is collected, store it in the `data_requirements` field in the state.

            Do not proceed to generate the prompt; your role is only to collect and store the requirements.
            """
        )

        data_gather_information_agent = self.utilities.create_agent(
            self.llm,
            [self.tools['placeholder']],
            system_prompt_template
        )
        return functools.partial(
            self.utilities.agent_node,
            agent=data_gather_information_agent,
            name="data_gather_information"
        )


    def data_prompt_generator(self):
        """Creates an agent that generates a prompt based on the defined data requirements."""
        system_prompt_template = (
            """
            Based on the following data requirements, write a clear and effective prompt for a SQL Execution:

            {data_requirements}

            The prompt should:

            - Clearly address the data objective.
            - Incorporate the specific data needs.
            - Respect the constraints.
            - Adhere to the specified requirements.

            Store the generated prompt in the `generated_prompt` field in the state.
            """
        )

        data_prompt_generator_agent = self.utilities.create_agent(
            self.llm,
            [self.tools['placeholder']],
            system_prompt_template
        )
        return functools.partial(
            self.utilities.agent_node,
            agent=data_prompt_generator_agent,
            name="data_prompt_generator"
        )

    def data_supervisor(self, members: List[str]):
        """Creates a supervisor agent that manages the data requirement gathering workflow."""
        system_prompt_template = (
            """
            You are the supervisor managing the data requirement gathering workflow. 
            Your role is to route the conversation either back to the user for additional input or to the next step in the process.
            If any agent requests more information, use 'FINISH' to prompt the user.
            If the latest input asks questions assume it is for the user and use 'FINISH' to prompt the user for the questions!

            Here is the current state of the conversation:
            {chat_history}

            Here are your available options to route the conversation:

            - **data_gather_information**: Collects data requirements from the user.
            - **data_prompt_generator**: Generates a prompt template based on collected requirements.
            - **sql_generation**: Passes control to the SQL generation team.
            - **FINISH**: Prompts the user for additional input.
            """
        )

        data_supervisor = self.utilities.create_team_supervisor(
            self.llm,
            system_prompt_template,
            members
        )
        return data_supervisor
