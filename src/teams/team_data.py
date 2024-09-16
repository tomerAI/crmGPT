import functools
from typing import List, TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_community.chat_models import ChatOpenAI
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
        """Creates an agent that captures the user's initial data request and stores it."""
        system_prompt_template = (
            """
            Your job is to collect data requirements from the user to create a prompt template.

            Here is the current state of the conversation:
            {chat_history}

            You should gather the following information:

            - **Data Objective**: What is the goal or purpose of the data the user wants?
            - **Variables**: What variables or parameters will be passed into the prompt template?
            - **Constraints**: Any constraints or limitations that the output should NOT include.
            - **Requirements**: Any specific requirements that the output MUST adhere to.

            Engage with the user to collect all necessary information. If any information is missing or unclear, ask the user for clarification.

            Once all information is collected, store it in the `data_requirements` field in the state.

            Do not proceed to generate the prompt; your role is only to collect and store the requirements.
            """
        )
        def system_prompt(state):
            # Access 'chat_history' from the state
            data_requirements = state.get('chat_history', 'No chat history available.')
            return system_prompt_template.format(data_requirements=data_requirements)

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
        """Creates an agent that generates a prompt template based on the stored data requirements."""
        system_prompt_template = (
            """
            Based on the following data requirements, write a clear and effective prompt template:

            {data_requirements}

            The prompt should:

            - Clearly address the data objective.
            - Incorporate the variables appropriately.
            - Respect the constraints.
            - Adhere to the specified requirements.

            Store the generated prompt in the `prompt_template` field in the state.
            """
        )
        def system_prompt(state):
            # Access 'data_requirements' from the state
            data_requirements = state.get('data_requirements', 'No data requirements provided.')
            return system_prompt_template.format(data_requirements=data_requirements)

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
            You are the supervisor managing the data requirement gathering workflow. Your primary responsibility is to oversee the agents and ensure that they perform their tasks correctly and in the proper sequence.

            Here is the current state of the conversation:
            {chat_history}

            Instructions:

            1. **Start with data_gather_information**: Begin by invoking `data_gather_information` to collect data requirements from the user.
            2. **Proceed to data_prompt_generator**: Once the data requirements are collected and stored, proceed to `data_prompt_generator`.
            3. **Transition to SQL Team**: After the prompt template is generated and stored in `prompt_template`, route the workflow to `sql_generation` in the SQL team.
            4. **User Engagement**: If at any point information is missing or unclear, use `FINISH` to prompt the user for additional input.

            Here are your available options to route the conversation:

            - **data_gather_information**: Collects data requirements from the user.
            - **data_prompt_generator**: Generates a prompt template based on collected requirements.
            - **sql_generation**: Passes control to the SQL generation team.
            - **FINISH**: Prompts the user for additional input.
            """
        )
        def system_prompt(state):
            # Access 'chat_history' from the state
            chat_history = state.get('chat_history', 'No chat history available.')
            return system_prompt_template.format(chat_history=chat_history)

        data_supervisor = self.utilities.create_team_supervisor(
            self.llm,
            system_prompt_template,
            members
        )
        return data_supervisor
