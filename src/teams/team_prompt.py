import functools
from typing import List, TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from utilities.helper import HelperUtilities
from tools.tool_empty import placeholder_tool
from tools.tool_metadata import fetch_metadata_as_json
import operator

class TeamPromptGenerator:
    def __init__(self, model):
        self.llm = ChatOpenAI(model=model)
        self.utilities = HelperUtilities()
        self.tools = {
            'placeholder': placeholder_tool,
            'metadata': fetch_metadata_as_json
        }    

    def prompt_generator(self):
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

            **Output Format:**

            {{
                "generated_prompt": "Your generated prompt here."
            }}

            **Do not include any code fences or extra text; output only the JSON object.**
            """
        )

        prompt_generator_agent = self.utilities.create_agent(
            self.llm,
            [self.tools['placeholder']],
            system_prompt_template
        )
        return functools.partial(
            self.utilities.agent_node,
            agent=prompt_generator_agent,
            name="prompt_generator"
        )


    def prompt_human_proxy(self):
        """Creates a human proxy agent that optimizes the generated prompt based on human feedback."""
        system_prompt_template = (
            """
            You are a human proxy agent responsible for optimizing the generated prompt.
            You optimize the prompt by using the following information:
            Chat History: {chat_history}
            Data Requirements: {data_requirements}
            Generated Prompt: {generated_prompt}
            """
        )
        prompt_human_proxy_agent = self.utilities.create_agent(
            self.llm,
            [self.tools['placeholder']],
            system_prompt_template
        )
        return functools.partial(
            self.utilities.agent_node,
            agent=prompt_human_proxy_agent,
            name="prompt_human_proxy"
        )


    def data_prompt_supervisor(self, members: List[str]):
        """Creates a supervisor agent that oversees the prompt generation process."""
        system_prompt_template = (
            """
            You are the supervisor for managing the prompt generation workflow.
            Your role is to route the conversation forward to the **sql_generation** team.

            Here is the current state of the conversation:
            {chat_history}

            Here are your available options to route the conversation:
            - **data_prompt_generator**: Generates a prompt template based on collected requirements.
            - **sql_generation**: Passes control to the SQL generation team.

            Now, based on the current conversation, route the conversation accordingly.
            """
        )

        data_prompt_supervisor = self.utilities.create_team_supervisor(
            self.llm,
            system_prompt_template,
            members
        )
        return data_prompt_supervisor