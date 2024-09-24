import functools
from typing import List, TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from utilities.helper import HelperUtilities
from tools.tool_empty import placeholder_tool
from tools.tool_metadata import fetch_metadata_as_yaml
import operator

class DataRequirementTeam:
    def __init__(self, model):
        self.llm = ChatOpenAI(model=model)
        self.utilities = HelperUtilities()
        self.tools = {
            'placeholder': placeholder_tool,
            'metadata': fetch_metadata_as_yaml
        }

    def data_gather_information(self):
        """Creates an agent that captures the user's expectations for the data and stores it."""
        system_prompt_template = (
            """
            Your job is to collect the user's data requirements and expectations to create a prompt template.
            Use your tool, 'fetch_metadata_as_yaml', to gather an understanding of the database schema.

            Here is the current state of the conversation:
            {chat_history}

            You should gather the following information:

            1. **Purpose of the Data**: Understand why the user needs the data. What decision or analysis will it support?
            2. **Specific Data Needs**: Identify which specific data points the user is interested in (e.g., sales figures, customer demographics).
            3. **Time Frame**: Determine if there is a specific time frame the user is interested in (e.g., last month, Q1 2024).
            4. **Filters/Criteria**: Ask for any specific conditions that should be applied to the data (e.g., region, product category).

            Engage with the user to collect all necessary information. If any information is missing or unclear, ask the user for clarification.

            **Once all information is collected**, output the collected information as a JSON object with the following structure:

            {{
                "purpose_of_data": "user's response",
                "specific_data_needs": "user's response",
                "time_frame": "user's response",
                "filters_criteria": "user's response"
            }}

            **Do not include any code fences or extra text; output only the JSON object.**
            """
        )

        data_gather_information_agent = self.utilities.create_agent(
            self.llm,
            [self.tools['metadata']],
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

            **Output Format:**

            {{
                "generated_prompt": "Your generated prompt here."
            }}

            **Do not include any code fences or extra text; output only the JSON object.**
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

    
    def data_prompt_human_proxy(self):
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

    def data_gather_supervisor(self, members: List[str]):
        """Creates a supervisor agent that oversees the data gathering process."""
        system_prompt_template = (
            """
            You are the supervisor for managing the data requirement gathering workflow.
            Your role is to route the conversation to the user, data_gather_information agent or data_prompt_generator agent
            Here is the chat history:
            {chat_history}
            
            Here are your available options to route the conversation:
            - **data_gather_information**: Collects data requirements from the user.
            - **data_prompt_generator**: Generates a prompt template based on collected requirements.
            - **FINISH**: Forwards the data_gather_information question to the user for additional input.

            Use the messages to route the conversation accordingly.

            If the data requirements are clear, you can route the conversation to the data_prompt_generator agent to generate a prompt template.
            Here are the data requirements collected:
            {data_requirements}

            Here are some examples of messages:
            Example 1: 
            Human: "I need data on customer demographics for the last quarter."
            data_gather_information: "Could you please provide more details about the specific data points you are interested in?"
            Output: **FINISH**

            Example 2:
            Human: "Hello, how are you!"
            data_gather_information: "Hello! How can I assist you with your data needs today?"
            Output: **FINISH**

            Now, based on the current conversation, route the conversation accordingly.
            """
        )

        data_gather_supervisor = self.utilities.create_team_supervisor(
            self.llm,
            system_prompt_template,
            members
        )
        return data_gather_supervisor


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
