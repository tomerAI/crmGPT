from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


class HelperUtilities:
    def __init__(self):
        pass

    def create_agent(self, llm: ChatOpenAI, tools: list, system_prompt: str) -> AgentExecutor:
        """
        Create a function-calling agent and add it to the graph.
        
        Args:
            llm: The language model to use (ChatOpenAI instance).
            tools: List of tools available to the agent.
            system_prompt: The system prompt that guides the agent's behavior.

        Returns:
            AgentExecutor: The agent executor ready to invoke the agent's chain.
        """
        system_prompt += (
            "\nWork autonomously according to your specialty, using the tools available to you."
            " Do not ask for clarification."
            " Your other team members (and other teams) will collaborate with you with their own specialties."
            " You are chosen for a reason! You are one of the following team members: {team_members}."
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        agent = create_openai_functions_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools)
        return executor

    def agent_node(self, state, agent: AgentExecutor, name: str, callback=None) -> dict:
        """
        Invoke the agent with the current state and return the result as a message.
        
        Args:
            state: The current state that the agent should use to make a decision.
            agent: The agent executor that will be invoked.
            name: The name of the agent for tracking purposes.
            callback: Optional callback function to handle the result after invocation.

        Returns:
            dict: A dictionary containing the result as a message.
        """
        result = agent.invoke(state)
        
        # If a callback is provided, execute it
        if callback:
            callback(state)
        
        return {"messages": [HumanMessage(content=result["output"], name=name)]}


    def create_team_supervisor(self, llm: ChatOpenAI, system_prompt: str, members: list) -> JsonOutputFunctionsParser:
        """
        Create an LLM-based team supervisor to route tasks to different team members.
        
        Args:
            llm: The language model to use (ChatOpenAI instance).
            system_prompt: The system prompt that guides the supervisor's behavior.
            members: List of team members who will be assigned tasks by the supervisor.

        Returns:
            JsonOutputFunctionsParser: The parser that routes tasks based on the conversation.
        """
        options = ["FINISH"] + members
        function_def = {
            "name": "route",
            "description": "Select the next role.",
            "parameters": {
                "title": "routeSchema",
                "type": "object",
                "properties": {
                    "next": {
                        "title": "Next",
                        "anyOf": [{"enum": options}],
                    },
                },
                "required": ["next"],
            },
        }
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
                (
                    "system",
                    "Given the conversation above, who should act next?"
                    " Or should we FINISH? Select one of: {options}",
                ),
            ]
        ).partial(options=str(options), team_members=", ".join(members))

        return (
            prompt
            | llm.bind_functions(functions=[function_def], function_call="route")
            | JsonOutputFunctionsParser()
        )
