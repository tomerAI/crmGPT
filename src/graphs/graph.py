from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage
from teams.team_sql import SQLTeam
from teams.team_data import DataRequirementTeam
from langchain_core.messages import HumanMessage
from typing import List, TypedDict, Annotated, Any
import operator

class CombinedTeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    chat_history: List[str]
    team_members: List[str]  # Keeping this to ensure the team members are passed correctly
    data_team_members: List[str]
    sql_team_members: List[str]
    next: str
    data_requirements: dict  # Stores parameters collected by DataRequirementTeam
    sql_query: str
    execution_results: Any

class PostgreSQLChain:
    def __init__(self, model):
        # Create instances of both teams
        self.sql_team = SQLTeam(model=model)
        self.data_team = DataRequirementTeam(model=model)
        self.graph = StateGraph(CombinedTeamState)  # Initialize the StateGraph with combined state

        # List of team members for supervisor agents
        self.data_team_members = [
            "data_user_input",
            "data_clarification",
            "data_schema_mapping"
        ]
        self.sql_team_members = [
            "sql_generation",
            "sql_validation",
            "sql_execution",
            "sql_result_formatting"
        ]
        self.team_members = self.data_team_members + self.sql_team_members

    def build_graph(self):
        """Build the combined data requirement and SQL execution graph."""
        # Add nodes for DataRequirementTeam agents
        self.graph.add_node("data_user_input", self.data_team.data_user_input_agent())
        self.graph.add_node("data_clarification", self.data_team.data_clarification_agent())
        self.graph.add_node("data_schema_mapping", self.data_team.data_schema_mapping_agent())
        self.graph.add_node("data_supervisor", self.data_team.data_supervisor(self.data_team_members))

        # Add nodes for SQLTeam agents
        self.graph.add_node("sql_generation", self.sql_team.sql_generation_agent())
        self.graph.add_node("sql_validation", self.sql_team.sql_validation_agent())
        self.graph.add_node("sql_execution", self.sql_team.sql_execution_agent())
        self.graph.add_node("sql_result_formatting", self.sql_team.sql_result_formatting_agent())
        self.graph.add_node("sql_supervisor", self.sql_team.sql_supervisor(self.sql_team_members))

        # Add conditional edges for dynamic routing
        self.graph.add_conditional_edges(
            "data_supervisor",
            lambda x: x["next"],
            {"data_user_input": "data_user_input",
             "data_clarification": "data_clarification", 
             "data_schema_mapping": "data_schema_mapping",
             "FINISH": END,
             "sql_generation": "sql_generation"}
        )

        # Build the workflow graph
        # DataRequirementTeam workflow
        self.graph.add_edge(START, "data_supervisor")
        self.graph.add_edge("data_user_input", "data_supervisor")
        self.graph.add_edge("data_clarification", "data_supervisor")
        self.graph.add_edge("data_schema_mapping", "data_supervisor")
    
        # SQLTeam workflow
        self.graph.add_edge("sql_generation", "sql_validation")
        self.graph.add_edge("sql_validation", "sql_execution")
        self.graph.add_edge("sql_execution", "sql_result_formatting")
        self.graph.add_edge("sql_result_formatting", "sql_supervisor")
        self.graph.add_edge("sql_supervisor", END)

    def compile_chain(self):
        """Compile the combined chain from the constructed graph."""
        return self.graph.compile()

    def enter_chain(self, message: str, chain, conversation_history: List[str]):
        """Enter the compiled chain with the user's message and the chat history, and return the final summary."""
        # Initialize messages with the user's input
        results = [HumanMessage(content=message)]
        
        input_data = {
            "messages": results,
            "chat_history": conversation_history,
            "team_members": self.team_members,  # Ensuring team members are passed correctly
            "agent_scratchpad": "",  # Add any additional placeholders needed by the agents
            "intermediate_steps": [],  # Ensure all expected variables are present
            "data_requirements": {},
            "sql_query": "",
            "execution_results": None,
            "next": None
        }

        # Execute the chain by invoking it with the input data
        chain_result = chain.invoke(input_data)

        if "messages" in chain_result and chain_result["messages"]:
            # Extract the final output from the messages
            final_output = chain_result["messages"][-1].content
        else:
            final_output = "No valid messages returned from the chain."

        return final_output

        """# Extract the final summary from the sql_result_formatting agent
        if "messages" in chain_result and chain_result["messages"]:
            final_summary = next(
                (msg.content for msg in chain_result["messages"] if msg.name == "sql_result_formatting"),
                "No summary generated."
            )
        else:
            final_summary = "No valid messages returned from the chain."

        return final_summary
"""