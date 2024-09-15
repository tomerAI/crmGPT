from langgraph.graph import StateGraph, START, END
from backend_py.teams.team_sql import SQLTeam, SQLTeamState
from langchain_core.messages import HumanMessage
import functools

class PostgreSQLChain:
    def __init__(self, model):
        # Create an instance of SQLTeam
        self.agents = SQLTeam(model=model)
        self.sql_graph = StateGraph(SQLTeamState)  # Initialize the StateGraph

    def build_graph(self, members):
        """Build the PostgreSQL query execution graph by adding nodes and edges."""

        # Add nodes using agent methods
        self.sql_graph.add_node("SQLCreationAgent", self.agents.sql_creation_agent())
        self.sql_graph.add_node("SQLReviewAgent", self.agents.sql_review_agent())
        self.sql_graph.add_node("SQLExecutionAgent", self.agents.sql_execution_agent())
        self.sql_graph.add_node("OutputFormattingAgent", self.agents.output_formatting_agent())
        self.sql_graph.add_node("supervisor", self.agents.supervisor_agent(members))

        # Add conditional edges for dynamic routing
        self.sql_graph.add_conditional_edges(
            "supervisor",
            lambda x: x["next"],
            {"SQLCreationAgent": "SQLCreationAgent", 
             "SQLExecutionAgent": "SQLExecutionAgent", 
             "FINISH": END},
        )
        
        # Manual graph construction
        self.sql_graph.add_edge(START, "SQLCreationAgent")
        self.sql_graph.add_edge("SQLCreationAgent", "SQLReviewAgent")
        # Let supervisor decide the next step
        self.sql_graph.add_edge("SQLReviewAgent", "supervisor") 
        self.sql_graph.add_edge("SQLExecutionAgent", "OutputFormattingAgent")
        # Let supervisor decide the next step
        self.sql_graph.add_edge("OutputFormattingAgent", "supervisor")

    def compile_chain(self):
        """Compile the PostgreSQL query execution chain from the constructed graph."""
        return self.sql_graph.compile()

    def enter_chain(self, message: str, chain, members):
        """Enter the compiled chain with the given query and return the final summary."""
        # Create a list of HumanMessage instances
        results = [HumanMessage(content=message)]

        input_data = {
            "messages": results,
            "team_members": members
        }

        # Execute the chain by passing the messages to it
        sql_chain = chain.invoke(input_data)

        # Assuming the chain returns a list of messages, get the summary output
        if "messages" in sql_chain and sql_chain["messages"]:
            final_summary = next((msg.content for msg in sql_chain["messages"] if msg.name == "OutputFormattingAgent"), "No summary generated.")
        else:
            final_summary = "No valid messages returned from the chain."
        
        # Return the final summary as the output
        return final_summary
