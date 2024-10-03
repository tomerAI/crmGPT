# graph_parent.py
from langgraph.graph import StateGraph, START, END
from graph_state import CombinedTeamState
from teams.team_sql import SQLTeam
from teams.team_data import DataRequirementTeam
from graph_data import DataRequirementTeamSubgraph
from graph_prompt import PromptTeamSubgraph
from graph_sql import SQLTeamSubgraph

class ParentGraph:
    def __init__(self, model):
        self.sql_team = SQLTeam(model=model)
        self.data_team = DataRequirementTeam(model=model)
        
        # Initialize subgraphs
        self.data_subgraph = DataRequirementTeamSubgraph(self.data_team).compile_graph()
        self.prompt_subgraph = PromptTeamSubgraph(self.data_team).compile_graph()
        self.sql_subgraph = SQLTeamSubgraph(self.sql_team).compile_graph()

        self.graph = StateGraph(CombinedTeamState)

    def build_graph(self):
        # Add subgraphs as nodes
        self.graph.add_node("data_subgraph", self.data_subgraph)
        self.graph.add_node("prompt_subgraph", self.prompt_subgraph)
        self.graph.add_node("sql_subgraph", self.sql_subgraph)

        # Add conditional edges based on state["next_subgraph"]
        self.graph.add_conditional_edges(
            "data_subgraph",
            lambda x: x.get("next_subgraph"),
            {
                "prompt_subgraph": "prompt_subgraph",
                "END": END,
            }
        )

        self.graph.add_conditional_edges(
            "prompt_subgraph",
            lambda x: x.get("next_subgraph"),
            {
                "sql_subgraph": "sql_subgraph",
                "END": END,
            }
        )

        self.graph.add_conditional_edges(
            "sql_subgraph",
            lambda x: x.get("next_subgraph"),
            {
                "END": END,
            }
        )

        # Start the graph
        self.graph.add_edge(START, "data_subgraph")

    def compile_graph(self):
        self.build_graph()
        return self.graph.compile()

    def enter_chain(self, message: str, chain, conversation_history: List[dict]):
        # Initialize messages with the user's input
        results = [HumanMessage(content=message)]

        input_data = {
            "messages": results,
            "chat_history": conversation_history,
            "team_members": self.data_team_members + self.sql_team_members,
            "data_team_members": self.data_team_members,
            "sql_team_members": self.sql_team_members,
            "agent_scratchpad": "",
            "intermediate_steps": [],
            "data_requirements": [],
            "generated_prompt": "",
            "sql_query": "",
            "execution_results": None,
            "next": None,
            "next_subgraph": None
        }

        # Execute the chain by invoking it with the input data
        chain_result = chain.invoke(input_data)

        if "messages" in chain_result and chain_result["messages"]:
            # Extract the final output from the messages
            final_output = chain_result["messages"][-1].content
        else:
            final_output = "No valid messages returned from the chain."

        return final_output
