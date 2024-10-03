# graph_data.py
from langgraph.graph import StateGraph, START, END
from graph_state import CombinedTeamState
from typing import List
from teams.team_data import TeamDataRequirement

class DataRequirementTeamSubgraph:
    def __init__(self, data_team: TeamDataRequirement):
        self.data_team = data_team
        self.graph = StateGraph(CombinedTeamState)
        self.data_team_members = ["data_gather_information"]

    def build_graph(self):
        self.graph.add_node("data_gather_information", self.data_team.data_gather_information())
        self.graph.add_node("data_gather_supervisor", self.data_gather_supervisor())

        self.graph.add_conditional_edges(
            "data_gather_supervisor",
            lambda x: x["next"],
            {
                "FINISH": END,
                "data_gather_information": "data_gather_information",
            }
        )

        self.graph.add_edge(START, "data_gather_information")
        self.graph.add_edge("data_gather_information", "data_gather_supervisor")

    def data_gather_supervisor(self):
        def supervisor_agent(state: CombinedTeamState):
            # Supervisor logic
            # Decide next action based on state
            # For example, if ready to proceed to prompt generation:
            if ready_for_prompt_generation(state):
                state["next_subgraph"] = "prompt_subgraph"
                state["next"] = "FINISH"  # Finish the subgraph
            else:
                state["next"] = "data_gather_information"  # Loop back
            return state
        return supervisor_agent

    def compile_graph(self):
        self.build_graph()
        return self.graph.compile()
