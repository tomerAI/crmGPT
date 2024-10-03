# graph_prompt.py
from langgraph.graph import StateGraph, START, END
from graph_state import CombinedTeamState
from teams.team_prompt import TeamPromptGenerator

class PromptTeamSubgraph:
    def __init__(self, data_team: TeamPromptGenerator):
        self.data_team = data_team
        self.graph = StateGraph(CombinedTeamState)
        self.team_members = ["data_prompt_generator"]

    def build_graph(self):
        self.graph.add_node("data_prompt_generator", self.data_team.data_prompt_generator())
        self.graph.add_node("data_prompt_supervisor", self.data_prompt_supervisor())

        self.graph.add_conditional_edges(
            "data_prompt_supervisor",
            lambda x: x["next"],
            {
                "FINISH": END,
                "data_prompt_generator": "data_prompt_generator",
            }
        )

        self.graph.add_edge(START, "data_prompt_generator")
        self.graph.add_edge("data_prompt_generator", "data_prompt_supervisor")

    def data_prompt_supervisor(self):
        def supervisor_agent(state: CombinedTeamState):
            # Supervisor logic
            # For example, if ready to proceed to SQL generation:
            if some_condition_met(state):
                state["next_subgraph"] = "sql_subgraph"
                state["next"] = "FINISH"  # Finish the subgraph
            else:
                state["next"] = "data_prompt_generator"  # Loop back
            return state
        return supervisor_agent

    def compile_graph(self):
        self.build_graph()
        return self.graph.compile()
