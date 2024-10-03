# graph_sql.py
from langgraph.graph import StateGraph, START, END
from graph_state import CombinedTeamState
from teams.team_sql import SQLTeam

class SQLTeamSubgraph:
    def __init__(self, sql_team: SQLTeam):
        self.sql_team = sql_team
        self.graph = StateGraph(CombinedTeamState)
        self.sql_team_members = ["sql_generation", "sql_execution", "sql_result_formatting"]

    def build_graph(self):
        self.graph.add_node("sql_generation", self.sql_team.sql_generation_agent())
        self.graph.add_node("sql_execution", self.sql_team.sql_execution_agent())
        self.graph.add_node("sql_result_formatting", self.sql_team.sql_result_formatting_agent())
        self.graph.add_node("sql_supervisor", self.sql_supervisor())

        self.graph.add_edge(START, "sql_generation")
        self.graph.add_edge("sql_generation", "sql_execution")
        self.graph.add_edge("sql_execution", "sql_result_formatting")
        self.graph.add_edge("sql_result_formatting", "sql_supervisor")
        self.graph.add_edge("sql_supervisor", END)

    def sql_supervisor(self):
        def supervisor_agent(state: CombinedTeamState):
            # Supervisor logic
            # Decide whether to end or loop
            state["next_subgraph"] = "END"
            return state
        return supervisor_agent

    def compile_graph(self):
        self.build_graph()
        return self.graph.compile()
