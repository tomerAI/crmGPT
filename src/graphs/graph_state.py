# state.py
from typing import List, TypedDict, Annotated, Any
from langchain_core.messages import BaseMessage
import operator

class CombinedTeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    chat_history: List[str]
    team_members: List[str]
    data_team_members: List[str]
    sql_team_members: List[str]
    next: str
    next_subgraph: str
    agent_scratchpad: str
    data_requirements: List[str]
    generated_prompt: str
    sql_query: str
    execution_results: Any
    intermediate_steps: List[str]
    metadata: List[dict]
