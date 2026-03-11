import operator
from typing import Annotated, TypedDict, List

class AgentState(TypedDict):
    """
    Represents the state of our multi-hop reasoning LangGraph.
    """
    question: str
    is_complex: bool
    sub_queries: List[str]
    # We use Annotated with operator.add so that parallel sub-query results get appended
    retrieved_context: Annotated[List[str], operator.add]
    hop_count: int
    final_answer: str
