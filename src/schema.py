from typing import List, Literal
from pydantic import BaseModel, Field

NodeType = Literal["Person", "Company", "Event", "Concept", "Document"]

# Base node schema to enforce ontology
class BaseNode(BaseModel):
    id: str = Field(description="Unique identifier for the node (usually its name, upper-cased)")
    type: NodeType = Field(description="The category of the entity, strictly enforced.")
    properties: dict = Field(default_factory=dict, description="Additional metadata about the entity")

# Dynamic relationship schema
class Relationship(BaseModel):
    source_id: str = Field(description="ID of the source node")
    target_id: str = Field(description="ID of the target node")
    type: str = Field(description="The relationship label (UPPERCASE like ACQUIRED, WORKS_FOR). The LLM can generate this dynamically.")
    properties: dict = Field(default_factory=dict, description="Additional context about the relationship")

class KnowledgeGraphExtraction(BaseModel):
    """Pydantic schema to enforce strict node types while allowing dynamic relationships."""
    nodes: List[BaseNode] = Field(description="List of extracted entities matching the strict base ontology")
    relationships: List[Relationship] = Field(description="List of relationships connecting the extracted nodes")
