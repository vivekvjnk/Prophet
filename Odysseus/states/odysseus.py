from pydantic import BaseModel, Field
from typing import List,Optional,TypedDict,Dict, Union
import networkx as nx

#---------------Bodhi Graph Extraction-Begin-------------#
class Relationship(BaseModel):
    source_entity: str = Field(..., description="Name of the source entity.")
    target_entity: str = Field(..., description="Name of the target entity.")
    description: str = Field(..., description="Explanation of the relationship between the source and target entity.")
    strength: float = Field(..., description="A numeric score indicating the strength of the relationship.", ge=0, le=10)

class RelationshipExtractionOutput(BaseModel):
    relationships: List[Relationship] = Field(..., description="A list of all relationships extracted from the text.")

class EntityValidationFeedback(BaseModel):
    reasons_to_loop: List[str]  # List of reasons why a loop-back is necessary
    loop_required: bool         # Boolean indicating if looping is needed
    additional_comments: Optional[str] = None  # Optional additional feedback or context


class Entity(BaseModel):
    name: str = Field(..., description="Capitalized name of the entity.")
    type: Optional[str] = Field(..., description="Entity type (must align with the provided or introduced types).")
    description: str = Field(..., description="Comprehensive description of the entity.")

class EntityExtractionOutput(BaseModel):
    entities: List[Entity] = Field(..., description="A list of all entities extracted from the text.")
#---------------Bodhi Graph Extraction-End---------------#

#---------------Bodhi state information-Begin-------------#
class BodhiState(TypedDict):
    source_path : str = Field("Path to input text file") 
    relationships: List[Relationship] = Field("Dictionary with list of relationships")
    entities : List[Entity] = Field("Dictionary with list of entities")
    entity_types: List[str] = Field("List of entity types")
    entity_iter_counter: int = Field("Iteration counter for EntityExtraction-Validation loop")
    relations_iter_counter: int = Field("Iteration counter for RelationExtraction-Validation loop") 
    entity_validation_feedback: EntityValidationFeedback = Field("Validation feedback for extracted entities")
    relations_validation_feedback: EntityValidationFeedback = Field("Validation feedback for extracted relations")
    text_chunks : List[str] = Field("List of text chunks extracted from document")
    number_of_chunks: int = Field("Total number of text chunks to be processed")
    chunk_index : int = Field("Current chunk index")
    ambiguities: str = Field("Any identified ambiguities in the entities and relationships")

    global_graph:nx.Graph = Field("Global graph of the document") # Graph embedding, Leiden Clustering etc
    
class BodhiInputState(TypedDict):
    source_path : str = Field("Path to input text file") 
#---------------Bodhi state information-End---------------#

#---------------GraphRAG summerization-Begin-------------#

#---------------Node summarization-Begin-------------#
class NodeSummary(BaseModel):
    """
    Represents the summary of a single node's information.
    """
    summary: str =Field("A concise summary of the node's descriptions and types.")
    types: Optional[List[str]] = Field(..., description="Entity types (must align with the provided types).")

class NodeSummarizationOutput(BaseModel):
    """
    Represents the output structure for the node summarization API.
    """
    nodes: Dict[str, NodeSummary] = Field("Mapping of node names to their summaries.")
#---------------Node summarization-End---------------#

class Findings(BaseModel):
    """
    Represent insight and its detailed description
    """
    insight: str = Field(..., description="A brief insight or observation about the community")
    description: str = Field(..., description="A detailed explanation or background of the insight")

class CommunityReport(BaseModel):
    title: str = Field(...,description="Community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.")
    key_highlights: str = Field(..., description="One or two sentences about the community's purpose.")
    summary: str = Field(..., description="An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.")
    detailed_findings: List[Findings] = Field(..., description="A list of insights about the community, each with a short summary and detailed explanation.")
    impact_severity_rating: float = Field(..., ge=0, le=10, description="A float score between 0-10 representing the severity of IMPACT posed by entities within the community.")
    rating_explanation: str = Field(..., description="A single sentence explanation of the IMPACT severity rating.")

class Highlights(BaseModel):
    key_highlights: str = Field(...,description= "Few sentences about the community's purpose.")
class SubCommunitySummary(BaseModel):
    summary: str = Field(...,description="An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.")
class Findings_summary(BaseModel):
    detailed_findings: List[Findings] = Field(..., description="A list of insights about the community, each with a short summary and detailed explanation.")
class Impact_severity(BaseModel):
    impact_severity_rating: float = Field(..., ge=0, le=10, description="A float score between 0-10 representing the severity of IMPACT posed by entities within the community.")
    rating_explanation: str = Field(..., description="A single sentence explanation of the IMPACT severity rating.")
#---------------GraphRAG summerization-End---------------#

#---------------GraphRAG LocalSearch -Begin-------------#


class CommunityMetadata(BaseModel):
    node_count: int = Field(...,description="Number of nodes in the community")
    title: str = Field(...,description="Name of the community")
    total_edges: int = Field(...,description="Number of edges in the community")

class RelevantCommunity(BaseModel):
    summary: str = Field(...,description="Summary of the community")
    metadata: CommunityMetadata = Field(...,description="Metadata of the community")


class LocalSearchState(TypedDict):
    query: str = Field(...,description="User query")
    source: str = Field(...,description="Source from which retrieval should happen")
    retrieved_entities: List[Entity] = Field(...,description="List of retrieved entities after filter")
    retrieved_text_units: List[str] = Field(...,description="List of retrieved text units")
    retrieved_communities: Dict[str,RelevantCommunity] = Field(...,description="Dictionary of relevant communities")
    entities: Dict[int,List[Entity]] = Field(...,description="Dictionary with list of retrieved entities at each level")
    relationships: Dict[int,List[Relationship]] = Field(...,description="Dictionary with list of retrieved relationships at each level")
    node_reln_level_counter:int = Field(...,description="Counter state variable to keep track of the level of graph traversal")
    level:int = Field(...,description="Variable specifying graph traversal depth for retrieval")

class RelevantEntity(BaseModel):
    entity_name: str = Field(...,description="Name of the entity")
    relevance: str = Field(...,description="Reason for inclusion. Why the entity is relevant to the query?")
class FilteredEntities(BaseModel):
    relevant_entities: List[RelevantEntity] = Field(...,description="List of relevant entities")


#---------------GraphRAG LocalSearch -End---------------#
