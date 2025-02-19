
from pydantic import BaseModel, Field
from typing import List,Optional,TypedDict,Dict, Union, Any
import networkx as nx

#---------------base models for retrieval-Begin-------------#
class CommunityAnswer(BaseModel):
    answer: str = Field(..., description="The extracted answer based on the provided community summary.")
    relevance_score: int = Field(..., ge=1, le=10, description="Relevance score of the community summary to the query (1-10).")

    class Config:
        schema_extra = {
            "example": {
                "answer": "Quantum entanglement enhances cryptographic security through Quantum Key Distribution (QKD), ensuring secure communication by detecting eavesdropping attempts.",
                "relevance_score": 9
            }
        }

class FinalAnswer(BaseModel):
    final_answer:str = Field(...,description="Final answer in string format. Make sure to wrap output in quotes")
#---------------base models for retrieval-End---------------#

class GRGlobalSearchState(TypedDict):
    query : str = Field("Query to the Prophet system") 
    retrieval_method : str = Field("Retrieval mechanism") 
    retrievals:List[Dict[str,Any]] = Field("Retrieved information from Odysseus")
    community_answers: List[CommunityAnswer] = Field(...,description="List of Community answers")
    final_answer: str = Field(...,description="Final answer from community")
