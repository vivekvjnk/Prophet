from pydantic import BaseModel, Field
from typing import List,Optional,TypedDict,Dict, Union, Any
import networkx as nx


#---------------Alchemist state information-Begin-------------#
class AlchemistState(TypedDict):
    query : str = Field("Query to the Prophet system") 
    retrieval_method : str = Field("Retrieval mechanism") 
    retrievals:List[Dict[str,Any]] = Field("Retrieved information from Odysseus")
    final_answer:str = Field("Final answer from alchemist")
    
class AlchemistInputState(TypedDict):
    query : str = Field("Query to the Prophet system") 

#---------------Alchemist state information-End---------------#
