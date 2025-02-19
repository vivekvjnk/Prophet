"""
GraphRAG Local Search Mechanism
================================
Source: https://microsoft.github.io/graphrag/query/local_search/

Overview
---------
The local search method identifies a set of entities from the knowledge graph that are semantically 
related to the user input. These entities serve as access points into the knowledge graph, enabling 
the extraction of further relevant details such as:
    - Connected entities
    - Relationships
    - Entity covariates
    - Community reports
Additionally, it retrieves relevant text chunks from the raw input documents associated with the 
identified entities. These candidate data sources are then prioritized and filtered to fit within 
a single context window of predefined size, which is used to generate a response.

Conceptualization
-----------------
Scope of **Alchemist**:
    - (Optional) Identify the vector databases to be searched.
    - Pass the user query to Odysseus for retrieval.
    - Collect the retrieved information from Odysseus.
    - Aggregate the retrieved knowledge into a coherent response.

"""

from typing import List, Dict, TypedDict, Union, Optional, Literal
from pydantic import BaseModel, Field

from langgraph.checkpoint.memory import MemorySaver
from langchain.output_parsers import YamlOutputParser
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import (
                                    ChatPromptTemplate,
                                    MessagesPlaceholder,
                                    SystemMessagePromptTemplate,
                                    HumanMessagePromptTemplate,
                                    AIMessagePromptTemplate,
                                    PromptTemplate)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langchain_ollama import OllamaLLM

from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver


import networkx as nx
import uuid

# from The_Prophet.Odysseus.states.bodhi import *
from infra.utils.utils import *
from infra.utils.cluster import *


#---------------Pipeline specific imports-Begin-------------#
from ..base import *
from .graphrag_state import *
#---------------Pipeline specific imports-End---------------#



class GraphRAGLocalSearch:
    def __init__(self,**kwargs):
        self.ipc = None
        if 'ipc' in kwargs:
            self.ipc=kwargs.get('ipc')

        # Langfuse
        self.parent_trace_id = kwargs.get('parent_trace_id')
        # Prompt related 
        self.prompt_dict = get_settings()
        config = Config()
        expt_llm = config["llm_model"]
        self.llm = OllamaLLM(temperature=0.2, model=expt_llm)

        # For persistence 
        self.checkpointer = MemorySaver()
        self.config = {"configurable": {"thread_id": uuid.uuid4()}}

        # graph related 
        self.workflow = StateGraph(GRGlobalSearchState
            # TODO: Add state once design is mature enough 
        )
        self.app = None
        self._setup_graph()

    def __call__():
        print(f"Hello from GRGlobalSearch\n")

    def invoke(self,state:GRGlobalSearchState):
        self.app.invoke(state,self.config)
    
    def get_state(self):
        return self.app.get_state(self.config)

    def _setup_graph(self):
        # Graph structure code 
        self.workflow.add_node("retrieve_info",self.retrieve_info)
        self.workflow.add_node("combine_answers",self.combine_answers)
        # self.workflow.add_node("gen_intermediate_answers",self.gen_intermediate_answers)

        self.workflow.add_edge(START,"retrieve_info")
        self.workflow.add_edge("retrieve_info","combine_answers")
        self.workflow.add_edge("combine_answers",END)
    
        self.app = self.workflow.compile(checkpointer = self.checkpointer)

    def retrieve_info(self,state:GRGlobalSearchState):
        '''
        input format 
        ---------

        Objective of this node
        -----------------------
        Retrieve relevant information from Odysseus based on the query 
        Steps 
        -----
        1. Read query from the state
        2. Create IPC(zeromq) query in proper format 
        3. Call IPC and collect response from Odysseus 
        4. Update loc_state with the response and return state
        Completed
        '''
        print("--Hello from get_community_summaries--\n")
        loc_state = state
        ody_results = {}
        if self.ipc:
            # print(f"{loc_state}")
            ody_results = self.ipc(loc_state)
            print(f"results from localsearch\n{"*"*30}\n{ody_results}")
            results = ody_results['retrievals']
            loc_state['retrievals'] = results
        
        #---------------new code-Begin-------------#
        
        relationships = results['relationships']
        for reln,item in relationships.items():
            relationships[reln] = self.reconstruct_relations(item)
        results['relationships'] = relationships
        #---------------new code-End---------------#

        final_results = self.format_graph_retrieval_results(results)
        print(f"Prompt\n{"*"*40}\n{final_results}")
        
        input_var_names = ["hierarchical_relationships","hierarchical_entities","retrieved_communities","query"]
        relationships = final_results['hierarchical_relationships']
        entities = final_results['hierarchical_entities']
        communities = final_results['retrieved_communities']
        query = final_results['query']
        text_units ="\n -".join(results['retrieved_text_units'])
        
        prompt_template = SystemMessagePromptTemplate.from_template(self.prompt_dict['local_search_answer_gen']['system'])
        chat_prompt_template = ChatPromptTemplate.from_messages([prompt_template])
        messages = chat_prompt_template.format_messages(text_units=text_units,entities=entities,relationships=relationships,query=query,communities=communities)
        # ollama_runnable = MultiOllamaClient(instances={"phi4:latest":"http://192.168.13.13:11434"},temperature=0.2)
        client = ollama.Client(host="http://192.168.13.13:11434")
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                raise ValueError(f"Unexpected message type: {type(msg)}")

            formatted_messages.append({"role": role, "content": msg.content})

        results = client.chat(
            model="phi4:latest", messages=formatted_messages, options={"temperature": 0.3}
        )
        print(results['message'])
        # process the results
        return loc_state
    
    def format_graph_retrieval_results(self,results):
        """Formats graph retrieval results into structured strings for LLM prompts."""

        query = results.get('query', 'No query provided')
        source = results.get('source', 'Unknown source')

        # Format Retrieved Entities
        retrieved_entities = results.get('retrieved_entities', {})
        retrieved_entities_formatted = "\n".join(
            # [f"- {name} (Type: {data['type']}, Distance: {data['distance']}, Community: {data['community']})\n  Description: {data['description']}"
            [f"- {name} (Type: {data['type']})\n  Description: {data['description']}"
            for name, data in retrieved_entities.items()]
            
        ) if retrieved_entities else "No entities retrieved."
        # Format Retrieved Communities
        retrieved_communities = results.get('retrieved_communities', {})
        retrieved_communities_formatted = "\n".join(
            # [f"- {name} (Nodes: {data['metadata']['node_count']}, Edges: {data['metadata']['total_edges']})\n  Title: {data['metadata']['title']}"
            [f"- {data['metadata']['title']}: {data['summary']}\n"
            for name, data in retrieved_communities.items()]
        ) if retrieved_communities else "No communities retrieved."

        # Format Retrieved Text Units
        retrieved_text_units = set(results.get('retrieved_text_units', []))
        retrieved_text_units_formatted = "\n".join(
            [f"- {text}" for text in retrieved_text_units]
        ) if retrieved_text_units else "No relevant text retrieved."

        # Format Hierarchical Entities
        entities = results.get('entities', {})
        # print(f"Entities:{entities}")
        # entities_formatted = "\n".join(
        #     [f"Level {level}:\n" + "\n".join(
        #         [f"  - {name} (Type: {data['type']}, Community: {data['community']})\n    Description: {data['description']}"
        #         # [f"  -{name}:{data}"
        #         for name, data in entity_dict.items()]
        #     ) for level, entity_dict in entities.items()]
        # ) if entities else "No hierarchical entities found."
        # For now only retrieve 1st level entities
        entities_formatted = "\n".join(
            [f"  - {name} (Type: {data['type']}, Community: {data['community']})\n    Description: {data['description']}"
                for name, data in entities['0'].items()]
        ) if entities else "No hierarchical entities found."

        # print(f"Formatted entities: {entities_formatted}")
        # Format Hierarchical Relationships
        relationships = results.get('relationships', {})
        # relationships_formatted = "\n".join(
        #     [f"Level {level}:\n" + "\n".join(
        #         [f"  - {name}: {data['description']}" for name, data in relationship_dict.items()]
        #     ) for level, relationship_dict in relationships.items()]
        # ) if relationships else "No hierarchical relationships found."
        # For now only retrieve 1st level relationships
        relationships_formatted = f"\n".join(
                [f"  - {name}: {data['description']}" for name, data in relationships['0'].items()]
        ) if relationships else "No hierarchical relationships found."
        
        return {
            "query": query,
            "source": source,
            "retrieved_entities": retrieved_entities_formatted,
            "retrieved_communities": retrieved_communities_formatted,
            "retrieved_text_units": retrieved_text_units_formatted,
            "hierarchical_entities": entities_formatted,
            "hierarchical_relationships": relationships_formatted
        }
    

    def reconstruct_relations(self,retrieved_docs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
            """
            TODO : This has to be moved back to Odysseus module. Implemented here for rapid prototyping
            Reconstructs the relations dictionary from a ChromaDB retrieval response.

            This function extracts entity names, types, descriptions, and retrieval distances
            to form a structured dictionary.

            Args:
                retrieved_docs (Dict[str, Any]): The dictionary returned from a ChromaDB retrieval query.
                distances (List[float], optional): A list of distances (lower means higher similarity).
                    - If provided, it overrides the distances extracted from `retrieved_docs`.

            Returns:
                Dict[str, Dict[str, Any]]: A dictionary where each key is an entity name, and the value contains:
                    - "type": The entity classification.
                    - "distance": The similarity score (lower is better).
                    - "description": A detailed explanation of the entity.

            Example Output:
                ```json
                {
                    "TOOLNODE": {
                        "type": "TECHNOLOGIES AND FRAMEWORKS",
                        "description": "A prebuilt component in LangGraph that handles tools returning `Command` objects."
                    }
                }
                ```
            """
            relationships = {}

            ids_list = retrieved_docs.get("ids", [[]])  # List of entity names
            documents_list = retrieved_docs.get("documents", [[]])  # List of descriptions
            metadatas_list = retrieved_docs.get("metadatas", [[]])  # List of metadata dictionaries
           
            num_results = min(len(ids_list), len(documents_list), len(metadatas_list))

            for i in range(num_results):
                relationship_name = ids_list[i]
                relationship_type = metadatas_list[i].get("type", "UNKNOWN") if metadatas_list else "UNKNOWN"
                weight = metadatas_list[i].get("weight", "UNKNOWN") if metadatas_list else "UNKNOWN"
                description = documents_list[i]

                relationships[relationship_name] = {
                    "type": relationship_type,
                    "description": description,
                    "weight":weight
                }

            return relationships
    

    def combine_answers(self,state:GRGlobalSearchState):
        '''
        '''
        loc_state = state
        
        return loc_state
    

if __name__ == "__main__":
    local_search = GraphRAGLocalSearch(parent_trace_id=113)
    local_search.invoke({"query":"hello"})
