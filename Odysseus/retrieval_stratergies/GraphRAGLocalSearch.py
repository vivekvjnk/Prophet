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
Scope of **Odysseus**:
    - Preparation of vector databases for:
        1. **Entities** - Stores embeddings of knowledge graph entities.
        2. **Relationships** - Stores embeddings of relationships between entities.
        3. **Community Information** - Stores embeddings of detected entity clusters.
        4. **Text Chunks** - Stores embeddings of document fragments.
    - Persistent storage and management of these vector databases.
    - Retrieval capabilities from these vector stores, either in sequence or in parallel.
        1. Given a query, retrieve all entities related to the query
            - Retrieve top 10(need to optimize) entities from **Entities** vector store
            - Serves as an access point to the graph knowledge base
        2. Use extracted entities for further extraction 
"""
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from typing import List,Dict, Any

import os
import io
import torch
import yaml
import uuid
#---------------langgraph imports-Begin-------------#
from infra.prompts.config_loader import get_settings
from langchain.output_parsers import YamlOutputParser
from langgraph.graph import StateGraph, START, END
from typing import List, Dict, TypedDict, Union, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import (
                                    ChatPromptTemplate,
                                    MessagesPlaceholder,
                                    SystemMessagePromptTemplate,
                                    HumanMessagePromptTemplate,
                                    AIMessagePromptTemplate,
                                    PromptTemplate)

from langchain_ollama import OllamaLLM

from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver

import networkx as nx
import uuid

from infra.utils.utils import *
from infra.utils.cluster import *
from ..states.odysseus import *
#---------------langgraph imports-End---------------#
#---------------Pipeline specific imports-Begin-------------#
from ..base import *
#---------------Pipeline specific imports-End---------------#

class LocalSearchUtils():
    def __init__(self,source,client,storage):
        self.client = client
        self.source = source
        self.storage = storage
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                                                    model_name="all-MiniLM-L6-v2")
        self.collections_dict = self.setup_vector_databases(client=self.client)
        self.collections_dict = self.load_collections(source=self.source,collections_dict=self.collections_dict)

    def setup_vector_databases(self,client):
        text_units_collection = client.get_or_create_collection(name="text_units",embedding_function=self.embedding_function)
        entities_collection = client.get_or_create_collection(name="entities",embedding_function=self.embedding_function)
        entity_descriptions = client.get_or_create_collection(name="entity_descriptions",embedding_function=self.embedding_function)
        relationships_collection = client.get_or_create_collection(name="relationships",embedding_function=self.embedding_function)
        community_summaries_collection = client.get_or_create_collection(name="community_summaries",embedding_function=self.embedding_function)
        return {"text_units":text_units_collection,"entities":entities_collection,"entity_descriptions":entity_descriptions,
                "relationships":relationships_collection,"community_summaries":community_summaries_collection}

    def load_collections(self,source,collections_dict):
        
        artifacts_path  = "Odysseus"
        bodhi_artifacts_path = "Bodhi"


        community_path  = f"graphRAG/{source}/{source}_communities.pt"
        graph_path      = f"{source}/{source}_graph.yml"
        text_units_path = f"{source}/{source}_text_units.yml"

        data_bytes = self.storage.read_file(community_path,artifacts_path)
        buffer = io.BytesIO(data_bytes)
        all_communities = torch.load(buffer)
       
        graph_bytes = self.storage.read_file(graph_path,bodhi_artifacts_path)
        graph = yaml.safe_load(graph_bytes)
        
        text_units_bytes = self.storage.read_file(text_units_path,bodhi_artifacts_path)
        text_units = yaml.safe_load(text_units_bytes)
       

        entities = {}
        for key,community in all_communities.items():
            if(key in ['needs_node_summary','needs_community_summary']):
                continue
            for entity_name, entity_data in community['nodes'].items():
                entity_data['community'] = key  # Embed the community name inside the entity data
                entities[entity_name] = entity_data
        relationships = graph['links']
        
        # Logic to check if the collections contain any values or not. 

        collections_dict['community_summaries'] = self.store_communities(collection=collections_dict['community_summaries'],communities=all_communities)
        collections_dict['entities']= self.store_entities(collection=collections_dict['entities'],entities=entities)
        collections_dict['entity_descriptions'] = self.store_entity_descriptions(collection=collections_dict['entity_descriptions'],entities=entities)
        collections_dict['relationships'] = self.store_relationships(collection=collections_dict['relationships'],relationships=relationships)
        collections_dict['text_units'] = self.store_text_units(collection=collections_dict['text_units'],text_unit_list=text_units)
        return collections_dict

    #---------------store-Begin-------------#
    def store_text_units(self,collection, text_unit_list: List[str]):
        """
        Stores text units as ChromaDB documents in the given collection.

        Args:
            collection: ChromaDB collection where text units will be stored.
            text_unit_list (List[str]): List of extracted text units.

        Returns:
            The updated ChromaDB collection.
        """
        # Handle empty input
        if not text_unit_list:
            return collection  # No change
        existing_docs = collection.get(include=["documents"])
        existing_ids = existing_docs["ids"] if existing_docs and "ids" in existing_docs else []
        # Generate document IDs using the text unit index
        doc_ids = [str(i) for i in range(len(text_unit_list))]
        missing_ids = [item for item in doc_ids if item not in existing_ids]
        print(f"Missing text_unit ids:{missing_ids}")
        if(missing_ids):
            # Store text units in ChromaDB
            collection.upsert(
                ids=doc_ids,
                documents=text_unit_list
            )
        return collection

    def store_entity_descriptions(self,collection, entities: Dict[str, Dict[str, str]]):
        """
        Stores entity nodes into a ChromaDB collection.

        Args:
            collection: The ChromaDB collection where entities will be stored.
            entities (Dict[str, Dict[str, str]]): A dictionary where:
                - Each key is a node name (acting as the node ID).
                - Each value is a dictionary containing:
                    - "description" (str): A detailed description of the entity.
                    - "summary" (str): A concise summary of the description.
                    - "type" (str): The category/type of the entity.

        Processing:
        - Each entity is converted into a document format for storage.
        - The node name is used as the unique document ID.
        - The "type" field is attached as metadata.
        - The stored document consists of the concatenated "summary" and "description".

        Returns:
            The updated ChromaDB collection.
        """

        # Handle empty input
        if not entities:
            return collection  # No change
        existing_docs = collection.get(include=["documents"])
        existing_ids = existing_docs["ids"] if existing_docs and "ids" in existing_docs else []

        # Prepare data for batch insertion
        doc_ids = list(entities.keys())  # Node names as IDs
        missing_doc_ids = [item for item in doc_ids if item not in existing_ids]
        print(f"Following are the missing_doc_ids:{missing_doc_ids}")
        # documents = [f"{entities[id]['summary']}\n{entities[id]['description']}" for id in missing_doc_ids]
        documents = [f"{entities[id]['summary']}" for id in missing_doc_ids]
        metadata_list = [{"type": entities[id]["type"],"community":entities[id]["community"]} for id in missing_doc_ids]

        if(missing_doc_ids):
            # Store entities in ChromaDB
            collection.upsert(
                ids=doc_ids,
                documents=documents,
                metadatas=metadata_list
            )
        return collection

    def store_entities(self,collection,entities):
        # Handle empty input
        if not entities:
            return collection  # No change
        existing_docs = collection.get(include=["documents"])
        existing_ids = existing_docs["ids"] if existing_docs and "ids" in existing_docs else []

        # Prepare data for batch insertion
        doc_ids = list(entities.keys())  # Node names as IDs
        missing_doc_ids = [item for item in doc_ids if item not in existing_ids]

        print(f"Following are the missing_doc_ids:{missing_doc_ids}")

        documents = [f"{entity}" for entity in entities.keys()]
        metadata_list = [{"type": entities[id]["type"]} for id in missing_doc_ids]

        if(missing_doc_ids):
            # Store entities in ChromaDB
            collection.upsert(
                ids=doc_ids,
                documents=documents,
                metadatas=metadata_list
            )
        return collection

    def store_relationships(self,collection, relationships: List[Dict[str, str]]):
        """
        Stores relationship data into a ChromaDB collection.

        Args:
            collection: The ChromaDB collection where relationships will be stored.
            relationships (List[Dict[str, str]]): A list of relationship dictionaries, 
                where each dictionary contains:
                - "description" (str): A textual explanation of the relationship.
                - "source" (str): The starting node of the relationship.
                - "target" (str): The ending node of the relationship.
                - "weight" (float): A numerical value representing the relationship strength.

        Processing:
        - First, check which relationship IDs are **already stored** in the collection.
        - Filter out existing IDs and insert **only missing relationships**.
        - The unique ID for each relationship is generated in the format: "(<source>,<target>)".
        - The "description" field is stored as the document content.
        - The metadata includes:
            - "source" (str): The starting node.
            - "target" (str): The ending node.
            - "weight" (float): Strength of the relationship.

        Returns:
            The updated ChromaDB collection.
        """

        # Handle empty input
        if not relationships:
            return collection  # No change

        # Step 1: Get existing document IDs in the collection
        existing_docs = collection.get(include=["documents"])
        existing_ids = set(existing_docs["ids"]) if existing_docs and "ids" in existing_docs else set()

        doc_ids = []
        documents = []
        metadata_list = []

        # Step 2: Filter out already existing relationships
        for rel in relationships:
            if "description" in rel and "source" in rel and "target" in rel and "weight" in rel:
                # Generate unique relationship ID
                rel_id = f"({rel['source']},{rel['target']})"

                if rel_id in existing_ids:
                    continue  # Skip existing relationships

                # Append only missing relationships
                doc_ids.append(rel_id)
                documents.append(rel["description"])
                metadata_list.append({
                    "source": rel["source"],
                    "target": rel["target"],
                    "weight": rel["weight"]
                })

        # Step 3: Store only new relationships in ChromaDB
        if doc_ids:  # Only store if there are new relationships
            collection.upsert(
                ids=doc_ids,
                documents=documents,
                metadatas=metadata_list
            )

        return collection

    def store_communities(self,collection, communities: Dict[str, Dict[str, str]]):
        """
        Stores community data into a ChromaDB collection.

        Args:
            collection: The ChromaDB collection where communities will be stored.
            communities (Dict[str, Dict[str, str]]): A dictionary containing various keys,
                but only the following keys are of interest:
                - "title" (str): The name or title of the community (used as metadata).
                - "summary" (str): A brief description summarizing the community.
                - "metadata" (dict): A dictionary containing:
                    - "community_id" (int): A unique ID derived from the GraphRAG community.
                    - "node_count" (int): Number of nodes in the community.
                    - "internal_edge_count" (int): Number of internal edges.
                    - "external_edge_count" (int): Number of external edges.
                    - "list_of_nodes" (list(str)): List of member nodes
        Processing:
        - First, check which community IDs are **already stored** in the collection.
        - Filter out existing IDs and insert **only missing communities**.
        - The "community_id" is used as the **unique document ID**.
        - The "summary" is stored as the document content.
        - Metadata includes:
            - "title" (str): The title of the community.
            - "node_count" (int): Total number of nodes.
            - "total_edges" (int): Sum of "internal_edge_count" and "external_edge_count".

        Returns:
            The updated ChromaDB collection.
        """
        # Handle empty input
        if not communities:
            return collection  # No change

        # Step 1: Get existing document IDs in the collection
        existing_docs = collection.get(include=["documents"])
        existing_ids = existing_docs["ids"] if existing_docs and "ids" in existing_docs else []
        missing_ids = communities.keys() - existing_ids
        print(f"Missing communities in the collection: {missing_ids}")

        doc_ids = []
        documents = []
        metadata_list = []

        # Step 2: Filter out already existing communities
        for community_id in missing_ids:
            data = communities[community_id]
            if "title" in data and "summary" in data and "metadata" in data: # avoid other keys except communities from the dictionary 
                # Convert community_id to string to match stored document IDs
                community_id_str = str(community_id)
                title = data["title"]
                summary = data["summary"]
                metadata = data["metadata"]
                metadata["nodes"] = data["nodes"].keys() 

                # Extract node count and total edges
                node_count = metadata.get("node_count", 0)
                total_edges = metadata.get("internal_edge_count", 0) + metadata.get("external_edge_count", 0)

                # Append only missing communities
                doc_ids.append(community_id_str)
                documents.append(summary)
                metadata_list.append({"node_count": node_count, "title": title, "total_edges": total_edges})

        # Step 3: Store only new communities in ChromaDB
        if doc_ids:  # Only store if there are new communities
            print(f"Adding following doc_ids to collection: {doc_ids}")
            collection.upsert(
                ids=doc_ids,
                documents=documents,
                metadatas=metadata_list
            )

        return collection
        #---------------store-End---------------#
    #---------------search-Begin-------------#
    def search_text_units(self,collection, query: str, k: int = 3) -> List[str]:
        """
        Given a query text, retrieve the top k most relevant text units from the collection.

        Args:
            collection: The ChromaDB collection for text units.
            query (str): The search query.
            k (int): The number of top results to retrieve.

        Returns:
            List[str]: A list of top k matching text units.
        """
        results = collection["text_units"].query(query_texts=[query], n_results=k)
        return results["documents"][0] if results and "documents" in results else []

    def search_entities(self, query: str,collection_dict: Dict[str, Any]= None, k: int = 3) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Searches for the most relevant entities based on a given query.

        This function performs vector similarity search on both:
        1. **Entity Names Collection** – Finds entities directly by name.
        2. **Entity Descriptions Collection** – Finds entities based on their descriptions.

        The results from both sources are merged using **harmonic mean** to prioritize entities that appear in both searches.

        Args:
            collection_dict (Dict[str, Any]): A dictionary of ChromaDB collections containing:
                - "entities": ChromaDB collection for entity names.
                - "entity_descriptions": ChromaDB collection for entity descriptions.
            query (str): The search query used for entity retrieval.
            k (int): The number of top results to retrieve from each search (default: 3).

        Returns:
            Dict[str, Dict[str, Dict[str, Any]]]: A dictionary containing:
                - "entity_name_store": Entities retrieved based on names.
                - "entity_desc_store": Entities retrieved based on descriptions.
                Each entity is stored as:
                ```json
                {
                    "<entity_name>": {
                        "type": "<entity_type>",
                        "distance": <retrieval_distance>,
                        "description": "<entity_description>"
                    }
                }
                ```

        Example:
            ```python
            collection_dict = {
                "entities": chromadb.PersistentClient().get_collection("entity_names"),
                "entity_descriptions": chromadb.PersistentClient().get_collection("entity_descriptions"),
            }
            results = search_entities(collection_dict, "AI research", k=3)
            print(results)
            ```
        """

        def reconstruct_entities(retrieved_docs: Dict[str, Any], distances: List[float] = []) -> Dict[str, Dict[str, Any]]:
            """
            Reconstructs the entities dictionary from a ChromaDB retrieval response.

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
                        "distance": 0.6034,
                        "description": "A prebuilt component in LangGraph that handles tools returning `Command` objects."
                    }
                }
                ```
            """

            entities = {}

            if distances:
                ids_list = retrieved_docs.get("ids", [[]])  # List of entity names
                documents_list = retrieved_docs.get("documents", [[]])  # List of descriptions
                metadatas_list = retrieved_docs.get("metadatas", [[]])  # List of metadata dictionaries
                distances_list = distances  # Use provided distances
            else:
                ids_list = retrieved_docs.get("ids", [[]])[0]
                documents_list = retrieved_docs.get("documents", [[]])[0]
                metadatas_list = retrieved_docs.get("metadatas", [[]])[0]
                distances_list = retrieved_docs.get("distances", [[]])[0]

            num_results = min(len(ids_list), len(documents_list), len(metadatas_list), len(distances_list))

            for i in range(num_results):
                entity_name = ids_list[i]
                entity_type = metadatas_list[i].get("type", "UNKNOWN") if metadatas_list else "UNKNOWN"
                community = metadatas_list[i].get("community","UNKNOWN") if metadatas_list else "UNKNOWN"
                distance = distances_list[i]
                description = documents_list[i]

                entities[entity_name] = {
                    "type": entity_type,
                    "distance": distance,
                    "description": description,
                    "community":community
                }

            return entities

        def merge_retrieved_dicts(dict1: Dict[str, Dict[str, Any]], dict2: Dict[str, Dict[str, Any]]):
            """
            Merges two retrieved entity dictionaries based on common keys.

            If an entity appears in both dictionaries, their 'distance' values are updated using **harmonic mean**:
                \[
                new\_distance = \frac{2}{(\frac{1}{d1} + \frac{1}{d2})}
                \]
            This ensures that frequently appearing entities are given **higher priority** in the retrieval results.

            Args:
                dict1 (Dict[str, Dict[str, Any]]): The primary dictionary to update.
                dict2 (Dict[str, Dict[str, Any]]): The secondary dictionary, from which matching elements are merged and removed.

            Returns:
                None: dict1 is updated in place, and matching keys are removed from dict2.

            Example:
                ```python
                dict1 = {"TOOLNODE": {"distance": 0.60}}
                dict2 = {"TOOLNODE": {"distance": 0.75}}
                merge_retrieved_dicts(dict1, dict2)
                print(dict1)  # TOOLNODE distance will be updated to ~0.667
                ```
            """
            common_keys = set(dict1.keys()) & set(dict2.keys())  # Find common entities

            for key in common_keys:
                d1 = dict1[key]["distance"]
                d2 = dict2[key]["distance"]

                # Compute harmonic mean of distances (lower is better)
                new_distance = 2 / ((1 / d1) + (1 / d2))

                dict1[key]["distance"] = new_distance  # Update dict1
                del dict2[key]  # Remove from dict2
        
        if(not collection_dict):
            collection_dict = self.collections_dict

        # Step 1: Query both collections (entity names and descriptions)
        entity_name_retrievals = collection_dict["entities"].query(query_texts=[query], n_results=k)
        entity_desc_retrievals = collection_dict["entity_descriptions"].query(query_texts=[query], n_results=k)

        # Step 2: Process retrievals
        results = {"entity_name_store": {}, "entity_desc_store": {}}

        if entity_name_retrievals:
            entities_to_get = entity_name_retrievals.get("ids", [[]])[0]
            distances = entity_name_retrievals.get("distances", [[]])[0]
            name_retrievals_with_desc = collection_dict["entity_descriptions"].get(ids=entities_to_get)
            results["entity_name_store"] = reconstruct_entities(name_retrievals_with_desc, distances=distances)

        if entity_desc_retrievals:
            results["entity_desc_store"] = reconstruct_entities(entity_desc_retrievals)

        # Step 3: Merge entity name and description retrievals
        merge_retrieved_dicts(results["entity_name_store"], results["entity_desc_store"])

        return results

    def get_relationships(self,collection,query, **kwargs) -> List[str]:
        """
        Given a query text, retrieve the top k most relevant relationships from the collection.

        Args:
            collection: The ChromaDB collection for relationships.
            query (str): The search query.
            k (int): The number of top results to retrieve.

        Returns:
            List[str]: A list of top k matching relationships.
        """
        if query:
            results = collection["relationships"].query(**kwargs)
        else:
            results = collection["relationships"].get(**kwargs)
        return results
    
    def get_entities(self,collection,query, **kwargs) -> List[str]:
        """
        Given a query text, retrieve the top k most relevant relationships from the collection.

        Args:
            collection: The ChromaDB collection for relationships.
            query (str): The search query.
            k (int): The number of top results to retrieve.

        Returns:
            List[str]: A list of top k matching relationships.
        """
        
        if query:
            results = collection["entity_descriptions"].query(**kwargs)
        else:
            results = collection["entity_descriptions"].get(**kwargs)
        return results
    
    def search_community_summaries(self,collection,query, **kwargs) -> List[str]:
        """
        Given a query text, retrieve the top k most relevant community summaries from the collection.

        Args:
            collection: The ChromaDB collection for community summaries.
            query (str): The search query.
            k (int): The number of top results to retrieve.

        Returns:
            List[str]: A list of top k matching community summaries.
        """
        if query:
            results = collection["community_summaries"].query(**kwargs)
        else:
            results = collection["community_summaries"].get(**kwargs)
        return results

#---------------search-End---------------#

#---------------loc_search graph-Begin-------------#
class LocalSearch:
    def __init__(self,**kwargs):
        # Local search specific
        cwd = os.getcwd()
        source_list = kwargs.get("sources",None)

        self.storage = kwargs.get("storage",None)

        self.artifacts_path = "Odysseus"
        
        self.client = {}
        self.vdb = {}
        if not source_list:
            raise(ValueError(f"Required argument missing!! \"sources:list\""))
        print(f"Sources : {source_list}\nartifacts path = {self.artifacts_path}")
        
        for source in source_list:
            global_search_artifacts_path = f"graphRAG/{source}/{source}_global_search.pt"
            # Check if communities info is available or not
            if self.storage.file_exists(global_search_artifacts_path,self.artifacts_path):  
                full_path = self.storage.get_absolute_path(None,self.artifacts_path)
                self.client[source] = chromadb.PersistentClient(
                            path=f"{full_path}/vector_dbs/graphRAG/{source}_vector_db",
                            settings=Settings(),
                            tenant=DEFAULT_TENANT,
                            database=DEFAULT_DATABASE,
                        )
                print(f"Loaded {source} vector db")
                self.vdb[source] = LocalSearchUtils(client=self.client[source],source=source,storage=self.storage)
        # Langfuse
        self.parent_trace_id = kwargs.get('trace_id') if 'trace_id' in kwargs else uuid.uuid4()
        # Prompt related 
        self.prompt_dict = get_settings()
        config = Config()
        expt_llm = config["llm_model"]
        self.llm = OllamaLLM(temperature=0.2, model=expt_llm)
        
        # For persistence 
        self.checkpointer = MemorySaver()
        self.config = None

        self.app = None
        # self._setup_graph()

    def invoke(self,state): # For every invoke generate new instance of graph 
        self.app = None
        self.config = {"configurable": {"thread_id": uuid.uuid4()}}
        self._setup_graph()
        self.app.invoke(state,self.config)
    
    def get_state(self):
        return self.app.get_state(self.config)

    def _setup_graph(self):
        # Graph structure code 
        # graph related 
        workflow = StateGraph(LocalSearchState)
        workflow.add_node("retrieve_entities",self.retrieve_entities)
        workflow.add_node("get_relationships",self.get_relationships)
        workflow.add_node("get_entities",self.get_entities)
        workflow.add_node("get_communities",self.get_communities)
        workflow.add_node("retrieve_text_units",self.retrieve_text_units)
        workflow.add_node("reduce",self.reduce)
        workflow.add_node("dummy",self.dummy)
        workflow.add_edge(START,"retrieve_entities")
        workflow.add_edge("retrieve_entities","get_relationships")
        workflow.add_edge("get_entities","get_relationships")
        
        workflow.add_edge("retrieve_entities","get_communities")
        workflow.add_edge("retrieve_entities","retrieve_text_units")

        workflow.add_edge(["get_communities","retrieve_text_units","dummy"],"reduce")

        workflow.add_edge("reduce",END)

        self.app = workflow.compile(checkpointer = self.checkpointer)
        graph_mage = self.app.get_graph().draw_mermaid_png()
        with(open("LocalSearch.png","wb")) as image:
            image.write(graph_mage)

    def retrieve_entities(self,state):
        # Invoke LLM and find most relevant entities with respect to the query
        # Objective : Given the retrieved entities, their descriptions, types and their distances from query embedding
        # filter out irrelevant retrievals from the list. Reply with the list of entity names which are relevant to the query
        # and explain why they are relevant 
        loc_state = state
        if('level' not in loc_state): # Fallback to set default entity-relation retrieval level at 1
            loc_state['level'] = 1

        retrieved_entities = self.vdb[loc_state['source']].search_entities(query=loc_state['query'],k=7)
        entities = self._format_retrieved_entities(retrieved_entities)
        input_var_names = ["entities"]
        model = sys_prompt_wrapped_call(pydantic_object=FilteredEntities,
                                        sys_prompt_template=self.prompt_dict['graphrag_loc_search_filter_entities']['system'],
                                        parent_trace_id=self.parent_trace_id,input_var_names_system=input_var_names,
                                        custom_llm="phi4:latest")
        
        s_prompt = model['system_template'].format(entities=entities)
        messages = [s_prompt]
        wrapped_state = {"messages":messages,"llm_output":"","error_status":"","iterations":0}
        answer = model['model'].invoke(wrapped_state)['llm_output']['relevant_entities']
        
        filtered_list = [entity['entity_name'] for entity in answer]
        entities = retrieved_entities.get('entity_name_store', {}) | retrieved_entities.get('entity_desc_store', {})
        filtered_entities = {}
        for name,entity in entities.items():
            if(name in filtered_list):
                filtered_entities[name] = entity

        loc_state['retrieved_entities'] = filtered_entities
        loc_state['node_reln_level_counter'] = 0 # Set graph extraction level to 0
        return loc_state
    
    def get_relationships(self,state)->Command[Literal["dummy","get_entities"]]:
        # we have filtered entities now. Based on the entities we need to filter out text chunks, relationships and 
        # community summaries
        loc_state = state
        # print(f"loc_state<get_relationships>:\n{"-"*20}\n{loc_state}")
        level = loc_state['node_reln_level_counter']
        if(level):
            entities = list(loc_state['entities'][level].keys())
        else: # Level zero case
            entities = list(loc_state['retrieved_entities'].keys())
            loc_state['entities'] = {}
            loc_state['entities'][level]=loc_state['retrieved_entities']
        
        # Retrieve relationships
        relationships = self.vdb[loc_state['source']].get_relationships(
                            collection=self.vdb[loc_state['source']].collections_dict,query=False,
                            where={"$or":[{"source": {"$in": entities}},{"target": {"$in": entities}}]},
                            include=["documents","metadatas"]
                            )
        if 'relationships' not in loc_state:
            loc_state['relationships'] = {}
        loc_state['relationships'] |= {loc_state['node_reln_level_counter']:relationships}
        
        loc_state['node_reln_level_counter'] +=1
        
        if(loc_state['node_reln_level_counter'] > loc_state['level']):
            goto = "dummy"
        else:
            sources = [reln['source'] for reln in relationships['metadatas']]
            targets = [reln['target'] for reln in relationships['metadatas']]
            all_entities = sources+targets
        
            secondary_entities_keys = set([entity for entity in all_entities if entity not in entities])
            secondary_entities = {entity:{} for entity in secondary_entities_keys}

            # print(f"Secondary entities\n{secondary_entities}")
            if 'entities' not in loc_state:
                loc_state['entities'] = {}    
            loc_state['entities'] |= {loc_state['node_reln_level_counter']:secondary_entities}
            goto = "get_entities"

        return Command(update=loc_state,goto=goto)
    
    def get_entities(self,state):
        loc_state = state
        level = loc_state['node_reln_level_counter']
        entities = list(loc_state['entities'][level].keys())
        extracted_entities = self.vdb[loc_state['source']].get_entities(
                            collection=self.vdb[loc_state['source']].collections_dict,query=False,
                            ids=entities,
                            include=["documents","metadatas"]
                            )

        def reconstruct_entities(retrieved_docs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
            """
            Reconstructs the entities dictionary from a ChromaDB retrieval response.

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

            entities = {}

            ids_list = retrieved_docs.get("ids", [[]])  # List of entity names
            documents_list = retrieved_docs.get("documents", [[]])  # List of descriptions
            metadatas_list = retrieved_docs.get("metadatas", [[]])  # List of metadata dictionaries
            
            num_results = min(len(ids_list), len(documents_list), len(metadatas_list))

            for i in range(num_results):
                entity_name = ids_list[i]
                entity_type = metadatas_list[i].get("type", "UNKNOWN") if metadatas_list else "UNKNOWN"
                community = metadatas_list[i].get("community", "UNKNOWN") if metadatas_list else "UNKNOWN"
                description = documents_list[i]

                entities[entity_name] = {
                    "type": entity_type,
                    "description": description,
                    "community":community
                }

            return entities
        
        entities = reconstruct_entities(extracted_entities)
        # print(f"retrieved entities\n{"*"*20}\n{entities}")
        for name,entity in entities.items():
            loc_state['entities'][level][name] = entity

        return {"entities":loc_state["entities"]}
    
    def get_communities(self,state):
        loc_state = state
        node_community_set = set([entity['community'] for entity in loc_state['retrieved_entities'].values()])
        print(f"Following are the community set:\n{"*"*20}\n{list(node_community_set)}")
        communities = self.vdb[loc_state['source']].search_community_summaries(collection=self.vdb[loc_state['source']].collections_dict,query=False,
            ids= list(node_community_set),
            include=["documents","metadatas"]
            )
        def reconstruct_communities(retrieved_docs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
            """
            Reconstructs the community dictionary from a ChromaDB retrieval response.

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
            """

            communities = {}

            ids_list = retrieved_docs.get("ids", [[]])  # List of entity names
            documents_list = retrieved_docs.get("documents", [[]])  # List of descriptions
            metadatas_list = retrieved_docs.get("metadatas", [[]])  # List of metadata dictionaries
            
            num_results = min(len(ids_list), len(documents_list), len(metadatas_list))

            for i in range(num_results):
                community_id = ids_list[i]
                metadata = metadatas_list[i]
                summary = documents_list[i]
                communities[community_id] = {
                    "metadata": metadata,
                    "summary": summary,
                }

            return communities
        
        rec_communities = reconstruct_communities(communities)
        # print(f"filtered communities:\n{"-"*20}\n {rec_communities}")
        # loc_state['retrieved_communities'] = rec_communities
        return {"retrieved_communities":rec_communities}
    
    def retrieve_text_units(self,state):
        loc_state = state
        text_units = []
        # print(f"retrieved_entities\n{loc_state['retrieved_entities']}")
        # retrieve most relevant text unit for each entity 
        entity_desc_list = [f"{name} : {entity["description"]}" for name,entity in loc_state['retrieved_entities'].items()]
        
        for item in entity_desc_list:
            unit = self.vdb[loc_state['source']].search_text_units(collection=self.vdb[loc_state['source']].collections_dict,query=item,k=1)
            text_units.append(unit[0])

        # print(f"Retrieved text units\n{"*"*20}\n{text_units}")
        return {"retrieved_text_units":text_units}

    def dummy(self,state):
        """
        Purpose of the dummy node : Avoid synchornization issues in the map-reduce design patter
            - Since we have a loop between get_relationships and get_entities, this will require multiple supersteps to complete
            - get_relationships, retrieve_text_units, get_communities all work in parallel. Hence to enable synchronism, after get_relationships we added a dummy node and synchronized dummy node with other parallel nodes
        """
        loc_state = state
        return {"entities":loc_state["entities"],"relationships":loc_state["relationships"],"node_reln_level_counter":loc_state["node_reln_level_counter"]}
    

    def reduce(self,state):
        """
        Reduce stage of map-reduce design pattern
        Combine all the retrievals and generate a coherent state
        """
        loc_state = state
        # results = f"loc_state"
        with(open("local_search_results.txt","w")) as file:
            file.write(str(loc_state))
        
        return loc_state
    
    def _format_retrieved_entities(self,entity_data: dict) -> str:
        def format_section(title: str, entities: dict) -> str:
            section = [f"{title}:\n" + "-" * 50]
            for name, details in entities.items():
                section.append(
                    f"{name} : {details['description']}\n"
                    f"Type : {details['type']}\n"
                    f"Distance : {details['distance']:.4f}\n"
                    f"{'-' * 30}"
                )
            return "\n".join(section)

        entity_name_results = format_section("Entities retrieved from entity name vector search", entity_data.get('entity_name_store', {}))
        entity_desc_results = format_section("Entities retrieved from entity description vector search", entity_data.get('entity_desc_store', {}))
        
        return f"{entity_name_results}\n\n{entity_desc_results}"
    
    # define node functions 

#---------------loc_search graph-End---------------#

class GraphRAGLocalSearch(RetrievalPlugin):
    def __init__(self,**kwds):
        trace_id = uuid.uuid4()
        self.local_search_graph = LocalSearch(trace_id=trace_id,**kwds)

    def __call__(self, *args, **kwds)->RetrievalOutput:
        query = kwds.get("query")
        source = kwds.get("source")
        if not(query and source):
            raise ValueError(f"Missing required attributes!! \"query:str\", \"source:List[str]\"")
        results = self.retrieve(query=query,**kwds)
        retrievals:RetrievalOutput = {}
        retrievals['modality'] = "text"
        retrievals['data'] = {results}
        retrievals['metadata'] = {}
        return results
    
    def retrieve(self, query, **kwds):
        source = kwds.get("source")
        state = {"query":query,"source":source}
        self.local_search_graph.invoke(state)
        results = self.local_search_graph.get_state()
        # print(f"Results: \n{results.keys()}")
        with(open("temp.txt","w")) as output:
            output.write(f"Results: \n{results.values}")
        return results.values
    
#---------------graph test-Begin-------------#
if __name__ == "__main__":
    kwargs = {"trace_id": 213,"sources":["low_level"]}
    loc_search_graph = LocalSearch(**kwargs)
    query = "What is langgraph?"
    loc_search_graph.invoke({"query":query,"level":1,"source":"low_level"})
#---------------graph test-End---------------#

if __name__ == "__main__1":
    cwd = os.getcwd()
    client = chromadb.PersistentClient(
                path=f"{cwd}/Odysseus/retrieval_stratergies/graphRAG_vector_db",
                settings=Settings(),
                tenant=DEFAULT_TENANT,
                database=DEFAULT_DATABASE,
            )
    
    source_name = "low_level"
    loc_search_db = LocalSearchUtils(client=client,source=source_name)
    
    #---------------Entity extraction pipeline-Begin-------------#
    query = "What is langgraph?"
    retrieved_entities            = loc_search_db.search_entities(query=query,k=10)
    # retrieved_text_units          = search_text_units(query=query,collection=loc_search_collections)
    # retrieved_relationships       = search_relationships(query=query,collection=loc_search_collections)
    # retrieved_community_summaries = search_community_summaries(query=query,collection=loc_search_collections)

    for _,entity in retrieved_entities.items():
        print(f"{_} : {entity}")

    #---------------Entity extraction pipeline-End---------------#



# TODO : LocalHost chromadb
    # client = chromadb.HttpClient(
    #             host="[::1]",
    #             port=8013,
    #             ssl=False,
    #             headers=None,
    #             settings=Settings(),
    #             tenant=DEFAULT_TENANT,
    #             database=DEFAULT_DATABASE,
    #             )    
#---------------Experiments-End---------------# 

