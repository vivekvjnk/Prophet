from typing import Dict, Any
import torch
from langchain_core.prompts import (
                                    ChatPromptTemplate,
                                    MessagesPlaceholder,
                                    SystemMessagePromptTemplate,
                                    HumanMessagePromptTemplate,
                                    AIMessagePromptTemplate,
                                    PromptTemplate)


import networkx as nx
import yaml
import pandas as pd
import io
import re

from infra.utils.utils import * # For prompt dictionary
from infra.utils.cluster import *
from infra.logs.prophet_logger import *

from ..states.odysseus import * # For state models

#---------------Pipeline specific imports-Begin-------------#
from ..base import *
#---------------Pipeline specific imports-End---------------#

from sentence_transformers import SentenceTransformer


# logger setup
log_file_handler = configure_rotating_file_handler("infra/logs/Odysseus_GR_GlobalSearch.log",max_bytes=500*1024)
logger = get_pipeline_logger(file_handler=log_file_handler,pipeline_name="Odysseus")

prompt_dict = get_settings()

embedding_model = SentenceTransformer('all-mpnet-base-v2')

#---------------Token counter-Begin-------------#
def _estimate_tokens_hf(text: str, model_name: str = "Qwen/Qwen2.5-Coder-14B") -> int:
    """
    Estimate the number of tokens using Hugging Face tokenizers.

    Args:
        text (str): The input text.
        model_name (str): The Hugging Face model name or path.
    
    Returns:
        int: The estimated number of tokens.
    """
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokens = tokenizer(text, return_tensors="pt")["input_ids"]
        return tokens.size(1)  # Token count
    except Exception as e:
        raise RuntimeError(f"Error loading Hugging Face tokenizer for {model_name}: {e}")
    
def _estimate_tokens_spm(text: str, model_path: str) -> int:
    """
    Estimate tokens using SentencePiece tokenizer.

    Args:
        text (str): The input text.
        model_path (str): Path to the SentencePiece model file (.model).

    Returns:
        int: Estimated token count.
    """
    import sentencepiece as spm
    try:
        sp = spm.SentencePieceProcessor(model_file=model_path)
        tokens = sp.encode(text, out_type=str)
        return len(tokens)
    except Exception as e:
        raise RuntimeError(f"Error loading SentencePiece model from {model_path}: {e}")
    
def _estimate_tokens_tiktoken(text: str, model: str = "gpt-4",fallback: str = "cl100k_base") -> int:
    """
    Estimate the number of tokens in a text string using the tiktoken library.

    Args:
        text (str): The input text to estimate token count for.
        model (str): The model name to determine the tokenizer (default: "gpt-4").
    
    Returns:
        int: The estimated number of tokens.
    """
    import tiktoken
    try:
        tokenizer = tiktoken.encoding_for_model(model)
    except KeyError:
        # Use fallback model
        tokenizer = tiktoken.get_encoding(fallback)
    return len(tokenizer.encode(text))
    
    # Encode the text and count the tokens
    token_count = len(tokenizer.encode(text))
    return token_count

def estimate_tokens(text: str, method: str = "hf", **kwargs) -> int:
    """
    Estimates the number of tokens in a given text using a specified tokenization method.

    This function acts as a wrapper for multiple token estimation methods and dynamically selects the appropriate method based on the `method` parameter. It logs the token estimation process, including the chosen method and the token count. If the token count is zero, a warning is logged.

    Args:
        text (str): 
            The input text for which the token count is to be estimated.
        method (str, optional): 
            The tokenization method to use. Defaults to `"hf"`. Supported methods are:
            - `"hf"`: Hugging Face tokenization.
            - `"spm"`: SentencePiece tokenization.
            - `"tiktoken"`: OpenAI's TikToken tokenization.
        **kwargs: 
            Additional keyword arguments passed to the underlying tokenization method for customization.

    Returns:
        int: 
            The estimated number of tokens in the input text.

    Raises:
        ValueError: 
            If an unsupported tokenization method is specified.

    Workflow:
        1. Log the selected tokenization method.
        2. Use the specified method to estimate the number of tokens:
            - If `method` is `"hf"`, call `_estimate_tokens_hf`.
            - If `method` is `"spm"`, call `estimate_tokens_spm`.
            - If `method` is `"tiktoken"`, call `estimate_tokens_tiktoken`.
        3. Log a warning if the estimated token count is zero.
        4. Log the final token count and return it.

    Sub-functions:
        - `_estimate_tokens_hf(text: str, **kwargs) -> int`: Estimates tokens using Hugging Face tokenization.
        - `estimate_tokens_spm(text: str, **kwargs) -> int`: Estimates tokens using SentencePiece tokenization.
        - `estimate_tokens_tiktoken(text: str, **kwargs) -> int`: Estimates tokens using OpenAI's TikToken.

    Example Usage:
        ```python
        text = "This is a sample text for token estimation."
        
        # Using Hugging Face tokenization
        token_count_hf = estimate_tokens(text, method="hf")
        
        # Using SentencePiece tokenization
        token_count_spm = estimate_tokens(text, method="spm")
        
        # Using TikToken tokenization
        token_count_tiktoken = estimate_tokens(text, method="tiktoken")
        ```

    Example Output:
        For the input text `"This is a sample text for token estimation."`, the token count may vary based on the method:
        - Hugging Face: 10 tokens
        - SentencePiece: 9 tokens
        - TikToken: 11 tokens

    Logging:
        - Logs the selected tokenization method at the beginning.
        - Logs a warning if the token count is zero, along with the input text.
        - Logs the estimated token count.

    Notes:
        - The function assumes that the tokenization methods `_estimate_tokens_hf`, `estimate_tokens_spm`, and `estimate_tokens_tiktoken` are implemented and available in the codebase.
        - The choice of tokenization method should align with the downstream processing system.

    Limitations:
        - The token count may differ across methods due to variations in tokenization algorithms.
        - If the input text is empty or poorly formatted, the token count may be zero, triggering a warning.
    """
    logger.info(f"Estimating tokens using {method} method.")
    if method == "hf":
        token_count = _estimate_tokens_hf(text, **kwargs)
    elif method == "spm":
        token_count = _estimate_tokens_spm(text, **kwargs)
    elif method == "tiktoken":
        token_count = _estimate_tokens_tiktoken(text, **kwargs)
    else:
        raise ValueError(f"Unsupported tokenization method: {method}")
    if(token_count==0):
        logger.warning(f"Zero token count for following chunk:\n{text}\n")
    logger.info(f"Token count: {token_count}")
    return token_count
#---------------Token counter-End---------------#


#---------------Graph generation-Begin-------------#
def generate_graph_from_dict(entities, relationships):
    """
    Generates a NetworkX graph from dictionaries of entities and relationships.

    Args:
        entities (dict): A dictionary where keys are entity names, and values are attributes like type and description.
        relationships (dict): A dictionary where keys are tuples of (source_entity, target_entity), 
                              and values are attributes like description and strength.

    Returns:
        nx.Graph: The constructed graph.
    """
    import networkx as nx
    import ast
    
    graph = nx.Graph()

    # Add nodes (entities) to the graph
    for entity_name, attributes in entities.items():
        graph.add_node(
            entity_name.strip().upper(),
            type=", ".join(attributes.get("type", [])).strip().upper(),
            description=" ; ".join(attributes.get("description", [])).strip(),
        )

    # Add edges (relationships) to the graph
    for relation, attributes in relationships.items():
        # Convert relationship string to source and target tuple
        (source_entity, target_entity) = ast.literal_eval(relation)
        source = source_entity.strip().upper()
        target = target_entity.strip().upper()
        weight = sum(attributes.get("strength", [5.0]))  # Sum all strengths if multiple are provided
        description = " ; ".join(attributes.get("description", [])).strip()

        # Add or update the edge
        if graph.has_edge(source, target):
            # Update weight and append description if edge already exists
            edge_data = graph.get_edge_data(source, target)
            edge_data['weight'] += weight
            edge_data['description'] += f" | {description}"
        else:
            graph.add_edge(
                source,
                target,
                weight=weight,
                description=description,
            )

    return graph
#---------------Graph generation-End---------------#


#---------------Process Community-Begin-------------#
def process_community_list(community_list, global_graph)->dict:
    """
    Processes a list of community clusters and constructs an intermediate dictionary 
    representation of the community knowledge graph.

    This function takes a list of detected communities and extracts relevant metadata, 
    including node and edge information, to facilitate further processing. It creates 
    a structured dictionary where each community is represented with its nodes, internal 
    edges, external edges, and metadata.

    Args:
        community_list (list): 
            A list of community tuples where each tuple contains:
            - community_level (int): The hierarchical level of the community.
            - community_id (int): Unique identifier for the community.
            - parent_community_id (int | None): ID of the parent community in the hierarchy.
            - node_list (list[str]): List of node IDs belonging to this community.
        
        global_graph (nx.Graph): 
            The full knowledge graph containing all nodes and edges.

    Returns:
        dict: 
            A dictionary where each key is `"community_<community_id>"` and each value 
            is an intermediate object containing:
            - `community_id` (int): Community identifier.
            - `metadata` (dict): Community metadata including:
                - `community_level` (int): Level in the hierarchy.
                - `parent_community_id` (int | None): Parent community ID.
                - `node_count` (int): Number of nodes in the community.
                - `internal_edge_count` (int): Count of edges within the community.
                - `external_edge_count` (int): Count of edges connecting to external nodes.
            - `summary` (str): Placeholder for a community summary.
            - `nodes` (dict): Dictionary of node details where:
                - Key: Node ID.
                - Value: Dictionary with:
                    - `type` (str): Node type (default `"UNKNOWN"` if unspecified).
                    - `description` (list[str]): List of descriptions (split from a semicolon-separated string).
            - `internal_edges` (dict): Dictionary of internal edges where:
                - Key: Tuple `(source, target)`.
                - Value: Dictionary with:
                    - `weight` (int): Edge weight (default `1`).
                    - `description` (str): Edge description.
            - `external_edges` (dict): Dictionary of external edges where:
                - Key: Tuple `(source, target)`.
                - Value: Dictionary with:
                    - `weight` (int): Edge weight (default `1`).
                    - `description` (str): Edge description.

    Example:
        >>> community_list = [
        ...     (1, 100, None, ["A", "B", "C"]),
        ...     (1, 101, None, ["D", "E"])
        ... ]
        >>> global_graph = nx.Graph()
        >>> global_graph.add_edge("A", "B", weight=2, description="Strong link")
        >>> global_graph.add_edge("B", "C", weight=1)
        >>> global_graph.add_edge("C", "D", weight=3, description="Weak link")
        >>> process_community_list(community_list, global_graph)
        {
            "community_100": {
                "community_id": 100,
                "metadata": {
                    "community_level": 1,
                    "parent_community_id": None,
                    "node_count": 3,
                    "internal_edge_count": 2,
                    "external_edge_count": 1
                },
                "summary": "",
                "nodes": { ... },
                "internal_edges": { ... },
                "external_edges": { ... }
            },
            "community_101": { ... }
        }

    Notes:
        - Node IDs are normalized to uppercase and stripped of whitespace.
        - Internal edges exist entirely within the community, while external edges connect 
          to nodes outside the community.
        - The function ensures structured retrieval of community-level knowledge while 
          preserving hierarchical relationships.
    """
    all_community_data = {}  # To store data for all communities

    for community_data in community_list:
        # Extract community information
        community_level, community_id, parent_community_id, node_list = community_data

        # Create subgraph for the community
        community_subgraph = global_graph.subgraph([node.strip().upper() for node in node_list])

        # Initialize Intermediate Object
        intermediate_object = {
            "community_id": community_id,
            "metadata": {
                "community_level": community_level,
                "parent_community_id": parent_community_id,
                "node_count": len(community_subgraph.nodes),
                "internal_edge_count": 0,
                "external_edge_count": 0
            },
            "summary":"",
            "nodes": {},
            "internal_edges": {},
            "external_edges": {}
        }

        # Populate Node Information
        for node, data in community_subgraph.nodes(data=True):
            intermediate_object["nodes"][node] = {
                "type": data.get("type", "UNKNOWN"),
                "description": data.get("description", "").split(" ; ")  # Convert back to list format
            }

        # Populate Internal Edge Information
        for source, target, data in community_subgraph.edges(data=True):
            internal_edge_key = (source, target)
            intermediate_object["internal_edges"][internal_edge_key] = {
                "weight": data.get("weight", 1),
                "description": data.get("description", "")
            }
        intermediate_object["metadata"]["internal_edge_count"] = len(intermediate_object["internal_edges"])

        # Populate External Edge Information
        for node in community_subgraph.nodes:
            for neighbor in global_graph.neighbors(node):
                if neighbor not in community_subgraph:
                    edge_data = global_graph.get_edge_data(node, neighbor, default={})
                    external_edge_key = (node, neighbor)
                    intermediate_object["external_edges"][external_edge_key] = {
                        "weight": edge_data.get("weight", 1),
                        "description": edge_data.get("description", "")
                    }

        intermediate_object["metadata"]["external_edge_count"] = len(intermediate_object["external_edges"])

        # Add this community's data to the global dictionary
        all_community_data[f"community_{community_id}"] = intermediate_object
        
    return all_community_data
#---------------Process Community-End---------------#


#---------------Community Summarization-Begin-------------#
def generate_and_summarize_community_graphs(source_name,nx_graph,storage):
    """
    Generates a network graph from entities and relationships, applies community detection, 
    and summarizes the nodes within each community, ensuring that all nodes are summarized.
    
    This function reads entities and relationships from intermediate YAML files, constructs 
    a networkX graph, applies the Leiden community algorithm to detect communities, and 
    generates summaries for each community's nodes. It ensures that all nodes are summarized 
    by implementing a fallback mechanism for any missed nodes. The results are saved into 
    YAML files containing both the community-level summaries and the final community data.

    Key Steps:
    1. **Graph Construction**: 
       - Reads entities and relationships from YAML files and constructs a networkX graph.
       
    2. **Community Detection**: 
       - Applies the Leiden community detection algorithm to partition the graph into communities.
       
    3. **Summarizing Nodes**:
       - Summarizes each community's nodes using the `summarize_community_nodes` function.
       - If any nodes are missed, the summarization is retried for those nodes until all are summarized.
       
    4. **Storing Results**:
       - Saves the community node summaries and the final community data into YAML files.

    Args:
        None: This function does not take input arguments as it reads predefined files.

    Returns:
        dict: A dictionary containing the summarized community data, including node summaries. Following is the schema of the dictionary
        all_communitites = {
            community_<community_id>:{
                    "community_id": community_id,
                    "metadata": {
                        "community_level": community_level,
                        "parent_community_id": parent_community_id,
                        "node_count": len(community_subgraph.nodes),
                        "internal_edge_count": 0,
                        "external_edge_count": 0
                    },
                    "summary":"",
                    "nodes": {},
                    "internal_edges": {},
                    "external_edges": {}
                }
            }
        
    Outputs:
        - `artifacts/community_node_summaries.yml`: A YAML file containing the summarized data for each community's nodes.
        - `artifacts/final_community_file.yml`: A YAML file containing the final community structure, including node summaries.

    Example:
        >>> generate_and_summarize_community_graphs()
    
    Logging:
        - Logs the progress of the summarization process and any missed nodes that will be retried.
        - Warnings are issued for missed nodes and errors are logged if no summaries are generated.

    """
    # Langfuse tracing 
    parent_trace_id = create_unique_trace_id()
    # Read the all_communities.pt file. Then check if all the dictionary keys are available. Depending on the 
    # available dictionary keys we will choose the pipeline. 
    import os 
    
    def keys_exist(keys, dictionary):
        """
        Checks if all keys in a list exist in a given dictionary.

        Args:
            keys: A list of keys to check.
            dictionary: The dictionary to check against.

        Returns:
            True if all keys in the list exist in the dictionary, False otherwise.
        """
        for key in keys:
            if key not in dictionary:
                return False
        return True
    def missing_keys(keys, dictionary):
        """
        Checks if all keys in a list exist in a given dictionary 
        and returns a list of missing keys.

        Args:
            keys: A list of keys to check.
            dictionary: The dictionary to check against.

        Returns:
            A list of missing keys. If all keys exist, returns an empty list.
        """
        missing_keys_list = []
        for key in keys:
            if key not in dictionary:
                missing_keys_list.append(key)
        return missing_keys_list
    
    community_state = {"intermediate_file":False,"intermediate_obj_keys":False,"community_summaries_found":False,
                       "node_summaries_found":False,"missing_keys":False,"completeness":False}
    artifact_path = "Odysseus"
    # community_path = storage.make_dir(f"{artifact_path}/graphRAG/{source_name}/")
    community_path = f"{artifact_path}/graphRAG/{source_name}/"
    
    all_communities_path = f"{source_name}_communities.pt"
    intermediate_obj_keys = ["external_edges","internal_edges","nodes","summary","metadata","community_id"]
    summary_key = ["summary"]
    all_communities = None
    logger.info(f"Checking if the community info file exists at path: {all_communities_path}")
    if(storage.file_exists(all_communities_path,community_path)):
        community_state['intermediate_file'] = True
        data = storage.read_file(all_communities_path,community_path)
        data_bytes = io.BytesIO(data)
        all_communities = torch.load(data_bytes)

        logger.info(f"Loaded community info from the file.\n{all_communities['needs_node_summary']}")

        if('needs_node_summary' in all_communities.keys()):
            if(not(all_communities['needs_node_summary'])):
                community_state['node_summaries_found'] = True
        else:
            community_state['missing_keys'] = True
            
        # Check if community summaries are available 
        if('needs_community_summary' in all_communities.keys()):
            if(not(all_communities['needs_community_summary'])):
                community_state['community_summaries_found'] = True
        else:
            community_state['missing_keys'] = True

        logger.info("Checking for required keys in the intermediate object.")

        _,sample_community = next(iter(all_communities.items()))
        # Check if any keys are missing from the intermediate object
        if keys_exist(intermediate_obj_keys, sample_community):
            logger.info("All required keys exist in the intermediate object.")
            community_state['intermediate_obj_keys'] = True
            
            community_state['completeness'] = community_state['node_summaries_found'] & community_state['community_summaries_found']
            logger.warning("Node summaries are missing required keys: %s", summary_key)
        else:
            missed_keys = missing_keys(intermediate_obj_keys,sample_community)
            logger.warning("Following required keys are missing from the intermediate object: %s", missed_keys)
            logger.warning(f"Loaded file content : {all_communities}")

    else:
        community_state['missing_keys'] = True
        logger.warning("Community info file not found at path: %s", all_communities_path)

    logger.info("Final community state: %s", community_state)

    if(community_state['completeness']):
        return all_communities
    
    elif(community_state['missing_keys']):
        communities = cluster_graph(nx_graph) # Leiden clustering 
        """ 
        Output of the clustering code is simply a list of clusters with nodes belonging to them. For further processing we need 
        more information about the cluster(like edges). Also clusters being a list will be hard to retrieve and process further 
        down in the pipeline. Hence we define an intermediate dictionary format to store the cluster knowledge graph. This is 
        achieved using process_community_list function.
        ------
        intermediate_object = {
            "community_id": community_id,
            "metadata": {
                "community_level": community_level,
                "parent_community_id": parent_community_id,
                "node_count": len(community_subgraph.nodes),
                "internal_edge_count": 0,
                "external_edge_count": 0
            },
            "summary":"",
            "nodes": {},
            "internal_edges": {},
            "external_edges": {}
        }
        ------
        process_community_list function returns a dictionary with community id as key and community intermediate object as value.
        """
        all_communities = process_community_list(community_list=communities,global_graph=nx_graph) 
        community_keys = list(all_communities.keys()) # take snapshot 
        all_communities['needs_community_summary'] = community_keys
        all_communities['needs_node_summary'] = list(community_keys) # snapshot to isolate both lists
    
    logger.info(f"needs node summary : {all_communities['needs_node_summary']}\nneeds community summary: {all_communities['needs_community_summary']}")
    
    # If we already have node summaries in intermediate object, proceed with final community summarization
    if(not(all_communities['needs_node_summary'])):
        logger.info(f"Found node_summaries in all_communities. Proceeding to community summarization")
        add_community_summaries_to_intermediate_object(all_communities=all_communities,
                                                       file_name=all_communities_path,
                                                       file_path=community_path,
                                                       storage=storage,
                                                       parent_trace_id=parent_trace_id)
    else:
        all_communities = add_node_summaries_to_intermediate_object(all_communities=all_communities,
                                                                    parent_trace_id=parent_trace_id,
                                                                    file_name=all_communities_path,
                                                                    file_path=community_path,
                                                                    storage=storage,
                                                                    )
        
        # Node summarization would take too much time. Hence backup all_communities after node_summarization
        # We already have logic to check if node summaries are available. 
        
        all_communities = add_community_summaries_to_intermediate_object(all_communities=all_communities,
                                                                         file_name=all_communities_path,
                                                                         file_path=community_path,
                                                                         storage=storage,
                                                                         parent_trace_id=parent_trace_id)

    # Save generated community information to file 
    communities_path = f"{source_name}_communities.yml"
    all_communities_bytes = b"".join(s.encode('utf-8') for s in yaml.dump(all_communities)) 
    storage.save_file(filename=communities_path,data=all_communities_bytes,storage_type=community_path)

    return all_communities

def summarize_community_nodes(nodes: Dict[str, Any], token_limit: int,parent_trace_id) -> dict:
    """
    Summarizes a community of nodes with a fallback mechanism for handling cases where the number of tokens exceeds a specified limit.

    This function uses a language model (LLM) to generate a summarized description of a community of nodes. If the total token count of the node descriptions exceeds the specified token limit, it recursively splits the nodes into smaller groups and summarizes them individually. In case of failure or single-node groups, a fallback mechanism is employed.

    Args:
        nodes (Dict[str, Any]): 
            A dictionary representing the community of nodes. 
            Each key is a node ID, and the value is another dictionary containing:
                - `type` (List[str]): The types of the node.
                - `description` (List[str]): A list of descriptive strings about the node.
        token_limit (int): 
            The maximum allowable token count for the input to the LLM.

    Returns:
        Dict[str,Any]: 
            A dictionary of node summaries grouped by community. If the process fails for any reason, it returns an error message.

    Raises:
        Exception: 
            Any exception raised during LLM interaction or processing is caught and logged.

    Workflow:
        1. Combine all node descriptions into a single text block and estimate the token count.
        2. If the token count is within the specified limit:
            - Use the LLM to summarize the entire community in one step.
        3. If the token count exceeds the limit:
            - Split the nodes into two smaller groups.
            - Recursively summarize each group.
        4. Handle special cases for groups:
            - For empty groups, return an error message.
            - For single-nodes(which exceeds token limit), summarize the individual node using a fallback mechanism.
        5. Combine the summaries of the groups into a final summary.

    Sub-functions:
        summarize_single_node(node_name: str, node_data: Dict[str, Any]) -> str:
            Summarizes a single node's description using the LLM.
        
        process_group(group: Dict[str, Any]) -> str:
            Processes a group of nodes by summarizing them recursively or falling back to single-node summarization if needed.

    Example Input:
        nodes = {
            "node1": {"type": ["Person"], "description": ["Alice is a software engineer."]},
            "node2": {"type": ["Location"], "description": ["Paris is the capital of France."]},
        }
        token_limit = 100

    Example Output:
        {
            "nodes": {
                "node1": "Alice is a software engineer and key contributor.",
                "node2": "Paris is a global cultural hub."
            }
        }

    Logging:
        - Logs token count, errors, and intermediate summaries for debugging.
        - Logs warnings for single-node fallback cases and empty groups.

    Notes:
        - The function uses a Ollama LLM setup (`OllamaLLM`) with specified prompt templates and parsing logic.
        - It handles both YAML and system message formatting for prompts.
        - If summarization fails at any step, it logs the error and returns an appropriate error message.

    Limitations:
        - The function assumes the availability of a pre-defined `prompt_dict` for LLM prompts.
        - Recursive splitting may increase latency for large communities.
        - Summarization quality depends on the LLM's capabilities and prompt effectiveness.
    """
    # Temporary code for LLM setup
    config = Config()
    expt_llm = config["llm_model"]
    llm = OllamaLLM(temperature=0.2, model=expt_llm)
    yml_parser =YamlOutputParser(pydantic_object=NodeSummarizationOutput)     
    s_prompt_template = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
        template=prompt_dict['graphrag_node_summarization_prompt']['system'],    
        input_variables = ["node_descriptions"],
        partial_variables = {"format_instructions": yml_parser.get_format_instructions()}    
        )
    )
    
    chat_prompt = ChatPromptTemplate.from_messages(
                [MessagesPlaceholder(variable_name="messages")]
                )
    llm_chain = chat_prompt | llm | extract_code
    wrapped_model = ValidationWrapper(parser=yml_parser,graph=llm_chain,pydantic_bm=NodeSummarizationOutput,parent_trace_id=parent_trace_id)
            
    def summarize_single_node(node_name: str, node_data: Dict[str, Any]) -> dict:
        # Summarize a single node's description using the LLM
        node_types = ",".join(node_data['type']) if isinstance(node_data['type'], list) else node_data['type']
        full_text = f"\nNode name = \"{node_name}\"\nNode type = {node_types}\nNode description: {','.join(node_data['description'])}"
        node_prompt_template = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template=prompt_dict['graphrag_single_node_summarization_prompt']['system'],    
                input_variables = ["node_descriptions"],
                partial_variables = {"format_instructions": yml_parser.get_format_instructions()}    
            )
        )
        prompt = node_prompt_template.format(node_descriptions=full_text)
        messages = [prompt]
        try:
            wrapped_state = {"messages":messages, "llm_output":"","error_status":"","iterations":0}
            summarized_description = wrapped_model.invoke(wrapped_state)['llm_output']
            return summarized_description
        except Exception as e:
            logger.error(f"Failed to summarize node `{node_name}`: {e}")
            return {nodes:{node_name:{"summary":"summary failed"}}}

    # Step 1: Combine all node descriptions into a single text
    node_descriptions = {
        node_id: f"\nNode name = \"{node_id}\"\nNode type = {node_data['type']}\nNode description: {','.join(node_data['description'])}"
        for node_id, node_data in nodes.items()
    }
    full_text = "\n".join(node_descriptions.values())
    token_count = estimate_tokens(full_text)

    logger.info(f"Token count for full community: {token_count}, Token limit: {token_limit}")

    # Step 2: Check if the full text exceeds the token limit
    if token_count <= token_limit:
        if(1==len(nodes.keys())):
            node_id, node_data = next(iter(nodes.items()))
            node_summary = summarize_single_node(node_name=node_id,node_data=node_data)
            return node_summary['nodes']
        # If within limit, directly summarize the entire community
        try:
            s_prompt = s_prompt_template.format(node_descriptions=full_text)
            messages = [s_prompt]
            wrapped_state = {"messages":messages, "llm_output":"","error_status":"","iterations":0}
            community_summary = wrapped_model.invoke(wrapped_state)['llm_output']
            logger.info(f"---Community summary : {community_summary}\n")
            return community_summary['nodes']
        except Exception as e:
            logger.error(f"Failed to summarize the full community: {e}")
            return {"community_summarization_error":"Error: Community summarization failed."}

    # Step 3: Split nodes into two groups and recursively summarize
    grouped_nodes = list(nodes.items())
    mid = len(grouped_nodes) // 2
    group_1 = dict(grouped_nodes[:mid])
    group_2 = dict(grouped_nodes[mid:])

    # Step 4: Handle empty or single-node groups using fallback
    def process_group(group: Dict[str, Any]) -> dict:
        if len(group) == 0:
            logger.error("Encountered an empty group during splitting.")
            # Hope was we would never reach here. Yet here we are..
            # If one single node description exceeds the token limit, we reach here.
            # Previous implementation returned a string indicating error from here.
            # This broke the combined_summary = summary_1|summary_2 line because 
            # either one side of | operator is a dictionary and other is a string.
            # Hence we are returning an empty dictionary from here.
            return {} 
        elif len(group) == 1:
            # Fallback: Summarize the single node
            logger.warning(f"Group has a single node exceeding token limit.\n group info:{group}\nApplying fallback.")
            node_id, node_data = next(iter(group.items()))
            summarized_node = summarize_single_node(node_id, node_data)
            return summarized_node['nodes']
        else:
            # Recursively summarize the group
            return summarize_community_nodes(group, token_limit,parent_trace_id=parent_trace_id)

    # Recursively summarize both groups
    summary_1 = process_group(group_1)
    summary_2 = process_group(group_2)
    logger.info(f"Summary_1 = {summary_1}\nSummary_2={summary_2}\n")
    # Step 5: Combine summaries of the two groups
    combined_summary = summary_1|summary_2
    return combined_summary



#---------------support functions-Begin-------------#

def normalize_string(s):
    """Normalize a string by removing special characters and converting to lowercase."""
    return re.sub(r'[\W_]+', '', s).lower()

def find_key_matches(first_list, second_list):
    """Find exact and approximate matches between two list of strings"""
    
    # Step 1: Direct exact match check
    exact_matches = set(second_list) & set(first_list)
    remaining_strings = set(first_list) - exact_matches
    remaining_keys = set(second_list) - exact_matches
    
    # Step 2: Normalization
    normalized_keys = {normalize_string(key): key for key in remaining_keys}
    
    approximate_matches = {}
    missing_keys = []
    
    for s in remaining_strings:
        normalized_s = normalize_string(s)
        if normalized_s in normalized_keys:
            approximate_matches[s] = normalized_keys[normalized_s]
        else:
            missing_keys.append(s)
    
    return missing_keys, approximate_matches

def replace_approx_matches(approximate_matches, data_dict):
    """Replace values in approximate_matches with corresponding values from data_dict."""
    
    for key, value in approximate_matches.items():
        if value in data_dict:  # Check if the value part of approximate_matches exists as a key in data_dict
            data_dict[key] = data_dict[value] # Map it to the corresponding value in data_dict
            del data_dict[value] # remove redundant entry
    
    return data_dict
#---------------support functions-End---------------#

def add_node_summaries_to_intermediate_object(all_communities,file_name,file_path,storage,parent_trace_id):
    """
    Node Summarization with Fallback for Missed Nodes

    This logic is designed to summarize all nodes from the `community['nodes']` dictionary using a summarization function. 
    It ensures no node is left unsummarized by implementing a fallback mechanism for any missed nodes.

    Steps:
    1. **Initial Summarization**: 
    - The function `summarize_community_nodes` is called with the current list of nodes to generate summaries.
    - It returns a dictionary where the node names are keys and the generated summaries are values.

    2. **Detecting Missed Nodes**:
    - After each iteration, the function `find_missing_keys` compares the `community['nodes']` (superset) with `summaries` (subset) to detect nodes that are still missing.

    3. **Fallback Mechanism**:
    - If any nodes are missed, a warning is logged, and the summarization function is called again for the missed nodes.
    - The newly generated summaries are merged into the `summaries` dictionary using the pipe (`|`) operator.

    4. **Termination**:
    - The loop continues until all nodes are successfully summarized (i.e., no missing nodes are found).

    Key Functions:
    - `summarize_community_nodes(nodes, token_limit, parent_trace_id)`: Summarizes the given list of nodes.
    - `find_missing_keys(superset_dict, subset_dict)`: Identifies keys missing in the subset compared to the superset.
    """
    # create list of communities with missing node summmaries
    # if('node_summarized_communities' in all_communities):
    #     node_summerized_communities = all_communities['node_summarized_communities']
    #     graph_with_communities = {key:value for key,value in all_communities.items() if key not in node_summerized_communities}
    # else:
    #     graph_with_communities=all_communities
    # community_names = list(graph_with_communities.keys())
    community_names = list(all_communities['needs_node_summary']) # get snapshot of the list

    # for community_name,community in graph_with_communities.items():
    for community_name in community_names:
        logger.info(f"---{community_name} of {len(all_communities.keys())}\n---")
        community = all_communities[community_name]
        node_list = community['nodes']
        summaries = {}
        while node_list:
            logger.info(f"Summary keys at beginning of loop:\n{summaries.keys()}\n")
            # Generate summaries for the current node list
            new_summaries = summarize_community_nodes(nodes=node_list, token_limit=1000, parent_trace_id=parent_trace_id)
            logger.info(f"new summary keys:\n{new_summaries.keys()}")
            if not new_summaries:
                logger.error("No summaries were generated. Exiting loop to avoid infinite iteration.")
            # Merge new summaries with the existing ones
            summaries = summaries | new_summaries

            logger.info(f"Summary keys after summarize_community_nodes:\n{summaries.keys()}\n")
            # Find any missed nodes
            missed_node_keys, approximate_matches = find_key_matches(community['nodes'].keys(),summaries.keys())
            summaries = replace_approx_matches(approximate_matches,summaries)

            missed_nodes = {key:community['nodes'][key] for key in missed_node_keys}

            if approximate_matches:
                logger.warning(f"Following nodes approximate matches are replaced in summary:\n{approximate_matches}\n")
            if missed_nodes:
                logger.warning(f"Following nodes are missed in summarization:\n{missed_node_keys}\nRetrying summarization.")
            else:    
                logger.info(f"All nodes are summarized successfully.")

            # Update the node list for the next iteration
            node_list = missed_nodes
            
        for node_name in community['nodes']:
            community['nodes'][node_name]['summary'] = summaries[node_name]['summary']

        all_communities[community_name] = community
        all_communities['needs_node_summary'].remove(community_name) # Remove community name from the list 
        
        buffer = io.BytesIO()
        torch.save(all_communities,buffer) # save for persistence   
        buffer_bytes = buffer.getvalue()
        storage.save_file(filename=file_name,data=buffer_bytes,storage_type=file_path)
        logger.info(f"Saved node summaries for {community_name}")
    
    return all_communities

def add_community_summaries_to_intermediate_object(all_communities,file_name,file_path,storage,parent_trace_id):
    """
    Summarize a community's nodes, internal edges, and external edges based on size and complexity. 
    **Algorithm Overview**:
    1. Combine all node summaries, internal edges, and external edges within a community into a single prompt.
    2. Calculate the token usage for the prompt.
    3. If the token count exceeds the limit:
        - Determine the number of splits required using the modulus operator: `Token count of prompt % Token limit`.
        - Split the community into smaller chunks:
            - Split internal edges proportionally to the calculated split ratio.
            - Add all nodes connected to the internal edges into the group.
            - Split external edges similarly and include connected nodes.
            - Prepare a new prompt with the collected information and summarize it to generate a community report.
    4. If the token count is within the limit, summarize the community directly.

    **Important Notes**:
    - **No Fallback Mechanism**: This algorithm assumes that the smallest split will fall within the token limit of the language model. 
      If issues arise, increasing the token limit is preferred, considering advancements in models with extended context length.
    - **Edge Weighting**: While edge prioritization is a potential optimization, it is not implemented in the current version. 
      The algorithm treats all splits as equally important to maintain simplicity and modularity.
    - The "best part is no part" philosophy drives the design, aiming for minimal complexity and robust functionality.

    **Parameters**:
    - `all_communities` (List[Dict]): A list where each dictionary represents a community with:
        - Nodes and their summaries.
        - Internal edges connecting nodes within the community.
        - External edges connecting nodes to other communities.

    **Returns**:
    - Updates the intermediate object with summarized information for each community.
    - Summaries are concise, modular, and structured to maintain key details for downstream tasks.
    

    Contex notes: Next steps 
    - Method to read the all_community.pt file and feed it to this function from outside : Done, implemented in generate_and_summarize_community_graphs()
    - implement token count and modulus based split logic.
    - implement summarization for the most simple pathway of the split logic. Ignore other alternatives
    - Test 
    """
    token_limit = 1000
    communities_count = 0
    summary_missing_communities = []
    communities_count = len(all_communities.keys()) - 3 # 3 keys are used for metadata in all_communities

    # if "summarized_communities" in all_communities:
    #     summary_missing_communities = all_communities.keys() - all_communities['summarized_communities']
    # else:
    #     all_communities['summarized_communities'] = []
    #     summary_missing_communities = all_communities.keys()

    summary_missing_communities = list(all_communities['needs_community_summary']) # Get snapshot of the list

    logger.info(f"List of summary missing communities: {summary_missing_communities}")
    for community_name in summary_missing_communities:
        # if(community_name=='node_summarized_communities'):
        #     continue
        community = all_communities[community_name]
        node_info = "\n".join([f"{name} : {node['summary']}" for name, node in community['nodes'].items()])
        internal_edge_info = "\n".join([f"{name} : {edge['description']}" for name, edge in community['internal_edges'].items()])
        external_edge_info = "\n".join([f"{name} : {edge['description']}" for name, edge in community['external_edges'].items()])

        community_summary_prompt = prompt_dict['graphrag_community_summarization_prompt']['system']
        input_var_names = ["external_edge_info", "internal_edge_info", "node_info"]
        model = sys_prompt_wrapped_call(input_var_names_system=input_var_names,
                                        sys_prompt_template=community_summary_prompt,
                                        parent_trace_id=parent_trace_id,
                                        pydantic_object=CommunityReport
                                        )


        s_prompt = model['system_template'].format(
            node_info=node_info,
            internal_edge_info=internal_edge_info,
            external_edge_info=external_edge_info
        )
        total_tokens = estimate_tokens(s_prompt.content)
        logger.info(f"{total_tokens} of {token_limit}\n")

        # Handle token limit exceeding cases
        if total_tokens > token_limit:
            # Calculate the number of splits required
            num_splits = -(-total_tokens // token_limit)  # Ceiling division
            internal_edges = list(community['internal_edges'].items())
            external_edges = list(community['external_edges'].items())
            
            # Determine the number of edges per split
            internal_split_size = max(1, len(internal_edges) // num_splits)
            external_split_size = max(1, len(external_edges) // num_splits)
            split_summaries = []
            for i in range(num_splits):
                # Split internal edges 
                internal_edge_subset = internal_edges[i * internal_split_size:(i + 1) * internal_split_size]
            
                # Split external edges
                external_edge_subset = external_edges[i * external_split_size:(i + 1) * external_split_size]
                # Add nodes connected to the split external edges
                
                connected_internal_nodes = {
                    edge[0][0] for edge in internal_edge_subset
                }.union({
                    edge[0][1] for edge in internal_edge_subset
                }).union({
                    edge[0][0] for edge in external_edge_subset
                })

                connected_external_nodes = {
                    edge[0][0] for edge in external_edge_subset
                }
                
                # Prepare node subset for this split
                split_nodes = connected_internal_nodes.union(connected_external_nodes)
                logger.info(f"List of connected nodes : \n {split_nodes}")
                
                node_subset = {
                    name: community['nodes'][name]
                    for name in split_nodes
                }
                
                # Prepare new prompt for the split subset
                split_node_info = "\n".join([f"{name} : {node['summary']}" for name, node in node_subset.items()])
                split_internal_edge_info = "\n".join([f"{name} : {edge['description']}" for name, edge in internal_edge_subset])
                split_external_edge_info = "\n".join([f"{name} : {edge['description']}" for name, edge in external_edge_subset])
                
                split_prompt = model['system_template'].format(
                    node_info=split_node_info,
                    internal_edge_info=split_internal_edge_info,
                    external_edge_info=split_external_edge_info
                )
                
                # Summarize the split using LLM
                messages = [split_prompt]
                wrapped_state = {"messages": messages, "llm_output": "", "error_status": "", "iterations": 0}
                split_summary = model['model'].invoke(state=wrapped_state)['llm_output']
                split_summaries.append(split_summary)
            
            # Aggregate split summaries
            '''
            Lets prepare prompts to summarize each individual sections of the split_summaries elements
            Following are the sections of subcommunity summaries. 
            [Key Highlights,Summary,Detailed Findings,Impact Severity Rating,Rating Explanation]
            We dont have to summarize all parts. 
            1. Title : <decide later>
            2. Key Highlights : 
                - check for semantic similarity. If similar sentences found, merge
                - If the token count exceeds 50, then only summarize 
            3. Summary : 
                - check for semantic similarity. If similar sentences found, merge
                - If the token count exceeds 200, then only summarize 
            4. Detailed Findings : 
                - check for semantic similarity. If similar sentences found, merge
                - If the token count exceeds 1000, then only summarize 
            5. Rating explanations and impact severity rating
                - Use rating explanations as prompt input and prompt llm to re-evaluate the rating
                - Also generate rating explanation
            
            Most of these steps rely on semantic similarity check based merging and token count based summarization decision. Lets conceptualize a function to implement this redundant part.
            merge_and_summarize_section() : The generated function
            '''
            # Key highlights merging 
            title = ""
            community_findings = []
            summaries = []
            key_highlights = []
            for summary in split_summaries:
                community_findings.extend(summary['detailed_findings'])
                summaries.append(summary['summary'])
                key_highlights.append(summary['key_highlights'])
                findings_input_str = [f"{finding['description']}" for finding in community_findings]
                title = summary['title']
            
            highlights = merge_and_summarize_section_with_logs(content_list=key_highlights,
                                               similarity_threshold=0.88,
                                               summarize_fn=merge_and_summarize_key_highlights,
                                               token_counter=estimate_tokens,token_limit=50,parent_trace_id=parent_trace_id)
            logger.debug(f"key highlights after merge and summarize:\n{highlights}\n")
            summary = merge_and_summarize_section_with_logs(content_list=summaries,
                                                    similarity_threshold=0.88,
                                                    summarize_fn=merge_and_summarize_summary,
                                                    token_counter=estimate_tokens,token_limit=500,parent_trace_id=parent_trace_id)
            logger.debug(f"Summary after merge and summarize:\n{summary}\n")
            new_findings = merge_and_summarize_section_with_logs(content_list=community_findings,content_string_list=findings_input_str,
                                                    similarity_threshold=0.88,
                                                    summarize_fn=merge_and_summarize_findings,
                                                    token_counter=estimate_tokens,token_limit=800,parent_trace_id=parent_trace_id)
            logger.debug(f"Findings after merge and summarize:\n{new_findings}\n")

            community['title'] = title
            community['summary'] = summary
            community['key_highlights'] = key_highlights
            community['detailed_findings'] = new_findings
            community = calculate_impact_severity_rating(community,parent_trace_id=parent_trace_id)
            all_communities[community_name] = community

            all_communities['needs_community_summary'].remove(community_name)
            buffer = io.BytesIO()
            torch.save(all_communities,buffer) # save for persistence   
            buffer_bytes = buffer.getvalue()
            storage.save_file(filename=file_name,data=buffer_bytes,storage_type=file_path)
            logger.info(f"Saved summary for {community_name} of {communities_count} communities")
        else:
            # Token count is within limit, summarize directly
            messages = [s_prompt]
            wrapped_state = {"messages":messages, "llm_output":"","error_status":"","iterations":0}
            # Store or process the summary as needed
            summary = model['model'].invoke(state=wrapped_state)['llm_output']
            community['summary'] = summary
            all_communities[community_name] = community
            
            all_communities['needs_community_summary'].remove(community_name)
            
            buffer = io.BytesIO()
            torch.save(all_communities,buffer) # save for persistence   
            buffer_bytes = buffer.getvalue()
            storage.save_file(filename=file_name,data=buffer_bytes,storage_type=file_path)
            logger.info(f"Saved summary for {community_name} of {communities_count} communities")
    
        


    return all_communities


#---------------Merge&Summarize Logic-Begin-------------#
def merge_and_summarize_section_with_logs(content_list, token_limit, similarity_threshold,
                                           summarize_fn, token_counter,parent_trace_id,
                                           content_string_list=None):
    """
    Merges semantically similar sentences from a content list and summarizes the result 
    if the token count exceeds the specified limit.

    Args:
        content_list (list): List of strings to be processed.
        token_limit (int): Maximum token count allowed before summarization.
        similarity_threshold (float): Minimum cosine similarity for grouping sentences.
        summarize_fn (callable): A function to summarize text, taking text and token limit as input.
        token_counter (callable): A function to calculate the token count of a text.

    Returns:
        str: Merged and optionally summarized text.

    Raises:
        ValueError: If `content_list` is empty.
        TypeError: If `summarize_fn` or `token_counter` is not callable.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    

    logger.info("Starting merge_and_summarize_section function.")

    def create_sublist_by_ignoring_indices(lst, ignore_set:set):
        """
        Creates a sublist from lst by ignoring elements at specified indices.

        Parameters:
        - lst: The original list from which elements are to be included in the sublist.
        - ignore_sets: A set of indices specifying positions of elements to ignore.

        Returns:
        - A new list with elements not at the ignored indices.
        """
        # Use a list comprehension to filter out the unwanted indices
        sublist = [elem for idx, elem in enumerate(lst) if idx not in ignore_set]

        return sublist

    # Validate inputs
    if not content_list:
        logger.error("The content_list is empty.")
        raise ValueError("The content_list cannot be empty.")
    if not content_string_list:
        # Special handling for key findings
        logger.warning("The content_string_list is empty.Assigning content_list list to content_string_list")
        logger.info(f"Content list = \n{content_list}")
        content_string_list = content_list
    if not callable(summarize_fn):
        logger.error("summarize_fn is not a callable function.")
        raise TypeError("summarize_fn must be a callable function.")
    if not callable(token_counter):
        logger.error("token_counter is not a callable function.")
        raise TypeError("token_counter must be a callable function.")

    logger.debug(f"Content information: \n{content_list}")
    logger.debug("Initializing sentence embedding model.")
    # Initialize a pre-trained sentence embedding model
    

    logger.info("Calculating sentence embeddings for content list.")
    # Step 1: Calculate sentence embeddings
    embeddings = embedding_model.encode(content_string_list)
    logger.debug(f"Calculated embeddings for {len(content_string_list)} sentences.")
    logger.debug(f"Content: \n{"\n".join(content_string_list)}")
    # Step 2: Semantic similarity check and merging
    merged_content = []
    used_indices = set()
    redundant_indices = set()

    logger.info("Starting semantic similarity check and merging process.")
    for i, sentence in enumerate(content_string_list):
        if i in used_indices:
            logger.debug(f"Sentence at index {i} already used, skipping.")
            continue
        
        group = [sentence]
        for j in range(i + 1, len(content_string_list)):
            if j in used_indices:
                logger.debug(f"Sentence at index {j} already used, skipping.")
                continue
            
            similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            logger.debug(f"Calculated similarity between sentence {i} and {j}: {similarity:.4f}")

            if similarity >= similarity_threshold:
                logger.info(f"Sentences at index {i} and {j} are semantically similar (similarity: {similarity:.4f})\nFirst sentence : {content_string_list[i]}\nSecond sentence: {content_string_list[j]}")
                used_indices.add(j)
                redundant_indices.add(j)

        merged_content.append(" ".join(group))
        used_indices.add(i)

    logger.info("Merging process complete.")
    logger.debug(f"Merged content: {merged_content}")

    # Step 3: Check token count after merging
    logger.info("Checking token count of merged content.")
    merged_text = " ".join(merged_content)
    total_tokens = token_counter(merged_text)
    logger.debug(f"Total token count: {total_tokens}")
    logger.info(f"Content list length: {len(content_list)}")
    new_content_list = create_sublist_by_ignoring_indices(lst=content_list,ignore_set=redundant_indices)
    logger.info(f"New content list length: {len(new_content_list)}")

    # Step 4: Summarization if token count exceeds limit
    if total_tokens > token_limit:
        logger.info(f"Token count ({total_tokens}) exceeds the limit ({token_limit}). Summarizing content.")
        number_of_chunks = -(-total_tokens // token_limit)
        content_split_ratio = (len(new_content_list)//number_of_chunks)
        summarizer_token_limit = token_limit // content_split_ratio
        if(content_split_ratio > 1):
            split_content_list = split_list(new_content_list,content_split_ratio)
            logger.info(f"Number of chunks= {number_of_chunks}, content split = {content_split_ratio}\nList length = {len(split_content_list[0])}, Summarizer token limit = {summarizer_token_limit}")
            summarized_findings = []
            for chunk_list in split_content_list:
                fndg_summary = summarize_fn(chunk_list, summarizer_token_limit,parent_trace_id)
                if(isinstance(fndg_summary,list)):
                    summarized_findings.extend(fndg_summary)
                elif(isinstance(fndg_summary,str)):
                    summarized_findings.append(fndg_summary)
            # merged_text = "\n".join(summarized_findings)
            merged_text = str(summarized_findings)
        else:
            logger.error(f"Number of chunks > content list. Need to implement fallback logic")
    else:
        logger.info(f"Token count ({total_tokens}) is within the limit ({token_limit}). No summarization needed.")

    logger.info(f"Finished processing.\nFinal Token count = {token_counter(merged_text)}\n Returning final text.")
    return merged_text

def merge_and_summarize_key_highlights(community_highlights,token_limit,parent_trace_id):
    highlights_list = "- "+"\n -".join(community_highlights)
    input_var_names = ['highlights_list','token_limit']
    template = prompt_dict['graphrag_community_key_highlights_summary']['system']
    model = sys_prompt_wrapped_call(pydantic_object=Highlights,sys_prompt_template=template,input_var_names_system=input_var_names,parent_trace_id=parent_trace_id)
    s_prompt = model['system_template'].format(highlights_list=highlights_list,token_limit=token_limit)
    messages = [s_prompt]
    wrapped_state = {"messages":messages,"llm_output":"","error_status":"","iterations":0}
    summary = model['model'].invoke(wrapped_state)['llm_output']
    return summary['key_highlights']

def merge_and_summarize_summary(community_summaries,token_limit,parent_trace_id):
    summary_list = "- "+"\n -".join(community_summaries)
    input_var_names = ['summary_list','token_limit']
    template = prompt_dict['graphrag_sub_community_summary']['system']
    model = sys_prompt_wrapped_call(pydantic_object=SubCommunitySummary,
                                    sys_prompt_template=template,input_var_names_system=input_var_names,
                                    parent_trace_id=parent_trace_id)
    s_prompt = model['system_template'].format(summary_list=summary_list,token_limit=token_limit)
    messages = [s_prompt]
    wrapped_state = {"messages":messages,"llm_output":"","error_status":"","iterations":0}
    summary = model['model'].invoke(wrapped_state)['llm_output']
    return summary['summary']

def merge_and_summarize_findings(findings_input,token_limit,parent_trace_id):
    logger.info(f"[merge_and_summarize_findings] Findings : {findings_input}\n Length : {len(findings_input)}")
    findings_string = [f"- Insight : {details['insight']}\n- Description : {details['description']}\n" for details in findings_input] 
    findings = "\n".join(findings_string)

    input_var_names = ['findings','token_limit']
    template = prompt_dict['graphrag_findings_consolidation']['system']
    model = sys_prompt_wrapped_call(pydantic_object=Findings_summary,
                                    sys_prompt_template=template,input_var_names_system=input_var_names,
                                    parent_trace_id=parent_trace_id,temperature=0.4)
    s_prompt = model['system_template'].format(findings=findings,token_limit=token_limit)
    messages = [s_prompt]
    wrapped_state = {"messages":messages,"llm_output":"","error_status":"","iterations":0}
    findings_summary = model['model'].invoke(wrapped_state)['llm_output']
    return findings_summary['detailed_findings']

def calculate_impact_severity_rating(community:CommunityReport,parent_trace_id,token_limit=100)->CommunityReport:
    logger.info(f"[calculate_impact_severity_rating]:")
    findings = str(community['detailed_findings'])
    summary = str(community['summary'])
    highlights = str(community['key_highlights'])  
            

    input_var_names = ['findings','summary','highlights','token_limit']
    template = prompt_dict['graphrag_impact_severity_rating']['system']
    model = sys_prompt_wrapped_call(pydantic_object=Impact_severity,
                                    sys_prompt_template=template,input_var_names_system=input_var_names,
                                    parent_trace_id=parent_trace_id,temperature=0.4)
    s_prompt = model['system_template'].format(findings=findings,token_limit=token_limit,highlights=highlights,summary=summary)
    messages = [s_prompt]
    wrapped_state = {"messages":messages,"llm_output":"","error_status":"","iterations":0}
    impact = model['model'].invoke(wrapped_state)['llm_output']
    community['impact_severity_rating'] = impact['impact_severity_rating']
    community['rating_explanation'] = impact['rating_explanation']
    return community
#---------------Merge&Summarize Logic-End---------------#
#---------------Community Summarization-End---------------#

'''
### Init
1. Read networkX graph : wrapper function to read the graph. Now following sources are supported
    - From intermediate storage
    - From in memory variable (state)
    Refer GraphLoader for implementation details

2. Apply the community detection and summarization
    - Create communities using Leiden community detection 
    - Community summarisation
        - Generate community summaries
        - Store summaries in vector database
'''

class GraphRAGGlobalSearch(RetrievalPlugin):
    def __init__(self,**kwargs):
        sources = kwargs.get("sources",[])
        self.storage = kwargs.get("storage",None)
        super().__init__(sources=sources)
        
        self.global_search_df:PandasDataFrame = None

        self.artifact_path = "Odysseus"
        self.bodhi_artifacts_path = "Bodhi"

        self.process()
        
    def __call__(self, *args, **kwds)->RetrievalOutput:
        # check if all graphs are filled properly and instance of netx
        return self.retrieve(*args, **kwds)
    
    # Preprocessing pipeline
    def process(self): 
        # Check if already processed artifacts are available or not
        # Graph loading and Community summarization 

        for source in self.source_list:
            storage_path =f"{self.artifact_path}/graphRAG/{source}/"
            final_graph_output = f"{source}_global_search.pt"
            if(self.storage.file_exists(final_graph_output,storage_path)):
                graph = self.storage.read_file(final_graph_output,storage_path)
                graph_bytes = io.BytesIO(graph)
                self.graph = torch.load(graph_bytes)

            else:
                # Assumption: Bodhi pipeline executed before Odysseus. Hence netX graphs are available in predefined locations
                graph_path = f"{source}_graph.yml"
                bodhi_path = f"{self.bodhi_artifacts_path}/{source}/"
                graph_bytes = self.storage.read_file(graph_path,bodhi_path)
                graph_yml = yaml.safe_load(graph_bytes)
                # Add logger statement to print the path 
                self.graph[source] = {}
                self.graph[source]["nx_graph"] = nx.node_link_graph(graph_yml)
                self.graph[source]["communities"] = generate_and_summarize_community_graphs(source_name=source,
                                                                                                nx_graph=self.graph[source]["nx_graph"],
                                                                                                storage=self.storage)
                # Temporarily Save processed graph
                buffer = io.BytesIO()
                torch.save(self.graph,buffer) # save for persistence   
                buffer_bytes = buffer.getvalue()
                self.storage.save_file(data=buffer_bytes,filename=final_graph_output,storage_type=storage_path)
                
        gs_df_community = {}
        # save the community information to dataframe for gloabl search
        for source,values in self.graph.items():
            communities = values['communities']
            del communities['needs_node_summary']
            del communities['needs_community_summary']
            # Define required keys
            required_keys = ["title", "summary","key_highlights","impact_severity_rating","rating_explanation"]  # Only include these keys
            filtered_community_info = {
                k: {key: v[key] for key in required_keys}|{'community_level':v['metadata']['community_level']} 
                for k, v in communities.items()
                }

            gs_df_community[source] = pd.DataFrame(filtered_community_info.values())
        
        self.global_search_df = gs_df_community

    def retrieve(self,query,source,community_level=0)->RetrievalOutput:
        # Retrieval pipeline
        # 1. Get all community summaries 
        df = self.global_search_df[source]
        
        retrievals:RetrievalOutput = {}
        retrievals['modality'] = "text"
        retrievals['data'] = {index: row.dropna().to_dict() for index, row in df.iterrows()}
        retrievals['metadata'] = {}
        
        # print(retrievals['data'][0]['summary'])
        return retrievals






#---------------Test code-Begin-------------#  
#---------------Summarize community test code-Begin-------------#
if __name__ == "__main__":
    # global_search = GraphRAGGlobalSearchPlugin(source_list=['low_level'])
    # global_search(source="low_level")
    global_search = GraphRAGGlobalSearch(source_list=['lfs'])
    global_search(source="lfs")
#---------------Summarize community test code-End---------------#

#---------------Summarize nodes test code-Begin-------------#
if __name__ == "__main__3":
    # Configure logging
    logger.basicConfig(
                        filename="Bodhi.log",
                        level=logger.INFO,
                        filemode="w",
                        format='%(asctime)s - %(levelname)s - %(message)s'
                        )
    def read_yml(file_path):
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
            return data    
    entities = read_yml("deduplicated_file_qwen2.5.yml")['entities']
    relations = read_yml("deduplicated_relns_qwen2.5.yml")
    # logger.info(f"---entities---\n{entities}\n---relations---\n{relations}")
    # Langfuse tracing 
    parent_trace_id = create_unique_trace_id()

    nx_graph = generate_graph_from_dict(entities=entities,relationships=relations)
    communities = cluster_graph(nx_graph)
    all_communities = process_community_list(community_list=communities,global_graph=nx_graph)
    add_node_summaries_to_intermediate_object(all_communities=all_communities,parent_trace_id=parent_trace_id)
    
    with(open("artifacts/community_node_summaries.yml",mode="w")) as node_summaries:
        yaml.dump(community_summaries,node_summaries,default_flow_style=False)
    with(open("artifacts/final_community_file.yml",mode="w")) as final_community_file:
        yaml.dump(all_communities,final_community_file,default_flow_style=False)
#---------------Summarize nodes test code-End---------------#

#---------------Tokenizer test code-Begin-------------#
def test_token_estimators(text: str):
    print("Testing Hugging Face:")
    print(_estimate_tokens_hf(text))

    # print("Testing SentencePiece:")
    # print(estimate_tokens_spm(text, model_path="/path/to/spm.model"))

    print("Testing Tiktoken:")
    print(_estimate_tokens_tiktoken(text))

if __name__ == "__main__2":
    text = "LTTNG_MY_SUBSYS_H is the header file where the tracepoint event 'my_subsys_my_event' is defined."
    test_token_estimators(text)
#---------------Tokenizer test code-End---------------#
#---------------Test code-End---------------# 
