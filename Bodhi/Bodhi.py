import networkx as nx
import uuid
import yaml
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

from langchain_ollama import OllamaLLM
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver

from .states.bodhi import *

from infra.prompts.config_loader import get_settings
from infra.utils.utils import *
# Storage
from Sanchayam.Sanchayam import Sanchayam

from infra.logs.prophet_logger import *




class Bodhi:
    def __init__(self,parent_trace_id):
        # Langfuse
        self.parent_trace_id = parent_trace_id

        #storage 
        cwd = os.getcwd()
        config = Config(config_path=f"{cwd}/config.yml")
        
        config_path = f"{cwd}/config.yml"
        self.storage = Sanchayam(config_path=config_path)

        # Prompt related 
        self.prompt_dict = get_settings()
        expt_llm =config["llm_model"]
        self.llm = OllamaLLM(temperature=0.2, model=expt_llm)
        
        self.similarity_model = SemanticSimilarity()

        # logger setup
        log_file_handler = configure_rotating_file_handler("infra/logs/Bodhi.log",max_bytes=500*1024)
        self.logger = get_pipeline_logger(file_handler=log_file_handler,pipeline_name="Bodhi")

        # For persistence 
        self.checkpointer = MemorySaver()
        self.config = {"configurable": {"thread_id": uuid.uuid4()},"recursion_limit":4000}

        self.artifacts_dir = "Bodhi"
        self.storage_dir = "data"
        self.source_name = None
        self.graph_info_storage = None
        self.dedup_interm_data_path = None
        self.interm_data_path = None
        self.text_unit_path = None
        self.graph_path = None

        self.token_limit = config["token_limit"]

        # graph related 
        self.workflow = StateGraph(BodhiState,input=BodhiInputState,output=BodhiState)
        self.app = None
        self._setup_graph()

    def invoke(self,state):
        self.app.invoke(state,self.config)
    
    def get_state(self):
        return self.app.get_state(self.config)

    def get_state_history(self):
        return self.app.get_state_history(self.config)

    def _setup_graph(self):
        # Graph structure code 
        self.workflow.add_node("extract_graph",self.extract_graph)
        self.workflow.add_node("extract_entities",self.extract_entities)    
        self.workflow.add_node("validate_entities",self.validate_entities)
        self.workflow.add_node("extract_relationships",self.extract_relationships)
        self.workflow.add_node("validate_relationships",self.validate_relationships)
        self.workflow.add_node("build_graph",self.build_graph)

        self.workflow.add_edge(START,"extract_graph")
        self.workflow.add_edge("extract_entities","validate_entities")
        self.workflow.add_edge("extract_relationships","validate_relationships")
        self.workflow.add_edge("build_graph",END)

        self.app = self.workflow.compile(checkpointer = self.checkpointer)

    # define node functions 
    
    def extract_graph(self, state) -> Command[Literal["build_graph","extract_entities"]]:
        """
        Processes the input file to extract text units, orchestrates entity and relationship extraction, 
        and manages intermediate file storage for the Bodhi pipeline.

        This function verifies the presence of required files, converts PDFs to Markdown if necessary, 
        and determines the appropriate next step in the extraction pipeline. 
        It ensures continuity by maintaining intermediate states and saving extracted information at each step.

        Args:
            state (dict): The current state of the extraction pipeline, containing metadata and progress details.

        Returns:
            Command: A command object that updates the pipeline state and directs execution to the next node.

        Workflow:
        1. **Graph Database Content Verification**
            - Checks if required artifacts exist in the artifacts directory.
            - Identifies missing artifacts and determines processing needs.
        
        2. **Text Extraction and Processing**
            - If source files exist but text units are not yet extracted, converts PDFs to Markdown.
            - Splits text into smaller chunks for processing.
        
        3. **State Management and Orchestration**
            - Determines the appropriate next step based on available intermediate files.
            - If deduplicated intermediate data exists, proceeds to graph building.
            - If partial extraction data exists, resumes entity extraction from the last processed chunk.
            - If no extracted data is found, initializes text extraction from the source.
        
        4. **Intermediate Data Handling**
            - Saves extracted entities and relationships after each chunk is processed.
            - Maintains continuity by updating chunk indices and resetting temporary state variables.
        
        Transitions:
            - "build_graph": If all text chunks are processed, move to graph construction.
            - "extract_entities": If text extraction is incomplete, continue extracting entities.
        
        Raises:
            ValueError: If the specified source file does not exist.
        """

        node_name = "extract_graph"
        self.logger.info(f"\n---NODE: {node_name}---")
        loc_state = state
        
        if(self.source_name is None):
            if 'source_path' not in loc_state or loc_state['source_path'] is None:
                raise ValueError("Missing or None source_path in loc_state")
            self.source_name = os.path.splitext(loc_state['source_path'])[0]        # Remove file extention
            self.graph_info_storage = f"{self.artifacts_dir}/{self.source_name}/"
            self.dedup_interm_data_path = f"{self.source_name}_dd_intermediate_data.yml"
            self.interm_data_path = f"{self.source_name}_intermediate_data.yml"
            self.text_unit_path = f"{self.source_name}_text_units.yml"
            self.raw_text_path = f"{self.source_name}_text.yml"
            self.graph_path = f"{self.source_name}_graph.yml"

        # Check if deduplicated graph info is available in filesystem
        if self.storage.file_exists(self.dedup_interm_data_path,self.graph_info_storage):
            self.logger.info(f"Found depulicated intermediate file. Proceeding with graph building..")
            goto = "build_graph"

        # check if chunks file and intermediate_data files are available 
        # Continue from the intermediate state
        elif ('number_of_chunks' not in loc_state)&\
             self.storage.file_exists(self.text_unit_path,self.graph_info_storage)&\
             (self.storage.file_exists(self.interm_data_path,self.graph_info_storage))\
            :
            text_units = self.storage.read_file(self.text_unit_path,self.graph_info_storage)
            loc_state['text_chunks'] = yaml.safe_load(text_units)
            loc_state['number_of_chunks'] = len(loc_state['text_chunks'])

            intermediate_content = self.storage.read_file(self.interm_data_path,self.graph_info_storage)
            intermediate_content = yaml.safe_load(intermediate_content)

            loc_state['entities'] = intermediate_content.get('entities', [])
            loc_state['relationships'] = intermediate_content.get('relationships', [])

            loc_state['chunk_index'] = max(intermediate_content['chunk_indices'])+1 # start extraction from next chunk  
           
            self.logger.info(f"Found intermediate file, Continuing from index {loc_state['chunk_index']} of {loc_state['number_of_chunks']}")
            
            if((loc_state['chunk_index'])>=(loc_state['number_of_chunks'])):
                goto = "build_graph"
            else:
                goto = "extract_entities"
        else: 
            if('number_of_chunks' not in loc_state.keys()): # => Very first iteration. Need to extract text chunks from source file
                if(self.storage.file_exists(filename=self.text_unit_path,storage_type=self.graph_info_storage)):
                    # Extracted text units file is available. Load from the the file. 
                    text_units = self.storage.read_file(filename=self.text_unit_path,storage_type=self.graph_info_storage)
                    text_units = yaml.safe_load(text_units)
                else: 
                    if self.storage.file_exists(loc_state['source_path'],self.storage_dir):
                        file_path = self.storage.get_absolute_path(loc_state['source_path'],self.storage_dir)

                        text_units = self._extract_text_chunks(file_path=file_path,token_limit=self.token_limit)
                        if not isinstance(text_units, list) or len(text_units) == 0:
                            raise ValueError(f"Text extraction failed for {file_path}. No chunks generated.")
                        #---------------<source>_chunks.yml : 1st intermediate file-Begin-------------#
                        text_unit_bytes = b"".join(s.encode('utf-8') for s in yaml.dump(text_units)) # convert to bytes
                        self.storage.save_file(filename=self.text_unit_path,
                                            data=text_unit_bytes,
                                            storage_type=self.graph_info_storage)
                        self.logger.info(f"Saved {len(text_units)} text units")
                        #---------------<source>_chunks.yml : 1st intermediate file-End-------------#
                    else:
                        raise(ValueError(f"File doesn't exist{loc_state['source_path']}"))

                loc_state['number_of_chunks'] = len(text_units)
                loc_state['chunk_index'] = 0
                loc_state['text_chunks'] = text_units

            self.logger.info(f"\n---NODE: {node_name}---INFO: Chunk Index={loc_state['chunk_index']+1}/{loc_state['number_of_chunks']}---\n")
            
            # Check if reached last text chunk
            if((loc_state['chunk_index'])>=(loc_state['number_of_chunks']-1)): 
                self.logger.info(f"---NODE: {node_name}---INFO: Reached last chunk. Going to build_graph---")
                goto = "build_graph" # Go to graph builder
            else: # Repeat extraction for the next chunk 
                # Before repeating the extraction pipeline, we have to store the already extracted information 
                # into yml/json formatted file & then clean up the entity/relations state variables 
                # Save the extracted information 
                if('entities' in loc_state):
                    self.logger.info(f"---NODE: {node_name}---INFO: saving to intermediate file---")
                    #---------------<source>_intermediate_data.yml-Begin-------------#
                    self.save_nodes_n_relns_to_intermediate_file(loc_state['entities']['entities'],
                                                loc_state['relationships'],type=True,
                                                chunk_index=loc_state['chunk_index']
                                                )
                    #---------------<source>_intermediate_data.yml-End-------------#
                    # Reset the relationships and entitites state variables
                    loc_state['entities'] = []
                    loc_state['relationships'] = []
                    loc_state['chunk_index'] += 1
                else:
                    self.logger.info(f"\n---NODE: {node_name}---INFO:No entities in loc_state. Proceed with entity extraction---\n")
                goto = "extract_entities"
                
        return Command(update = loc_state,goto=goto)

    def extract_entities(self,state):
        # print("\n---hello from extract_entities---\n")
        node_name = "extract_entities"
        self.logger.info(f"\n---NODE: {node_name}---")
        loc_state = state
        yml_parser = YamlOutputParser(pydantic_object=EntityExtractionOutput)     
            
        if ("entity_validation_feedback" in loc_state and
            loc_state["entity_validation_feedback"]['loop_required'] == True):
            print("\n--entity_validation_feedback--\n")
            # reset loop variable 
            loc_state["entity_validation_feedback"]['loop_required'] = False
            reasons_to_loop = f"{loc_state["entity_validation_feedback"]["reasons_to_loop"]} \n {loc_state["entity_validation_feedback"]["additional_comments"]}"
            content = loc_state['text_chunks'][loc_state['chunk_index']]  
            entity_types = loc_state['entity_types']

            # Prompt setup
            extract_entities_reflection = self.prompt_dict['extract_entities_reflection']

            s_prompt_template = SystemMessagePromptTemplate(
                prompt=PromptTemplate(
                template=extract_entities_reflection['system'],    
                input_variables = ["content","reasons_to_loop","entity_types"],
                partial_variables = {"format_instructions":yml_parser.get_format_instructions()}
                )
            )
            s_prompt = s_prompt_template.format(content=content,
                                                reasons_to_loop = reasons_to_loop,entity_types=entity_types)
            messages = [s_prompt]
            chat_prompt = ChatPromptTemplate.from_messages(
                        [MessagesPlaceholder(variable_name="messages")]
                        )
            llm_chain = chat_prompt | self.llm | extract_code  
            wrapped_state = {"messages":messages, "llm_output":"","error_status":"","iterations":0}
            wrapped_model = ValidationWrapper(llm=self.llm,parent_trace_id=self.parent_trace_id,node_name=node_name,
                                            parser=yml_parser,graph=llm_chain,pydantic_bm=EntityExtractionOutput)
            entities = wrapped_model.invoke(wrapped_state)['llm_output']
            loc_state['entities'] = self._update_entities(loc_state['entities'],new_entities=entities)
            entity_types = loc_state.get('entity_types')
            for entity in entities.get('entities', []):
                entity_type = entity.get('type')
                entity_types.add(entity_type)
            loc_state['entity_types'] = entity_types
            
        else:
            print("\n--entity_extraction--\n")
            extract_entities_prompt = self.prompt_dict['extract_entities']
            human_prompt_template = HumanMessagePromptTemplate(
                    prompt=PromptTemplate(
                        template=extract_entities_prompt['user'],
                        input_variables=["content"],
                        partial_variables={"format_instructions": yml_parser.get_format_instructions()}    
                    )
            )
            s_prompt_template = SystemMessagePromptTemplate(
                prompt=PromptTemplate(
                template=extract_entities_prompt['system'],    
                input_variables = ["entity_types"]
                )
            )
            print(f"chunk index = {loc_state['chunk_index']+1}")
            content = loc_state['text_chunks'][loc_state['chunk_index']]  
            entity_types = ["Concepts","Components","Processes","DataTypes","Inputs and Outputs",
                            "Configuration and Parameters","Entities (Real-World Objects)","Validation and Evaluation",
                            "Technologies and Frameworks","Constraints and Requirement","Results and Outcomes"]
            # repeated code 
            s_prompt = s_prompt_template.format(entity_types=entity_types)
            h_prompt = human_prompt_template.format(content=content)    
            messages = [s_prompt,h_prompt]
            chat_prompt = ChatPromptTemplate.from_messages(
                        [MessagesPlaceholder(variable_name="messages")]
                        )
            llm_chain = chat_prompt | self.llm | extract_code  
            wrapped_state = {"messages":messages, "llm_output":"","error_status":"","iterations":0}
            wrapped_model = ValidationWrapper(llm=self.llm,parent_trace_id=self.parent_trace_id,node_name=node_name,
                                            parser=yml_parser,graph=llm_chain,pydantic_bm=EntityExtractionOutput)
            
            entities = wrapped_model.invoke(wrapped_state)['llm_output']
            # Parse entity types from the output
            entity_types = set()
            for entity in entities.get('entities', []):
                entity_type = entity.get('type')
                entity_types.add(entity_type)
              
            loc_state['entities'] = entities
            loc_state['entity_types'] = entity_types
            loc_state['entity_iter_counter'] = 0

        return loc_state

    def validate_entities(self,state)->Command[Literal["extract_relationships","extract_entities"]]:
        # print("\n---hello from validate_entities---\n")
        node_name="validate_entities"
        self.logger.info(f"\n---NODE: {node_name}---")
        loc_state = state
        yml_parser =YamlOutputParser(pydantic_object=EntityValidationFeedback)     
        extract_entities_prompt = self.prompt_dict['missing_entities_prompt']
        loc_state['entity_iter_counter'] += 1
        max_iterations = 4

        s_prompt_template = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
            template=extract_entities_prompt['system'],    
            input_variables = ["content","extracted_entities","entity_iter_counter"],
            partial_variables = {"format_instructions": yml_parser.get_format_instructions()}    
            )
        )
        extracted_entities = loc_state['entities']
        content = loc_state['text_chunks'][loc_state['chunk_index']]  
        # repeated code 
        s_prompt = s_prompt_template.format(extracted_entities=extracted_entities,content=content,
                                            iteration_counter=loc_state['entity_iter_counter'],max_iterations=max_iterations)
        messages = [s_prompt]
        chat_prompt = ChatPromptTemplate.from_messages(
                    [MessagesPlaceholder(variable_name="messages")]
                    )
        llm_chain = chat_prompt | self.llm | extract_code  
        wrapped_state = {"messages":messages, "llm_output":"","error_status":"","iterations":0}
        wrapped_model = ValidationWrapper(llm=self.llm,parent_trace_id=self.parent_trace_id,node_name=node_name,
                                        parser=yml_parser,graph=llm_chain,pydantic_bm=EntityValidationFeedback)
        
        entity_validation_out = wrapped_model.invoke(wrapped_state)['llm_output']
        loc_state['entity_validation_feedback'] = entity_validation_out
        if(entity_validation_out['loop_required']):
            goto = "extract_entities"
            
        else: 
            goto = "extract_relationships"
        return Command(update=loc_state,goto=goto)

    def extract_relationships(self,state):
        # print("\n---hello from extract_relationships---\n")
        node_name = "extract_relationships"
        self.logger.info(f"\n---NODE: {node_name}---")
        loc_state = state
        yml_parser =YamlOutputParser(pydantic_object=RelationshipExtractionOutput) 
        if "relations_validation_feedback" in loc_state and loc_state["relations_validation_feedback"]['loop_required'] == True:
            print("\n--relations_validation_feedback--\n")
            # reset loop variable 
            loc_state["relations_validation_feedback"]['loop_required'] = False
            reasons_to_loop = f"{loc_state["relations_validation_feedback"]["reasons_to_loop"]} \n {loc_state["relations_validation_feedback"]["additional_comments"]}"
            content = loc_state['text_chunks'][loc_state['chunk_index']]  
            
            # Prompt setup
            extract_relations_feedback = self.prompt_dict['extract_relations_feedback']
            extracted_entities = loc_state['entities']
            extracted_entity_names = [entity["name"] for entity in extracted_entities["entities"]]
            s_prompt_template = SystemMessagePromptTemplate(
                prompt=PromptTemplate(
                template=extract_relations_feedback['system'],    
                input_variables = ["content","extracted_entity_names","validation_reflection"],
                partial_variables = {"format_instructions":yml_parser.get_format_instructions()}
                )
            )
            s_prompt = s_prompt_template.format(content=content,
                                                extracted_entity_names= extracted_entity_names,validation_reflection=reasons_to_loop)
            messages = [s_prompt]
            chat_prompt = ChatPromptTemplate.from_messages(
                        [MessagesPlaceholder(variable_name="messages")]
                        )
            llm_chain = chat_prompt | self.llm | extract_code  
            wrapped_state = {"messages":messages, "llm_output":"","error_status":"","iterations":0}
            wrapped_model = ValidationWrapper(llm=self.llm,parent_trace_id=self.parent_trace_id,node_name=node_name,
                                            parser=yml_parser,graph=llm_chain,pydantic_bm=EntityExtractionOutput)
            relations = wrapped_model.invoke(wrapped_state)['llm_output']
            loc_state['relationships'] = self._update_relationships(loc_state['relationships'],new_relationships=relations)
            return loc_state
            
        else:
            print("\n--extract_relationships--\n")
            extract_relations_prompt = self.prompt_dict['extract_relationships']
            
            s_prompt_template = SystemMessagePromptTemplate(
                prompt=PromptTemplate(
                template=extract_relations_prompt['system'],    
                input_variables = ["content","extracted_entities","extracted_entities_names"],
                partial_variables={"format_instructions": yml_parser.get_format_instructions()}    
                )
            )
            extracted_entities = loc_state["entities"]
            content = loc_state['text_chunks'][loc_state['chunk_index']]  
            extracted_entities_names = [entity["name"] for entity in extracted_entities["entities"]]
            # repeated code 
            s_prompt = s_prompt_template.format(extracted_entities=extracted_entities,extracted_entities_names=extracted_entities_names,
                                                content=content)
            messages = [s_prompt]
            chat_prompt = ChatPromptTemplate.from_messages(
                        [MessagesPlaceholder(variable_name="messages")]
                        )
            llm_chain = chat_prompt | self.llm | extract_code  
            wrapped_state = {"messages":messages, "llm_output":"","error_status":"","iterations":0}
            wrapped_model = ValidationWrapper(llm=self.llm,parent_trace_id=self.parent_trace_id,node_name=node_name,
                                            parser=yml_parser,graph=llm_chain,pydantic_bm=EntityExtractionOutput)
            
            relations = wrapped_model.invoke(wrapped_state)['llm_output']
            loc_state['relationships'] = relations['relationships']
            loc_state['relations_iter_counter'] = 0
        return loc_state

    def validate_relationships(self,state)->Command[Literal["extract_graph","extract_relationships"]]:
        # print("\n---hello from validate_relationships---\n")
        node_name = "validate_relationships"
        self.logger.info(f"\n---NODE: {node_name}---")
        loc_state = state
        yml_parser =YamlOutputParser(pydantic_object=EntityValidationFeedback)     
        extract_entities_prompt = self.prompt_dict['missing_relations_prompt']
        loc_state['relations_iter_counter'] += 1
        
        s_prompt_template = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
            template=extract_entities_prompt['system'],    
            input_variables = ["extracted_relationships","extracted_entities_names",
                               "content","iteration_counter","max_iterations"],
            partial_variables = {"format_instructions": yml_parser.get_format_instructions()}    
            )
        )

        extracted_entities = loc_state['entities']
        extracted_entities_names = [entity["name"] for entity in extracted_entities["entities"]]
        extracted_relationships = loc_state['relationships']
        iteration_counter = loc_state['relations_iter_counter']
        max_iterations = 4
        content = loc_state['text_chunks'][loc_state['chunk_index']]  
        # repeated code 
        s_prompt = s_prompt_template.format(max_iterations=max_iterations,content=content,
                                            extracted_relationships=extracted_relationships,
                                            extracted_entities_names=extracted_entities_names,
                                            iteration_counter=iteration_counter)
        messages = [s_prompt]
        chat_prompt = ChatPromptTemplate.from_messages(
                    [MessagesPlaceholder(variable_name="messages")]
                    )
        llm_chain = chat_prompt | self.llm | extract_code  
        wrapped_state = {"messages":messages, "llm_output":"","error_status":"","iterations":0}
        wrapped_model = ValidationWrapper(llm=self.llm,parent_trace_id=self.parent_trace_id,node_name=node_name,
                                        parser=yml_parser,graph=llm_chain,pydantic_bm=EntityValidationFeedback)
        
        relation_validation_out = wrapped_model.invoke(wrapped_state)['llm_output']
        loc_state['relations_validation_feedback'] = relation_validation_out
        if(relation_validation_out['loop_required']):
            goto = "extract_relationships"
            
        else: 
            goto = "extract_graph"
        
        return Command(update=loc_state,goto=goto)
    
    def build_graph(self,state):
        '''
        Objectives 
        1. Deduplication of entities and relations 
        2. Save the deduplicated information to intermediate file
        2. Generate networkX graph from the optimized entity-relation information
        '''
        node_name = "build_graph"
        self.logger.info(f"\n---NODE: {node_name}---")
        loc_state = state
        
        # Check if deduplicated graph is already available 
        if(self.storage.file_exists(filename=self.dedup_interm_data_path,storage_type=self.graph_info_storage)):
            dd_file = self.storage.read_file(filename=self.dedup_interm_data_path,storage_type=self.graph_info_storage)
            self.logger.info(f"Found deduplication file.Loading..")
            dd_content = yaml.load(dd_file,Loader=yaml.FullLoader)
            deduplicated_entities = dd_content['entities']
            updated_relationships = dd_content['relationships']
        else:
            # TODO : Replace following state preservation logic with langgraph persistence 
            self.logger.info(f"Deduplication pipeline started")
            entities,relations = self.load_nodes_n_relns_from_intermediate_file()

            #---------------Deduplication logic-Begin-------------#
            # Initialize deduplicator
            deduplicator = EntityDeduplicator(self.similarity_model)
            deduplicated_entities, deduplication_map = deduplicator(entities)
            updated_relationships, relationship_updates = self._update_deduplicate_relationships(relationships=relations,deduplication_map=deduplication_map)
            #---------------Deduplication logic-End-------------#
            
            self.logger.debug(f"Deduplicated relationships:\n {updated_relationships}")
            #---------------<source>_dd_intermediate_data.yml-Begin-------------#
            self.save_nodes_n_relns_to_intermediate_file(deduplicated_entities,
                                                updated_relationships,type=False)
            

            self.logger.info(f"Deduplication pipeline end. Saved to intermediate file.")
        #---------------<source>_dd_intermediate_data.yml-End---------------#
        
        nx_graph = self._generate_graph_from_dict(deduplicated_entities,updated_relationships)
        # Serialize graph before attaching to state. Checkpointer doesn't support networX graphs
        ser_graph = nx.node_link_data(nx_graph)
        
        # Save the graph to yaml file
        relation_dict_string = b"".join(s.encode('utf-8') for s in yaml.dump(ser_graph,default_flow_style=False, sort_keys=False)) # convert to bytes
        self.storage.save_file(filename=self.graph_path,
                                data=relation_dict_string,
                                storage_type=self.graph_info_storage)

        loc_state['global_graph'] = ser_graph
        # Serialized graph is the final output of Bodhi. All other information in State of Bodhi is not supposed to 
        # be used outside Bodhi. 
        # NOTE : state['entities'] and state['relationships'] are not updated with deduplicated information. 
        # Deduplicated entities and relationships follow distinct structure from the entities and relations present in state variables.
        # This resulted from independent development of certain sections. Yet once they are integrated into the nx_graph
        # unified and reliable interface is ensured. We don't see any rationale to merge these as of now. 

        return loc_state
    
    #---------------Support functions-Begin-------------#
    #---------------Intermediate file handling-Begin-------------#
    def save_nodes_n_relns_to_intermediate_file(self,entities, relationships,type=False,chunk_index=None):
        """
        Saves extracted entities/nodes and relationships to a YAML file.
        
        Args:
            entities (list of dict): A list of entity dictionaries in the specified format.
            relationships (list of dict): A list of relationship dictionaries in the specified format.
            file_path (str): Path to the YAML file where data will be stored.
            type (bool): Flag to decide the information type, deduplicated or just intermediate.
        Notes:
            file_path = artifacts/graph_extraction/<file_name>_intermediate_data.yml
        """
        
        try:
            # TODO : Need to revise the type logic later. 
            if(type): # type = 1 => just intermediate data 
                file_path = self.interm_data_path
            else:    
                file_path = self.dedup_interm_data_path
            # Check if file exists and load existing data if it does
            try:
                f = self.storage.read_file(file_path,self.graph_info_storage)
                existing_data = yaml.safe_load(f) or {}

            except FileNotFoundError:
                self.logger.error(f"File doesn't exist: {file_path}. Creating file..")
                existing_data = {}

            # Merge new data with existing data
            if "entities" in existing_data:
                existing_data["entities"].extend(entities)
            else:
                existing_data["entities"] = entities

            if "relationships" in existing_data:
                existing_data["relationships"].extend(relationships)
            else:
                existing_data["relationships"] = relationships

            # Save chunk indices to the intermediate_data.yml file
            if(type)&(chunk_index is not None):
                if "chunk_indices" in existing_data:
                    existing_data["chunk_indices"].append(chunk_index)
                else:
                    existing_data["chunk_indices"] = [chunk_index]

            # Write updated data back to the file
            intermediate_data = b"".join(s.encode('utf-8') for s in yaml.dump(existing_data)) # convert to bytes
            self.storage.save_file(filename=file_path,
                                    data=intermediate_data,
                                    storage_type=self.graph_info_storage)
            
            self.logger.info(f"Nodes & relations saved to {self.graph_info_storage}{file_path}")
        
        except Exception as e:
            self.logger.error(f"An error occurred while saving data: {e}")
    
    def load_nodes_n_relns_from_intermediate_file(self):
        """Loads entities and relationships from a YAML file into separate dictionaries."""
        
        file_path = self.interm_data_path
        try:
            file = self.storage.read_file(file_path,self.graph_info_storage)
            data = yaml.safe_load(file)
            if data is None:
                print(f"Warning: File {file_path} is empty. Returning empty dictionaries.")
                return {}, {}

            if not isinstance(data, dict) or "entities" not in data or "relationships" not in data:
                print(f"Warning: File {file_path} does not contain expected 'entities' and 'relationships' structure. Returning empty dictionaries.")
                return {}, {}
            
            entities_dict = {}
            relationships_dict = {}

            for entity in data["entities"]:
                name = entity["name"]
                entities_dict.setdefault(name, {"type": [], "description": []})
                entities_dict[name]["type"].append(entity.get("type")) #Use get to avoid keyerror
                entities_dict[name]["description"].append(entity.get("description"))

            for relationship in data["relationships"]:
                source = relationship["source_entity"]
                target = relationship["target_entity"]
                key = (source, target)
                relationships_dict.setdefault(key, {"description": [], "strength": []})
                relationships_dict[key]["description"].append(relationship.get("description"))
                relationships_dict[key]["strength"].append(relationship.get("strength"))
                
            return entities_dict, relationships_dict

        except FileNotFoundError:
            print(f"File not found: {file_path}. Returning empty dictionaries.")
            return {}, {}
        except yaml.YAMLError as e:
            print(f"YAML parsing error: {e}. Returning None, None.")
            return None, None
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Returning None, None.")
            return None, None
    #---------------Intermediate file handling-End-------------# 
    
    def _generate_graph_from_dict(self,entities, relationships):
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
                # type=", ".join(attributes.get("type", [])).strip().upper(),
                type = ", ".join(map(lambda x: str(x) if x is not None else "None", attributes.get("type", []))).strip().upper(),
                description=" ; ".join(attributes.get("description", [])).strip(),
            )

        # Add edges (relationships) to the graph
        for relation, attributes in relationships.items():
            # Convert relationship string to source and target tuple(In case if we are reading from yml file)
            # (source_entity, target_entity) = ast.literal_eval(relation) 
            (source_entity, target_entity) = relation
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
    
    def _update_entities(self,global_entities, new_entities):
        """
        Updates the global entity list with new entities, ensuring no duplicates are added.

        Args:
            global_entities (list): The current list of global entities.
            new_entities (list): The list of new entities to be added.

        Returns:
            list: The updated global entity list.
        """
        for new_entity in new_entities['entities']:
            # Check if the new entity is already in the global list
            is_duplicate = any(
                existing_entity["name"].strip().lower() == new_entity["name"].strip().lower() and
                existing_entity["type"].strip().lower() == new_entity["type"].strip().lower()
                for existing_entity in global_entities['entities']
            )
            # Add new entity if not a duplicate
            if not is_duplicate:
                global_entities['entities'].append(new_entity)
        print(f"\n--final entities after update entities--\n{global_entities}\n")
        return global_entities

    def _update_relationships(self,global_relationships, new_relationships):
        """
        Updates the global relationships list with new relationships, ensuring no duplicates are added.

        Args:
            global_relationships (list): The current list of global relationships.
            new_relationships (list): The list of new relationships to be added.

        Returns:
            list: The updated global relationships list.
        """
        for new_relationship in new_relationships['relationships']:
            # Check if the new relationship is already in the global list
            is_duplicate = any(
                existing_relationship["source_entity"].strip().lower() == new_relationship["source_entity"].strip().lower() and
                existing_relationship["target_entity"].strip().lower() == new_relationship["target_entity"].strip().lower() and
                existing_relationship["description"].strip().lower() == new_relationship["description"].strip().lower() and
                existing_relationship["strength"] == new_relationship["strength"]
                for existing_relationship in global_relationships
            )
            # Add new relationship if not a duplicate
            if not is_duplicate:
                global_relationships.append(new_relationship)
        print(f"\n--final relationships after update--\n{global_relationships}\n")
        return global_relationships

    def _extract_text_chunks(self,file_path: str, token_limit: int = 400) -> list[str]:
        """
        Extract meaningful/complete text chunks from a source file.

        Args:
            file_path (str): The path to the source document (text format).
            token_limit (int): Maximum token limit per chunk.

        Returns:
            list[str]: List of text chunks, each below the token limit.
        """
        import spacy
        # Load spaCy for tokenization
        nlp = spacy.load("en_core_web_sm")
        nlp.max_length = 2000000
        def extract_text_from_file(file_path: str) -> str:
            """Extract text from a file (text or PDF)."""
            try:
                if file_path.lower().endswith(".pdf"):  # Check if it's a PDF
                    import pymupdf4llm
                    output = pymupdf4llm.to_markdown(file_path)
                    return output  # Return the markdown output
                elif file_path.lower().endswith(('.txt', '.text','.md')): # if it is a text file
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()
                        return text
                else:
                    return "Unsupported file type" # if it is of any other type

            except Exception as e: # Handle any exception
                return f"Error extracting text: {e}"

        def split_into_chunks(text: str, token_limit: int) -> list[str]:
            """Split text into chunks based on the token limit."""
            chunks = []
            current_chunk = ""
            current_token_count = 0

            # Tokenize the entire text
            doc = nlp(text)

            for sentence in doc.sents:
                sentence_text = sentence.text.strip()
                sentence_token_count = len(sentence)

                if current_token_count + sentence_token_count <= token_limit:
                    current_chunk += f"{sentence_text} "
                    current_token_count += sentence_token_count
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence_text
                    current_token_count = sentence_token_count

            if current_chunk:
                chunks.append(current_chunk.strip())

            return chunks

        # Step 1: Extract raw text from the file
        raw_text = extract_text_from_file(file_path)
        text_bytes = b"".join(s.encode('utf-8') for s in yaml.dump(raw_text)) # convert to bytes
        self.storage.save_file(filename=self.raw_text_path,
                            data=text_bytes,
                            storage_type=self.graph_info_storage)
        # Step 2: Split text into chunks based on token limit
        text_chunks = split_into_chunks(raw_text, token_limit)

        return text_chunks    

    def _update_deduplicate_relationships(self,relationships, deduplication_map):
        """
        Updates relationships based on deduplications made on entities and tracks changes.

        Args:
            relationships (dict): Dictionary containing relationships in the format:
                { (source, destination): {"description": [...], "strength": [...]}}
            deduplication_map (dict): Map of deduplicated entities in the format:
                { original_entity: deduplicated_entity }

        Returns:
            tuple: (updated_relationships, relationship_updates)
                - updated_relationships (dict): Updated relationships with deduplicated entities.
                - relationship_updates (dict): Map of original relationships to their updated relationships.
        """
        updated_relationships = {}
        relationship_updates = {}

        for (source, destination), rel_data in relationships.items():
            # Check if source or destination has been deduplicated
            updated_source = deduplication_map.get(source, source)
            updated_destination = deduplication_map.get(destination, destination)

            # Define the updated relationship key
            updated_key = (updated_source, updated_destination)
            original_key = (source, destination)

            # Track the relationship update
            if original_key != updated_key:
                relationship_updates[original_key] = updated_key

            # Merge data if the relationship already exists with the updated key
            if updated_key in updated_relationships:
                existing_data = updated_relationships[updated_key]

                # Merge descriptions
                existing_data["description"].extend(rel_data["description"])
                existing_data["description"] = list(set(existing_data["description"]))  # Remove duplicates

                # Merge strengths
                existing_data["strength"].extend(rel_data["strength"])
                existing_data["strength"] = list(set(existing_data["strength"]))  # Remove duplicates

            else:
                # Add the relationship to the updated relationships
                updated_relationships[updated_key] = {
                    "description": rel_data["description"][:],  # Copy the list
                    "strength": rel_data["strength"][:],        # Copy the list
                }

        return updated_relationships, relationship_updates

    #TODO : Integrate this function in next version 
    def _update_deduplicate_relationships_optimized(
        relationships, deduplication_map, similarity_threshold=0.8
    ):
        """
        Updates relationships based on deduplications made on entities and tracks changes.

        Args:
            relationships (dict): Dictionary containing relationships in the format:
                { (source, destination): {"description": [...], "strength": [...]}}
            deduplication_map (dict): Map of deduplicated entities in the format:
                { original_entity: deduplicated_entity }
            similarity_threshold (float): Threshold for semantic similarity of descriptions (default=0.8).

        Returns:
            tuple: (updated_relationships, relationship_updates)
                - updated_relationships (dict): Updated relationships with deduplicated entities.
                - relationship_updates (dict): Map of original relationships to their updated relationships.
        """
        updated_relationships = {}
        relationship_updates = {}
        similarity_checker = SemanticSimilarity()  # Use your semantic similarity class

        for (source, destination), rel_data in relationships.items():
            # Check if source or destination has been deduplicated
            updated_source = deduplication_map.get(source, source)
            updated_destination = deduplication_map.get(destination, destination)

            # Define the updated relationship key
            updated_key = (updated_source, updated_destination)
            original_key = (source, destination)

            # Track the relationship update
            if original_key != updated_key:
                relationship_updates[original_key] = updated_key

            # Merge data if the relationship already exists with the updated key
            if updated_key in updated_relationships:
                existing_data = updated_relationships[updated_key]

                # Merge descriptions using semantic similarity
                for new_desc in rel_data["description"]:
                    if not any(
                        similarity_checker.is_similar(new_desc, existing_desc, similarity_threshold)
                        for existing_desc in existing_data["description"]
                    ):
                        existing_data["description"].append(new_desc)

                # Merge strengths with an aggregation method
                if isinstance(rel_data["strength"][0], (int, float)):
                    # Use an aggregation method like mean or max for numeric strengths
                    existing_data["strength"] = [
                        max(existing_data["strength"] + rel_data["strength"])
                    ]
                else:
                    # Default to deduplicating categorical strengths
                    existing_data["strength"].extend(rel_data["strength"])
                    existing_data["strength"] = list(set(existing_data["strength"]))

            else:
                # Add the relationship to the updated relationships
                updated_relationships[updated_key] = {
                    "description": rel_data["description"][:],  # Copy the list
                    "strength": rel_data["strength"][:],        # Copy the list
                }

        return updated_relationships, relationship_updates

    #---------------Support functions-End-------------#

#---------------Class for entity deduplication mechanism-Begin-------------#
class EntityDeduplicator:
    def __init__(self, similarity_model: SemanticSimilarity, similarity_threshold=0.9, description_threshold=0.6):
        self.similarity_model = similarity_model
        self.similarity_threshold = similarity_threshold
        self.description_threshold = description_threshold
        self.name_embeddings = {}  # Cache for name embeddings
    def __call__(self, intermediate_entities):
        return (self.deduplicate_entities_batch(intermediate_entities=intermediate_entities))
    
    def deduplicate_entities_batch(self, intermediate_entities):
        """
        Deduplicates or merges entities from the intermediate file format.

        Args:
            intermediate_entities (dict): Dictionary containing entities with names, types, and descriptions.

        Returns:
            tuple: Deduplicated entities (dictionary) and deduplication map (dictionary).
        """
        deduplicated_entities = {}
        deduplication_map = {}

        # Helper function to remove duplicates from a list
        def remove_duplicates_from_list(data):
            return list(set(data))

        # Helper function to compute embeddings
        def get_name_embedding(name):
            if name not in self.name_embeddings:
                self.name_embeddings[name] = self.similarity_model.get_or_compute_embedding(name)
            return self.name_embeddings[name]

        # Process all entities
        for entity_name, entity_data in intermediate_entities.items():
            entity_type = entity_data["type"]
            descriptions = entity_data["description"]

            # Get embedding for the current entity name
            entity_name_embedding = get_name_embedding(entity_name)

            # Track the best match
            best_match_name = None
            best_name_similarity = 0
            best_description_similarity = 0

            for existing_name, existing_data in deduplicated_entities.items():
                existing_type = existing_data["type"]
                existing_descriptions = existing_data["description"]

                # Check name similarity
                existing_name_embedding = get_name_embedding(existing_name)
                name_similarity = self.similarity_model.compute_similarity(
                    entity_name, existing_name
                )

                if name_similarity >= self.similarity_threshold:
                    # Name is semantically similar, check description similarity
                    max_description_similarity = max(
                        [
                            self.similarity_model.compute_similarity(current_desc, existing_desc)
                            for current_desc in descriptions
                            for existing_desc in existing_descriptions
                        ]
                    )

                    # Deduplicate if both name and descriptions are highly similar
                    if max_description_similarity >= self.similarity_threshold:
                        deduplication_map[entity_name] = existing_name
                        break

                    # Otherwise, track for merging based on description similarity
                    if max_description_similarity >= self.description_threshold:
                        if max_description_similarity > best_description_similarity:
                            best_match_name = existing_name
                            best_name_similarity = name_similarity
                            best_description_similarity = max_description_similarity

            # If a match is found, merge descriptions into the best match
            if best_match_name:
                deduplicated_entity = deduplicated_entities[best_match_name]
                for description in descriptions:
                    if description not in deduplicated_entity["description"]:
                        deduplicated_entity["description"].append(description)
                deduplicated_entity["type"].extend(entity_type)
                deduplicated_entity["type"] = remove_duplicates_from_list(deduplicated_entity["type"])
                deduplicated_entity["description"] = remove_duplicates_from_list(deduplicated_entity["description"])

                deduplication_map[entity_name] = best_match_name
            else:
                # Add the new entity to deduplicated entities
                deduplicated_entities[entity_name] = {
                    "type": remove_duplicates_from_list(entity_type),
                    "description": remove_duplicates_from_list(descriptions),
                }

        return deduplicated_entities, deduplication_map
#---------------Class for entity deduplication mechanism-End---------------#

# Bodhi test code 
if __name__ == "__main__1":
    # Langfuse tracing 
    parent_trace_id = create_unique_trace_id()

    bodhi = Bodhi(parent_trace_id=parent_trace_id)
    data_path = os.path.join(os.path.dirname(__file__),"../storage/data")
    init_state = {"source_path":f"{data_path}/graphrag_paper.pdf"}
    # init_state = {"source_path":"data/lttng_guide.md"}
    final_state = bodhi.invoke(init_state)
    print(final_state)
    final_state = bodhi.get_state()

    print(final_state.values)
    state_history = bodhi.get_state_history()
    sh_string = ""
    for snapshot in state_history:
        sh_string += f"{snapshot.created_at}:{snapshot.values}\n"
    with open("logging/lfs_langgraph.md","w") as bodhi_results:
        bodhi_results.write(sh_string)
