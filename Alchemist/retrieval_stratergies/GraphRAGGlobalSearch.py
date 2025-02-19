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


import networkx as nx
import uuid

# from The_Prophet.Odysseus.states.bodhi import *
from infra.utils.utils import *
from infra.utils.cluster import *


#---------------Pipeline specific imports-Begin-------------#
from ..base import *
from .graphrag_state import *
#---------------Pipeline specific imports-End---------------#



class GraphRAGGlobalSearch:
    def __init__(self,**kwargs):
        self.ipc = None
        if 'ipc' in kwargs:
            self.ipc=kwargs.get('ipc')
            # print("added ipc\n")
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
        self.workflow.add_node("get_community_summaries",self.get_community_summaries)
        self.workflow.add_node("combine_answers",self.combine_answers)
        self.workflow.add_node("gen_intermediate_answers",self.gen_intermediate_answers)

        self.workflow.add_edge(START,"get_community_summaries")
        self.workflow.add_edge("get_community_summaries","gen_intermediate_answers")
        self.workflow.add_edge("gen_intermediate_answers","combine_answers")
        self.workflow.add_edge("combine_answers",END)
    
        self.app = self.workflow.compile(checkpointer = self.checkpointer)

    def get_community_summaries(self,state:GRGlobalSearchState):
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
        results = {}
        if self.ipc:
            results = self.ipc(loc_state)
            loc_state = results
        
        return loc_state
    
    def gen_intermediate_answers(self,state:GRGlobalSearchState):
        '''
        input format 
        ---------------
        {'query': 'Explain what is langgraph', 'retrieval_method': 'GraphRAGGlobalSearch', 'retrievals': {'modality': 'text', 'data': {'0': {'title': 'LangGraph Community: Message Passing and Graph Processing', 'summary': 'The LangGraph community is structured around key entities such as RECURSION LIMIT, SUPER-STEPS, LANGGRAPH, GRAPHRECURSIONERROR, and STATE...','key_highlights':[....,...],'impact_severity_rating':8.2,'rating_explanation':'.....'},'1':{...}}
        Objective of this node 
        --------------------- 
        Generate answer to the user query from the context of each community summaries.
        Steps 
        -----
        1. Read query, all community summaries from the state
        2. Prepare and prompt llm to answer the user query from the perspective of each community 
        3. Repeat the step 2 until all communities are covered 
        4. Store answers in the state and pass to combine_answers node
        '''
        loc_state = state
        retrieved_files = loc_state['retrievals']['data']
        query = loc_state['query']
        community_answers = []
        # print(retrieved_files)
        for community_id, community in retrieved_files.items():
            summary = community['summary']
            key_highlights = community['key_highlights']
            impact_severity = community['impact_severity_rating']
            rating_explanation = community['rating_explanation']
            input_var_names= ['community_highlights','community_summary','query']
            template = self.prompt_dict['commuity_answer_extraction']['system']
            model = sys_prompt_wrapped_call(pydantic_object=CommunityAnswer,
                                    sys_prompt_template=template,input_var_names_system=input_var_names,
                                    parent_trace_id=self.parent_trace_id,temperature=0.4)
            s_prompt = model['system_template'].format(community_highlights=key_highlights,community_summary=summary,query=query)
            messages = [s_prompt]
            wrapped_state = {"messages":messages,"llm_output":"","error_status":"","iterations":0}
            answer = model['model'].invoke(wrapped_state)['llm_output']
            answer['community_name'] = community['title']
            community_answers.append(answer)
            print(loc_state)

        loc_state['community_answers'] = community_answers
        print(loc_state)
        return loc_state


    def combine_answers(self,state:GRGlobalSearchState):
        '''
        input format
        -----------

        Objective of this node
        ---------------------
        Generate final answers from the list of answers generated from communities 
        Steps
        ------
        1. Read list of answers from the state
        2. Prepare and prompt llm to summarize and generate a coherent answer based on the list of answers 
        3. Update state with final answer 
        '''
        loc_state = state
        answers = ""
        for community_answers in loc_state["community_answers"]:
            if community_answers['relevance_score'] < 5:
                continue
            answer = f"---\nCommunity name : {community_answers['community_name']}\nAnswer: {community_answers['answer']}\n Relevance score:{community_answers['relevance_score']}\n---\n"
            answers += answer
        
        input_var_names= ['query','answers']
        template = self.prompt_dict['final_answer_extraction']['system']
        model = sys_prompt_wrapped_call(pydantic_object=FinalAnswer,
                                sys_prompt_template=template,input_var_names_system=input_var_names,
                                parent_trace_id=self.parent_trace_id,temperature=0.4)
        s_prompt = model['system_template'].format(query=loc_state['query'],answers=answers)
        messages = [s_prompt]
        wrapped_state = {"messages":messages,"llm_output":"","error_status":"","iterations":0}
        final_answer = model['model'].invoke(wrapped_state)['llm_output']
        loc_state['final_answer'] = final_answer
        return loc_state
    

