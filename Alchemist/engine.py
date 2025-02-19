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
from typing import Any,Optional,TypedDict,Dict, Union


# from The_Prophet.Odysseus.states.bodhi import *

from infra.utils.utils import *

#---------------Pipeline specific imports-Begin-------------#
from .base import *
from .states.alchemist_state import *
#---------------Pipeline specific imports-End---------------#


class Alchemist:
    def __init__(self,parent_trace_id):
        # Langfuse 
        self.parent_trace_id = parent_trace_id
        # Prompt related 
        self.prompt_dict = get_settings()
        config = Config()
        expt_llm = config["llm_model"]
        self.llm = OllamaLLM(temperature=0.2, model=expt_llm)

        # For persistence 
        self.checkpointer = MemorySaver()
        self.config = {"configurable": {"thread_id": uuid.uuid4()}}

        # graph related 
        self.workflow = StateGraph(AlchemistState)
        self.app = None
        self._setup_graph()
        
        # 1. Register all available retrieval plugins
        self.plugins:RetrievalFactory = RetrievalFactory(ipc=trigger_workflow,parent_trace_id=parent_trace_id)
        
    def invoke(self,state):
        self.app.invoke(state,self.config)

    def get_state(self):
        return self.app.get_state(self.config)

    def _setup_graph(self):
        # Graph structure code 
        self.workflow.add_node(self.analyze,"analyze") # Query analysis node (Command node,Chooses retrieval stratergy based on query type)
        self.workflow.add_node(self.plugin,"plugin") # Plugin invoke node  (Execute retrieval stratergy)
        self.workflow.add_node(self.answer,"answer") # Answering node      (Capture retrieved results and build final answer)

        self.workflow.add_edge(START,"analyze")
        self.workflow.add_edge("plugin","answer")
        self.workflow.add_edge("answer",END)
        self.app = self.workflow.compile(checkpointer = self.checkpointer)

    def answer(self,state):
        print("Hello from answer\n")
        pass
    def plugin(self,state:AlchemistState):
        print("Hello from plugin\n")
        loc_state = state
        results = self.plugins.invoke(strategy=loc_state['retrieval_method'],state=loc_state)
        # results = self.plugins.get_state().values
        loc_state['answer'] = results
        return loc_state
    
    def analyze(self,state:AlchemistState)->Command:
        # LLM node to analyze if query is of global,local, or intermediate scope
        print("Hello from analyze\n")
        loc_state = state
        loc_state["retrieval_method"] = "GraphRAGGlobalSearch"
        loc_state["retrieval_method"] = "GraphRAGLocalSearch"
        goto = "plugin"
        
        return Command(goto=goto,update=loc_state)
    # define node functions

#---------------zero mq client-Begin-------------#

def trigger_workflow(request_data=None):
    import zmq
    from .retrieval_stratergies.SchemaLoader import SchemaLoader  # Use the global instance
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    schema_loader = SchemaLoader()

    if not request_data:
        request_data = {
            "query": "Find relevant research papers",
            "retrieval_method": "GraphRAGGlobalSearch",
            "retrievals": []
        }
    print(f"Request data={request_data}")
    is_valid, error = schema_loader.validate(request_data)
    if not is_valid:
        print(f"Validation Error: {error}")
        return

    print("Sending request to Alchemist...")
    socket.send_json(request_data)

    response = socket.recv_json()
    is_valid,error = schema_loader.validate(response)
    if not is_valid:
        print(f"Invalid Response: {error}\n{response}")
        return
    
    return response

#---------------zero mq client-End---------------# 

if __name__ == "__main__1":
    alchemist = Alchemist(101)
    alchemist.trigger_workflow()

if __name__ == "__main__":
    alchemist = Alchemist(101)
    state = {"query":"Explain concept of checkpointers?"}
    alchemist.invoke(state=state)
