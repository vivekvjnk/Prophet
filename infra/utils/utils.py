import os
import re
import inspect
import uuid  # For generating unique trace IDs
import yaml
import torch
import logging
import numpy as np
import threading
from pydantic import Field

from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModel

import ollama
from typing import List,TypedDict,get_type_hints #,Dict, Any, Optional
from langchain.tools.base import BaseTool
from langfuse.callback import CallbackHandler
from langchain.output_parsers.yaml import YamlOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_ollama import OllamaLLM
from langchain.schema.runnable import RunnableSerializable
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


from infra.prompts.config_loader import get_settings
from infra.logs.prophet_logger import *

langfuse_handler = CallbackHandler(
    public_key="pk-lf-0a64c71e-c587-45dd-aed2-cd729ba34dc6",
    secret_key="sk-lf-798d71dd-f314-474a-94f6-b3ad4eceb1f5",
    host="http://localhost:3000"
)

#---------------Custom yaml loader for tuples-Begin-------------#
# Custom constructor to handle the !!python/tuple tag
def tuple_constructor(loader, node):
    # Extract values from ScalarNode objects and create a tuple
    return tuple([loader.construct_scalar(n) for n in node.value])

# Register the custom constructor with PyYAML
yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor,yaml.FullLoader)
#---------------Custom yaml loader for tuples-End---------------#


#---------------Validation wrapper-Begin---------------#


"""
ValidationWrapper
=================

This module defines the `ValidationWrapper` class, designed to handle formatting errors in outputs
from a LangGraph-based module (LLM or any other runnable module). The wrapper validates and corrects
the module’s output, ensuring it adheres to specified Pydantic constraints. If errors occur, it
reflects the issue back to the module, requesting corrections and retrying execution.

Key Features
------------
- Handles structured output validation.
- Iteratively requests fixes for formatting errors.
- Works with any LangGraph runnable module, making it highly reusable.
- Uses an LLM-powered reflection mechanism to identify and resolve errors.
- Maintains a state-driven workflow using `StateGraph`.

Classes
-------
ValidationWrapper
    A wrapper class that ensures structured output validation for a given LangGraph module.

Usage Example
-------------
```python
wrapper = ValidationWrapper(parser, graph, pydantic_bm)
state = ValidationWrapperState(messages=[{"role": "user", "content": "Generate a response"}])
validated_state = wrapper.invoke(state)
print(validated_state.llm_output)
```

Class Documentation
-------------------
"""

class ValidationWrapperState(TypedDict):
    messages: List
    llm_output:str
    error_status:bool
    error_description:str
    iterations:int

class ValidationWrapper:
    """
    A wrapper to handle LLM formatting errors and validate output against Pydantic constraints.

    This class provides an automated error-handling mechanism by:
    - Running an LLM-based module.
    - Checking for formatting errors in the output.
    - Reflecting errors back to the module for correction.
    - Validating the corrected output against a Pydantic model.
    
    Attributes:
    -----------
    prompt_dict : dict
        Dictionary containing prompt settings.
    llm : OllamaLLM
        The LLM instance used for error reflections and analysis.
    parent_trace_id : str, optional
        Parent trace ID for logging/debugging (default is None).
    workflow : StateGraph
        The validation workflow defined as a state graph.
    app : Runnable
        The compiled validation workflow.
    yml_parser : callable
        Function to parse and validate LLM responses.
    pydantic_bm : Type[BaseModel]
        The Pydantic model used for response validation.
    module_graph : Runnable
        The LangGraph module to be validated.
    node_name : str
        Name of the validation node (default: "generic_module").
    logger : logging.Logger
        Logger instance for tracking validation steps.
    
    Methods:
    --------
    __init__(self, parser: callable, graph: Runnable, pydantic_bm: Type[BaseModel], logger: Optional[logging.Logger] = None, node_name: str = "generic_module", llm: Optional[OllamaLLM] = None, parent_trace_id: Optional[str] = None)
        Initializes the validation wrapper with necessary components.
    
    _setup_graph(self) -> None
        Configures the validation workflow with nodes and edges.
    
    invoke(self, state: ValidationWrapperState) -> ValidationWrapperState
        Invokes the validation workflow and returns the final validated state.
    
    module_call(self, state: ValidationWrapperState) -> ValidationWrapperState
        Executes the LangGraph module and handles error corrections.
    
    llm_error_check(self, state: ValidationWrapperState) -> Command
        Parses the LLM response and checks for formatting errors.
    
    llm_error_status_fn(self, state: ValidationWrapperState) -> str
        Determines the next action based on whether an error occurred.
    
    error_reflections(self, state: ValidationWrapperState) -> ValidationWrapperState
        Generates an error analysis and suggests a fix using the LLM.
     
    """
    def __init__(self,parser,graph,pydantic_bm,logger=None,node_name="generic_module",llm=None,parent_trace_id=None):
        self.prompt_dict = get_settings()
        config = Config()
        expt_llm = config["llm_model"]
        self.llm = OllamaLLM(temperature=0.3, model=expt_llm)
        self.parent_trace_id = parent_trace_id
        self.workflow = StateGraph(ValidationWrapperState)
        self.app = None
        self.yml_parser = parser
        self.pydantic_bm = pydantic_bm
        self.module_graph = graph
        self.node_name = node_name
        
        self._setup_graph()
        if(logger):
            self.logger = logger
        else:
            validator_log_handler = configure_rotating_file_handler(log_file_path="infra/logs/llm_validator.log")
            self.logger  = get_pipeline_logger(file_handler=validator_log_handler,pipeline_name="llm_validator")

    def _setup_graph(self):

        self.workflow.add_node("module_call",self.module_call)
        self.workflow.add_node("llm_error_check",self.llm_error_check)
        self.workflow.add_node("error_reflections",self.error_reflections)
        
        self.workflow.add_edge(START, "module_call")
        self.workflow.add_edge("module_call","llm_error_check")
        self.workflow.add_edge("error_reflections","module_call")
        
        self.app = self.workflow.compile(checkpointer=False) # Avoid multiple subgraph error 

    def invoke(self,state:ValidationWrapperState)->ValidationWrapperState:
        result = self.app.invoke(state)
        return result 
    
    def module_call(self,state:ValidationWrapperState):
        loc_state = state
        print(f"\n------Hello from wrapper:module_call:i={loc_state['iterations']}------\n")
        self.logger.debug(f"\n------ValidationWrapper:{self.node_name}:i={loc_state['iterations']}------\n")
        if(loc_state["error_status"]):
            correction_prompt = f"""
                    Actual output from the LLM:
                    {loc_state['llm_output']}

                    Structuring issues in the LLM output:
                    {loc_state['error_description']}

                    Please fix the issue and provide a valid structured response.
                    Additionally, ensure that every string in the LLM response is enclosed in quotes to treat it as a single string value.

                    """
                    # {self.yml_parser.get_format_instructions()}
            loc_state['iterations'] += 1
            # chain = self.llm | extract_code
            message = [('system',correction_prompt)]
            chain_kwargs = {"config":{"callbacks": [langfuse_handler], "metadata": {"langfuse_session_id": self.parent_trace_id}} if self.parent_trace_id else {}}
            # loc_state['llm_output'] = self.module_graph.invoke({"messages":message}, 
            #                                           config={"callbacks": [langfuse_handler], "metadata": {"langfuse_session_id": self.parent_trace_id}}) 
            loc_state['llm_output'] = self.module_graph.invoke({"messages":message},**chain_kwargs)

        else:
            loc_state['iterations'] += 1
            # loc_state['llm_output'] = self.module_graph.invoke({"messages":loc_state['messages']},
            #                                                     config={"callbacks": [langfuse_handler], "metadata": {"langfuse_session_id": self.parent_trace_id}})
            chain_kwargs = {"config":{"callbacks": [langfuse_handler], "metadata": {"langfuse_session_id": self.parent_trace_id}} if self.parent_trace_id else {}}
            loc_state['llm_output'] = self.module_graph.invoke({"messages":loc_state['messages']},**chain_kwargs)
        return loc_state

    def llm_error_check(self, state)->Command:
        """
        Checks for errors in the LLM response using the output parser.
        Updates the state if an error occurs.
        """

        print("\n------Hello from wrapper:llm_error_check------\n")
        llm_response = state["llm_output"]
        if((isinstance(llm_response,str))&("EC_01" in llm_response)):
            state["error_status"] = False # Lets retry generation without error description, hope llm would correct issues
            state["error_description"] = f"No codeblocks found in LLM response.Please wrap the response in ```yml <response> ```\n"
            goto = "module_call"
        else:
            try:
                parsed_output = self.yml_parser.invoke(llm_response)
                state["llm_output"] = parsed_output.dict()
                state["error_status"] = False
                goto = END
            except OutputParserException as e:
                self.logger.error(f"Error while parsing LLM response.\nError Description: {str(e)}\n")
                state["error_status"] = True
                state["error_description"] = str(e)
                goto = "error_reflections"
        return Command(update=state,goto=goto)
    
    def llm_error_status_fn(self, state):
        """
        Routing function to check if an error occurred.
        """
        return "error" if state.get("error_status", False) else "no_error"
    
    def error_reflections(self, state):
        loc_state = state
        error_reflection_prompt = f"""
            The previous LLM output(in yml format) had the following structuring issue:

            {loc_state['error_description']}

            Following pydantic basemodel used for parsing:
            {str(inspect.getsource(self.pydantic_bm))}

            Please analyze the issue and provide a brief passage describing:

            1. The root cause of the issue. Specifically analyze if the issue can be solved by quoting the values.
            2. A suggested solution in clear terms.
            3. The exact location of the problem.
            Additionally, ensure that every string in the LLM response is enclosed in quotes to treat it as a single string value. Respond within 75 words.
            """
        chain = self.llm | self.do_nothing
        chain_kwargs = {"config":{"callbacks": [langfuse_handler], "metadata": {"langfuse_session_id": self.parent_trace_id}} if self.parent_trace_id else {}}
        response = chain.invoke(error_reflection_prompt,**chain_kwargs)
        loc_state['error_description'] = response
        # print(f"Hello from route_to_end: {state}\n")

        return loc_state
    
    def do_nothing(self,state):
        """Passes the state unchanged."""
        return state
#---------------Validation wrapper-End---------------#


def sys_prompt_wrapped_call(pydantic_object,sys_prompt_template,
                            input_var_names_system:List[str],parent_trace_id,
                            model = "phi4:latest",temperature=0.2,human_prompt_template = None,
                            ai_prompt_template = None,input_var_names_ai=None,input_var_names_human=None,
                            custom_llm=None
                            ):
    """
    Creates a wrapped LLM model using LangChain components, simplifying prompt setup and execution.
    
    This function encapsulates LangChain configurations, providing a structured way to define and
    execute system, human, and AI message prompts while integrating validation and parsing logic.
    
    Args:
        pydantic_object: The Pydantic model to parse the output.
        sys_prompt_template (str): System message template for the prompt.
        input_var_names_system (List[str]): List of variable names for the system prompt.
        parent_trace_id (str): Unique trace identifier for tracking requests.
        model (str, optional): The LLM model identifier (default: "phi4:latest").
        temperature (float, optional): Sampling temperature for the model (default: 0.2).
        human_prompt_template (Optional[str], optional): Template for human messages (default: None).
        ai_prompt_template (Optional[str], optional): Template for AI messages (default: None).
        input_var_names_ai (Optional[List[str]], optional): List of AI message input variable names (default: None).
        input_var_names_human (Optional[List[str]], optional): List of human message input variable names (default: None).
        custom_llm (Optional[str], optional): Custom LLM model name(eg: phi4:latest) (default: None).
    
    Returns:
        dict: A dictionary containing the wrapped model and prompt templates.
            - "model": The validation-wrapped model.
            - "system_template": The system prompt template.
            - "human_template" (if applicable): The human prompt template.
    
    Notes:
        - If a `custom_llm` is provided, it overrides the default LLM instance with MultiOllamaClient. For now this is a workaround to access multiple ollama servers from the same codebase
        - Uses `YamlOutputParser` to ensure structured output validation.
        - Wraps the LangChain model execution pipeline inside a `ValidationWrapper`.
    """
    from langchain_core.prompts import (
                                    ChatPromptTemplate,
                                    MessagesPlaceholder,
                                    SystemMessagePromptTemplate,
                                    HumanMessagePromptTemplate,
                                    AIMessagePromptTemplate,
                                    PromptTemplate)   
    from langchain_core.runnables.base import RunnableLambda

    if custom_llm:
        ollama_runnable = MultiOllamaClient(instances={custom_llm:"http://192.168.13.13:11434"},temperature=0.2)
        llm = RunnableLambda(ollama_runnable.invoke)
    else:
        llm = OllamaLLM(temperature=temperature, model=model)
    yml_parser = YamlOutputParser(pydantic_object=pydantic_object)
    s_prompt_template = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            template=sys_prompt_template,
            input_variables=input_var_names_system,
            partial_variables={"format_instructions": yml_parser.get_format_instructions()}
        )
    )
    if(human_prompt_template):
        h_prompt_template = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=human_prompt_template,
            input_variables=input_var_names_human,
            partial_variables={"format_instructions": yml_parser.get_format_instructions()}
            )
        )
    if(human_prompt_template):
        ai_prompt_template = AIMessagePromptTemplate(
        prompt=PromptTemplate(
            template=human_prompt_template,
            input_variables=input_var_names_ai,
            partial_variables={"format_instructions": yml_parser.get_format_instructions()}
            )
        )     

    chat_prompt = ChatPromptTemplate.from_messages(
        [MessagesPlaceholder(variable_name="messages")]
    )
    llm_chain = chat_prompt | llm | extract_code
    wrapped_model = ValidationWrapper(parser=yml_parser, graph=llm_chain, 
                                      pydantic_bm=pydantic_object, parent_trace_id=parent_trace_id)
    model = {"model":wrapped_model,"system_template":s_prompt_template}
    if human_prompt_template is not None:
        model |= {"human_template":h_prompt_template} 
    if ai_prompt_template is not None:
        #TODO : To be implemented 
        pass
    return (model)

def extract_code(llm_response):
    """
    Extracts Python code blocks from an LLM response.

    Args:
    llm_response: The text response from the LLM.

    Returns:
    A list of code blocks found in the response.
    """
    code_blocks = re.findall(r'```(?:.*?)\n(.*?)```', llm_response, re.DOTALL)
    if code_blocks:
        return code_blocks[0]
    else:
        logging.error(f"EC_01:No codeblocks found in LLM response\nActual LLM Response:\n{llm_response}\n")
        return (f"EC_01:No codeblocks found in LLM response\nActual LLM Response:\n{llm_response}\n")

def is_instance_of_any_typeddict(data, typeddict_types):
    if not isinstance(data, dict):
        return False
    
    for typeddict_type in typeddict_types:
        hints = get_type_hints(typeddict_type)
        if all(key in data and isinstance(data[key], hints[key]) for key in hints):
            return True
    return False

def create_unique_trace_id():
    """Generate a unique trace ID."""
    return str(uuid.uuid4())

#---------------ollama custom client -Begin-------------#
class MultiOllamaClient(RunnableSerializable):
    clients: dict = Field(default_factory=dict, exclude=True)  # Exclude from Pydantic
    instances: dict = Field(default_factory=dict, exclude=True)  # Exclude from Pydantic
    temperature: float = Field(default=0.2, exclude=True)  # Exclude from Pydantic

    def __init__(self, **kwargs):
        """
        Initializes multiple Ollama clients for different machines.

        Args:
        - instances (dict): A dictionary mapping models to their respective Ollama server URLs.
                            Example: {"graphrag-global": "http://192.168.13.13:11434"}
        - temperature (float): The temperature setting for the LLM responses.
        """
        instances = kwargs.get("instances", None)
        if instances is None:
            raise ValueError("You must provide an 'instances' dictionary.")

        object.__setattr__(self, "clients", {model: ollama.Client(host=host) for model, host in instances.items()})
        object.__setattr__(self, "temperature", kwargs.get("temperature", 0.2))  # ✅ Fixed Assignment

    def __call__(self, input_data):
        """
        Handles LangChain-style input.

        Args:
        - input_data (dict): Expected format {"model": "model_name", "prompt": "query"}

        Returns:
        - str: The model's response.
        """
        if isinstance(input_data, str):
            raise ValueError("Expected dictionary input with 'model' and 'prompt' keys.")
        messages = input_data.messages# Format messages
        # print(f"Following are the messages : \n{messages}\n")
        # prompt = "\n".join([msg.content for msg in messages])  # Convert to string
        formatted_messages = []
        model = "phi4:latest"  # Default model if not provided
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
        
        # model = input_data.get("model", "phi4:latest")
        # prompt = input_data.get("prompt")
        # prompt = "\n".join([msg.content for msg in messages])
        if model not in self.clients:
            raise ValueError(f"No Ollama instance configured for model '{model}'")

        response = self.clients[model].chat(
            model=model, messages=formatted_messages, options={"temperature": self.temperature}
        )
        return response["message"]["content"]
    
    def invoke(self, input_data):
        """
        Handles LangChain-style input.

        Args:
        - input_data (dict): Expected format {"model": "model_name", "prompt": "query"}

        Returns:
        - str: The model's response.
        """
        return self.__call__(input_data)  # ✅ Now calls __call__()

    def list_models(self, model):
        """
        Lists available models from a specific Ollama instance.

        Args:
        - model (str): The model associated with a specific Ollama instance.

        Returns:
        - list: A list of available models.
        """
        if model not in self.clients:
            raise ValueError(f"No Ollama instance configured for model '{model}'")

        return self.clients[model].list()
#---------------ollama custom client -End---------------#

def rename_files(path):
  """
  Renames text and markdown files in a given directory, replacing spaces with underscores.

  Args:
    path: The path to the directory containing the files.
  """
  for filename in os.listdir(path):
    if filename.endswith((".txt", ".md")):
      new_filename = filename.replace(" ", "_")
      os.rename(os.path.join(path, filename), os.path.join(path, new_filename))

#---------------Token counter-Begin-------------#
def estimate_tokens_hf(text: str, model_name: str = "Qwen/Qwen2.5-Coder-14B") -> int:
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
            - If `method` is `"hf"`, call `estimate_tokens_hf`.
            - If `method` is `"spm"`, call `estimate_tokens_spm`.
            - If `method` is `"tiktoken"`, call `estimate_tokens_tiktoken`.
        3. Log a warning if the estimated token count is zero.
        4. Log the final token count and return it.

    Sub-functions:
        - `estimate_tokens_hf(text: str, **kwargs) -> int`: Estimates tokens using Hugging Face tokenization.
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
        - The function assumes that the tokenization methods `estimate_tokens_hf`, `estimate_tokens_spm`, and `estimate_tokens_tiktoken` are implemented and available in the codebase.
        - The choice of tokenization method should align with the downstream processing system.

    Limitations:
        - The token count may differ across methods due to variations in tokenization algorithms.
        - If the input text is empty or poorly formatted, the token count may be zero, triggering a warning.
    """
    logging.info(f"Estimating tokens using {method} method.")
    if method == "hf":
        token_count = estimate_tokens_hf(text, **kwargs)
    elif method == "spm":
        token_count = _estimate_tokens_spm(text, **kwargs)
    elif method == "tiktoken":
        token_count = _estimate_tokens_tiktoken(text, **kwargs)
    else:
        raise ValueError(f"Unsupported tokenization method: {method}")
    if(token_count==0):
        logging.warning(f"Zero token count for following chunk:\n{text}\n")
    logging.info(f"Token count: {token_count}")
    return token_count
#---------------Token counter-End-------------#

#---------------Logging configuration-Begin-------------#

def configure_logging(logger_name,log_folder="logging", log_level=logging.INFO):
    """
    Configures logging for the program, ensuring logs are written to a file
    corresponding to the script that initiated program execution.

    Args:
        log_folder (str): The folder where log files will be stored. Defaults to "logging".
        log_level (int): The logging level. Defaults to logging.INFO.

    Returns:
        None
    """
    import logging
    import os
    import inspect
    # Get the topmost caller file (the file that initiated the program)
    top_caller_file = inspect.stack()[-1].filename
    top_caller_filename = os.path.basename(top_caller_file)
    log_file_name = os.path.splitext(top_caller_filename)[0] + ".log"

    # Create the log folder if it doesn't exist
    os.makedirs(log_folder, exist_ok=True)

    # Full path for the log file
    log_file_path = os.path.join(log_folder, log_file_name)

    # Configure logging
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    # Set a formatter for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Create a logger and attach the handler
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Disable propagation to prevent duplicate logs
    logger.propagate = False

    # logging.info(f"Logging configured. Logs will be saved to: {log_file_path}")
    return logger
#---------------Logging configuration-End---------------#

def split_list(lst, n):
    """
    Splits a list into approximately equal-sized sublists.

    This function divides a given list `lst` into `n` sublists of nearly equal length.
    If the length of `lst` is not perfectly divisible by `n`, the remainder elements are
    distributed among the last few sublists, with any remaining extra elements being added
    to the final sublist.

    Parameters:
    lst (list): The list to be split into sublists.
    n (int): The number of sublists to create. Must be greater than 0.

    Returns:
    list: A list containing `n` sublists, where each sublist is a segment of the input list.
          The final sublist contains any remaining elements if `len(lst) % n != 0`.

    Raises:
    ValueError: If `n` is less than or equal to 0.

    Examples:
    >>> split_list([1, 2, 3, 4, 5, 6, 7, 8, 9], 3)
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    >>> split_list([1, 2, 3, 4, 5, 6, 7], 3)
    [[1, 2, 3], [4, 5], [6, 7]]

    >>> split_list(['a', 'b', 'c'], 4)
    [['a'], ['b'], ['c'], []]
    """
    if n <= 0:
        raise ValueError("The number of sublists 'n' must be greater than 0.")
    # Calculate size of each chunk
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

#---------------Semantic similarity check class-Begin-------------#
class SemanticSimilarity:
    """
    A class for computing semantic similarity between sentences using a pre-trained
    transformer model from the Sentence-Transformers library.

    This class utilizes a caching mechanism to store previously computed sentence embeddings,
    reducing redundant computations and improving efficiency.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer for processing input text.
        model (AutoModel): Pre-trained transformer model for embedding extraction.
        embedding_cache (dict): Dictionary that caches computed sentence embeddings to avoid
            redundant computations.
    """
    def __init__(self, embedding_cache: dict = None):
        """
        Initializes the SemanticSimilarity class with a tokenizer, model, and an optional cache.

        Args:
            embedding_cache (dict, optional): A dictionary to store sentence embeddings.
                If None, an empty dictionary is initialized.
        """
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_cache = embedding_cache if embedding_cache is not None else {}

    def compute_similarity(self, sentence1: str, sentence2: str) -> float:
        """
        Computes the cosine similarity between two sentences.

        Args:
            sentence1 (str): The first sentence.
            sentence2 (str): The second sentence.

        Returns:
            float: A similarity score between -1 and 1, where 1 indicates identical sentences
                and -1 indicates completely dissimilar sentences.
        """
        embedding1 = self.get_or_compute_embedding(sentence1)
        embedding2 = self.get_or_compute_embedding(sentence2)
        similarity = 1 - cosine(embedding1, embedding2)
        return similarity

    def sentence_to_embedding(self, sentence: str) -> np.ndarray:
        """
        Converts a single sentence into its embedding representation.

        This method first checks if the sentence embedding exists in the cache.
        If found, the cached embedding is returned. Otherwise, the embedding is computed,
        stored in the cache, and then returned.

        Args:
            sentence (str): The sentence to be embedded.

        Returns:
            np.ndarray: The sentence embedding as a NumPy array.
        """
        if sentence in self.embedding_cache:
            return self.embedding_cache[sentence]
        
        inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        self.embedding_cache[sentence] = embedding
        return embedding

    def batch_sentence_to_embeddings(self, sentences: list) -> torch.Tensor:
        """
        Converts a batch of sentences into their respective embedding representations.

        Args:
            sentences (list): A list of sentences to be embedded.

        Returns:
            torch.Tensor: A tensor containing embeddings for the batch of sentences.
        """
        inputs = self.tokenizer(sentences, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def batch_semantic_similarity(self, new_sentence: str, existing_sentences: list) -> list:
        """
        Computes the semantic similarity between a new sentence and a batch of existing sentences.

        Args:
            new_sentence (str): The sentence to compare against the batch.
            existing_sentences (list): A list of sentences to compare with the new sentence.

        Returns:
            list: A list of similarity scores between the new sentence and each existing sentence.
        """
        new_embedding = torch.tensor(self.get_or_compute_embedding(new_sentence)).unsqueeze(0)
        existing_embeddings = self.batch_sentence_to_embeddings(existing_sentences)
        similarities = 1 - torch.nn.functional.cosine_similarity(new_embedding, existing_embeddings)
        return similarities.tolist()

    def get_or_compute_embedding(self, sentence: str) -> np.ndarray:
        """
        Retrieves a sentence embedding from the cache if available, otherwise computes it.

        This method ensures that redundant computations are minimized by storing previously
        computed embeddings in a cache.

        Args:
            sentence (str): The sentence whose embedding is required.

        Returns:
            np.ndarray: The embedding representation of the sentence.
        """
        if sentence in self.embedding_cache:
            return self.embedding_cache[sentence]
        embedding = self.sentence_to_embedding(sentence)
        return embedding
#---------------Semantic similarity check class-End-------------#

#---------------Configuration loader-Begin-------------#

class Config:
    _instance = None  # Singleton instance

    def __new__(cls, config_path=None):
        """Ensure only one instance of Config exists (Singleton)."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialize(config_path)  # ✅ Initialize here
        return cls._instance

    def _initialize(self, config_path):
        """Initialize the config only once."""
        if hasattr(self, "_config"):  # ✅ Prevent re-initialization
            return
        self.config_path = config_path or os.getenv("PROPHET_CONFIG_PATH", "infra/config.yaml")
        self._config = self._load_config()  # ✅ Initialize _config

    def _load_config(self):
        """Load YAML config."""
        try:
            with open(self.config_path, "r") as file:
                return yaml.safe_load(file) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML: {e}")

    def __getitem__(self, key):
        return self._config.get(key)

    def __getattr__(self, key):
        """Ensure _config exists before accessing it to avoid recursion."""
        if "_config" not in self.__dict__:  # ✅ Check if _config is initialized
            raise AttributeError(f"Config not initialized yet: Missing key '{key}'")
        try:
            return self._config[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def as_dict(self):
        return self._config
#---------------Configuration loader-End-------------#