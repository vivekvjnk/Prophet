import os

from Bodhi.Bodhi import Bodhi
from Odysseus.engine import Odysseus
from Alchemist import engine

from infra.utils.utils import create_unique_trace_id

if __name__ == "__main__":
    parent_trace_id = create_unique_trace_id()

    bodhi = Bodhi(parent_trace_id=parent_trace_id)
    data_path = os.path.join(os.path.dirname(__file__),"../storage/data")
    init_state = {"source_path":f"langgraph_application_structure.md"}
    
    final_state = bodhi.invoke(init_state)
    
    odysseus = Odysseus(sources=["langgraph_application_structure"])
    odysseus() # start odysseus server


    