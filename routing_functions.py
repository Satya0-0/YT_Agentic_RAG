from state import *
from retrievals import global_vector_store

# Defining the Routing Functions

def vector_db_exists(state: State) -> bool:
    """Checks if the VectorDB already exists for the video"""
    if state.vectorDB_flg and global_vector_store is not None:
        return True
    else:
        return False



def retrieved_docs_relevant(state: State) -> str:
    """Checks if the documents retired are relevant to determine the next course of action.
       Also checks if the query is re-written"""

    if state.retrieval_sync:
        next_node = "Response"
    elif not state.retrieval_sync and state.rewritten_flg:
        next_node = "webSearch"
    elif not state.retrieval_sync and not state.rewritten_flg:
        next_node = "Rewrite"
    else:
        # Fallback (shouldn't reach here)
        return "Response"

    return next_node


def graph_exit(state: State) -> bool:
    """Determines if the graph should be exited based on user input"""
    return state.graph_exit