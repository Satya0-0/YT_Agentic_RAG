from src.state import *
from src.retrievals import global_vector_store
import logging

logger = logging.getLogger(__name__)

# Defining the Routing Functions

def vector_db_exists(state: State) -> bool:
    """Checks if the VectorDB already exists for the video"""
    if state.vectorDB_flg and global_vector_store is not None:
        logger.debug("[ROUTER] VectorDB exists.")
        return True
    else:
        logger.error("[ROUTER] VectorDB does not exist.")
        return False

def acceptable_for_demo(state: State) -> bool:
    """Checks if the video is acceptable for the demo based on duration and content"""
    logger.debug(f"[ROUTER] Proceed with demo is set to: {state.proceed_with_demo}")
    return state.proceed_with_demo

def retrieved_docs_relevant(state: State) -> str:
    """Checks if the documents retired are relevant to determine the next course of action.
       Also checks if the query is re-written"""
    
    logger.debug(f"[ROUTER] Checking document relevance basis Retrieval_sync: {state.retrieval_sync}, and  Rewritten_flg: {state.rewritten_flg} flags")
    
    if state.retrieval_sync:
        next_node = "Response"
    elif not state.retrieval_sync and state.rewritten_flg:
        next_node = "webSearch"
    elif not state.retrieval_sync and not state.rewritten_flg:
        next_node = "Rewrite"
    else:
        # Fallback (shouldn't reach here)
        next_node = "Response"

    logger.debug(f"[ROUTER] Next node determined: {next_node}")
    return next_node


def graph_exit(state: State) -> bool:
    """Determines if the graph should be exited based on user input"""
    logger.debug(f"[ROUTER] Checking if graph should exit. Graph exit flag: {state.graph_exit}")
    return state.graph_exit