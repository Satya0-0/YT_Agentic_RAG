# Imports
from src.state import *
import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
# from src.yt_video import node1_video_download, node2_transcription, node3_clean_up
# from src.retrievals import node4_vectordb,
from src.video_transcriber import step1_video_download, step2_InitiateVectorDB
from src.retrievals import node5_retriever
from src.query_optimizations import node6_llm_judge, node7_query_rewriter
from src.websearch import node8_web_search
from src.responses import node9_generate_response #, node10_get_user_input
from src.routing_functions import vector_db_exists, retrieved_docs_relevant, graph_exit, acceptable_for_demo
import sys
import os

# Handling Logging
ENVIRONMENT = os.getenv("ENVIRONMENT", "local")

if ENVIRONMENT == "local":
    LOG_LEVEL = logging.DEBUG
    handlers = logging.FileHandler('logs/yt_video_rag.log', encoding='utf-8', mode='w')
else:
    LOG_LEVEL = logging.INFO
    handlers = logging.StreamHandler(sys.stdout)

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[handlers])
logger = logging.getLogger(__name__)


# Defning the main function
def main():
    logger.info("Starting the YouTube Q&A Application")
    
    # Taking in YT Video URL
    yt_url = input(">> Enter the YouTube URL for Q&A: ")
    logger.info("Downloading and trancribing the video.")
    transcription = step1_video_download(yt_url)
    if transcription != False:
        local_vector_store = step2_InitiateVectorDB(transcription)
    else:
        logger.error("Issue with Video. Exiting the app!")
        sys.exit(1)

    # Instantiating the Graph
    graph = StateGraph(State)

    # Adding nodes
    graph.add_node("5_Retriever", node5_retriever)
    graph.add_node("6_llmJudge", node6_llm_judge)
    graph.add_node("7_QueryRewriter", node7_query_rewriter)
    graph.add_node("8_WebSearch", node8_web_search)
    graph.add_node("9_GenerateResponse", node9_generate_response)
    # graph.add_node("10_getUserInput", node10_get_user_input)

    # Adding Edges to the "graph" instance

    graph.add_edge(START, "5_Retriever")
    graph.add_edge("5_Retriever", "6_llmJudge")
    graph.add_conditional_edges("6_llmJudge", retrieved_docs_relevant, {"Response": "9_GenerateResponse", "Rewrite": "7_QueryRewriter", "webSearch": "8_WebSearch"})
    graph.add_edge("7_QueryRewriter", "5_Retriever")
    graph.add_edge("8_WebSearch", "9_GenerateResponse")
    graph.add_edge("9_GenerateResponse", END)

    # Compling the Graph
    # Using a memory saver so the state persists between runs
    app = graph.compile(checkpointer=MemorySaver())

    # A unique ID for the conversation
    thread_id = "chat-1"

    # Taking in user queries
    user_query = input(">> Enter your query: ")

    # Starting the Graph with Initial User Query
    try:
        app.update_state(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "recursion_limit": 100,
                    "vector_store": local_vector_store
                }
            },
            values={
                "user_query": user_query
            }
        )
    except Exception as e:
        logger.error(f"Error occurred:{e}\nExiting the application!")
        sys.exit(1)

    logger.info("Invoking the application")
    result = app.invoke({}, config={"configurable": {"thread_id": thread_id, "vector_store": local_vector_store}})
    logger.info("Application finished execution")

if __name__ == "__main__":
    main()