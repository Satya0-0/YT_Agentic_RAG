# Imports
from src.state import *
import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.video_transcriber import step1_video_download, step2_InitiateVectorDB
from src.retrievals import node1_retriever
from src.query_optimizations import node2_llm_judge, node3_query_rewriter
from src.websearch import node4_web_search
from src.responses import node5_generate_response
from src.routing_functions import vector_db_exists, retrieved_docs_relevant, graph_exit, acceptable_for_demo
import sys
import os
import asyncio

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

# Initializing ChromaDB variable to `None`
vector_store = None

# Defining the Graph function

def BuildGraph() -> None:
    """
    Function used to build and compile the RAG pipeline to facilitate futher use as Q&A app
    """

    # Instantiating the Graph
    graph = StateGraph(State)

    # Adding nodes
    graph.add_node("1_Retriever", node1_retriever)
    graph.add_node("2_llmJudge", node2_llm_judge)
    graph.add_node("3_QueryRewriter", node3_query_rewriter)
    graph.add_node("4_WebSearch", node4_web_search)
    graph.add_node("5_GenerateResponse", node5_generate_response)

    # Adding Edges to the "graph" instance

    graph.add_edge(START, "1_Retriever")
    graph.add_edge("1_Retriever", "2_llmJudge")
    graph.add_conditional_edges("2_llmJudge", retrieved_docs_relevant, {"Response": "5_GenerateResponse", "Rewrite": "3_QueryRewriter", "webSearch": "4_WebSearch"})
    graph.add_edge("3_QueryRewriter", "1_Retriever")
    graph.add_edge("4_WebSearch", "5_GenerateResponse")
    graph.add_edge("5_GenerateResponse", END)

    # Compling the Graph
    # Using a memory saver so the state persists between runs
    compiled_graph = graph.compile(checkpointer=MemorySaver())
    return compiled_graph

# Build the graph   
app = BuildGraph()


# Running the transcription and initializing the ChromaDB
async def Transcribe(yt_url: str):
    """
    Function used to download the YouTube video -> Generate it's transcript and embedd it-> Store embeddings in ChromaDB 
    """
    logger.info("Starting the YouTube Q&A Application")
    
    # Taking in YT Video URL
    logger.info("Downloading and trancribing the video.")
    transcription = await step1_video_download(yt_url)
    if transcription and transcription.strip() != "":
        vector_store = await step2_InitiateVectorDB(transcription)
        return vector_store
    else:
        logger.error("Issue with Video. Exiting the app!")
        raise ValueError("Invalid URL.")

# Q&A: Taking User query and generating the response
async def QnALoop(user_query:str, vector_store) -> str:
    """
    Function used to accept a user query along with the previously created ChromaDB instance to generate responses from the RAG pipeline
    """
    # A unique ID for the conversation
    thread_id = "chat-1"
    
    # Check if ChromaDB exists or not
    if not vector_store:
        logger.info("ChormaDB not initialized. Exitting the loop")
        raise ValueError("No ChromaDB instance present!")
    
 
    # Starting the Graph with Provided User Query
    try:
        await app.aupdate_state(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "recursion_limit": 30,
                    "vector_store": vector_store
                }
            },
            values={
                "user_query": user_query
            }
        )
    except Exception as e:
        logger.error(f"Error occurred:{e}\nExiting the application!")
        raise ValueError("Inalid request. Unable to initiate the application")

    logger.info("Invoking the application")
    result = await app.ainvoke({}, config={"configurable": {"thread_id": thread_id, "vector_store": vector_store}})

    return result["graph_output"]