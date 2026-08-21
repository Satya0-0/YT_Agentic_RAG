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


# Defning the main function
async def main():
    logger.info("Starting the YouTube Q&A Application")
    
    # Taking in YT Video URL
    yt_url = input(">> Enter the YouTube URL for Q&A: ")
    logger.info("Downloading and trancribing the video.")
    transcription = await step1_video_download(yt_url)
    if transcription and transcription.strip() != "":
        local_vector_store = await step2_InitiateVectorDB(transcription)
    else:
        logger.error("Issue with Video. Exiting the app!")
        sys.exit(1)

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
    app = graph.compile(checkpointer=MemorySaver())

    # A unique ID for the conversation
    thread_id = "chat-1"
    
    
    # Loop control vairable 
    enter_loop = True

    while(enter_loop):
        # Taking in user queries
        user_query = input("\n>> Enter your query: ")
        print("\n")
    
        # Starting the Graph with Provided User Query
        try:
            await app.aupdate_state(
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
        result = await app.ainvoke({}, config={"configurable": {"thread_id": thread_id, "vector_store": local_vector_store}})

        print(f"\nGenerated response: {result["graph_output"]}\n")

        # Taking user input to decide if they want to continue or not
        user_choice_quit = input("\nDo you have further queries? (Yes/No): ")

        if user_choice_quit.strip().upper() =="YES" or user_choice_quit.strip().upper() == "Y":
            enter_loop = True
        elif user_choice_quit.strip().upper() =="NO" or user_choice_quit.strip().upper() == "N":
            enter_loop = False
        else:
            enter_loop = False
            print("Response Unrecognized. Exiting the app!")
            logger.info("User entered invalid input on Loop_Continue decision. Exited the loop!")

    logger.info("Application finished execution")

if __name__ == "__main__":
    asyncio.run(main())