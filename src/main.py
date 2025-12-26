# Imports
from src.state import *
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.yt_video import node1_video_download, node2_transcription, node3_clean_up
from src.retrievals import node4_vectordb, node5_retriever
from src.query_optimizations import node6_llm_judge, node7_query_rewriter
from src.websearch import node8_web_search
from src.responses import node9_generate_response, node10_get_user_input
from src.routing_functions import vector_db_exists, retrieved_docs_relevant, graph_exit
import sys

def main():
    # Instantiating the Graph
    graph = StateGraph(State)

    # Adding nodes

    graph.add_node("1_YTVideoDownload", node1_video_download)
    graph.add_node("2_Transcription", node2_transcription)
    graph.add_node("3_CleanUp", node3_clean_up)
    graph.add_node("4_vectorDB", node4_vectordb)
    graph.add_node("5_Retriever", node5_retriever)
    graph.add_node("6_llmJudge", node6_llm_judge)
    graph.add_node("7_QueryRewriter", node7_query_rewriter)
    graph.add_node("8_WebSearch", node8_web_search)
    graph.add_node("9_GenerateResponse", node9_generate_response)
    graph.add_node("10_getUserInput", node10_get_user_input)

    # Adding Edges to the "graph" instance

    graph.add_conditional_edges(START, vector_db_exists, {True: "5_Retriever", False: "1_YTVideoDownload"})
    graph.add_edge("1_YTVideoDownload", "2_Transcription")
    graph.add_edge("2_Transcription", "3_CleanUp")
    graph.add_edge("3_CleanUp", "4_vectorDB")
    graph.add_edge("4_vectorDB", "5_Retriever")
    graph.add_edge("5_Retriever", "6_llmJudge")
    graph.add_conditional_edges("6_llmJudge", retrieved_docs_relevant, {"Response": "9_GenerateResponse", "Rewrite": "7_QueryRewriter", "webSearch": "8_WebSearch"})
    graph.add_edge("7_QueryRewriter", "5_Retriever")
    graph.add_edge("8_WebSearch", "9_GenerateResponse")
    graph.add_edge("9_GenerateResponse", "10_getUserInput")
    graph.add_conditional_edges("10_getUserInput", graph_exit, {True: END, False: "5_Retriever"})

    # Compling the Graph
    # Using a memory saver so the state persists between runs
    app = graph.compile(checkpointer=MemorySaver())

    # A unique ID for the conversation
    thread_id = "chat-1"

    # Taking in user arguments
    yt_url = input(">> Enter the YouTube URL for Q&A: ")
    initial_user_query = input(">> Enter your query: ")

    # Starting the Graph with Initial User Query
    try:
        app.update_state(
            config={
                "configurable": {
                    "thread_id": thread_id
                }
            },
            values={
                "user_query": initial_user_query,
                "youtubeURL": yt_url
            }
        )
    except Exception as e:
        print(f"Error occurred:{e}\nExiting the application!")
        sys.exit(1)

    result = app.invoke({}, config={"configurable": {"thread_id": thread_id}})


if __name__ == "__main__":
    main()