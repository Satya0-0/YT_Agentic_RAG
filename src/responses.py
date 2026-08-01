from src.state import *
from src.services.llm_provider import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging

logger = logging.getLogger(__name__)

# Global Variable
OUTPUT_PARSER = StrOutputParser()

# Node-9: Response Generator Node

def node9_generate_response(state: State) -> dict:
    """Final LLM which uses the retrieved docs/ web search results to generate User output"""
    global OUTPUT_PARSER
    
    generate_llm = get_llm()

    response_sys_message = """
    You are a smart assistant which takes in the given query and generates a response using the provided context data.
    If the provided data is insufficient to answer the question, reply with I don't know.
    The provided context data can be from retrieved documents or from web. Always mention the source of getting this data.
    Context is : {context}
    \n
    Query is: {user_input}"""

    generation_prompt = ChatPromptTemplate(
        [
            ("system", response_sys_message),
            ("human", "{user_input}")
        ]
    )

    response_chain = (generation_prompt | generate_llm | OUTPUT_PARSER)

    query = state.rewritten_query if state.rewritten_flg else  state.user_query

    context = state.documents if state.retrieval_sync else state.webResults

    result = response_chain.invoke({"context": context, "user_input": query})

    logger.info("Node-9 Executed!")
    print(f"Generated response: {result}")

    return {"graph_output" : result}


# Node-10: Placeholder Query. Used for "interrupt()"

# def node10_get_user_input(state: State) -> dict:
#     """Used for taking in new user input"""

#     # Taking in the New User Input
#     next_query = input("\n>>What's your next question (or type 'quit' to exit): ")

#     # Check if the graph reached END during the last run
#     try:
#         exit_decision = True if next_query.lower() == "quit" else False
#     except Exception as e:
#         logger.error("\nError: ", e)
#         exit_decision = True

#     logger.info("Node-10 Executed!")
#     logger.debug("All the variables initialized to their default values. Ready to receive new input!")
#     return {"graph_exit": exit_decision,
#             "user_query": next_query,
#             "rewritten_query": None,  # Reset for new query
#             "rewritten_flg": False,  # Reset flag
#             "documents": [],  # Clear old docs
#             "webResults": None  # Clear old web results
#             }