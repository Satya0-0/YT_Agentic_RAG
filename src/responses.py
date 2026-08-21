from src.state import *
from src.services.llm_provider import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging

logger = logging.getLogger(__name__)

# Global Variable
OUTPUT_PARSER = StrOutputParser()

# Node-5: Response Generator Node

async def node5_generate_response(state: State) -> dict:
    """Final LLM which uses the retrieved docs/ web search results to generate User output"""
    global OUTPUT_PARSER
    
    generate_llm = get_llm()

    response_sys_message = """
    You are a smart assistant which takes in the given query and generates a response using the provided context data.
    If the provided data is insufficient to answer the question, reply with I don't know.
    The provided context data can be from retrieved documents or from web. Always mention the source of getting this data.
    Also, be as precise in your response as possible and be brief.
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

    logger.info("Node-5: GenerateResponse Executed!")
    
    if state.retrieval_sync:
        logger.debug("\n\n========Retrieved contents:========\n")
        for id, doc in enumerate(context):
            doc_id = id+1
            logger.debug(f"Retrieved_Doc_{doc_id};\nContents:\n{doc}\n\n")

        logger.debug("\n\n========End========\n")

    return {"graph_output" : result}