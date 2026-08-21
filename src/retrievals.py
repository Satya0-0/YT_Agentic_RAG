from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import get_config
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.runnables import RunnableConfig
from typing import List, Any
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.state import *
from src.services.nlp_transformers import get_sentence_transformer
import logging

logger = logging.getLogger(__name__)

# Global Variable
global_vector_store = None

# Node-1: Retriever
async def node1_retriever(state: State, config: RunnableConfig) -> dict:
    """Retrieves documents from the ChromaDB collection basis semantic Search with User query"""
   
   # Access from LG's config
    vector_store = config["configurable"]["vector_store"]
    query = state.rewritten_query if state.rewritten_flg else state.user_query
    docs = await vector_store.asimilarity_search(query, k=5)
    docs_list = [doc.page_content for doc in docs]
    
    # Logging the retrievals
    if docs_list:
        logger.debug("\n\n========Retrieved contents:========\n")
        for id, doc in enumerate(docs_list):
            doc_id = id+1
            logger.debug(f"Retrieved_Doc_{doc_id};\nContents:\n{doc}\n\n")
        logger.debug("\n\n========End========\n")
    logger.info("Node-1:Retriever Executed!")    
    return {"documents": docs_list}