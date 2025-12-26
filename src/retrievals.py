from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import get_config
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List, Any
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.state import *
from src.services.nlp_transformers import get_sentence_transformer


# Global Variable
global_vector_store = None


# Node-4: Creating a VectorDB Collection

def node4_vectordb(state: State) -> dict:
    """Creates the ChromaDB collection for the video transcription"""
    global global_vector_store

    # Splitting the transcribed text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=get_config("text_splitter.chunk_size"),
        chunk_overlap=get_config("text_splitter.overlap"),
        length_function=get_config("text_splitter.length"),
        is_separator_regex=get_config("text_splitter.is_separator_regex")
    )

    docs = text_splitter.create_documents([state.transcription])
    sentence_transformer = get_sentence_transformer()
    vector_store = Chroma(
        collection_name = get_config("vectorDB_collection_name"),
        embedding_function = sentence_transformer
    )

    # Adding documents to ChromaDB Collection - "YT_RAG"

    document_ids = list(f"id_{x}" for x in range(len(docs)))

    vector_store.add_documents(documents=docs, ids=document_ids)

    global_vector_store = vector_store

    print("Node-4 Executed!")

    return {"vectorDB_flg": True}


# Node-5: Retriever
def node5_retriever(state: State) -> dict:
    """Retrieves documents from the ChromaDB collection basis semantic Search with User query"""
    global global_vector_store

    # Access from global variable
    vector_store = global_vector_store

    class MyRetriever(BaseRetriever):
        class Config:
            arbitrary_types_allowed = True

        vdb: Chroma
        top_k: int = 3

        def _get_relevant_documents(self, qry, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
            results = self.vdb.similarity_search(query=qry, k=self.top_k)
            return results

        def get_relevant_documents(self, query):
            return self._get_relevant_documents(query, CallbackManagerForRetrieverRun)

    retriever = MyRetriever(vdb = vector_store, top_k=3)
    query = state.rewritten_query if state.rewritten_flg else state.user_query
    docs = retriever.get_relevant_documents(query)
    docs_list = list(doc.page_content for doc in docs)

    print("Node-5 Executed!")
    return {"documents": docs_list}