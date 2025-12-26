# Imports
from services.llm_provider import get_llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from state import *

# -----------------
# Global Variable
output_parser = StrOutputParser()


# Node-6: LLM Judge (determines the relevance of retrieved documents)

def node6_llm_judge(state: State) -> dict:
    """Acts as an LLM Judge to determine retrieved document relevance"""
    llm_judge = get_llm()
    global output_parser

    judge_sys_message = """You are acting as a Judge for determining the relevance of provided input documents in the context
    and comparing it with the query.
    If you think the provided prompts contain the relevant details required to answer the question, say "True", otherwise, say "False".
    You will not generate any additional response other than a single word - "True" or "False".
    Context is : {context}
    \n
    Query is: {user_input}"""

    judge_prompt = ChatPromptTemplate([
        ("system", judge_sys_message),
        ("human", "Context: {context} \n Query: {user_input}")
    ])

    judge_chain = (
            judge_prompt
            | llm_judge
            | output_parser
    )

    if state.rewritten_flg:
        query_to_use = state.rewritten_query
    else:
        query_to_use = state.user_query

    judgement = judge_chain.invoke({"context": state.documents, "user_input": query_to_use})

    judgement_result = judgement.lower() == "true"

    print("Node-6 Executed!")

    return {"retrieval_sync": judgement_result}



# Node-7: Conditional Node - Rewriting the query (depends on the State variable "retrieval_sync")

def node7_query_rewriter(state: State) -> dict:
    """Rewrites the initial user query using another LLM for better retrievals"""
    llm_query_rewrite = get_llm()
    global output_parser

    rewrite_sys_message = """You are an expert **Query Rewriter** for a sophisticated **Agentic RAG (Retrieval-Augmented Generation) system**. Your sole task is to take a user's initial query and rewrite it into a **highly optimized, self-contained, and comprehensive search query** that will maximize the chances of retrieving relevant documents from a technical knowledge base.

    **Key principles for the rewritten query:**
    1.  **Contextualization:** Expand acronyms, define ambiguous terms, or add missing context implied by the original conversation (if applicable, though for a first-pass query, focus on being self-contained).
    2.  **Explicitness:** Convert vague questions (e.g., "how to fix that?") into concrete statements or specific requests (e.g., "what are the troubleshooting steps for error code 503 on the Kubernetes control plane?").
    3.  **Keyword Density:** Increase the number of relevant, specific technical terms that an indexing system would recognize.
    4.  **Stand-alone Clarity:** The rewritten query **must** make sense and be an effective search term even if the original query is lost.

    Original User Query: {user_input}"""

    rewrite_prompt = ChatPromptTemplate([
        ("system", rewrite_sys_message),
        ("human", "{user_input}")
    ])

    rewrite_chain = (rewrite_prompt | llm_query_rewrite | output_parser)

    rewritten_query = rewrite_chain.invoke({"user_input": state.user_query})

    print("Node-7 Executed!")

    return {"rewritten_query": rewritten_query, "rewritten_flg": True}