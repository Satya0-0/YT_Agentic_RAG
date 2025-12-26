from state import *
from langchain_community.tools import DuckDuckGoSearchRun


# Node-8: Conditional Node - Performs a Websearch if the re-written query also doesn't fetch the required docs

def node8_web_search(state: State) -> dict:
    """Used to perform a web search using DuckDuckGo for the given (rewritten) query"""

    if not state.rewritten_query:
        # to prevent a failed tool call.
        print("ERROR: WebSearch Node reached without a rewritten_query. Skipping web search.")
        return {"webResults": "Web search skipped due to missing rewritten query."}

    search = DuckDuckGoSearchRun()
    web_result = search.invoke(state.rewritten_query)

    print("Node-8 Executed!")
    print(f"Search result: {web_result}")  # for Debugging

    return {"webResults": web_result}