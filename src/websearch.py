from src.state import *
from langchain_community.tools import DuckDuckGoSearchRun
import logging

logger = logging.getLogger(__name__)

# Node-4: Conditional Node - Performs a Websearch if the re-written query also doesn't fetch the required docs

async def node4_web_search(state: State) -> dict:
    """Used to perform a web search using DuckDuckGo for the given (rewritten) query"""

    if not state.rewritten_query:
        # to prevent a failed tool call.
        logger.error("ERROR: WebSearch Node reached without a rewritten_query. Skipping web search.")
        return {"webResults": "Web search skipped due to missing rewritten query."}

    search = DuckDuckGoSearchRun()
    web_result = search.invoke(state.rewritten_query)

    logger.info("Node-4: WebSearch Executed!")
    logger.debug(f"Search result: {web_result}")  # for Debugging

    return {"webResults": web_result}