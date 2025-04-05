from typing import Optional, Literal
from smolagents import Tool
from litellm import completion, utils
from opendeepsearch.ods_agent import OpenDeepSearchAgent
from opendeepsearch.prompts import ANALYSIS_PROMPT

class OpenDeepSearchTool(Tool):
    name = "web_search"
    description = """
    Performs web search based on your query (think a Google search) then returns the final answer that is processed by an llm.
    This tool cannot access URLs directly."""
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query to perform",
        },
    }
    output_type = "string"

    def __init__(
        self,
        model_name: Optional[str] = None,
        reranker: str = "infinity",
        search_provider: Literal["serper", "searxng"] = "serper",
        serper_api_key: Optional[str] = None,
        searxng_instance_url: Optional[str] = None,
        searxng_api_key: Optional[str] = None
    ):
        super().__init__()
        self.search_model_name = model_name  # LiteLLM model name
        self.reranker = reranker
        self.search_provider = search_provider
        self.serper_api_key = serper_api_key
        self.searxng_instance_url = searxng_instance_url
        self.searxng_api_key = searxng_api_key

    def forward(self, query: str):
        answer = self.search_tool.ask_sync(query, max_sources=2, pro_mode=False)
        return answer

    def setup(self):
        self.search_tool = OpenDeepSearchAgent(
            self.search_model_name,
            reranker=self.reranker,
            search_provider=self.search_provider,
            serper_api_key=self.serper_api_key,
            searxng_instance_url=self.searxng_instance_url,
            searxng_api_key=self.searxng_api_key
        )


class QueryRephrasingTool(Tool):
    name = "query_rephrasing"
    description = """
    Use this as your first tool, all the time.
    Detects, understands and explains complex queries.
    Only relevant information is retained. Ambiguous statements are rephrased using simple language."""
    inputs = {
        "query": {
            "type": "string",
            "description": "The input query.",
        },
    }
    output_type = "string"

    def __init__(
        self,
        model_name: str,
        system_prompt: str = ANALYSIS_PROMPT,
        temperature: float = 0.2, # Slight variation while maintaining reliability
        top_p: float = 0.3
    ):
        super().__init__()
        self.model = model_name  # LiteLLM model name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p

    def forward(self, query: str):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Query:\n{query}\n"}
        ]

        response = completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p
        )

        print(response.choices[0].message.content)
