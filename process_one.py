from multiprocessing import Manager
from tqdm.contrib.concurrent import process_map

from opendeepsearch import OpenDeepSearchTool, QueryRephrasingTool
from smolagents import CodeAgent, LiteLLMModel, LogLevel
import os
from opendeepsearch.prompts import ANALYSIS_PROMPT
from litellm import completion, utils

# Set environment variables for API keys
os.environ["SERPER_API_KEY"] = "e638d76ed48e55d1a0f52634096d1e2bbfcba6c2"  # If using Serper
# Or for SearXNG
# os.environ["SEARXNG_INSTANCE_URL"] = "https://your-searxng-instance.com"
# os.environ["SEARXNG_API_KEY"] = "your-api-key-here"  # Optional

os.environ["FIREWORKS_API_KEY"] = "fw_3ZbxviZGMfmYbiJXsWEADriD"
os.environ["JINA_API_KEY"] = "jina_41f45e97bb0e4f0584932a80299f7aa2OHnEzv2Y9IHLCegx8DrhslQa30HI"


os.environ["WOLFRAM_ALPHA_APP_ID"] = "Q6Y5YV-8TGGQ6TGKG"


# model_lite_llm = "fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct"
# model_lite_llm = "fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct"
model_anal = "fireworks_ai/accounts/fireworks/models/qwq-32b"
model_ods = "fireworks_ai/accounts/fireworks/models/qwen2p5-coder-32b-instruct"
model_lite_llm = "fireworks_ai/accounts/fireworks/models/qwen2p5-coder-32b-instruct"


def process_one(query):
# Using Serper (default)
    search_agent = OpenDeepSearchTool(
        model_name=model_ods,
        reranker="jina"
    )

    model = LiteLLMModel(
        model_lite_llm,
        temperature=0.2
    )

    code_agent = CodeAgent(tools=[search_agent], model=model, max_steps=7, verbosity_level=LogLevel.ERROR)

    messages = [
        {"role": "system", "content": ANALYSIS_PROMPT},
        {"role": "user", "content": f"Query:\n{query}\n"}
    ]

    response = completion(
        model=model_anal,
        messages=messages,
        max_tokens=8192,
        temperature=0.1,
        top_p=0.1
    )

    new_query = response.choices[0].message.content.rsplit("BREAKDOWN")[-1].strip().strip(":")

    final_query = "Actual query:\n" + query + "\n\nHere is a suggestion on how the question could be broken down:\n" + new_query
    result = code_agent.run(final_query)

    with open("log.txt", "a") as f:
        f.write(f"{query}, {result}\n")

    return [query, result]
