from openai import OpenAI
from mem0 import Memory
import os
BASE_URL = "https://openkey.cloud/v1"
API_KEY = "sk-wiHpoarpNTHaep0t54852a32A75a4d6986108b3f6eF7B7B9"
openai_client = OpenAI(base_url="https://openkey.cloud/v1",api_key="sk-wiHpoarpNTHaep0t54852a32A75a4d6986108b3f6eF7B7B9")
config = {
    # "graph_store": {
    #     "provider": "neo4j",
    #     "config": {
    #         "url": "neo4j://127.0.0.1:7687",
    #         "username": "zyantine",
    #         "password": "632147Air",
    #         "database": "neo4j",
    #     }
    # },
    "vector_store": {
        "provider": "milvus",
        "config": {
            "collection_name": "default",
                "url": "http://localhost:19530",
                "token": "",
            }
    },
    "llm": {
        "provider": "openai",
        "config": {
            "openai_base_url":BASE_URL,
            "api_key":API_KEY
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-large",
            "openai_base_url":BASE_URL,
            "api_key":API_KEY
        }
    }
}

memory = Memory.from_config(config)

conversation = [
    {"role": "user", "content": "你还记得我叫什么名字吗"},
]

memory.add(conversation, user_id="demo-user")

results = memory.search(
    "用户叫什么名字？",
    user_id="demo-user",
    limit=3,
    rerank=True,
)

for hit in results["results"]:
    print(hit["memory"])