from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

# Load Token from Environment
hf_api_token = os.getenv("HF_API_TOKEN")

# Initialize LLM
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    huggingfacehub_api_token=hf_api_token
)

# Use Chat Model
model = ChatHuggingFace(llm=llm)
result = model.invoke("What is the capital of India?")

print("\n".join(result.content.split('.')))