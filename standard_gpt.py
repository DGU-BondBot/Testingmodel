from langchain_community.chat_models import ChatOpenAI

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("OPENAI_API_KEY")

chat_model = ChatOpenAI(openai_api_key=api_key)

chat_model.predict("잠이 안 올 때는 어떻게 하면 좋을지 대답해줘")