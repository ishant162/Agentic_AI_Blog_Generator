import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv


class GroqLLM:
    """A class to initialize and manage the Groq LLM."""
    def __init__(self):
        load_dotenv()

    def get_llm(self):
        """
        Initializes the Groq LLM with the API key from environment variables.
        """
        try:
            self.groq_api_key = os.environ["GROQ_API_KEY"] = os.getenv(
                "GROQ_API"
            )
            self.llm = ChatGroq(
                model="llama-3.1-8b-instant",
                api_key=self.groq_api_key
            )
            return self.llm
        except Exception as e:
            raise ValueError(f"Error initializing GroqLLM: {e}")
