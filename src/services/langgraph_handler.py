from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledGraph
from langgraph.graph.message import add_messages
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from torch import Tensor
from langchain_core.runnables.graph_mermaid import MermaidDrawMethod
from langchain_postgres.vectorstores import PGVector
from langgraph.prebuilt import create_react_agent
from langmem import create_manage_memory_tool, create_search_memory_tool
from langchain_core.tools import tool
from dotenv import load_dotenv

from src.services.tools import (
    get_weather,
    get_wiki_summary,
    get_tavily_search_tool,
    calendar_events_handler,
)
from src.crud.db_handler import DataBaseHandler

from sentence_transformers import SentenceTransformer

from pathlib import Path

import json
import uuid
import os

from typing import Annotated
from typing_extensions import TypedDict, Tuple

chats_by_session_id = {}


class State(TypedDict):
    messages: Annotated[list, add_messages]
    metadata: dict
    telegram_id: int
    summary: str | None


class IsValid(TypedDict):
    is_valid: bool


class EvaluationResponse(TypedDict):
    evaluation_success: int


class LangGraphHandler:
    def __init__(self, telegram_id: int = 0):
        load_dotenv()
        try:
            self.telegram_id = telegram_id
        except TypeError:
            self.telegram_id = 0

        with DataBaseHandler() as db_handler:
            self.summary = db_handler.get_chat_summary_from_db(self.telegram_id)

        self.llm = self._build_model()

        self.workflow = self._build_workflow()
        self.guidelines = self._load_guidelines()

    def validate_query(self, state: State) -> State:
        print("Validating query against ethical guidelines...")
        validation_prompt = f"""
        You are an AI assistant. Your task is to validate the user's query against ethical guidelines {self.guidelines}.
        If the query is valid, return True. If the query is invalid, return False."""
        structured_output = self.llm.with_structured_output(IsValid)
        try:
            user_message = state["messages"][-1].content
            validation_response = structured_output.invoke(
                [
                    SystemMessage(content=validation_prompt),
                    HumanMessage(content=user_message),
                ]
            )
            print(f"Validation response: {validation_response}")
            is_valid = validation_response["is_valid"]
            state["metadata"]["is_valid"] = is_valid
            if validation_response == "True":
                print(f"Query within guidelines: {state['metadata']['is_valid']}")
            if not is_valid:
                state["messages"].append(
                    AIMessage(content="Sorry, I can't assist with that. (Guidelines)")
                )

            return state
        except Exception as e:
            print(f"Error in input validation: {e}")
            state["metadata"]["is_valid"] = True
            return state

    def create_agent(self) -> CompiledGraph:
        """Creates a new agent."""
        namespace = (
            "agent_memories",
            str(self.telegram_id),
        )  # converted to str for namespace
        checkpointer = InMemorySaver()
        return create_react_agent(
            self.llm,
            tools=[
                create_manage_memory_tool(namespace),
                create_search_memory_tool(namespace),
                get_weather,
                get_wiki_summary,
                get_tavily_search_tool,
                calendar_events_handler,
            ],
            checkpointer=checkpointer,
        )

    def draw_graph(self) -> None:
        with open("graph.png", "wb") as f:
            f.write(
                self.workflow.get_graph().draw_mermaid_png(
                    max_retries=5,
                    retry_delay=2.0,
                    draw_method=MermaidDrawMethod.PYPPETEER,  # ← use browser-based rendering
                )
            )

    def chatbot_handler(self, user_query: str) -> Tuple[str, int]:
        """Handles the chatbot interaction."""
        chat_history = self._get_chat_history()
        chat_history.add_user_message(user_query)
        system_message = SystemMessage(
            content="""
        You are an AI assistant called Vita Samara. Your task is to assist the user with their queries. You have access
        to a summary of previous messages in the conversation: {self.chat_summary}. Answer the questions where
        appropriate and provide relevant information. Be precise and concise.
        """
        )
        print("processing user query...")
        print(f"User query: {user_query}")

        messages = chat_history.messages + [
            system_message,
            HumanMessage(content=user_query),
            self.summary,
        ]

        state = {
            "messages": messages,
            "metadata": {"is_valid": True},
            "telegram_id": self.telegram_id,
            "summary": self.summary,
        }
        thread_id = str(uuid.uuid4())
        config = RunnableConfig(
            configurable={"session_id": self.telegram_id, "thread_id": thread_id}
        )

        result = self.workflow.invoke(state, config=config)

        ai_responses = [
            m.content for m in result["messages"] if isinstance(m, AIMessage)
        ]

        ai_response = (
            ai_responses[-1]
            if ai_responses
            else "sorry, I couldn't generate a response."
        )
        print(f"AI response: {ai_response}")
        evaluation = self._evaluate_response(
            user_query=user_query, ai_response=ai_response
        )

        self.summary = self._update_summary(user_query, ai_response)
        self._save_summary_to_db(self.summary)
        self.add_to_chat_vector_store(user_query, ai_response)

        return ai_response, int(evaluation)

    def main(self, user_query: str):
        response = self.chatbot_handler(user_query)
        print(response)

    @staticmethod
    def _load_guidelines():
        """Load assistant guidelines from file."""
        base_path = Path(__file__).parent.parent.parent / "src" / "services"
        try:
            guidelines_path = f"{base_path}/guidelines.json"
            with open(guidelines_path, "r") as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading guidelines: {e}")
            return {
                "prohibited_content": [
                    "harmful content",
                    "illegal activities",
                ],
                "privacy": "Do not share personal information",
            }

    @staticmethod
    def _build_model() -> ChatOpenAI:
        """Builds the model."""
        model = "gpt-4o-mini"
        llm = ChatOpenAI(
            model=model,
            temperature=0,
            max_tokens=1000,
        )
        return llm

    def _build_workflow(self) -> CompiledGraph:
        workflow = StateGraph(State)
        agent = self.create_agent()

        workflow.add_node("validate_query", self.validate_query)

        workflow.add_node("agent", agent)

        workflow.add_edge(START, "validate_query")
        workflow.add_conditional_edges(
            "validate_query",
            lambda state: "agent" if state["metadata"].get("is_valid", True) else END,
        )
        workflow.add_edge("agent", END)

        return workflow.compile()

    def _update_summary(self, user_query, ai_response) -> str:
        """Update the conversation summary."""
        print("Updating Summary...")
        summary_prompt = f"""
        Previous summary: {self.summary}
        New exchange:
        User: {user_query}, AI: {ai_response} ,
        Provide a concise summary of the entire conversation so far. Mote Places, dates, names, and other entities
        """
        # Generate new summary
        new_summary = self.llm.invoke([HumanMessage(content=summary_prompt)])
        print(f"New summary:\n{new_summary.content}")
        return new_summary.content

    def _save_summary_to_db(self, summary) -> None:
        """Save the summary to the database."""
        print("Saving chat summary to the database...")
        with DataBaseHandler() as db_handler:
            db_handler.write_chat_summary_to_db(
                telegram_id=self.telegram_id,
                summary=summary,
            )

    def _get_chat_history(self) -> InMemoryChatMessageHistory:
        """Get chat history from the database."""
        chat_history = chats_by_session_id.get(self.telegram_id)
        if chat_history is None:
            chat_history = InMemoryChatMessageHistory()
            chats_by_session_id[self.telegram_id] = chat_history
        return chat_history

    def _evaluate_response(self, user_query, ai_response) -> str:
        "evaluate response against query"
        print("Evaluating response...")

        evaluation_prompt = f"""

        Evaluate, how well the generated response {ai_response} fulfills the given query {user_query}.
        Compares the generated response to the input query and calculate the degree to which the response satisfies the
        query's intent and content. The result (evaluation_success) is returned as an integer between 0 and 100, where 100
        indicates a perfect match and lower values indicate partial fulfillment of the query.
        """
        structured_output = self.llm.with_structured_output(EvaluationResponse)

        evaluation = structured_output.invoke(
            [SystemMessage(content=evaluation_prompt)]
        )

        print(f"Response accuracy: {evaluation['evaluation_success']} %")

        return evaluation["evaluation_success"]

    @staticmethod
    def create_embedding(user_query: str, ai_response: str) -> Tensor:
        """Create an embedding for the given text."""
        text = f"User: {user_query}, AI: {ai_response}"
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding = model.encode(text)
        return embedding

    @staticmethod
    def create_user_query_embedding(user_query: str) -> Tensor:
        """Create an embedding for the user query."""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding = model.encode(user_query)
        return embedding

    @tool
    def search_chat_history(self, embedded_user_query: Tensor) -> list[str]:
        """Search the chat history for relevant messages."""
        print("Searching chat history...")
        pass

    def add_to_chat_vector_store(self, user_query: str, ai_response: str) -> None:
        """Add the user query and AI response to the chat vector store."""
        print("Adding to chat vector store...")

        # Load environment variables
        load_dotenv()
        connection_string = os.getenv("DATABASE_URL")

        # The text that will be added to the chat vector store
        text = f"User: {user_query}, AI: {ai_response}"

        # Generate embeddings for the text using the SentenceTransformer model
        embedding_function = EmbeddingFunctionWrapper("all-MiniLM-L6-v2")

        # Create PGVector using the embeddings
        vector_store = PGVector(
            connection=connection_string,
            collection_name="chat_history",  # Use the wrapper for embedding
            use_jsonb=True,
            embeddings=embedding_function,
        )
        # Add the text data along with the embeddings
        vector_store.add_texts(
            texts=[text],
            metadatas=[{"telegram_id": self.telegram_id}],
            namespace=self.telegram_id,
        )


class EmbeddingFunctionWrapper:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Use the encode method to generate embeddings
        return self.model.encode(texts, convert_to_tensor=False)

    def embed_query(self, text: str) -> list[float]:
        # For single query embedding
        return self.model.encode(text, convert_to_tensor=False)
