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
from langchain.embeddings.base import Embeddings
from torch import Tensor
from langchain_postgres.vectorstores import PGVector
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langchain.schema import Document
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
import os

from typing import Annotated
from typing_extensions import TypedDict, Tuple

import numpy.typing as npt
import numpy as np

chats_by_session_id = {}
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class State(TypedDict):
    """State of the workflow."""

    messages: Annotated[list, add_messages]
    metadata: dict
    telegram_id: int
    summary: str | None


class IsValid(TypedDict):
    """Validation response."""

    is_valid: bool


class EvaluationResponse(TypedDict):
    """Evaluation response."""

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

        self.tools = self._tools()

        self.llm = self._build_model()
        self.workflow = self._build_workflow()
        self.guidelines = self._load_guidelines()
        self.vector_store_history = self._vector_store_history()

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

    def refine_prompt(self, state: State) -> State:
        """Improves the prompt for the agent."""
        print("Improving prompt...")
        print(f"User query: {state['messages'][-1].content}")
        user_message = (
            state["messages"][-1].content
            if isinstance(state["messages"][-1], HumanMessage)
            else ""
        )
        prompt = f"""
        You are an AI assistant. Your task is to improve the user's query {user_message}
        to make it more specific and clear. Provide a revised version of the query.
        """
        improved_prompt = self.llm.invoke([SystemMessage(content=prompt)])
        print(f"Improved prompt: {improved_prompt.content}")
        state["messages"].append(HumanMessage(content=improved_prompt.content))
        return state

    def create_agent(self) -> CompiledGraph:
        """Creates a new agent."""
        print("Creating agent...")
        return create_react_agent(
            self.llm,
            self.tools,
        )

    def draw_graph(self) -> None:
        try:
            # Get the Mermaid diagram source
            mermaid_src = self.workflow.get_graph().draw_mermaid()

            # Optional: Print or inspect the Mermaid source
            print("Mermaid diagram source:\n", mermaid_src)

            # Save it to PNG using the Mermaid source
            png_data = self.workflow.get_graph(xray=True).draw_mermaid_png()

            with open("workflow_graph.png", "wb") as f:
                f.write(png_data)
            print("Graph saved as workflow_graph.png")

        except Exception as e:
            print(f"Failed to generate graph: {e}")

    def chatbot_handler(self, user_query: str) -> Tuple[str, int]:
        """Handles the chatbot interaction."""
        system_message = SystemMessage(
            content="""
        You are an AI assistant called Vita Samara. Your task is to assist the user with their queries. Answer the
        questions where appropriate and provide relevant information. Be precise and concise.
        """
        )
        print("processing user query...")
        print(f"User query: {user_query}")

        messages = [system_message, HumanMessage(content=user_query)]

        state = {
            "messages": messages,
            "metadata": {"is_valid": True},
            "telegram_id": self.telegram_id,
            "summary": self.summary,
        }

        result = self.workflow.invoke(state)
        print(f"result: {result['messages']}")

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
        workflow.add_node("refine_prompt", self.refine_prompt)
        workflow.add_node("agent", agent)

        workflow.add_edge(START, "validate_query")

        workflow.add_conditional_edges(
            "validate_query",
            lambda state: "refine_prompt"
            if state["metadata"].get("is_valid", True)
            else END,
        )
        workflow.add_edge("refine_prompt", "agent")
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

    def search_chat_history(self, user_query: str) -> list[tuple[Document, float]]:
        """Search the chat history for relevant messages."""
        print("Searching chat history...")
        results = self.vector_store_history.similarity_search_with_score(
            query=user_query,
            k=5,
        )
        return results

    @staticmethod
    def _vector_store_history() -> PGVector:
        """creates a vector store for the chat history"""
        load_dotenv()
        connection_string = os.getenv("DATABASE_URL")
        return PGVector(
            connection=connection_string,
            collection_name="chat_history",
            use_jsonb=True,
            embeddings=EmbeddingFunctionWrapper("all-MiniLM-L6-v2"),
        )

    def _tools(self):
        """Create tools for the agent."""
        return [
            self.history_retriever_tool(),
            get_weather,
            get_wiki_summary,
            get_tavily_search_tool,
            calendar_events_handler,
        ]

    def history_retriever_tool(self):
        """Create a retriever tool for the agent."""
        print("Creating history retriever tool...")

        history_retriever = self._vector_store_history().as_retriever()
        return create_retriever_tool(
            retriever=history_retriever,
            name="retrieve_chat_history",
            description="""Check the chat history for similar messages top the user prompt and return the most relevant
                        "ones.""",
        )

    def print_stream(self, stream):
        for s in stream:
            message = s["messages"][-1]
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()


class EmbeddingFunctionWrapper(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> npt.NDArray[np.float32]:
        # Use the encode method to generate embeddings
        return self.model.encode(texts, convert_to_tensor=False)

    def embed_query(self, text: str) -> Tensor:
        # For single query embedding
        return self.model.encode(text, convert_to_tensor=False)


if __name__ == "__main__":
    # Example usage
    inputs = {"messages": [("user", "who built you?")]}

    handler = LangGraphHandler(telegram_id=123456789)
    result = handler.chatbot_handler("What is the weather like in New York?")
