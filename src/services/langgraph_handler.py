from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledGraph
from langchain.agents import Tool
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
from src.services.schemas import State, IsValid, EvaluationResponse

from sentence_transformers import SentenceTransformer

from pathlib import Path

import json
import os

from typing_extensions import Tuple, List

import numpy.typing as npt
import numpy as np

chats_by_session_id = {}
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
        self.vector_store_support_docs = self._vector_store_support_docs()

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

    def chatbot_handler(self, user_query: str, timestamp: int) -> Tuple[str, int]:
        """Handles the chatbot interaction."""
        system_message = SystemMessage(
            content="""
        You are an AI assistant called Vita Samara. Your task is to assist the user with their queries. Answer the
        questions where appropriate and provide relevant information. Be precise and concise. You have access to the
        chat history and the support documentation via tools in the agent. You can also use the additiona tools provided
        to assist the user.
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
        with DataBaseHandler() as db_handler:
            db_handler.save_message(
                user_query=user_query,
                ai_response=ai_response,
                evaluation=int(evaluation),
                telegram_id=self.telegram_id,
                timestamp=timestamp,
            )
        self.summary = self._update_summary(user_query, ai_response)
        self._save_summary_to_db(self.summary)
        self.add_to_vector_store_history(user_query, ai_response)

        return ai_response, int(evaluation)

    def main(self, user_query: str):
        response = self.chatbot_handler(user_query, timestamp=1699999999)
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
        Provide a concise summary of the entire conversation so far. Note Places, dates, names, and other entities
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

    def search_chat_history(self, user_query: str) -> list[tuple[Document, float]]:
        """Search the chat history for a specific user based on the query."""
        print("Searching chat history...")
        metadata_filter = {"telegram_id": self.telegram_id}
        results = self.vector_store_history.similarity_search_with_score(
            query=user_query,
            filter=metadata_filter,
            k=5,
        )
        if not results:
            print("No results found in chat history")
        else:
            print("Results found in chat history")
        return results

    def search_support_docs(self, user_query: str) -> list[tuple[Document, float]]:
        """Search the support documentation for mental health issues"""
        print("Searching support documentation...")
        results = self.vector_store_history.similarity_search_with_score(
            query=user_query,
            k=5,
        )
        if not results:
            print("No results found in support documentation")
        else:
            print("Results found in support documentation.")
        return results

    def refine_prompt(self, user_query) -> str:
        """Improves the prompt for the agent for more specific and clear queries."""
        print("Improving prompt...")
        print(f"User query: {user_query}")

        prompt = f"""
        You are an AI assistant. Your task is to improve the user's query {user_query}
        to make it more specific and clear. Provide a revised version of the query.
        """
        improved_prompt = self.llm.invoke([SystemMessage(content=prompt)]).content
        print(f"Improved prompt: {improved_prompt}")
        return improved_prompt

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

    @staticmethod
    def _vector_store_support_docs() -> PGVector:
        """creates a vector store mental_health_docs"""
        load_dotenv()
        connection_string = os.getenv("DATABASE_URL")
        return PGVector(
            connection=connection_string,
            collection_name="mental_health_docs",
            use_jsonb=True,
            embeddings=EmbeddingFunctionWrapper("all-MiniLM-L6-v2"),
        )

    def create_text_embedding(self, text: str) -> list[float]:
        """Creates an embedding for the user query."""
        print("Creating text embedding...")
        embedding = self.vector_store_history.embeddings.embed_query(text)
        return embedding

    def _tools(self):
        """Create tools for the agent."""
        return [
            Tool.from_function(
                func=self.refine_prompt,
                name="refine_prompt",
                description="Improves vague or unclear user queries before sending them to the main agent",
            ),
            self.history_retriever_tool(),
            self.support_docs_retriever_tool(),
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

    def support_docs_retriever_tool(self):
        """Create a retriever tool for the agent."""
        print("Creating support doc retriever tool...")
        support_doc_retriever = self._vector_store_support_docs().as_retriever()
        return create_retriever_tool(
            retriever=support_doc_retriever,
            name="retrieve_support_docs",
            description="""Check the mental health support documentation for similar messages top the user prompt and
            return the most relevant ones.""",
        )

    def add_to_vector_store_history(self, user_query: str, ai_response: str) -> None:
        """Embeds and stores the user query and AI response into the vector store."""
        print("Adding user query and AI response to vector store history...")

        combined_text = f"User: {user_query}\nAI: {ai_response}"
        document = Document(
            page_content=combined_text,
            metadata={
                "telegram_id": self.telegram_id,
            },
        )
        print(f"Document: {document}")
        self.vector_store_history.add_documents(
            documents=[document],
        )

    def add_to_vector_store_support_docs(self, split_docs: List[Document]) -> None:
        """Embeds and stores the user query and AI response into the vector store."""
        print("Adding user query and AI response to vector store support docs...")

        self.vector_store_support_docs.add_documents(
            documents=split_docs,
        )


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
    lbh = LangGraphHandler(9999999999)
    timestamp = 1699999999
    lbh.main("Using the chat history - do you know what my name is?")
