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

from langchain_core.runnables.graph_mermaid import MermaidDrawMethod

from langgraph.prebuilt import create_react_agent
from langmem import create_manage_memory_tool, create_search_memory_tool

from dotenv import load_dotenv

from src.services.tools import (
    get_weather,
    get_wiki_summary,
    get_tavily_search_tool,
    calendar_events_handler,
)
from src.crud.db_handler import DataBaseHandler

from pathlib import Path

import json
import uuid

from typing import Annotated
from typing_extensions import TypedDict

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

    def chatbot_handler(self, user_query: str) -> str:
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
        messages = chat_history.messages + [
            system_message,
            HumanMessage(content=user_query),
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

        return (
            ai_responses[-1]
            if ai_responses
            else "Sorry, I couldn't generate a response."
        )

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
        workflow.add_node("summarize", self._update_summary)
        workflow.add_node("save_summary", self._save_summary_to_db)
        workflow.add_node("evaluate_response", self.evaluate_response)

        workflow.add_edge(START, "validate_query")
        workflow.add_conditional_edges(
            "validate_query",
            lambda state: "agent" if state["metadata"].get("is_valid", True) else END,
        )
        workflow.add_edge("agent", "summarize")
        workflow.add_edge("summarize", "evaluate_response")
        workflow.add_edge("evaluate_response", "save_summary")
        workflow.add_edge("save_summary", END)

        return workflow.compile()

    def _update_summary(self, state: State) -> State:
        """Update the conversation summary."""
        print("Updating Summary:")
        try:
            if len(state["messages"]) >= 2:  # Ensure sufficient history
                # Get the most recent user and assistant messages
                recent_messages = state["messages"][-2:]

                # Format the messages for the summary prompt
                user_msg = ""
                ai_msg = ""
                for msg in recent_messages:
                    if isinstance(msg, HumanMessage):
                        user_msg = msg.content
                    elif isinstance(msg, AIMessage):
                        ai_msg = msg.content

                # Create a summary prompt for the LLM
                summary_prompt = f"""
                Previous summary: {state["summary"]}

                New exchange:
                User: {user_msg}
                Assistant: {ai_msg}

                Provide a concise summary of the entire conversation so far.
                """

                # Generate new summary
                summary_response = self.llm.invoke(
                    [HumanMessage(content=summary_prompt)]
                )

                # Return the updated state with the new summary
                return {
                    "messages": state["messages"],
                    "summary": summary_response.content,
                    "metadata": state["metadata"],
                    "telegram_id": self.telegram_id,
                }
            return state
        except Exception as e:
            print(f"Error updating summary: {e}")
            return state

    def _save_summary_to_db(self, state: State) -> None:
        """Save the summary to the database."""
        print("Saving chat summary to the database")
        telegram_id = state["telegram_id"]
        summary = state["summary"]
        with DataBaseHandler() as db_handler:
            db_handler.write_chat_summary_to_db(
                telegram_id=telegram_id,
                summary=summary,
            )

    def _get_chat_history(self) -> InMemoryChatMessageHistory:
        """Get chat history from the database."""
        chat_history = chats_by_session_id.get(self.telegram_id)
        if chat_history is None:
            chat_history = InMemoryChatMessageHistory()
            chats_by_session_id[self.telegram_id] = chat_history
        return chat_history

    def evaluate_response(self, state: State) -> State:
        "evaluate response against query"
        print("Evaluating response")

        user_query = state["messages"][-2].content
        ai_response = state["messages"][-1].content

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

        print(f"Response accuracy: {evaluation} %")

        return state


if __name__ == "__main__":
    lgh = LangGraphHandler(7594929889)

    lgh.main("What does my name mean  ?")
