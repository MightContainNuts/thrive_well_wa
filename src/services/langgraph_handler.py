from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledGraph
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_core.runnables.graph_mermaid import MermaidDrawMethod
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langmem import create_manage_memory_tool, create_search_memory_tool
from dotenv import load_dotenv

from src.services.tools import (
    get_weather,
    get_wiki_summary,
    get_tavily_search_tool,
    calendar_events_handler,
)

import json
import uuid

from typing import Annotated
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str
    metadata: dict


class IsValid(TypedDict):
    is_valid: bool


class LangGraphHandler:
    def __init__(self, telegram_id: int = 0):
        load_dotenv()
        try:
            self.telegram_id = str(telegram_id)
        except TypeError:
            self.telegram_id = str(0)
        thread_id = str(uuid.uuid4())
        self.config = {
            "configurable": {
                "user_id": self.telegram_id,
                "thread_id": thread_id,
            }
        }

        self.llm = self._build_model()
        self.graph_builder = self._build_workflow()
        self.workflow = self.graph_builder.compile()
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
        namespace = ("agent_memories", self.telegram_id)
        checkpointer = InMemorySaver()
        return create_react_agent(
            self.llm,
            tools=[
                get_weather,
                get_wiki_summary,
                get_tavily_search_tool,
                calendar_events_handler,
                create_manage_memory_tool(namespace),
                create_search_memory_tool(namespace),
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
        input_messages: State = {
            "messages": [HumanMessage(content=user_query)],
            "metadata": {"is_valid": True},
            "summary": "",
        }
        result = self.workflow.invoke(input_messages)

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
        try:
            guidelines_path = "guidelines.json"
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

    def _build_workflow(self) -> StateGraph:
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

        return workflow


if __name__ == "__main__":
    query = "What is the weather in Edinburgh"
    db_handler = LangGraphHandler()
    # db_handler.draw_graph()
    db_handler.main(query)
