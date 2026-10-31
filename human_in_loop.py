import os
from typing import Literal

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

load_dotenv()

Technical_database = [
    {
        "project": "ProjectX",
        "language": "Python",
        "code_snippet": "this is a sample Python code snippet for ProjectX."
    },
    {
        "project": "ProjectY",
        "language": "JavaScript",
        "code_snippet": "this is a sample JavaScript code snippet for ProjectY."
    },
    {
        "project": "ProjectZ",
        "language": "Java",
        "code_snippet": "this is a sample Java code snippet for ProjectZ."
    },
    {
        "project": "ProjectA",
        "language": "C++",
        "code_snippet": "this is a sample C++ code snippet for ProjectA."
    }
]


class AgentState(BaseModel):
    """One schema for every field that can travel through the graph."""
    message: str
    topic: Literal["Invoice", "Technical", "Company Policies", "Other"] | None = None
    response: str | None = None
    intent: Literal["update_total_amount", "update_due_date", "update_status", "get_invoice_details", "get_invoice_pdf", "code_review", "bug_fix", "feature_request", "language_update"] | None = None
    invoice_id: str | None = None
    invoice_date: str | None = None
    due_date: str | None = None
    total_amount: float | None = None
    status: str | None = None
    pdf_path: str | None = None
    project: str | None = None
    language: Literal["Python", "JavaScript", "Java", "C++"] | None = None

class RouteDecision(BaseModel):
    topic: Literal["Invoice", "Technical", "Company Policies", "Other"]


class InvoiceRequest(BaseModel):
    intent: Literal["update_total_amount", "update_due_date", "update_status", "get_invoice_details", "get_invoice_pdf"]
    invoice_id: str | None = Field(default=None, description="Invoice ID explicitly present in the message")
    invoice_date: str | None = Field(default=None, description="Invoice date explicitly present, otherwise null")
    due_date: str | None = Field(default=None, description="Due date explicitly present, otherwise null")
    total_amount: float | None = Field(default=None, description="Amount explicitly present, otherwise null")
    status: str | None = Field(default=None, description="Status explicitly present, otherwise null")


class TechnicalRequest(BaseModel):
    intent: Literal["code_review", "bug_fix", "feature_request", "language_update"]
    project: str | None = Field(default=None, description="Project name explicitly present in the message")
    language: Literal["Python", "JavaScript", "Java", "C++"] | None = Field(default=None, description="Programming language explicitly present in the message")


class ApprovalDecision(BaseModel):
    approved: bool
    note: str | None = None

LLM = ChatOllama(
    model="qwen3:4b",
    temperature=0,
    reasoning=False,
    num_ctx=4096,
    num_predict=256,
    keep_alive="30m",
)
ROUTER = LLM.with_structured_output(RouteDecision)
INVOICE_EXTRACTOR = LLM.with_structured_output(InvoiceRequest)
TECHNICAL_EXTRACTOR = LLM.with_structured_output(TechnicalRequest)

def agent_node(state: AgentState) -> dict:
    route = ROUTER.invoke(
        "Classify this request into exactly one topic: Invoice, Technical, "
        f"Company Policies, or Other. Request: {state.message!r}"
    )

    update: dict = {"topic": route.topic}
    if route.topic == "Invoice":
        request = INVOICE_EXTRACTOR.invoke(
            "Extract the invoice intent and only values explicitly stated by the user. "
            "Use null for every value that is not stated. "
            f"Request: {state.message!r}"
        )
        update.update(request.model_dump())

        # Read requests do not supply update values. Clearing them prevents a small
        # model from inventing irrelevant values that downstream code might trust.
        if request.intent in {"get_invoice_details", "get_invoice_pdf"}:
            update.update(
                invoice_date=None,
                due_date=None,
                total_amount=None,
                status=None,
            )
    elif route.topic == "Technical":
        request = TECHNICAL_EXTRACTOR.invoke(
            "Extract the technical request details and only values explicitly stated by the user. "
            "Use null for every value that is not stated. "
            f"Request: {state.message!r}"
        )
        update.update(request.model_dump())
    return update


def _require(value: object, field_name: str) -> None:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"{field_name} is required for this invoice operation")


def invoice_tool(state: AgentState) -> dict:
    _require(state.intent, "intent")
    _require(state.invoice_id, "invoice_id")

    update_values = {
        "update_total_amount": ("total_amount", state.total_amount),
        "update_due_date": ("due_date", state.due_date),
        "update_status": ("status", state.status),
    }

    # Approval is required for mutations. Code before interrupt() may run again on
    # resume, so keep side effects after the interrupt.
    if state.intent in update_values:
        column, new_value = update_values[state.intent]
        _require(new_value, column)
        decision = interrupt(
            {
                "question": (
                    f"Update {column} to {new_value!r} for invoice "
                    f"{state.invoice_id}?"
                ),
                "type": "approval",
                "allowed_responses": ["yes", "no"],
            },
        )
        if not decision.get("approved", False):
            return {
                "response": f"Update for invoice {state.invoice_id} cancelled."
            }

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("Set DATABASE_URL before executing an invoice database action")

    engine = create_engine(database_url, pool_pre_ping=True)
    with engine.begin() as conn:
        if state.intent in update_values:
            column, new_value = update_values[state.intent]
            # The column comes only from the fixed mapping above. User values are bound.
            result = conn.execute(
                text(f"UPDATE public.invoices SET {column} = :value WHERE invoice_id = :id"),
                {"value": new_value, "id": state.invoice_id},
            )
            if result.rowcount == 0:
                return {"response": f"Invoice {state.invoice_id} not found."}
            return {
                "response": (
                    f"Invoice {state.invoice_id} {column} updated to {new_value}."
                )
            }

        if state.intent == "get_invoice_details":
            row = conn.execute(
                text(
                    "SELECT invoice_id, invoice_date, due_date, total_amount, status "
                    "FROM invoices WHERE invoice_id = :id"
                ),
                {"id": state.invoice_id},
            ).mappings().first()
            if row is None:
                return {"response": f"Invoice {state.invoice_id} not found."}
            return {
                "invoice_id": row["invoice_id"],
                "invoice_date": str(row["invoice_date"]),
                "due_date": str(row["due_date"]),
                "total_amount": float(row["total_amount"]),
                "status": row["status"],
                "response": f"Invoice {state.invoice_id} loaded.",
            }

        if state.intent == "get_invoice_pdf":
            row = conn.execute(
                text("SELECT pdf_path FROM invoices WHERE invoice_id = :id"),
                {"id": state.invoice_id},
            ).mappings().first()
            if row is None:
                return {"response": f"Invoice {state.invoice_id} not found."}
            return {
                "pdf_path": row["pdf_path"],
                "response": f"PDF path for invoice {state.invoice_id} loaded.",
            }

    raise ValueError(f"Unsupported invoice intent: {state.intent}")


def technical_tool(state: AgentState) -> dict:
    _require(state.project, "project")
    _require(state.language, "language")

    if state.intent == "language_update":
        for entry in Technical_database:
            if entry["project"] == state.project:
                if entry["language"] != state.language:
                    decision = interrupt(
                        {
                            "question": (
                                f"Update language for project {state.project} "
                                f"from {entry['language']} to {state.language}?"
                            ),
                            "type": "approval",
                            "allowed_responses": ["yes", "no"],
                        },
                    )
                    if not decision.get("approved", False):
                        return {
                            "response": f"Language update for project {state.project} cancelled."
                        }
                    entry["language"] = state.language
                    return {
                        "response": f"Language for project {state.project} updated to {state.language}."
                    }
                else:
                    return {
                        "response": f"Project {state.project} already has language {state.language}."
                    }   

    elif state.intent in {"code_review", "bug_fix", "feature_request"}:
        for entry in Technical_database:
            if entry["project"] == state.project and entry["language"] == state.language:
                if state.code_snippet:
                    decision = interrupt(
                        {
                            "question": (f"Process {state.intent} for project {state.project} with the provided code snippet?"),
                            "type": "approval",
                            "allowed_responses": ["yes", "no"],
                        }
                    )
                    if not decision.get("approved", False):
                        return {
                            "response": f"{state.intent} for project {state.project} cancelled."
                        }
                    return {
                        "response": f"Processed {state.intent} for project {state.project} with provided code snippet."
                    }
                else:
                    return {
                        "response": f"No code snippet provided for {state.intent} in project {state.project}."
                    }
    return {"response": "Route this request to your technical knowledge/tool node."}


def company_policies_tool(state: AgentState) -> dict:
    return {"response": "Route this request to your policy retrieval/RAG node."}


def route_topic(state: AgentState) -> str:
    return {
        "Invoice": "invoice_tool",
        "Technical": "technical_tool",
        "Company Policies": "company_policies_tool",
        "Other": "end",
    }.get(state.topic, "end")


builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("invoice_tool", invoice_tool)
builder.add_node("technical_tool", technical_tool)
builder.add_node("company_policies_tool", company_policies_tool)
builder.set_entry_point("agent")
builder.add_conditional_edges(
    "agent",
    route_topic,
    {
        "invoice_tool": "invoice_tool",
        "technical_tool": "technical_tool",
        "company_policies_tool": "company_policies_tool",
        "end": END,
    },
)
builder.add_edge("invoice_tool", END)
builder.add_edge("technical_tool", END)
builder.add_edge("company_policies_tool", END)

# A checkpointer is mandatory for interrupt/resume.
app = builder.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    config = {
        "configurable": {"thread_id": "invoice-demo-1"}
    }

    first_result = app.invoke(
        {
            "message": "Update Invoice INV-1002 total_amount to 1500.75 and due_date to 2024-07-15.",
            "response": None,
        },
        config=config,
    )

    interrupts = first_result.get("__interrupt__", [])

    if interrupts:
        interrupt_payload = interrupts[0].value

        print("\nHuman approval required")
        print(interrupt_payload["question"])

        human_answer = input("Enter yes or no: ").strip().lower()
        approved = human_answer in {"yes", "y"}

        final_result = app.invoke(
            Command(
                resume={
                    "approved": approved
                }
            ),
            config=config,  # Must use the same thread ID
        )

        print("\nFinal result:")
        print(final_result)

    else:
        print("\nGraph completed without an interrupt:")
        print(first_result)
