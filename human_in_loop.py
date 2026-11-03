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

database_url = os.getenv("DATABASE_URL")
if not database_url:
    raise RuntimeError("Set DATABASE_URL before executing an invoice database action")

engine = create_engine(database_url, pool_pre_ping=True)

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


Company_Policies_database = [
    {
        "policy_id": "Policy1",
        "title": "Leave Policy",
        "content": "This is the content of the Leave Policy."
    },
    {
        "policy_id": "Policy2",
        "title": "Code of Conduct",
        "content": "This is the content of the Code of Conduct."
    },
    {
        "policy_id": "Policy3",
        "title": "Remote Work Policy",
        "content": "This is the content of the Remote Work Policy."
    }
]

class AgentState(BaseModel):
    """One schema for every field that can travel through the graph."""
    message: str
    topic: Literal["Invoice", "Technical", "Company Policies", "Other"] | None = None
    response: str | None = None
    intent: list[Literal["update_total_amount", "update_due_date", "update_status", "get_invoice_details", "get_invoice_pdf", "code_review", 
                         "bug_fix", "feature_request", "language_update", "get_policy_details", "update_sick_leave", "update_paid_leave"]] | None = None
    invoice_id: str | None = None
    invoice_date: str | None = None
    due_date: str | None = None
    total_amount: float | None = None
    status: str | None = None
    pdf_path: str | None = None
    project: str | None = None
    language: Literal["Python", "JavaScript", "Java", "C++"] | None = None
    code_snippet: str | None = None
    policy_id: str | None = None
    title: str | None = None
    employee_id: int | None = None
    employee_name: str | None = None
    sick_leave: int | None = None
    paid_leave: int | None = None


class RouteDecision(BaseModel):
    topic: Literal["Invoice", "Technical", "Company Policies", "Other"]


class InvoiceRequest(BaseModel):
    intent: list[Literal["update_total_amount", "update_due_date", "update_status", "get_invoice_details", "get_invoice_pdf"]]
    invoice_id: str | None = Field(default=None, description="Invoice ID explicitly present in the message")
    invoice_date: str | None = Field(default=None, description="Invoice date explicitly present, otherwise null")
    due_date: str | None = Field(default=None, description="Due date explicitly present, otherwise null")
    total_amount: float | None = Field(default=None, description="Amount explicitly present, otherwise null")
    status: str | None = Field(default=None, description="Status explicitly present, otherwise null")


class TechnicalRequest(BaseModel):
    intent: list[Literal["code_review", "bug_fix", "feature_request", "language_update"]]
    project: str | None = Field(default=None, description="Project name explicitly present in the message")
    language: Literal["Python", "JavaScript", "Java", "C++"] | None = Field(default=None, description="Programming language explicitly present in the message")
    code_snippet: str | None = Field(default=None, description="Code snippet explicitly present in the message, otherwise null")


class CompanyPolicyRequest(BaseModel):
    intent: list[Literal["get_policy_details", "update_sick_leave", "update_paid_leave"]]
    policy_id: str | None = Field(default=None, description="Policy ID explicitly present in the message")
    title: str | None = Field(default=None, description="Policy title explicitly present in the message, otherwise null")
    employee_id: int | None = Field(default=None, description="Employee ID explicitly present in the message, otherwise null")
    employee_name: str | None = Field(default=None, description="Employee name explicitly present in the message, otherwise null")
    sick_leave: int | None = Field(default=None, description="Sick leave days explicitly present in the message, otherwise null")
    paid_leave: int | None = Field(default=None, description="Paid leave days explicitly present in the message, otherwise null")


class ApprovalDecision(BaseModel):
    approved: bool
    note: str | None = None
    sick_leave_balance: int | None = None
    paid_leave_balance: int | None = None

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
COMPANY_POLICY_EXTRACTOR = LLM.with_structured_output(CompanyPolicyRequest)

def agent_node(state: AgentState) -> dict:
    route = ROUTER.invoke(
        "Classify this request into exactly one topic: Invoice, Technical, "
        f"Company Policies, or Other. Request: {state.message!r}")

    update: dict = {"topic": route.topic}
    if route.topic == "Invoice":
        request = INVOICE_EXTRACTOR.invoke(
            """Extract the invoice intent and only values explicitly stated by the user. 
            if any date is provided which can be in any format like YYYY-MM-DD identify it and return the date into string format keep the date inside "" like "2024-07-15".,
            Use null for every value that is not stated. """
            f"Request: {state.message!r}")
        
        update.update(request.model_dump())
        if any(intent in {"get_invoice_details", "get_invoice_pdf"} for intent in request.intent):
            update.update(
                invoice_date=None,
                due_date=None,
                total_amount=None,
                status=None,)
            
    elif route.topic == "Technical":
        request = TECHNICAL_EXTRACTOR.invoke(
            "Extract the technical request details and only values explicitly stated by the user. "
            "Use null for every value that is not stated. "
            f"Request: {state.message!r}")
        update.update(request.model_dump())

    elif route.topic == "Company Policies":
        request = COMPANY_POLICY_EXTRACTOR.invoke(
            "Extract the intent and company policy request details and only values explicitly stated by the user. "
            "If user is asking to apply means they want to update the leave balance. "
            "also if user want to update sick leave then intent would be update_sick_leave and if user want to update paid leave then intent would be update_paid_leave. "
            "Use null for every value that is not stated. "
            f"Request: {state.message!r}")
        update.update(request.model_dump())

        if request.intent == ["get_policy_details"]:
            update.update(
                employee_id=None,
                employee_name=None,
                sick_leave=None,
                paid_leave=None,)

        elif request.intent in ["update_sick_leave"]:
            update.update(
                title=None,
                employee_name=None,
                paid_leave=None,)

        elif request.intent in ["update_paid_leave"]:
            update.update(
                title=None,
                employee_name=None,
                sick_leave=None,)
    print(update)
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
        "update_status": ("status", state.status),}

    approved_updates = []
    rejected_updates = []

    for intent in state.intent:
        if intent not in update_values:
            continue  # Skip intents that are not update operations

        column, new_value = update_values[intent]
        _require(new_value, column)
        print(f"Processing approval for: {intent}")

        decision = interrupt(
            {
            "question": (
                f"Update {column} to {new_value!r} for invoice "
                f"{state.invoice_id}?"
            ),
            "type": "approval",
            "allowed_responses": ["yes", "no"],
            "approved": [True, False],},
            )
            
        approval = ApprovalDecision.model_validate(decision)
        print(f"{intent}: {approval.approved}")

        if approval.approved:
            approved_updates.append((column, new_value))
        else:
            rejected_updates.append((column, new_value))

    if not approved_updates:
        return {
            "response": (
                f"No changes were approved for invoice {state.invoice_id}.")}

    print("Approved changes:", approved_updates)
    print("Rejected changes:", rejected_updates)

    with engine.begin() as conn:
        invoice = conn.execute(
            text(
                "SELECT invoice_id FROM public.invoices "
                "WHERE invoice_id = :id"),
            {"id": state.invoice_id},).first()

        if invoice is None:
            return {"response": f"Invoice {state.invoice_id} not found."}

        updated_fields = []
        for column, new_value in approved_updates:
            result = conn.execute(
                text(
                    f"UPDATE public.invoices "
                    f"SET {column} = :value "
                    f"WHERE invoice_id = :id"
                ),
                {
                    "value": new_value,
                    "id": state.invoice_id,
                },
            )

            # ADDED: Check that the update affected the expected row.
            if result.rowcount != 1:
                raise RuntimeError(
                    f"Unable to update {column} for invoice {state.invoice_id}"
                )

            updated_fields.append(f"{column} = {new_value}")

    # CHANGED: Return one final summary after all updates are complete.
    return {
        "response": (
            f"Invoice {state.invoice_id} processing completed. "
            f"Updated: {', '.join(updated_fields)}. "
            f"Rejected: "
            f"{', '.join(column for column, _ in rejected_updates) or 'None'}."
        )
    }


def technical_tool(state: AgentState) -> dict:
    _require(state.project, "project")
    _require(state.language, "language")

    if "language_update" in state.intent:
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

    elif any(intent in {"code_review", "bug_fix", "feature_request"} for intent in state.intent):
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
    _require(state.intent, "intent")

    approved_updates = []
    rejected_updates = []

    if "get_policy_details" in state.intent:
        _require(state.policy_id, "policy_id")
        for entry in Company_Policies_database:
            if entry["policy_id"] == state.policy_id:
                return {
                    "response": f"Policy Details for {state.policy_id}: Title: {entry['title']}, Content: {entry['content']}"
                }

    elif any(intent in {"update_sick_leave", "update_paid_leave"} for intent in state.intent):
        _require(state.employee_id, "employee_id")
   
        if "update_sick_leave" in state.intent:
            _require(state.sick_leave, "sick_leave")

            with engine.begin() as conn:
                Total_sick_leave_balance, Total_paid_leave_balance, EMPLOYEE_NAME = conn.execute(text(
                        "SELECT sick_leave, paid_leave, employee_name FROM public.ATTENDANCE WHERE employee_id = :employee_id"),
                    {"employee_id": state.employee_id},).first()

                if Total_sick_leave_balance > state.sick_leave:
                    conn.execute(text(
                        "UPDATE public.ATTENDANCE "
                        "SET sick_leave = sick_leave - :sick_leave "
                        "WHERE employee_id = :employee_id"
                    ),
                    {
                        "sick_leave": state.sick_leave,
                        "employee_id": state.employee_id,
                    },
                    )
                else:
                    decision = interrupt(
                        {
                            "question": (f"The requested sick leave {state.sick_leave} exceeds the current balance of {Total_sick_leave_balance}. "
                                         f"Do you want to proceed with deducting the remaining days from paid leave? "
                                         f"You have {Total_paid_leave_balance} paid leave balance. Do you want to proceed?"),
                            "type": "Sick leave approval",
                            "allowed_responses": ["yes", "no"],
                            "approved": [True, False],
                        }
                    )

                    approval = ApprovalDecision.model_validate(decision)
                    if approval.approved:
                        approved_updates.append(("sick_leave", state.sick_leave))  

                        conn.execute(text(
                            "UPDATE public.ATTENDANCE "
                            "SET sick_leave = 0, paid_leave = paid_leave - :remaining_days "
                            "WHERE employee_id = :employee_id"
                        ),
                        {
                            "remaining_days": state.sick_leave - Total_sick_leave_balance,
                            "employee_id": state.employee_id,
                        },
                        ) 
                            
                    else:
                        rejected_updates.append(("sick_leave", state.sick_leave))
                        
        elif "update_paid_leave" in state.intent:
            _require(state.paid_leave, "paid_leave")

            with engine.begin() as conn:
                Total_paid_leave_balance, EMPLOYEE_NAME = conn.execute(text(
                        "SELECT paid_leave, employee_name FROM public.ATTENDANCE WHERE employee_id = :employee_id"),
                    {"employee_id": state.employee_id},).first()

                decision = interrupt(
                    {
                        "question": (f"Employee {EMPLOYEE_NAME} (ID: {state.employee_id}) has current paid leave balance of "
                        f"{Total_paid_leave_balance}. Do you still want to update it to {state.paid_leave}?"),
                        "type": "approval",
                        "allowed_responses": ["yes", "no"],
                    }
                )
                approval = ApprovalDecision.model_validate(decision)
                if approval.approved:
                    approved_updates.append(("paid_leave", state.paid_leave))

                    conn.execute(text(
                        "UPDATE public.ATTENDANCE "
                        "SET paid_leave = paid_leave - :paid_leave "
                        "WHERE employee_id = :employee_id"
                    ),
                    {
                        "paid_leave": state.paid_leave,
                        "employee_id": state.employee_id,
                    },
                    )
                else:
                    rejected_updates.append(("paid_leave", state.paid_leave))

    with engine.begin() as conn:
        REMAINING_SICK_LEAVE_BALANCE, REMAINING_PAID_LEAVE_BALANCE = conn.execute(text(
            "SELECT sick_leave, paid_leave FROM public.ATTENDANCE WHERE employee_id = :employee_id"),
            {"employee_id": state.employee_id},
        ).first()

    return {
        "response": (
            f"Company policy processing completed. "
            f"Remaining sick leave balance: {REMAINING_SICK_LEAVE_BALANCE}, Remaining paid leave balance: {REMAINING_PAID_LEAVE_BALANCE}. "
            f"Approved updates: {', '.join(f'{col} = {val}' for col, val in approved_updates) or 'None'}. "
            f"Rejected updates: {', '.join(f'{col} = {val}' for col, val in rejected_updates) or 'None'}."
        )
    }
                

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
    config = {"configurable": {"thread_id": "invoice-demo-1"}}
    # CHANGED: Start the graph only once.
    result = app.invoke(
        {"message": ("Apply 6 days of sick leave for employee ID 789"),
         "response": None,},
        config=config,)

    # CHANGED: Keep processing until there are no more interrupts.
    while result.get("__interrupt__"):
        interrupt_payload = result["__interrupt__"][0].value
        print("\nHuman approval required")
        print(interrupt_payload["question"])

        human_answer = input("").strip().lower()

        # CHANGED: Convert the answer into the ApprovalDecision format.
        approved = human_answer in {"yes", "y"}

        # CHANGED: Resume the same graph thread.
        result = app.invoke(Command(
                resume={"approved": approved}
            ), config=config,)

    # CHANGED: Print the final result only after all interrupts finish.
    print("\nFinal result:")
    print(result.get("response", "No response generated."))
