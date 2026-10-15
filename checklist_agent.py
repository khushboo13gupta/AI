## imports
from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console
import gradio as gr
import json
import re

# load environment variables
load_dotenv(override=True)

# create console
console = Console()

# create OpenAI client
openai = OpenAI()

checklist = []
completed = []
step_notes: list[str] = []

MODEL_NAME = "gpt-5.5"


def show(text: str) -> None:
    try:
        console.print(text)
    except Exception as e:
        console.print(f"Error showing text: {e}")


def reset_checklist() -> None:
    checklist.clear()
    completed.clear()
    step_notes.clear()


def get_checklist_report() -> str:
    result = ""
    for index, item in enumerate(checklist):
        if completed[index]:
            result += f"Checklist #{index + 1}: [green][strike]{item}[/strike][/green]\n"
        else:
            result += f"Checklist #{index + 1}: {item}\n"
    show(result)
    return result


def create_checklist(descriptions: list[str]) -> str:
    checklist.extend(descriptions)
    completed.extend([False] * len(descriptions))
    return f"Checklist created with {len(descriptions)} items"


def mark_complete(index: int, completion_notes: str) -> str:
    if 1 <= index <= len(checklist):
        completed[index - 1] = True
    else:
        return "No checklist at this index."
    show(completion_notes)
    step_notes.append(f"**Step {index}:** {rich_to_markdown(completion_notes)}")
    return get_checklist_report()


create_checklist_json = {
    "name": "create_checklist",
    "description": "Add new checklist from a list of descriptions and return the full list",
    "parameters": {
        "type": "object",
        "properties": {
            "descriptions": {
                "type": "array",
                "items": {"type": "string"},
                "title": "Descriptions of checklist items",
            }
        },
        "required": ["descriptions"],
        "additionalProperties": False,
    },
}

mark_complete_json = {
    "name": "mark_complete",
    "description": "Mark complete the checklist item at the given position (starting from 1) and return the full list",
    "parameters": {
        "properties": {
            "index": {
                "description": "The 1-based index of the checklist item to mark as complete",
                "title": "Index",
                "type": "integer",
            },
            "completion_notes": {
                "description": "Notes about how you completed the checklist item in rich console markup",
                "title": "Completion Notes",
                "type": "string",
            },
        },
        "required": ["index", "completion_notes"],
        "type": "object",
        "additionalProperties": False,
    },
}

tools = [
    {"type": "function", "function": create_checklist_json},
    {"type": "function", "function": mark_complete_json},
]


def handle_tool_calls(tool_calls):
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        tool = globals().get(tool_name)
        result = tool(**arguments) if tool else {}
        results.append(
            {"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id}
        )
    return results


system_message = """
You are given a problem to solve, by using your checklist tools to plan a list of steps, then carrying out each step in turn.
Now create a plan, set the checklist, carry out the steps, and reply with the solution.
If any quantity isn't provided in the question, then include a step to come up with a reasonable estimate.
Provide your solution in Rich console markup without code blocks.
Do not ask the user questions or clarification; respond only with the answer after using your tools.
"""
system = [{"role": "system", "content": system_message}]


def rich_to_markdown(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\[strike\](.*?)\[/strike\]", r"~~\1~~", text, flags=re.DOTALL)
    text = re.sub(r"\[/?[a-zA-Z0-9_#= -]+\]", "", text)
    return text.strip()


def checklist_markdown() -> str:
    if not checklist:
        return "_Waiting for the agent to create a plan..._"
    done = sum(completed)
    lines = [f"**Progress:** {done} / {len(checklist)} steps complete", ""]
    for index, item in enumerate(checklist):
        if completed[index]:
            lines.append(f"- [x] ~~{item}~~")
        else:
            lines.append(f"- [ ] {item}")
    return "\n".join(lines)


def notes_markdown() -> str:
    if not step_notes:
        return "_No completed steps yet._"
    return "\n\n".join(step_notes)


def snapshot(status: str, solution: str = "") -> tuple[str, str, str, str]:
    return status, checklist_markdown(), notes_markdown(), solution


def run_agent(problem: str):
    reset_checklist()
    problem = (problem or "").strip()
    if not problem:
        yield snapshot("Enter a problem to solve.", "")
        return

    messages = system + [{"role": "user", "content": problem}]
    yield snapshot("Planning a checklist...", "")

    response = openai.chat.completions.create(
        model=MODEL_NAME, messages=messages, tools=tools
    )
    while response.choices[0].finish_reason == "tool_calls":
        assistant_message = response.choices[0].message
        results = handle_tool_calls(assistant_message.tool_calls)
        messages.append(assistant_message)
        messages.extend(results)
        yield snapshot("Working through the checklist...", "")
        response = openai.chat.completions.create(
            model=MODEL_NAME, messages=messages, tools=tools
        )

    solution = rich_to_markdown(response.choices[0].message.content or "")
    show(response.choices[0].message.content)
    yield snapshot("Done.", solution)


DEFAULT_PROBLEM = """who is the most influential person in the world?"""


if __name__ == "__main__":
    with gr.Blocks(title="Checklist Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Checklist Agent\n"
            "Submit a problem. The agent plans steps, works through them, and this board shows progress."
        )
        problem = gr.Textbox(
            label="Problem",
            value=DEFAULT_PROBLEM,
            lines=5,
        )
        run_btn = gr.Button("Run agent", variant="primary")
        status = gr.Markdown()
        with gr.Row():
            checklist_view = gr.Markdown(label="Checklist")
            notes_view = gr.Markdown(label="Step notes")
        solution = gr.Markdown(label="Solution")
        run_btn.click(
            fn=run_agent,
            inputs=[problem],
            outputs=[status, checklist_view, notes_view, solution],
        )

    demo.queue().launch()
