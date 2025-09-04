# src/ai_agents_01_call_llm/main.py
import os
import json
import argparse
import ast
import operator as op
import time
from typing import Any, List, cast
import math


from dotenv import load_dotenv
from openai import OpenAI

# Types to satisfy Pylance for messages/tools
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
    ChatCompletionMessageToolCallParam,
)

# ---------------------- Pretty printing helpers ----------------------
ANSI = {
    "reset": "\x1b[0m",
    "bold": "\x1b[1m",
    "dim": "\x1b[2m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "blue": "\x1b[34m",
    "magenta": "\x1b[35m",
    "red": "\x1b[31m",
    "gray": "\x1b[90m",
}

def color(text: str, name: str) -> str:
    if os.getenv("NO_COLOR"):
        return text
    return f"{ANSI.get(name, '')}{text}{ANSI['reset']}"

def banner(title: str) -> None:
    print(color(f"\n=== {title} ===", "blue"))

def kv(label: str, value: str) -> None:
    print(color(f"{label}: ", "dim") + f"{value}")

def pretty_json(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)

# ---------------------- Safe arithmetic evaluator ----------------------
_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,        
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}

ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
}


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    # --- allow specific safe functions like sqrt(x) ---
    if isinstance(node, ast.Call):
        # function must be a bare name like "sqrt", no attributes, no keywords
        if isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS:
            if node.keywords:
                raise ValueError("Keywords not allowed in function calls.")
            if len(node.args) != 1:
                raise ValueError(f"{node.func.id} expects exactly 1 argument.")
            arg_val = _eval_node(node.args[0])  # recurse to keep safety
            return float(ALLOWED_FUNCS[node.func.id](arg_val))
        raise ValueError("Only sqrt(x) is allowed as a function call.")

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPS:
            raise ValueError(f"Operator {op_type.__name__} not allowed.")
        return _ALLOWED_OPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPS:
            raise ValueError(f"Unary operator {op_type.__name__} not allowed.")
        return _ALLOWED_OPS[op_type](operand)

    if isinstance(node, ast.Expr):
        return _eval_node(node.value)

    # Reject everything else (names, attributes, subscripts, etc.)
    if isinstance(
        node,
        (
            ast.Name, ast.Attribute, ast.Subscript,
            ast.List, ast.Dict, ast.Set, ast.Tuple,
            ast.Compare, ast.BoolOp, ast.IfExp, ast.Lambda,
        ),
    ):
        raise ValueError("Only pure numeric arithmetic (and sqrt) is allowed.")

    raise ValueError(f"Unsupported expression: {ast.dump(node)}")

def safe_eval(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")
    return float(_eval_node(tree.body))

# ---------------------- Tool (function-calling) wiring ----------------------
def evaluate_expression(expression: str) -> dict:
    """Safely evaluate a numeric arithmetic expression and return a JSON-able dict."""
    try:
        value = safe_eval(expression)
        return {"ok": True, "expression": expression, "result": value}
    except Exception as e:
        return {"ok": False, "expression": expression, "error": str(e)}

TOOLS: List[ChatCompletionToolParam] = cast(List[ChatCompletionToolParam], [
    {
        "type": "function",
        "function": {
            "name": "evaluate_expression",
            "description": "Safely evaluate arithmetic (ints/floats; + - * / // % **, parentheses) and sqrt(x).",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression, e.g. (2 + 3*4) / 5",
                    }
                },
                "required": ["expression"],
            },
        },
    }
])

AVAILABLE_FUNCTIONS = {
    "evaluate_expression": evaluate_expression,
}

SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant. "
    "If the user asks for any calculation or numeric result, "
    "use the function tool `evaluate_expression` instead of estimating. "
    "Explain briefly and clearly."
)

def _content_to_text(content) -> str:
    """Normalize ChatCompletionMessage.content to plain text (SDK >=1.x)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    out = []
    for part in content:
        txt = getattr(part, "text", None)
        if txt:
            out.append(txt)
    return "".join(out)

def run_with_tools(client: OpenAI, model: str, user_prompt: str, trace: bool = False) -> str:
    messages: List[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_INSTRUCTIONS}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_prompt}],
        },
    ]

    if trace:
        banner("INPUT")
        kv("User", user_prompt)
        kv("Model", model)

    # --- First call (may request tools) ---
    t0 = time.perf_counter()
    first = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.2,
    )
    t_first = (time.perf_counter() - t0) * 1000

    msg = first.choices[0].message
    if trace:
        banner("FIRST RESPONSE")
        kv("Latency", f"{t_first:.0f} ms")
        print(color(pretty_json(msg.model_dump()), "gray"))

    # If the assistant asked to call tools, execute them
    if msg.tool_calls:
        # Build assistant.tool_calls param (branch by type; avoid importing missing symbols)
        tool_calls_param: List[ChatCompletionMessageToolCallParam] = []
        for tc in msg.tool_calls:
            if getattr(tc, "type", None) == "function" and getattr(tc, "function", None):
                fn = cast(Any, tc).function
                tool_calls_param.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": fn.name,
                        "arguments": fn.arguments,
                    },
                })
            else:
                if trace:
                    banner("TOOL (unsupported)")
                    kv("Type", str(getattr(tc, "type", None)))

        # Add the assistant message that requested tools
        assistant_with_calls: ChatCompletionMessageParam = cast(ChatCompletionMessageParam, {
            "role": "assistant",
            "content": None,  # valid when tool_calls are present
            "tool_calls": tool_calls_param,
        })
        messages.append(assistant_with_calls)

        # Execute tools and append tool results
        for tc in msg.tool_calls:
            if getattr(tc, "type", None) != "function" or not getattr(tc, "function", None):
                continue  # ignore non-function calls

            fn = cast(Any, tc).function
            fn_name = fn.name
            try:
                fn_args = json.loads(fn.arguments or "{}")
            except json.JSONDecodeError:
                fn_args = {"_raw": fn.arguments}

            py_fn = AVAILABLE_FUNCTIONS.get(fn_name)

            if trace:
                banner("TOOL")
                kv("Name", fn_name)
                kv("Args", pretty_json(fn_args))

            if py_fn is None:
                tool_payload = {"ok": False, "error": f"Unknown function: {fn_name}"}
            else:
                t_tool = time.perf_counter()
                tool_payload = py_fn(**fn_args)
                tool_latency = (time.perf_counter() - t_tool) * 1000
                if trace:
                    kv("Result", color(pretty_json(tool_payload), "green" if tool_payload.get("ok") else "red"))
                    kv("Latency", f"{tool_latency:.0f} ms")

            # Tool role expects STRING content
            tool_msg: ChatCompletionMessageParam = cast(ChatCompletionMessageParam, {
                "role": "tool",
                "tool_call_id": tc.id,
                "name": fn_name,
                "content": json.dumps(tool_payload),
            })
            messages.append(tool_msg)

        # --- Second call: ask the model to compose final answer using tool outputs ---
        t1 = time.perf_counter()
        second = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="none",  # now we want the final answer only
            temperature=0.3,
        )
        t_second = (time.perf_counter() - t1) * 1000
        final_msg = second.choices[0].message

        if trace:
            banner("SECOND RESPONSE (FINAL)")
            kv("Latency", f"{t_second:.0f} ms")

        return _content_to_text(final_msg.content) or "(no content)"
    else:
        # No tool needed; just return the assistant content
        return _content_to_text(msg.content) or "(no content)"

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment (.env).")

    parser = argparse.ArgumentParser(description="LLM + tool-calling demo")
    parser.add_argument("query", nargs="*", help="User prompt. If omitted, a demo prompt is used.")
    parser.add_argument("--trace", action="store_true", help="Show detailed agent trace.")
    args = parser.parse_args()

    user_prompt = " ".join(args.query) or "What is (2 + 3*4) / 5 ? Explain briefly."
    client = OpenAI(api_key=api_key)

    answer = run_with_tools(client, model, user_prompt, trace=args.trace)

    banner("FINAL ANSWER")
    print(answer)

if __name__ == "__main__":
    main()
