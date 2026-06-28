# 📘 Claude API Course Notes

> Structured notes from a comprehensive Anthropic Claude API course — covering everything from basic API access to advanced tool use, prompt engineering, and evaluation pipelines.

These notes are designed to be a **quick reference for developers and learners** who want to understand how to work with the Claude API effectively. Each section is concise, practical, and backed by real course content.

---

## 📋 Table of Contents

1. [Claude Model Families](#1-claude-model-families)
2. [Accessing the API](#2-accessing-the-api)
3. [Making Requests](#3-making-requests)
4. [Multi-Turn Conversations](#4-multi-turn-conversations)
5. [System Prompts](#5-system-prompts)
6. [Temperature](#6-temperature)
7. [Response Streaming](#7-response-streaming)
8. [Controlling Model Output](#8-controlling-model-output)
9. [Structured Data Generation](#9-structured-data-generation)
10. [Prompt Evaluation](#10-prompt-evaluation)
11. [Eval Workflow](#11-eval-workflow)
12. [Generating Test Datasets](#12-generating-test-datasets)
13. [Running the Eval](#13-running-the-eval)
14. [Model-Based Grading](#14-model-based-grading)
15. [Code-Based Grading](#15-code-based-grading)
16. [Prompt Engineering Techniques](#16-prompt-engineering-techniques)
17. [Introducing Tool Use](#17-introducing-tool-use)
18. [Project Overview — Reminders](#18-project-overview--reminders)
19. [Tool Functions](#19-tool-functions)
20. [Tool Schemas](#20-tool-schemas)
21. [Handling Message Blocks](#21-handling-message-blocks)
22. [Sending Tool Results](#22-sending-tool-results)
23. [Multi-Turn Conversations with Tools](#23-multi-turn-conversations-with-tools)
24. [Implementing Multiple Turns](#24-implementing-multiple-turns)
25. [Next Steps](#-next-steps)

---

## 1. Claude Model Families

Claude offers three model families, each optimised for different priorities:

| Model | Priority | Best For |
|-------|----------|----------|
| **Opus** | Highest intelligence | Complex, multi-step tasks requiring deep reasoning |
| **Sonnet** | Balanced (speed + intelligence) | Most practical use cases, strong coding abilities |
| **Haiku** | Speed & cost efficiency | Real-time interactions, high-volume processing |

**Selection framework:**
- Intelligence required → **Opus**
- Speed required → **Haiku**
- Balanced needs → **Sonnet**

> 💡 A common pattern is to use **multiple models in the same application** — routing different tasks to the best-fit model.

---

## 2. Accessing the API

API requests follow a **5-step flow**:

1. **Client → Server** — User input is sent to your server (never call the Anthropic API directly from the client — this exposes your API key)
2. **Server → Anthropic API** — Server sends the request using the SDK with: `API key`, `model name`, `messages list`, `max_tokens`
3. **Text Generation** — Four internal stages:
   - **Tokenization** → input broken into tokens (words/parts/symbols)
   - **Embedding** → tokens converted to numerical vectors
   - **Contextualization** → embeddings adjusted based on neighbouring tokens
   - **Generation** → model outputs next-token probabilities and selects iteratively
4. **Stop Condition** — Generation stops when `max_tokens` is reached or an `end_of_sequence` token is produced
5. **Response** — API returns generated text + usage counts + `stop_reason`

---

## 3. Making Requests

**Setup steps:**

```bash
pip install anthropic python-dotenv
```

```env
# .env file
ANTHROPIC_API_KEY="your_api_key_here"
```

**Basic request structure:**

```python
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()
model = "claude-3-5-sonnet-20241022"

message = client.messages.create(
    model=model,
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "What is quantum computing?"}
    ]
)

# Extract just the text
print(message.content[0].text)
```

**Key parameters:**
- `model` — which Claude model to use
- `max_tokens` — safety limit for generation length (not a target)
- `messages` — list of conversation exchanges

---

## 4. Multi-Turn Conversations

The Claude API is **stateless** — it stores no memory between requests. You must manually maintain conversation history and send the full history with every follow-up.

```python
messages = []

def add_user_message(messages, text):
    messages.append({"role": "user", "content": text})

def add_assistant_message(messages, text):
    messages.append({"role": "assistant", "content": text})

def chat(messages):
    response = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=messages
    )
    return response.content[0].text

# Conversation flow
add_user_message(messages, "What is Python?")
reply = chat(messages)
add_assistant_message(messages, reply)

add_user_message(messages, "Give me a simple example.")
reply = chat(messages)  # Claude has full context
```

---

## 5. System Prompts

System prompts customise **how Claude responds**, not what it responds to. They are passed as a plain string using the `system` keyword argument.

```python
message = client.messages.create(
    model=model,
    max_tokens=1000,
    system="You are a patient math tutor. Give hints and guidance, never direct answers.",
    messages=[
        {"role": "user", "content": "How do I solve x² + 5x + 6 = 0?"}
    ]
)
```

> Same question, different system prompt → completely different response style.

---

## 6. Temperature

Temperature (range `0–1`) controls the **randomness** of token selection during generation.

| Temperature | Behaviour | Use Case |
|-------------|-----------|----------|
| `0` | Deterministic — always picks highest probability token | Data extraction, factual tasks |
| `0.3–0.5` | Slightly varied | Summarisation, classification |
| `0.7–1.0` | Creative / unexpected outputs | Brainstorming, creative writing, marketing |

```python
message = client.messages.create(
    model=model,
    max_tokens=500,
    temperature=0.9,   # high creativity
    messages=[{"role": "user", "content": "Write a tagline for a coffee brand."}]
)
```

---

## 7. Response Streaming

Streaming displays responses **chunk-by-chunk** as they're generated — eliminating the wait for a full response (which can take 10–30 seconds).

```python
# Simplified streaming with text_stream
with client.messages.stream(
    model=model,
    max_tokens=1000,
    messages=[{"role": "user", "content": "Explain neural networks."}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

# Capture final assembled message for storage
final_message = stream.get_final_message()
```

**Key event types:**
- `message_start` — acknowledgement
- `content_block_delta` — actual text chunks ← most important
- `message_stop` — generation complete

---

## 8. Controlling Model Output

Two techniques to steer Claude's output without changing the core prompt:

### Pre-filling Assistant Messages
Manually add an assistant message at the end of the conversation — Claude continues from exactly where you left off.

```python
messages = [
    {"role": "user", "content": "Which is better, coffee or tea?"},
    {"role": "assistant", "content": "Coffee is better because"}  # Claude continues from here
]
```

### Stop Sequences
Force Claude to stop generating when a specific string appears.

```python
message = client.messages.create(
    model=model,
    max_tokens=100,
    stop_sequences=["five"],
    messages=[{"role": "user", "content": "Count from one to ten."}]
)
# Output stops before "five"
```

---

## 9. Structured Data Generation

Combine **pre-filling + stop sequences** to extract clean, raw structured data without Claude's default explanatory text.

```python
messages = [
    {"role": "user", "content": "Return a JSON object with name, age, and city for a sample user."},
    {"role": "assistant", "content": "```json"}   # pre-fill opening delimiter
]

message = client.messages.create(
    model=model,
    max_tokens=500,
    stop_sequences=["```"],   # stop at closing delimiter
    messages=messages
)

# message.content[0].text now contains ONLY the raw JSON
import json
data = json.loads(message.content[0].text)
```

> Works for any structured format: JSON, Python code, regex, CSV, etc.

---

## 10. Prompt Evaluation

**Prompt evaluation** is automated testing of prompts using objective metrics — essential before deploying to production.

**Three paths after writing a prompt:**

| Path | Outcome |
|------|---------|
| ❌ Test once, deploy | Unreliable, misses edge cases |
| ❌ Test a few inputs, tweak | Still subjective and incomplete |
| ✅ Run through eval pipeline | Objective scores, systematic improvement |

> Engineers commonly under-test prompts. Use evaluation pipelines to get objective performance scores before iterating.

---

## 11. Eval Workflow

A **6-step iterative process** for systematic prompt improvement:

```
1. Write initial prompt draft
       ↓
2. Create evaluation dataset (3 examples → thousands)
       ↓
3. Interpolate each dataset input into the prompt template
       ↓
4. Get Claude's response for each variation
       ↓
5. Grade responses (e.g., 1–10 scale), average for overall score
       ↓
6. Modify prompt → repeat → compare versions
```

> No universal standard exists. Start simple with a custom implementation and iterate.

---

## 12. Generating Test Datasets

Use Claude (preferably **Haiku** for speed/cost) to automatically generate test cases:

```python
def generate_dataset(n=10):
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=2000,
        stop_sequences=["```"],
        messages=[
            {
                "role": "user",
                "content": f"Generate {n} test cases as a JSON array. Each object should have a 'task' field describing a user request."
            },
            {"role": "assistant", "content": "```json"}
        ]
    )
    return json.loads(response.content[0].text)
```

Save the output to `dataset.json` for consistent evaluation runs.

---

## 13. Running the Eval

Three core functions power the eval pipeline:

```python
def run_prompt(test_case, prompt_template):
    """Merge test case into prompt and call Claude."""
    prompt = prompt_template.format(**test_case)
    response = client.messages.create(...)
    return response.content[0].text

def run_test_case(test_case, prompt_template):
    """Run prompt + grade result."""
    output = run_prompt(test_case, prompt_template)
    score = grade(output, test_case)
    return {"output": output, "test_case": test_case, "score": score}

def run_eval(dataset, prompt_template):
    """Loop through entire dataset and collect results."""
    return [run_test_case(tc, prompt_template) for tc in dataset]
```

---

## 14. Model-Based Grading

Use a **second Claude call** to evaluate the quality of the first Claude's output.

```python
def model_grade(output, test_case):
    grading_prompt = f"""
    Evaluate this output on a scale of 1-10. Provide:
    - Strengths
    - Weaknesses
    - Reasoning
    - Score (integer 1-10)

    Output to grade: {output}
    Test case: {test_case}

    Respond only with JSON.
    """
    response = client.messages.create(
        model=model,
        max_tokens=500,
        stop_sequences=["```"],
        messages=[
            {"role": "user", "content": grading_prompt},
            {"role": "assistant", "content": "```json"}
        ]
    )
    result = json.loads(response.content[0].text)
    return result["score"]
```

> Ask for reasoning, not just a score — bare score requests tend to produce middling defaults.

---

## 15. Code-Based Grading

Programmatically validate outputs containing **code, JSON, or regex** using syntax parsing:

```python
import ast
import re

def validate_json(output: str) -> int:
    try:
        json.loads(output)
        return 10
    except json.JSONDecodeError:
        return 0

def validate_python(output: str) -> int:
    try:
        ast.parse(output)
        return 10
    except SyntaxError:
        return 0

def validate_regex(output: str) -> int:
    try:
        re.compile(output)
        return 10
    except re.error:
        return 0

# Combined score
def final_score(model_score, output, format_type):
    validators = {"json": validate_json, "python": validate_python, "regex": validate_regex}
    syntax_score = validators[format_type](output)
    return (model_score + syntax_score) / 2
```

---

## 16. Prompt Engineering Techniques

### 16.1 Being Clear and Direct
Use an **action verb in the first line** to specify exactly what you need.

```
# ❌ Vague
"Solar panels..."

# ✅ Clear
"Write three paragraphs explaining how solar panels convert sunlight into electricity."
```

### 16.2 Being Specific
Add **guidelines or steps** to direct the output:

- **Type A (Attributes)** — specify length, format, tone, structure
- **Type B (Steps)** — provide reasoning steps for Claude to follow

```
Generate a one-day meal plan for an athlete that:
- Meets their specific dietary restrictions
- Includes macro breakdowns for each meal
- Provides preparation time estimates
- Lists 3 meals and 2 snacks
```

### 16.3 Structure with XML Tags
Use XML tags to organise multi-part prompts and delineate injected content:

```python
prompt = f"""
Analyse the following code and documentation, then identify the bug.

<my_code>
{user_code}
</my_code>

<docs>
{documentation}
</docs>

Provide a step-by-step explanation of the issue.
"""
```

### 16.4 Providing Examples (Few-Shot Prompting)
Include **one or more input/output examples** to guide Claude's behaviour:

```python
prompt = """
Classify the sentiment of the following review.

<examples>
<example>
Input: "The product arrived broken and support was useless."
Output: negative
Reasoning: Expresses frustration with both product and service.
</example>

<example>
Input: "This is the best purchase I've made all year!"
Output: positive
Reasoning: Strong positive language and enthusiasm.
</example>
</examples>

Now classify: "{user_review}"
"""
```

---

## 17. Introducing Tool Use

**Tool use** allows Claude to access external data beyond its training knowledge by triggering functions on your server.

**Flow:**

```
1. Send request to Claude + list of available tools
       ↓
2. Claude evaluates if external data is needed
       ↓
3. Claude requests specific tool with arguments
       ↓
4. Your server executes the tool function
       ↓
5. Send tool result back to Claude
       ↓
6. Claude generates final response using retrieved data
```

> Example: "What's the weather today?" → Claude requests weather data → server calls weather API → Claude responds with current conditions.

---

## 18. Project Overview — Reminders

A practical tool use project: teaching Claude to **set time-based reminders**.

**Three problems requiring tools:**

| Problem | Tool Solution |
|---------|--------------|
| Claude doesn't know the exact current time | `get_current_datetime()` tool |
| Claude sometimes miscalculates time arithmetic | `add_duration_to_datetime()` tool |
| Claude has no mechanism to actually set reminders | `set_reminder()` tool |

---

## 19. Tool Functions

Tool functions are **plain Python functions** that Claude calls when it needs additional information.

```python
from datetime import datetime

def get_current_datetime(date_format: str = "%Y-%m-%d %H:%M:%S") -> str:
    if not date_format:
        raise ValueError("date_format cannot be empty")
    return datetime.now().strftime(date_format)

def add_duration_to_datetime(date_str: str, days: int = 0, hours: int = 0) -> str:
    from datetime import timedelta
    dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    result = dt + timedelta(days=days, hours=hours)
    return result.strftime("%Y-%m-%d %H:%M:%S")

def set_reminder(message: str, remind_at: str) -> str:
    print(f"⏰ Reminder set: '{message}' at {remind_at}")
    return f"Reminder successfully set for {remind_at}."
```

**Best practices:**
- Use descriptive function and argument names
- Validate inputs and raise errors with clear messages
- Error messages are visible to Claude — it can retry with corrections

---

## 20. Tool Schemas

Tool schemas are **JSON specifications** that tell Claude what tools are available, when to use them, and what arguments they require.

```python
from anthropic.types import ToolParam

get_current_datetime_schema = ToolParam({
    "name": "get_current_datetime",
    "description": "Returns the current date and time. Use this tool whenever the user references the current time, today's date, or asks about time-sensitive scheduling. Returns a formatted datetime string.",
    "input_schema": {
        "type": "object",
        "properties": {
            "date_format": {
                "type": "string",
                "description": "Python strftime format string (e.g., '%Y-%m-%d %H:%M:%S'). Defaults to standard datetime format."
            }
        },
        "required": []
    }
})
```

> 💡 **Shortcut:** Paste your function into Claude.ai and ask it to write the JSON schema following Anthropic's tool use documentation.

---

## 21. Handling Message Blocks

When tools are enabled, Claude's responses contain **multiple content blocks** — not just a single text block.

**Tool response structure:**

```python
# response.content now contains multiple blocks:
# [TextBlock(text="I'll check the time for you."), ToolUseBlock(name="get_current_datetime", ...)]

# ❌ Old approach — breaks with multi-block responses
text = response.content[0].text

# ✅ New approach — handle all blocks
def add_assistant_message(messages, response):
    messages.append({
        "role": "assistant",
        "content": response.content  # append ALL blocks, not just text
    })
```

---

## 22. Sending Tool Results

After executing a tool, send the result back to Claude as a **tool result block** inside a user message.

```python
def create_tool_result(tool_use_id: str, output: str, is_error: bool = False) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,   # must match the original tool use block ID
        "content": str(output),
        "is_error": is_error
    }

# Add tool result to conversation history
tool_result = create_tool_result(
    tool_use_id=tool_use_block.id,
    output=get_current_datetime()
)
messages.append({
    "role": "user",
    "content": [tool_result]
})
```

**Key rules:**
- `tool_use_id` must match the ID from Claude's tool use block
- Tool results go in a **user message**, not an assistant message
- Always include the original tool schemas in the follow-up request

---

## 23. Multi-Turn Conversations with Tools

Claude can chain **multiple tools sequentially** in a single conversation to answer one query.

```
User: "What day is 103 days from today?"
    ↓
Claude calls get_current_datetime()
    ↓
Server executes → returns "2024-03-15 10:30:00"
    ↓
Claude calls add_duration_to_datetime(date_str="2024-03-15...", days=103)
    ↓
Server executes → returns "2024-06-26 10:30:00"
    ↓
Claude: "103 days from today is June 26, 2024."
```

Use a `while` loop to handle arbitrary tool chains:

```python
def run_conversation(messages, tools):
    while True:
        response = chat(messages, tools)
        add_assistant_message(messages, response)

        if response.stop_reason != "tool_use":
            break  # Claude is done with tools — final answer reached

        tool_results = run_tools(response)
        messages.append({"role": "user", "content": tool_results})

    return response
```

---

## 24. Implementing Multiple Turns

Full implementation of the multi-turn tool loop:

```python
def run_tools(response) -> list:
    """Execute all tool use blocks in a response."""
    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            result = run_tool(block.name, block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result["output"]),
                "is_error": result["is_error"]
            })
    return tool_results


def run_tool(tool_name: str, tool_input: dict) -> dict:
    """Dispatch tool call to the correct function."""
    try:
        if tool_name == "get_current_datetime":
            output = get_current_datetime(**tool_input)
        elif tool_name == "add_duration_to_datetime":
            output = add_duration_to_datetime(**tool_input)
        elif tool_name == "set_reminder":
            output = set_reminder(**tool_input)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
        return {"output": output, "is_error": False}
    except Exception as e:
        return {"output": str(e), "is_error": True}
```

**Stop reason reference:**

| `stop_reason` | Meaning |
|--------------|---------|
| `"tool_use"` | Claude wants to call a tool — continue the loop |
| `"end_turn"` | Claude is done — return final response to user |
| `"max_tokens"` | Generation hit the token limit |

---

## 🚀 Next Steps

These notes cover the core of the Anthropic Claude API — but the learning doesn't stop here! Here are some ways to go further:

- 🔧 **Build something** — Pick a project from the notes (reminders system, eval pipeline, structured extractor) and implement it end-to-end
- 📖 **Explore the official docs** — [docs.anthropic.com](https://docs.anthropic.com) for the latest API features including RAG, MCP, Claude Code, and computer use
- 🧪 **Run your own evals** — Apply the eval workflow to a prompt you're actively using and see how the scores change as you apply engineering techniques
- 💬 **Open a Discussion** — Found something unclear, spotted an error, or want to share your implementation? Open a GitHub Discussion — all contributions are welcome!
- ⭐ **Star the repo** — If these notes helped you, a star helps others discover this resource

---

<div align="center">

Made with 🧠 by [Karthik Boodidha](https://www.linkedin.com/in/karthik-boodidha)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/karthik-boodidha)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Karthik-1221)
[![Portfolio](https://img.shields.io/badge/Portfolio-FF6B35?style=flat-square&logo=codeforces&logoColor=white)](https://codebasics.io/portfolio/Karthik-Boodidha)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/boodidhakarthik)

</div>
