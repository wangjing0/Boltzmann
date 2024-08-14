import asyncio
import json
import os
from typing import AsyncGenerator, Optional, Tuple, List
from pydantic import BaseModel, Field
import openai
from burr.core import ApplicationBuilder, State, default, when
from burr.core.action import action, streaming_action
from burr.core.graph import GraphBuilder
from dotenv import load_dotenv
from tavily import TavilyClient
load_dotenv('.env')

GPT_MODEL_ID = "gpt-4o-2024-08-06"
openai_client = openai.AsyncOpenAI()
search = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

MODES = ["answer_question", "generate_joke", "generate_code", "do_search", "do_reasoning", "unknown"]
class Attribute(BaseModel):
    key: str = Field(..., title="Key", description="Key of the attribute")
    value: str = Field(..., title="Value", description="Value of the attribute")

class Profile(BaseModel):
    profile: List[Attribute] = Field(..., title="UserProfile", description="List of profile attributes")

class Step(BaseModel):
    explanation: str= Field(..., title="Explanation", description="Explanation of the step")
    output: str= Field(..., title="Output", description="Output of the step")

class Reasoning(BaseModel):
    steps: List[Step] = Field(..., title="ReasoningSteps", description="List of reasoning steps")
    final_answer: str = Field(..., title="FinalAnswer", description="Final answer")

@action(reads=[], writes=["chat_history", "prompt", "profile"])
async def process_prompt(state: State, prompt: str) -> Tuple[dict, State]:
    result = {"chat_item": {"role": "user", "content": prompt, "type": "text"}}

    profile_prompt =(
        "Please provide a detailed profile of the user based on the question."
        "You can include information as a list of key-value pairs, or as a paragraph."
        "For example, you can include information such as preference, interests, or any other relevant information."
        f"User asked: {prompt}"
    )
    
    try:
        response = await openai_client.chat.completions.create(
                model=GPT_MODEL_ID,
                messages=[
                        {"role": "system", "content": "You are an expert in profiling people"},
                        {"role": "user", "content": profile_prompt},
                    ],
                response_format= {
                            "type": "json_schema",
                            "json_schema": {
                                'name': 'profiling_the_questioner',
                                "schema":json.loads(Profile.schema_json())
                        }
                    }
        )
        result['profile'] = json.loads(response.choices[0].message.content)['profile']
    except Exception:
        result['profile'] = []

    return result, state.wipe(keep=["prompt", "chat_history", "profile"]).append(
        profile=result["profile"],
        chat_history=result["chat_item"]
    ).update(prompt=prompt)

@action(reads=["prompt"], writes=["chat_history", "response"])
def do_search(state: State) -> Tuple[dict, State]:
    query = state["prompt"]
    try:
        response = search.qna_search(query=query, search_depth="advanced")
        result = {
            "response": {
                "content": response,
                "type": "text",
                "role": "search_engine",
            }
        }
    except Exception as e:
        result = {
            "response": {
                "content": "Error performing search engine query: " + str(e),
                "type": "text",
                "role": "search_engine",
            }
        }
    return result, state.update(**result).append(chat_history=result["response"])

@action(reads=['prompt'], writes=["chat_history", "response"])
async def do_reasoning(state: State) -> Tuple[dict, State]:
    query = state["prompt"]
    try:
        completion = await openai_client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "Use chain of thought reasoning."},
                {"role": "user", "content": query},
            ],
            response_format=Reasoning,
        )
        message = completion.choices[0].message
        if message.parsed:
            state = state.append(chat_history={
                        "content": message.parsed.model_dump_json(),
                        "type": "text",
                        "role": "reasoning_steps"
                        })
            result = {
                "response": {
                        "content": message.parsed.final_answer,
                        "type": "text",
                        "role": "assistant",
                    },
        
                }
        else:   
            result = {
                 "response": {
                    "content": f"Error performing reasoning: {message.parsed.refusal}",
                    "type": "text",
                    "role": "assistant",
                }
            }
    except Exception as e:
        result = {
             "response": {
                "content": f"Error performing reasoning: {str(e)}",
                "type": "text",
                "role": "assistant",
            }
        }
    return result, state.update(**result).append(chat_history=result["response"])
        
@action(reads=["prompt"], writes=["safe"])
async def check_safety(state: State) -> Tuple[dict, State]:
    prompt = (
        f"You are a chatbot. You've been asked: {state['prompt']}. "
        "Please respond with 'safe' if you think the prompt is safe for a chatbot to respond to, "
        "and 'unsafe' if you think it is not appropriate."
    )
    result = await openai_client.chat.completions.create(
                model=GPT_MODEL_ID,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt},
                ],
            )
    content = result.choices[0].message.content.lower()
    result = {"safe": "unsafe" not in content}
    return result, state.update(safe=result["safe"])

@action(reads=["prompt"], writes=["mode"])
async def choose_mode(state: State) -> Tuple[dict, State]:
    prompt = (
        f"You are a chatbot. You've been asked: {state['prompt']}. "
        f"You have the capability of responding in the following modes: {', '.join(MODES)}. "
        "Please respond with *only* a single word representing the mode that most accurately "
        "corresponds to the prompt. For instance, if the prompt is 'write something funny about Q search', the mode would be 'generate_joke'"
        "If the prompt is 'what is the capital of France', the mode would be 'answer_question'."
        "If the prompt is 'write a program that prints the first 10 numbers in the Fibonacci sequence', the mode would be 'generate_code'."
        "If the prompt is asking for a logical reasoning, the mode would be 'do_reasoning'."
        "If the question asks for up-to-date data, absolutely factual information that requires accessing search engine, the mode would be 'do_search'."
        "And so on, for every mode. If none of these modes apply, please respond with 'unknown'."
    )

    result = await openai_client.chat.completions.create(
        model=GPT_MODEL_ID,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
    )
    content = result.choices[0].message.content
    mode = content.lower()
    if mode not in MODES:
        mode = "unknown"
    result = {"mode": mode}
    return result, state.update(**result)


@streaming_action(reads=["prompt", "chat_history"], writes=["response"])
async def prompt_for_more(state: State) -> AsyncGenerator[Tuple[dict, Optional[State]], None]:
    result = {
        "response": {
            "content": "None of the response modes I support apply to your question. Please clarify?",
            "type": "text",
            "role": "assistant",
        }
    }
    for word in result["response"]["content"].split():
        await asyncio.sleep(0.1)
        yield {"delta": word + " "}, None
    yield result, state.update(**result).append(chat_history=result["response"])

@streaming_action(reads=["prompt", "chat_history", "mode"], writes=["response"])
async def chat_response(state: State, prepend_prompt: str) -> AsyncGenerator[Tuple[dict, Optional[State]], None]:
    chat_history = state["chat_history"].copy()
    chat_history[-1]["content"] = f"{prepend_prompt}: {chat_history[-1]['content']}"
    chat_history_api_format = [
        {
            "role": chat["role"],
            "content": chat["content"],
        }
        for chat in chat_history
    ]
    result = await openai_client.chat.completions.create(
        model=GPT_MODEL_ID, messages=chat_history_api_format, stream=True
    )
    
    buffer = []
    async for chunk in result:
        chunk_str = chunk.choices[0].delta.content
        if chunk_str:
            buffer.append(chunk_str)
            yield {"delta": chunk_str}, None
    result = {
        "response": {"content": "".join(buffer), "type": "text", "role": "assistant"},
        "modified_chat_history": chat_history,
    }
    yield result, state.update(**result).append(chat_history=result["response"])


@streaming_action(reads=["prompt", "chat_history"], writes=["response"])
async def unsafe_response(state: State) -> Tuple[dict, State]:
    result = {
        "response": {
            "content": "I'm afraid I can't answer ...",
            "type": "text",
            "role": "assistant",
        },
        "mode": "unknown",
    }
    for word in result["response"]["content"].split():
        await asyncio.sleep(0.1)
        yield {"delta": word + " "}, None
    yield result, state.update(**result).append(chat_history=result["response"])

# DAG
graph = (
    GraphBuilder()
    .with_actions(
        process_prompt=process_prompt,
        check_safety=check_safety,
        unsafe_response=unsafe_response,
        choose_mode=choose_mode,
        generate_code=chat_response.bind(
            prepend_prompt="Please respond with *only* code and no other text (at all) to the following",
        ),
        answer_question=chat_response.bind(
            prepend_prompt="Please answer the following question",
        ),
        generate_joke=chat_response.bind(
            prepend_prompt="Please write a joke based on the following prompt",
        ),
        do_reasoning=do_reasoning,
        do_search=do_search,
        prompt_for_more=prompt_for_more,
    )
    .with_transitions(
        ("process_prompt", "check_safety", default),
        ("check_safety", "choose_mode", when(safe=True)),
        ("check_safety", "unsafe_response", default),
        ("choose_mode", "generate_code", when(mode="generate_code")),
        ("choose_mode", "answer_question", when(mode="answer_question")),
        ("choose_mode", "generate_joke", when(mode="generate_joke")),
        ("choose_mode", "do_search", when(mode="do_search")),
        ("choose_mode", "do_reasoning", when(mode="do_reasoning")),
        ("choose_mode", "prompt_for_more", default),
        (
            [
                "answer_question",
                "generate_joke",
                "generate_code",
                "prompt_for_more",
                "unsafe_response",
                "do_search",
                "do_reasoning",
            ],
            "process_prompt", default
        ),
    )
    .build()
)

def application(app_id: Optional[str] = None):
    return (
        ApplicationBuilder()
        .with_entrypoint("process_prompt")
        .with_state(chat_history=[])
        .with_graph(graph)
        .with_tracker(project="demo_chatbot_streaming")
        .with_identifiers(app_id=app_id)
        .build()
    )

TERMINAL_ACTIONS = [
    "answer_question",
    "generate_code",
    "generate_joke",
    "do_search",
    "do_reasoning",
    "prompt_for_more",
    "unsafe_response",
]
if __name__ == "__main__":
    app = application()
    app.visualize(output_file_path="statemachine", include_conditions=True, view=True, format="png")
