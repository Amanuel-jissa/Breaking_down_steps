from openai import OpenAI
import json
import os
import re
#sentenceTransformer to Embed reasoning steps into vectors using MiniLM
#We can instead look into other methods such as using the attention but i think this is sufficient for now.
from sentence_transformers import SentenceTransformer, util
import torch
client = OpenAI(api_key="Your api key")
embedding_modl = SentenceTransformer('all-MiniLM-L6-v2')
#as the threshold increases we expect a finner step structure
#which means we should expect more steps to be embedded.
Coherence_thershold = float(0.90)
#we don't want to iteratively continue to add the steps so for now i have minimized it to 10 steps.
#we can only get at most 10 reasoning steps for any question
max_step = 10
#my file has only one question for now. I advise that we test this in fewer questions as the iteration will be expensive.
with open('/Users/perfectoid/Downloads/metadata-2.jsonl') as f:
    questions = [json.loads(line) for line in f if line.strip()]
responses = {}
def computing_granularity(steps):
    # This function will score our reasoning finness/granularity. It is the number of steps divided by total embedding norm.
    #if the score is high it means the reasoning has a fine structure and is less dense with "information".
    if not steps:
        return 0.0
    #embedding steps into vectors
    embeddings = embedding_modl.encode(steps, convert_to_tensor=True)
    #we need to find the norm of each vector embedding for the same reasons we want to normalize any vector
    norms = [torch.norm(e).item() for e in embeddings]
    tot_norm = sum(norms)
    return len(steps) / (1.0 + tot_norm)
def agent_prompt(task, num_steps):
    #I tried to come up with a prompt to force the output to be similar to agentic responses.
    # I don't have experience with agents so some tweeking might be needed here.
    return (
        "You are an AI planner operating in a tool-using agent system. "
        "You must break the following task into a sequence of minimal, logically necessary steps "
        "that an execution agent can perform. Each step should be atomic, aligned with possible tool actions "
        "(e.g., search, browse, parse, calculate, extract, compare), and necessary for the completion of the task. "
        f"Provide exactly {num_steps} steps.\n"
        "Format your output as:\n"
        "Steps: <comma-separated list of steps>\n"
        "Number of Steps: <integer>\n"
        "Final Answer: <your final answer based on these steps>\n"
        "Do not output explanations.\n"
        f"Task: {task}"
    )
#Each question goes through this process however for this experiment this loop is not useful
for i in questions:
    task_id = i["task_id"]
    task_text = i["Question"]
    num_steps = 2
    final_response = None
    last_score = 0.0
    while True:
        # Checking the max steps first
        if num_steps > max_step:
            print(f"[{task_id}] we have reached max step limit without passing threshold.")
            responses[task_id] = {"error": "we have reached max step limit"}
            break
        prompt = agent_prompt(task_text, num_steps)
        #requesting OpenAI API for num_steps reasoning steps for a given question.
        #Thanks to the platform https://community.openai.com/t/how-to-pass-prompt-to-the-chat-completions-create/592629 for the detailed discussion
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}]
        )
        output = response.choices[0].message.content
        # THis extracts the steps
        match = re.search(r"Steps:\s*(.+?)\nNumber of Steps:", output, re.DOTALL)
        if not match:
            print(f"[{task_id}] Malformed response:\n{output}")
            responses[task_id] = {"error": "Malformed response"}
            break
        steps_raw = match.group(1)
        steps = [step.strip() for step in steps_raw.split(',') if step.strip()]
        if len(steps) < 2:
            print(f"[{task_id}] Too few steps: {len(steps)}")
            responses[task_id] = {"error": "Too few steps generated"}
            break
        # Calculates how good the reasoning is broken into steps
        score = computing_granularity(steps)
        # the gradient is the numerical difference between the current and previous granularity score.
        # It's not important to our calculation but it measures how much the granularity score increased or decreased after adding a step.
        print(f"[{task_id}] {num_steps} steps -> granularity score: {score:.4f} (Δ = {score - last_score:.4f})")
        # Check if threshold is met
        if score >= Coherence_thershold:
            # kept gettifng a several errors earlier so added this section
            try:
                final_answer = re.search(r"Final Answer:\s*(.+)", output).group(1).strip()
                final_response = {
                    "steps": steps,
                    "granularity_score": score,
                    "delta_score": score - last_score,
                    "num_steps": len(steps),
                    "final_answer": final_answer
                }
                print(f"[{task_id}] Threshold met: score={score:.4f} >= {Coherence_thershold}")
                responses[task_id] = final_response
                break
            except AttributeError:
                print(f"[{task_id}] Failed to extract Final Answer:\n{output}")
                responses[task_id] = {"error": "Failed to extract Final Answer"}
                break
        # Update for next iteration
        last_score = score
        num_steps += 1
    with open("all_responses.json", 'w') as f:
        json.dump(responses, f, indent=2)
