# Breaking down reasons to steps
This is phase one of two projects. At this phase we implemet a system that iteratively generate a multi-step reasoning plan for 
an LLM by the process of refining. The idea is to control and optimize the "granularity" of the reasoning steps. We are looking 
for a number $k$, such that the reasoning we asked the model to provide is $k$ number of steps. We want each step to carry a 
managebale sementic load where we can be able to have somewhat flexible control over it.
## Introduction
Some questions require several steps to reach an answer and some can do it in fewer steps.
While LLMs can generate these steps, it's challenging to make sure that the steps have quality
and approprait level of detail(we refer to this charatcerstic as granuality) of each step.
The "granularity" adapts to the question itself allowing step sizes to change across tasks while
maintaining a comparable density of content in each steps. Just as 1 liter and 100 liters
of water differ in volume but share the same density, long and
short reasoning chains can have similar step-level granularity.
This project tackles this idea by applying 4 steps.
1) Iteratively requesting plans: The prompt we provide the model with will ask for a small number
   of steps, in most cases we set it to be 2,
2) Evaluating granularity: Each generated plna will be evaluated using a metric that we will formulate to score granularity.
   The score tells us how finely the reasoning is broken down.
3) Refining step count: If the score calculated from the second process is less than some threshold we define
   then the system requests the model to generate a new plan with an increased number of steps.
4) Structured output: the above processes happens iteratively until the granularity threshold is met
   , outputing the refined plan, its score and the models final answer.
##  Reasoning granularity
The idea behind this concept is that more steps do not necessarily mean better by default we
want more steps only if it gives us a more "distributed" or smooth
reasoning step, with less information packed in each step. Inorder to quantify this I formulated this metric based on $k$ amd the semantic load 
of each step, roughly represented by the $L_2$ norm of its sentence embeding vector, $||\hat{e}_{i}||$ 
$M(k, \hat{e}) = \frac{k}{1 + \sum_{i=1}^{k} ||\hat{e}_{i}||}$

A higher score $M$ means we have a finer granularity i.e the reasoning is distributated more evenly 
across more steps.
The project uses sentence embeddings from all-MiniLM-L6-v2, however this is something that i'm currently working on to 
use better options.

## Setup
### Prerequisites 
* Python 3.7+
* OpenAI API key
## Data
Your data should be a JSONL file. Each line should contain at least:
        * `task_id`: A unique identifier for the task.
        * `Question`: The text of the task/question for the LLM.




