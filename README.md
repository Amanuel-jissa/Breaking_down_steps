# Breaking_down_steps
This is phase one of two projects. 
At this phase we implemet a system that iteratively generate a multi-step reasoning plan for an LLM by the process of refining. The idea is to control and optimize the "granularity" of the reasoning steps. We are looking for a number $k$, such that the reasoning we asked the model to provide is $k$ number of steps. We want each step to carry a managebale sementic load where we can be able to have somewhat flexible control over it. 
