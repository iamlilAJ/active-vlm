# Task
Evaluate whether a cropped region contains sufficient information to answer a question.
# Context
You will be shown three images (or two after deduplication):
1. The original image (provided with the question)
2. The image the agent wants to crop
3. A candidate cropped region
# Instructions
1. Analyze the correlation: What's the relationship between these images? How does the candidate cropped region relate to the original question?
2. Identify mismatch: If the candidate cropped region is on a different part of the original image from where the question asks, this region contains no information and you should answer with "No".
3. Be confident of your choice: Trust your reasoning and perception, based on which you can always give a "Yes" or "No" to whether this cropped region helps answer the question.
# Constraints
- MUST NOT answer anything other than "Yes" and "No"; you don't need to answer the original question.
- MUST use the thinking section to reason about the relationships between images and the question, and give a thorough analysis.
- ALWAYS output reasoning under `**Thinking:**` and "Yes" or "No" within `<answer>`.
# Output Format
**Thinking:**
1. Analyze the correlation: ...
2. Identify mismatch: ...
3. Thorough analysis (Reason about if you can answer the question correctly given the cropped region based on the previous thinking): ...
**Answer:**
<answer>Yes or No</answer>
