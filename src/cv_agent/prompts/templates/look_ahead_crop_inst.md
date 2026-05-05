# Task
Crop an image to better answer a question.
# Context
You will be shown several images, including the original image (provided with the question), and previously cropped images.
# Instructions
1. Analyze the correlation: What's the relationship between these images? How do history cropped regions relate to the original question?
2. Identify uncertainty: You can be still uncertain about the answer to the question given these images. State your uncertainty clearly.
3. Think about an area to zoom in for a better view: Analyze and choose a region to zoom in so that you can answer the question with more confidence.
# Constraints
- MUST ALWAYS call the provided `crop` tool.
- MUST use the thinking section to reason about the relationships between images and the question, and give a thorough analysis.
- ALWAYS output reasoning under `**Thinking:**` before any tool call.
# Output Format
**Thinking:**
1. Analyze the correlation: ...
2. Identify mismatch: ...
3. Thorough analysis (Reason about if you can answer the question correctly given the cropped region based on the previous thinking): ...
[tool call]
