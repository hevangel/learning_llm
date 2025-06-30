#!/usr/bin/env python3
"""
Script to compare DeepLearning.AI course list with actual directories
"""

# Course titles from the markdown file (course number: title)
courses_from_markdown = {
    80: "ChatGPT Prompt Engineering for Developers",
    79: "Building Systems with the ChatGPT API", 
    78: "LangChain for LLM Application Development",
    77: "How Diffusion Models Work",
    76: "LangChain: Chat with Your Data",
    75: "Building Generative AI Applications with Gradio",
    74: "Evaluating and Debugging Generative AI Models Using Weights and Biases",
    73: "Large Language Models with Semantic Search",
    72: "Finetuning Large Language Models",
    71: "How Business Thinkers Can Start Building AI Plugins With Semantic Kernel",
    70: "Understanding and Applying Text Embeddings",
    69: "Pair Programming with a Large Language Model",
    68: "Functions, Tools and Agents with LangChain",
    67: "Vector Databases: from Embeddings to Applications",
    66: "Quality and Safety for LLM Applications",
    65: "Building and Evaluating Advanced RAG Applications",
    64: "Reinforcement Learning from Human Feedback",
    63: "Advanced Retrieval for AI with Chroma",
    62: "Build LLM Apps with LangChain.js",
    61: "LLMOps",
    60: "Automated Testing for LLMOps",
    59: "Building Applications with Vector Databases",
    58: "Serverless LLM Apps with Amazon Bedrock",
    57: "Prompt Engineering with Llama 2 & 3",
    56: "Open Source Models with Hugging Face",
    55: "Knowledge Graphs for RAG",
    54: "Efficiently Serving LLMs",
    53: "JavaScript RAG Web Apps with LlamaIndex",
    52: "Red Teaming LLM Applications",
    51: "Preprocessing Unstructured Data for LLM Applications",
    50: "Quantization Fundamentals with Hugging Face",
    49: "Getting Started With Mistral",
    48: "Prompt Engineering for Vision Models",
    47: "Quantization in Depth",
    46: "Building Agentic RAG with LlamaIndex",
    45: "Building Multimodal Search and RAG",
    44: "Multi AI Agent Systems with crewAI",
    43: "Introduction to On-Device AI",
    42: "AI Agentic Design Patterns with AutoGen",
    41: "AI Agents in LangGraph",
    40: "Building Your Own Database Agent",
    39: "Function-Calling and Data Extraction with LLMs",
    38: "Carbon Aware Computing for GenAI Developers",
    37: "Prompt Compression and Query Optimization",
    36: "Pretraining LLMs",
    35: "Federated Learning",
    34: "Embedding Models: From Architecture to Implementation",
    33: "Improving Accuracy of LLM Applications",
    32: "Building AI Applications with Haystack",
    31: "Large Multimodal Model Prompting with Gemini",
    30: "AI Python for Beginners",
    29: "Multimodal RAG: Chat with Videos",
    28: "Retrieval Optimization: From Tokenization to Vector Quantization",
    27: "Introducing Multimodal Llama 3.2",
    26: "Serverless Agentic Workflows with Amazon Bedrock",
    25: "Practical Multi AI Agents and Advanced Use Cases with crewAI",
    24: "Safe and Reliable AI via Guardrails",
    23: "Building an AI-Powered Game",
    22: "Collaborative Writing and Coding with OpenAI Canvas",
    21: "Reasoning with o1",
    20: "Build Long-Context AI Apps with Jamba",
    19: "Building Towards Computer Use with Anthropic",
    18: "How Transformer LLMs Work",
    17: "Attention in Transformers: Concepts and Code in PyTorch",
    16: "Evaluating AI Agents",
    15: "Build Apps with Windsurf's AI Coding Agents",
    14: "Event-Driven Agentic Document Workflows",
    13: "Long-Term Agentic Memory with LangGraph",
    12: "Vibe Coding 101 with Replit",
    11: "Getting Structured LLM Output",
    10: "Building AI Browser Agents",
    9: "Building Code Agents with Hugging Face smolagents",
    8: "LLMs as Operating Systems: Agent Memory",
    7: "Building AI Voice Agents for Production",
    6: "MCP: Build Rich-Context AI Apps with Anthropic",
    5: "Reinforcement Fine-Tuning LLMs with GRPO",
    4: "DSPy: Build and Optimize Agentic Apps",
    3: "Orchestrating Workflows for GenAI Applications",
    2: "Building with Llama 4",
    1: "ACP: Agent Communication Protocol"
}

# Directory names from the file system
directories = [
    "1. ChatGPT Prompt Engineering for Developers/",
    "10. How Business Thinkers Can Start Building AI Plugins WIth Semantic Kernel/",
    "11. Understanding and Applying Text Embeddings/",
    "12. Pair Programming with a Large Language Model/",
    "13. Functions, Tools and Agents with LangChain/",
    "14. Vector Databases from Embeddings to Applications/",
    "15. Quality and Safety for LLM Applications/",
    "16. Building and Evaluating Advanced RAG/",
    "17. Reinforcement Learning from Human Feedback/",
    "18. Advanced Retrieval for AI with Chroma/",
    "19. Build LLM Apps with LangChain.js/",
    "2. Building Systems with the ChatGPT API/",
    "20. LLMOps/",
    "21. Automated Testing for LLMOps/",
    "22. Building Applications with Vector Databases/",
    "23. _Serverless LLM apps with Amazon Bedrock/",
    "24. Prompt Engineering with Llama 2 & 3/",
    "25. Open Source Models with Hugging Face/",
    "26. Knowledge Graphs for RAG/",
    "27. _Efficient Serving LLMs/",
    "28. _JavaScript RAG Web Apps with LlamaIndex/",
    "29. _Red Teaming LLM Applications/",
    "3. LangChain for LLM Application Development/",
    "30. _Preprocessing Unstructured Data for LLM Applications/",
    "31. Quantization Fundamentals with Hugging Face/",
    "32. Getting Started with Mistral/",
    "33. _Prompt Engineering for Vision Models/",
    "34. Quantization in Depth/",
    "35. Building Agentic RAG with Llamaindex/",
    "36. _Building Multimodel Search and RAG/",
    "37. Multi AI Agent Systems with crewAI/",
    "38. AI Agentic Design Patterns with AutoGen/",
    "39. AI Agents in LangGraph/",
    "4. How Diffusion Models Work/",
    "40. _Building Your Own Database Agent/",
    "41. _Function-Calling and Data Extraction with LLMs/",
    "42. Carbon Aware Computing for GenAI Developers/",
    "43. _Prompt Compression and Query Optimization/",
    "44. _Pretraining LLMs/",
    "45. _Federated Learning/",
    "46. _Embedding Models From Architecture to Implementation/",
    "47. _Improving Accuracy of LLM Applications/",
    "48. _Building AI Application with Haystack/",
    "49. Large Multimodel Model Prompting with Gemini/",
    "5. LangChain Chat with Your Data/",
    "50. AI Python for Beginners/",
    "51. Multimodel RAG Chat with Videos/",
    "52. _Retrieval Optimization From Tokenization to Vector Quantization/",
    "53. Introducing Multimodal Llama 3.2/",
    "54. Practical Multi AI Agents and Advanced Use Cases with crewAi/",
    "55. _LLMs as Operating Systems Agent Memory/",
    "56. _Safe and Reliable AI via Guardrails/",
    "57. Building an AI-Powered Game/",
    "58. Collaborative Writing and Coding with OpenAI Canvas/",
    "59. Reasoning with o1/",
    "6. Building Generative AI Applications with Gradio/",
    "60. Build Long-Context AI Apps with Jamba/",
    "61. Building Towards Computer Use with Anthropic/",
    "62. How Transformer LLMs Work/",
    "63. _Attention in Transformers Concepts and Code in PyTorch/",
    "64. Evaluating AI Agents/",
    "65. Build Apps with Windsurf's AI Coding Agents/",
    "66. _Event-Driven Agentic Document Workflows/",
    "67. _Long-Term Agentic Memory with LangGraph/",
    "68. Vibe Coding 101 with Replit/",
    "69. Getting Structured LLM Output/",
    "7. Evaluating and Debugging Generative AI/",
    "70. Building AI Browswer Agents/",
    "71. Building Code Agents with Hugging Face smolagents/",
    "72. _Building AI Voice Agents for Production/",
    "73. MCP Build Rich-Context AI Apps with Anthropic/",
    "74. _Reinforcement Fine-Tuning LLMs with GRPO/",
    "75. _DSPy Build and Optimize Agentics Apps/",
    "76. _Orchestrating Workflows for GenAI Applications/",
    "77. _Building with Llama 4/",
    "78. _ACP Agent Communication Protocol/",
    "8. Large Language Models with Semantic Search/",
    "9. Finetuning Large Language Models/"
]

def normalize_title(title):
    """Normalize title for comparison by removing common variations"""
    # Remove underscores, extra spaces, normalize punctuation
    normalized = title.lower()
    normalized = normalized.replace('_', ' ')
    normalized = normalized.replace(':', '')
    normalized = normalized.replace('&', 'and')
    normalized = ' '.join(normalized.split())  # normalize whitespace
    return normalized

def extract_dir_title(dir_name):
    """Extract title from directory name, removing number prefix and trailing slash"""
    # Remove number prefix and trailing slash
    title = dir_name.split('. ', 1)[1].rstrip('/')
    # Remove leading underscore if present
    if title.startswith('_'):
        title = title[1:]
    return title

# Create mapping from directory number to title
dir_mapping = {}
for dir_name in directories:
    if '. ' in dir_name and not dir_name.startswith('c'):  # Skip specialization courses
        num_str = dir_name.split('.')[0]
        try:
            num = int(num_str)
            title = extract_dir_title(dir_name)
            dir_mapping[num] = title
        except ValueError:
            continue

# Find missing courses
missing_courses = []
mismatched_courses = []

print("=== COMPARISON ANALYSIS ===\n")

for course_num, course_title in courses_from_markdown.items():
    # Convert course numbering: course 80 -> dir 1, course 79 -> dir 2, etc.
    dir_num = 81 - course_num
    
    if dir_num not in dir_mapping:
        missing_courses.append((course_num, course_title))
        print(f"MISSING: Course #{course_num} '{course_title}' (should be directory {dir_num})")
    else:
        dir_title = dir_mapping[dir_num]
        if normalize_title(course_title) != normalize_title(dir_title):
            mismatched_courses.append((course_num, course_title, dir_title))
            print(f"MISMATCH: Course #{course_num}")
            print(f"  Markdown: '{course_title}'")
            print(f"  Directory: '{dir_title}'")
            print()

print(f"\n=== SUMMARY ===")
print(f"Total courses in markdown: {len(courses_from_markdown)}")
print(f"Total directories found: {len(dir_mapping)}")
print(f"Missing courses: {len(missing_courses)}")
print(f"Mismatched titles: {len(mismatched_courses)}")

if missing_courses:
    print(f"\nMISSING COURSES:")
    for course_num, title in missing_courses:
        dir_num = 81 - course_num
        print(f"  #{course_num}: '{title}' (should be directory {dir_num})")

if mismatched_courses:
    print(f"\nMISMATCHED TITLES:")
    for course_num, md_title, dir_title in mismatched_courses:
        print(f"  #{course_num}: '{md_title}' vs '{dir_title}'")
