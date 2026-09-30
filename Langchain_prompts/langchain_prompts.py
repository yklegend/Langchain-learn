import json
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

load_dotenv()

# 1. Model Setup
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    # Note: 1.2 is very high for summaries; 0.7 is usually better for factual tasks
    temperature=0.7 
)
model = ChatHuggingFace(llm=llm)

st.header("Research Paper Summarizer")

# 2. UI Components
paper_input = st.selectbox("Select the Research Paper name", [
    "1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities (Wang et al., 2025)",
    "Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond) (Jiang et al., 2025)",
    "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free (Qiu et al., 2025)",
    "The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery (Lu et al., 2024/2025)",
    "In-Place Test-Time Training (In-Place TTT) (Feng et al., 2026)",
    "Cosmos Policy: Adapting a Large Pretrained Video Model into an Effective Robot Policy (NVIDIA, 2026)",
    "RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots (Qu et al., 2026)",
    "Kimi K1.5: Scaling Reinforcement Learning with LLMs (Kimi Team, 2025)",
    "Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training (NeurIPS 2025)",
    "TRecViT: A Recurrent Video Transformer (Google DeepMind, 2026)",
    "The Abstraction Fallacy: Why AI Can Simulate But Not Instantiate Consciousness (DeepMind, 2026)"
])

style_input = st.selectbox("Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])
length_input = st.selectbox("Select explanation length", ["Short(1-2 paragraph)", "Medium(3-5 paragraph)", "Long(detailed explanation)"])

# 3. Prompt Template Logic
# Assuming your template.json looks like: {"prompt": "Summarize {paper_input} in a {style_input} way..."}
try:
    with open("template.json", "r") as f:
        template_data = json.load(f)
        # Use the specific key from your JSON, e.g., 'prompt'
        prompt_string = template_data.get("prompt", "Summarize the paper {paper_input} in a {style_input} style with {length_input} length.")
        prompt = PromptTemplate.from_template(prompt_string)
except FileNotFoundError:
    st.error("template.json not found. Please ensure the file exists.")
    prompt = None

# 4. Execution
if st.button("Summarize"):
    if prompt:
        with st.spinner("Thinking..."):
            # Create the chain: Prompt -> Model
            chain = prompt | model
            
            # You MUST assign the result of invoke to a variable
            result = chain.invoke({
                'paper_input': paper_input,
                'style_input': style_input,
                'length_input': length_input 
            })
            
            # Display the content
            st.subheader("Summary")
            st.write(result.content)
    else:
        st.warning("Prompt template could not be loaded.")