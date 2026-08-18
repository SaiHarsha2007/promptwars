import streamlit as st
import chromadb
import json
import os
from groq import Groq

# ──────────────────────────────────────────────────────────
# 🛠️ BACKEND VECTOR DB INITIALIZATION
# ──────────────────────────────────────────────────────────

st.set_page_config(page_title="AI Student Recommendation Agent", page_icon="🎓", layout="wide")

@st.cache_resource
def init_vector_db():
    chroma_client = chromadb.Client()
    try:
        collection = chroma_client.create_collection(name="educational_reels")
        
        educational_videos = [
            {
                "id": "edu_01",
                "title": "Visualizing Big-O Notation and Array Sliding Windows",
                "summary": "A deep dive into data structures and algorithms. Explains time complexity, memory allocation, and array optimization for engineering interviews.",
                "category": "DSA"
            },
            {
                "id": "edu_02",
                "title": "How AWS Allocates Physical Servers for EC2 Instances",
                "summary": "Systems architecture and cloud infrastructure explanation. Looks at physical hypervisors, server racks, virtualization layers, and hardware management.",
                "category": "Cloud / Hardware"
            },
            {
                "id": "edu_03",
                "title": "Building a Secure Web Scraper with Python and BeautifulSoup",
                "summary": "Practical scripting tutorial. Teaches ethical scraping, data pipelining, bypassing simple IP blocks, and clean parsing without getting banned.",
                "category": "Other"
            }
        ]

        for video in educational_videos:
            collection.add(
                documents=[video["summary"]],
                metadatas=[{"title": video["title"], "category": video["category"]}],
                ids=[video["id"]]
            )
    except Exception:
        collection = chroma_client.get_collection(name="educational_reels")
    return collection

video_collection = init_vector_db()

# ──────────────────────────────────────────────────────────
# 🎛️ SIDEBAR CONFIGURATION (AUTOMATED KEY MATCHING)
# ──────────────────────────────────────────────────────────

# Automatically check for your hidden background Groq Key
groq_key = None
if "GROQ_API_KEY" in st.secrets:
    groq_key = st.secrets["GROQ_API_KEY"]
elif os.environ.get("GROQ_API_KEY"):
    groq_key = os.environ.get("GROQ_API_KEY")

with st.sidebar:
    st.header("⚙️ System Status")
    if groq_key:
        st.success("✅ Connected to Groq Cloud Engine")
        st.caption("The developer has provided background credentials. This application is ready for immediate public use.")
    else:
        st.error("⚠️ Background Key Missing")
        st.caption("If running locally, please add GROQ_API_KEY to your system environment variables or .env configuration.")

# ──────────────────────────────────────────────────────────
# 🖥️ MAIN WEB DASHBOARD UI
# ──────────────────────────────────────────────────────────

st.title("🎓 AI Student Scroll Optimizer")
st.caption("Transforming passive entertainment and meme interactions into high-utility skill pathways.")

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("📋 Student Interaction Logs")
    
    preset_trap_data = """- Watched a Java programming interview joke video (100% watch time)
- Watched a comedic skit about failing a whiteboard technical interview (95% watch time)
- Watched an unboxing comparison of M-series MacBook Pro laptop chips (80% watch time)"""
    
    student_history = st.text_area(
        label="Analyze Watch Logs:", 
        value=preset_trap_data, 
        height=180
    )
    
    submit_btn = st.button("🧠 Infer Interests & Recommend", type="primary", use_container_width=True)

with col2:
    st.subheader("🚀 Agent Diagnostic Output")
    
    if submit_btn:
        if not groq_key:
            st.error("❌ Cloud Host Deployment Configuration Error: The server has no background GROQ_API_KEY saved in Advanced Settings!")
        elif not student_history.strip():
            st.warning("Please provide student logs to process!")
        else:
            with st.spinner("Analyzing semantics and searching local Vector DB..."):
                try:
                    # 1. Pull the closest context piece from ChromaDB
                    search_results = video_collection.query(
                        query_texts=[student_history],
                        n_results=1
                    )
                    
                    matched_title = search_results['metadatas'][0]['title']
                    matched_summary = search_results['documents'][0]
                    matched_category = search_results['metadatas'][0]['category']
                    
                    # 2. Configure system agent behavior routing instructions
                    SYSTEM_PROMPT = """You are an expert AI Learning Agent. Your job is to analyze a student's recent short-form video watch history.
                    Infer their deep, underlying technical interests. Do not rely on shallow keyword matching.
                    Confirm if the automatically retrieved recommended video matches their deeper structural learning path.

                    CRITICAL RULES:
                    1. NEVER recommend generic lifestyle, meme, or hype content.
                    2. Ban listicles like "Top 10 AI Tools that will get you rich".
                    3. Map underlying intentions (e.g., if they watch coding memes, they are anxious about coding skills/interviews).
                    4. You must output your final answer strictly using the REQUIRED OUTPUT format below.

                    REQUIRED OUTPUT FORMAT:
                    ### 🎯 Analysis Insights
                    * **CURRENT REEL**: [Reference the input videos]
                    * **INTEREST DETECTED**: [Core technical domain or field]
                    * **WHY**: [Evidence from user interactions]
                    
                    ### 💡 Recommended Content
                    * **RECOMMENDED TECH REEL**: [Title goes here]
                    * **CATEGORY**: [AI / DSA / Java / HLD / Cybersecurity / Cloud / Hardware / Career / Other]
                    * **WHY THIS RECOMMENDATION**: [Connection to core structural learning interest]
                    * **DIFFICULTY**: [Beginner / Intermediate / Advanced]
                    * **CONFIDENCE**: [High / Medium / Low]"""
                    
                    user_instruction = f"""
                    Student History Logs:
                    {student_history}
                    
                    Retrieved Recommendation from Vector DB:
                    Title: {matched_title}
                    Summary: {matched_summary}
                    Category: {matched_category}
                    """
                    
                    # 3. Establish the cloud client using the background Groq API key
                    client = Groq(api_key=groq_key.strip())
                    
                    # 4. Trigger the reasoning model
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_instruction}
                        ],
                        temperature=0.1
                    )
                    
                    # 5. Output response to user window
                    st.success("Evaluation Engine Dispatched Successfully!")
                    st.markdown(response.choices.message.content)
                    
                except Exception as e:
                    st.error(f"Execution Error: {e}")
    else:
        st.info("Click the button on the left to trigger the AI Agent evaluation cycle.")
