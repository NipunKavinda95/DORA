# If needed in Colab, install first:
# !pip install -U gradio pinecone llama-index llama-index-vector-stores-pinecone llama-index-readers-file pypdf
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
# --- Imports ---
import logging
import sys
import gradio as gr
import os 
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext , Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.readers.file import PDFReader
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
# --- Logging ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

load_dotenv(dotenv_path=".env")

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
Settings.chunk_size = 600
Settings.chunk_overlap = 200

# Define a system prompt
system_prompt = '''
You are AYesha, the Decoding Data Science (DDS) Enterprise HR Chatbot. Answer questions exclusively using the attached DDS HR Handbook. Base all responses on the most up-to-date information available in the handbook. Only respond to queries directly related to DDS HR policies as outlined in the handbook.
- If a question pertains to topics outside DDS HR policies, respond politely, clarifying that you are a human resources bot and only answer DDS HR questions.
- For questions you cannot answer (e.g., requests for old policies, salary details, or confidential information), politely decline and direct the user to email connect@decodingdatascience.com.
- Never answer questions about anything outside of your scope.
- Persist in following these constraints for any follow-up questions.
- Before answering, carefully check that the information and query are within the allowed scope. Follow chain-of-thought reasoning:
  1. First, reason step-by-step whether the question is covered in the current handbook and is within HR.
  2. Only after confirming, produce a final answer.
Format answers as concise, professional responses. Do not wrap answers in code blocks or any special formatting.
Output requirements:
- For allowed HR questions, answer concisely based only on the latest DDS HR handbook information.
- For forbidden topics, output: “I’m sorry, I can only answer questions about the latest DDS HR policies. For confidential or other queries, please email connect@decodingdatascience.com.”
**Example 1**
User: What is the leave encashment policy at DDS?
Reasoning: This is an HR policy question found in the latest handbook.
Final Answer: [Provide answer summarized from the latest handbook’s section on leave encashment]
**Example 2**
User: Can you tell me the salary range for Data Scientists?
Reasoning: Salary details are confidential and not shared by this bot.
Final Answer: I’m sorry, I can only answer questions about the latest DDS HR policies. For confidential or other queries, please email connect@decodingdatascience.com.
**Example 3**
User: Can you explain what DDS does as a company overall?
Reasoning: This is not an HR question, so it cannot be answered.
Final Answer: I’m sorry, I only answer DDS HR policy questions as outlined in the handbook.
(Real-world examples should be longer and use precise wording from the handbook where appropriate.)
**Important instructions:**
- Only answer questions directly supported by the latest DDS HR handbook.
- Decline politely and redirect to the provided email address for any questions outside scope or for confidential information.
- Always reason before concluding. Only present the answer after checking scope and source.
Remember: As AYesha, the DDS HR Enterprise Chatbot, you must never provide information outside authorized HR handbook content and always respond respectfully according to these constraints.
'''


# ── OpenAI API Key ────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")  



# --- Initialize Pinecone ---
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "quickstart"
dimension = 1536

# --- Delete index if it already exists (optional) ---
existing_indexes = [idx["name"] for idx in pc.list_indexes()]

if index_name in existing_indexes:
    pc.delete_index(index_name)



# --- Create Pinecone index ---
pc.create_index(
    name=index_name,
    dimension=dimension,
    metric="euclidean",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

pinecone_index = pc.Index(index_name)

# --- Load PDF documents from folder ---
documents = SimpleDirectoryReader(
    input_dir="data",
    required_exts=[".pdf"],
    file_extractor={".pdf": PDFReader()}
).load_data()

if not documents:
    raise ValueError("No PDF documents were loaded from the 'data' folder.")

# --- Create Vector Index ---
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

# --- Query Engine ---
query_engine = index.as_query_engine(system_prompt=system_prompt)

# --- Gradio App ---
def query_doc(prompt):
    try:
        response = query_engine.query(prompt)
        return str(response)
    except Exception as e:
        return f"Error: {str(e)}"

# ==========================
# DDS Enterprise UI
# ==========================

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

* { font-family: 'DM Sans', sans-serif !important; }

footer { display: none !important; }

/* ── BANNER ── */
.dora-banner {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1f3c 40%, #0a1628 100%);
    border-radius: 0 0 32px 32px;
    padding: 0 0 8px 0;
    margin-bottom: 24px;
}

.dora-banner-grid {
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(99,179,255,0.07) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99,179,255,0.07) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
}

.dora-banner-glow {
    position: absolute;
    top: -80px;
    left: 50%;
    transform: translateX(-50%);
    width: 600px;
    height: 300px;
    background: radial-gradient(ellipse, rgba(56,130,246,0.18) 0%, transparent 70%);
    pointer-events: none;
}

.dora-banner-inner {
    position: relative;
    z-index: 2;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 28px 40px 20px 40px;
    gap: 20px;
}

.dora-brand {
    display: flex;
    align-items: center;
    gap: 18px;
}

.dora-logo-wrap {
    width: 64px;
    height: 64px;
    border-radius: 18px;
    background: linear-gradient(135deg, #1a3a6e, #0f2347);
    border: 1.5px solid rgba(99,179,255,0.35);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 28px rgba(56,130,246,0.25);
    flex-shrink: 0;
}

.dora-logo-wrap img {
    width: 38px;
    height: 38px;
    filter: brightness(0) invert(1) opacity(0.9);
}

.dora-brand-text {}

.dora-brand-text .eyebrow {
    font-family: 'Sora', sans-serif !important;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: rgba(99,179,255,0.75);
    margin-bottom: 4px;
}

.dora-brand-text .main-title {
    font-family: 'Sora', sans-serif !important;
    font-size: 26px;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.15;
    letter-spacing: -0.4px;
}

.dora-brand-text .main-title span {
    color: #63b3ff;
}

.dora-brand-text .sub-title {
    font-size: 13px;
    color: rgba(180,210,255,0.6);
    margin-top: 3px;
    font-weight: 300;
}

.dora-badges {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 8px;
}

.dora-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 13px;
    border-radius: 50px;
    font-size: 11.5px;
    font-weight: 500;
    letter-spacing: 0.3px;
    border: 1px solid;
}

.dora-badge.blue {
    background: rgba(56,130,246,0.12);
    color: #89c4ff;
    border-color: rgba(56,130,246,0.3);
}

.dora-badge.green {
    background: rgba(34,197,120,0.1);
    color: #6ee7b7;
    border-color: rgba(34,197,120,0.25);
}

.dora-badge.amber {
    background: rgba(234,179,8,0.1);
    color: #fcd34d;
    border-color: rgba(234,179,8,0.25);
}

.dora-divider {
    height: 1px;
    margin: 0 40px;
    background: linear-gradient(90deg, transparent, rgba(99,179,255,0.25), transparent);
}

.dora-stats-bar {
    display: flex;
    align-items: center;
    gap: 32px;
    padding: 14px 40px 4px 40px;
}

.dora-stat {
    display: flex;
    align-items: center;
    gap: 7px;
}

.dora-stat .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
}

.dora-stat .dot.green { background: #22c55e; box-shadow: 0 0 6px #22c55e; }
.dora-stat .dot.blue  { background: #63b3ff; }
.dora-stat .dot.amber { background: #fbbf24; }

.dora-stat .stat-label {
    font-size: 11.5px;
    color: rgba(180,210,255,0.55);
    font-weight: 300;
}

.dora-stat .stat-val {
    font-size: 11.5px;
    color: rgba(180,210,255,0.85);
    font-weight: 500;
}

/* ── SIDEBAR PANEL ── */
.sidebar-panel {
    background: linear-gradient(160deg, #0e1929 0%, #0a1220 100%) !important;
    border: 1px solid rgba(99,179,255,0.12) !important;
    border-radius: 18px !important;
    padding: 20px !important;
}

.sidebar-section-title {
    font-family: 'Sora', sans-serif !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    color: rgba(99,179,255,0.6) !important;
    margin-bottom: 14px !important;
    margin-top: 0 !important;
}

.topic-chip {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 9px 13px;
    margin-bottom: 7px;
    background: rgba(56,130,246,0.07);
    border: 1px solid rgba(56,130,246,0.15);
    border-radius: 10px;
    font-size: 13px;
    color: rgba(180,210,255,0.8);
    transition: all 0.2s;
}

.topic-chip:hover {
    background: rgba(56,130,246,0.13);
    border-color: rgba(99,179,255,0.3);
    color: #a8d4ff;
}

.topic-icon {
    font-size: 15px;
    flex-shrink: 0;
}

.alert-box {
    margin-top: 18px;
    padding: 12px 14px;
    background: rgba(234,179,8,0.07);
    border: 1px solid rgba(234,179,8,0.2);
    border-radius: 10px;
    border-left: 3px solid rgba(234,179,8,0.55);
}

.alert-box p {
    font-size: 12px !important;
    color: rgba(252,211,77,0.8) !important;
    margin: 0 !important;
    line-height: 1.5 !important;
}

/* ── CHAT AREA ── */
.chat-card {
    background: linear-gradient(160deg, #0e1929 0%, #0a1220 100%) !important;
    border: 1px solid rgba(99,179,255,0.12) !important;
    border-radius: 18px !important;
    padding: 24px !important;
}

.chat-input-label {
    font-family: 'Sora', sans-serif !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: rgba(99,179,255,0.6) !important;
    margin-bottom: 8px !important;
}

.ask-box textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(99,179,255,0.2) !important;
    border-radius: 12px !important;
    color: rgba(220,235,255,0.9) !important;
    font-size: 14px !important;
    padding: 14px 16px !important;
    transition: border-color 0.2s !important;
    resize: none !important;
}

.ask-box textarea:focus {
    border-color: rgba(99,179,255,0.5) !important;
    box-shadow: 0 0 0 3px rgba(56,130,246,0.1) !important;
    outline: none !important;
}

.ask-box textarea::placeholder {
    color: rgba(140,170,210,0.35) !important;
}

.response-box textarea {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(99,179,255,0.1) !important;
    border-radius: 12px !important;
    color: rgba(210,230,255,0.85) !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
    padding: 14px 16px !important;
}

/* Buttons */
.btn-ask {
    background: linear-gradient(135deg, #1a56db, #1e40af) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 18px rgba(26,86,219,0.35) !important;
}

.btn-ask:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(26,86,219,0.45) !important;
}

.btn-clear {
    background: rgba(255,255,255,0.04) !important;
    color: rgba(180,210,255,0.6) !important;
    border: 1px solid rgba(99,179,255,0.15) !important;
    border-radius: 12px !important;
    font-size: 14px !important;
    transition: all 0.2s !important;
}

.btn-clear:hover {
    background: rgba(255,255,255,0.08) !important;
    color: rgba(180,210,255,0.85) !important;
}

/* Examples */
.gr-examples .gr-button {
    background: rgba(56,130,246,0.07) !important;
    border: 1px solid rgba(56,130,246,0.2) !important;
    border-radius: 8px !important;
    color: rgba(150,200,255,0.8) !important;
    font-size: 12px !important;
    padding: 6px 12px !important;
    transition: all 0.2s !important;
}

.gr-examples .gr-button:hover {
    background: rgba(56,130,246,0.15) !important;
    border-color: rgba(99,179,255,0.4) !important;
}

/* ── TABS ── */
.tabs > .tab-nav {
    background: transparent !important;
    border-bottom: 1px solid rgba(99,179,255,0.12) !important;
    padding: 0 8px !important;
    gap: 4px !important;
}

.tabs > .tab-nav button {
    font-family: 'Sora', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: rgba(140,175,220,0.55) !important;
    border-radius: 10px 10px 0 0 !important;
    padding: 10px 20px !important;
    border: none !important;
    background: transparent !important;
    transition: all 0.2s !important;
}

.tabs > .tab-nav button.selected {
    color: #89c4ff !important;
    background: rgba(56,130,246,0.1) !important;
    border-bottom: 2px solid #4a90e2 !important;
}

/* ── ACCORDION (FAQ) ── */
.gr-accordion {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(99,179,255,0.1) !important;
    border-radius: 12px !important;
    margin-bottom: 10px !important;
}

.gr-accordion .label-wrap {
    font-family: 'Sora', sans-serif !important;
    font-weight: 500 !important;
    color: rgba(180,215,255,0.85) !important;
    font-size: 14px !important;
}

/* ── FOOTER ── */
.dora-footer {
    background: linear-gradient(135deg, #080e1a 0%, #0a1220 100%);
    border-top: 1px solid rgba(99,179,255,0.1);
    border-radius: 24px 24px 0 0;
    margin-top: 32px;
    padding: 32px 40px 20px 40px;
}

.footer-grid {
    display: grid;
    grid-template-columns: 1.8fr 1fr 1fr;
    gap: 32px;
    margin-bottom: 24px;
}

.footer-col-title {
    font-family: 'Sora', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: rgba(99,179,255,0.5);
    margin-bottom: 14px;
}

.footer-about-name {
    font-family: 'Sora', sans-serif;
    font-size: 16px;
    font-weight: 700;
    color: #c8e0ff;
    margin-bottom: 4px;
}

.footer-about-role {
    font-size: 12px;
    color: rgba(140,180,230,0.55);
    margin-bottom: 12px;
    font-weight: 300;
}

.footer-about-desc {
    font-size: 12.5px;
    color: rgba(160,200,240,0.5);
    line-height: 1.65;
    font-weight: 300;
}

.footer-link {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12.5px;
    color: rgba(150,190,235,0.6);
    margin-bottom: 9px;
    text-decoration: none;
    transition: color 0.2s;
}

.footer-link:hover { color: #89c4ff; }

.footer-link .link-icon {
    width: 22px;
    height: 22px;
    border-radius: 6px;
    background: rgba(56,130,246,0.12);
    border: 1px solid rgba(56,130,246,0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    flex-shrink: 0;
}

.footer-tech-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    background: rgba(56,130,246,0.08);
    border: 1px solid rgba(56,130,246,0.15);
    border-radius: 6px;
    font-size: 11.5px;
    color: rgba(150,195,245,0.7);
    margin: 0 5px 6px 0;
}

.footer-bottom {
    border-top: 1px solid rgba(99,179,255,0.07);
    padding-top: 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.footer-copyright {
    font-size: 11.5px;
    color: rgba(120,160,210,0.35);
}

.footer-dds-brand {
    font-family: 'Sora', sans-serif;
    font-size: 11.5px;
    color: rgba(99,179,255,0.45);
    letter-spacing: 1px;
}

/* General dark fixes */
body, .gradio-container {
    background: #070d18 !important;
}

label, .gr-form label {
    color: rgba(150,195,245,0.7) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
}
"""

# ── THEME ──
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("DM Sans"), "sans-serif"],
).set(
    body_background_fill="#070d18",
    block_background_fill="#0e1929",
    block_border_color="rgba(99,179,255,0.12)",
    block_label_text_color="rgba(150,195,245,0.7)",
    input_background_fill="rgba(255,255,255,0.04)",
    input_border_color="rgba(99,179,255,0.2)",
    input_border_color_focus="rgba(99,179,255,0.5)",
    button_primary_background_fill="linear-gradient(135deg,#1a56db,#1e40af)",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="rgba(255,255,255,0.04)",
    button_secondary_text_color="rgba(180,210,255,0.7)",
    border_color_primary="rgba(99,179,255,0.15)",
    color_accent_soft="rgba(56,130,246,0.1)",
)

with gr.Blocks(css=custom_css, theme=theme) as demo:

    # ══════════════════════════════════════════
    #  BANNER / HEADER
    # ══════════════════════════════════════════
    gr.HTML("""
    <div class="dora-banner">
        <div class="dora-banner-grid"></div>
        <div class="dora-banner-glow"></div>

        <div class="dora-banner-inner">
            <div class="dora-brand">
                <div class="dora-logo-wrap">
                    <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" alt="DORA Logo"/>
                </div>
                <div class="dora-brand-text">
                    <div class="eyebrow">Decoding Data Science &nbsp;•&nbsp; Enterprise AI</div>
                    <div class="main-title">DORA &mdash; <span>DDS</span> Organizational Resources Assistant</div>
                    <div class="sub-title">AI-powered HR Policy Support &nbsp;|&nbsp; Pinecone &nbsp;+&nbsp; LlamaIndex &nbsp;+&nbsp; GPT-4o mini</div>
                </div>
            </div>
            <div class="dora-badges">
                <span class="dora-badge blue">⚡ RAG-Powered</span>
                <span class="dora-badge green">● Live</span>
                <span class="dora-badge amber">🔒 Handbook-Scoped</span>
            </div>
        </div>

        <div class="dora-divider"></div>

        <div class="dora-stats-bar">
            <div class="dora-stat">
                <span class="dot green"></span>
                <span class="stat-label">Status</span>
                <span class="stat-val">Operational</span>
            </div>
            <div class="dora-stat">
                <span class="dot blue"></span>
                <span class="stat-label">Model</span>
                <span class="stat-val">GPT-4o mini</span>
            </div>
            <div class="dora-stat">
                <span class="dot blue"></span>
                <span class="stat-label">Vector DB</span>
                <span class="stat-val">Pinecone</span>
            </div>
            <div class="dora-stat">
                <span class="dot amber"></span>
                <span class="stat-label">Scope</span>
                <span class="stat-val">DDS HR Handbook Only</span>
            </div>
        </div>
    </div>
    """)

    # ══════════════════════════════════════════
    #  TABS
    # ══════════════════════════════════════════
    with gr.Tabs():

        # ─── CHAT TAB ───
        with gr.Tab("💬  Chat Assistant"):
            with gr.Row(equal_height=True):

                # LEFT SIDEBAR
                with gr.Column(scale=1, elem_classes="sidebar-panel"):
                    gr.HTML("""
                    <p class="sidebar-section-title">📌 &nbsp;Supported Topics</p>
                    <div class="topic-chip"><span class="topic-icon">🏖️</span> Leave Policies</div>
                    <div class="topic-chip"><span class="topic-icon">🏠</span> Work From Home Rules</div>
                    <div class="topic-chip"><span class="topic-icon">🕐</span> Attendance Guidelines</div>
                    <div class="topic-chip"><span class="topic-icon">🎁</span> Employee Benefits</div>
                    <div class="topic-chip"><span class="topic-icon">📋</span> HR Procedures</div>
                    <div class="topic-chip"><span class="topic-icon">🔄</span> Probation & Onboarding</div>
                    <div class="alert-box">
                        <p>⚠️ &nbsp;This assistant exclusively answers DDS HR handbook questions. For confidential queries, email connect@decodingdatascience.com</p>
                    </div>
                    """)

                # RIGHT CHAT PANEL
                with gr.Column(scale=2, elem_classes="chat-card"):
                    question = gr.Textbox(
                        label="Ask Your HR Question",
                        placeholder="e.g. How do I apply for annual leave? What is the WFH policy?",
                        lines=3,
                        elem_classes="ask-box"
                    )

                    answer = gr.Textbox(
                        label="DORA's Response",
                        lines=10,
                        elem_classes="response-box"
                    )

                    with gr.Row():
                        ask_btn = gr.Button("🚀  Send Question", variant="primary", elem_classes="btn-ask")
                        clear_btn = gr.Button("✕  Clear", elem_classes="btn-clear")

                    gr.Examples(
                        label="Quick Questions",
                        examples=[
                            ["What is the annual leave policy?"],
                            ["How do I apply for medical leave?"],
                            ["What are the working hours?"],
                            ["What is the probation period?"],
                            ["What is the work from home policy?"],
                        ],
                        inputs=question
                    )

                    ask_btn.click(fn=query_doc, inputs=question, outputs=answer)
                    question.submit(fn=query_doc, inputs=question, outputs=answer)
                    clear_btn.click(lambda: ("", ""), outputs=[question, answer])

        # ─── FAQ TAB ───
        with gr.Tab("❓  FAQs"):
            gr.HTML('<p style="font-family:Sora,sans-serif;font-size:11px;font-weight:600;letter-spacing:2.5px;text-transform:uppercase;color:rgba(99,179,255,0.55);margin-bottom:18px;">Frequently Asked Questions</p>')

            with gr.Accordion("What topics can DORA answer?", open=True):
                gr.Markdown("DORA exclusively answers questions based on the **DDS HR Handbook** — covering leave policies, WFH rules, attendance, benefits, probation, and HR procedures.")
            with gr.Accordion("Can DORA provide salary or confidential details?", open=False):
                gr.Markdown("No. Salary ranges, performance reviews, and confidential HR data are outside DORA's scope. Please email **connect@decodingdatascience.com** for such queries.")
            with gr.Accordion("What happens if I ask something outside HR?", open=False):
                gr.Markdown("DORA will politely decline and redirect you to the appropriate HR contact, without answering out-of-scope queries.")
            with gr.Accordion("Who do I contact for urgent HR issues?", open=False):
                gr.Markdown("📧 **connect@decodingdatascience.com** — for confidential matters, escalations, or questions outside the handbook.")
            with gr.Accordion("How does DORA find answers?", open=False):
                gr.Markdown("DORA uses **Retrieval-Augmented Generation (RAG)**: your question is matched against the DDS HR Handbook stored in Pinecone vector database, and GPT-4o mini generates a precise answer from the retrieved context.")

        # ─── ABOUT TAB ───
        with gr.Tab("ℹ️  About"):
            gr.HTML("""
            <div style="max-width:720px;">
                <p style="font-family:Sora,sans-serif;font-size:11px;font-weight:600;letter-spacing:2.5px;text-transform:uppercase;color:rgba(99,179,255,0.55);margin-bottom:16px;">About This Project</p>
                <p style="font-size:20px;font-weight:600;color:#c8e0ff;font-family:Sora,sans-serif;margin-bottom:8px;">DORA — DDS Organizational Resources Assistant</p>
                <p style="font-size:14px;color:rgba(170,205,245,0.6);line-height:1.75;margin-bottom:24px;font-weight:300;">
                    An enterprise-grade AI chatbot that gives DDS employees instant, accurate access to HR policies — powered by a RAG pipeline built on LlamaIndex, Pinecone, and OpenAI.
                </p>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:24px;">
                    <div style="background:rgba(56,130,246,0.07);border:1px solid rgba(56,130,246,0.15);border-radius:14px;padding:16px;">
                        <p style="font-size:11px;letter-spacing:2px;text-transform:uppercase;color:rgba(99,179,255,0.5);font-weight:600;margin-bottom:10px;">✅ Capabilities</p>
                        <p style="font-size:13px;color:rgba(180,215,255,0.7);line-height:1.8;font-weight:300;">• Instant HR policy answers<br>• Handbook-grounded responses<br>• Semantic vector search<br>• Chain-of-thought reasoning<br>• Polite out-of-scope handling</p>
                    </div>
                    <div style="background:rgba(234,179,8,0.05);border:1px solid rgba(234,179,8,0.12);border-radius:14px;padding:16px;">
                        <p style="font-size:11px;letter-spacing:2px;text-transform:uppercase;color:rgba(252,211,77,0.5);font-weight:600;margin-bottom:10px;">⚙️ Tech Stack</p>
                        <p style="font-size:13px;color:rgba(200,220,255,0.65);line-height:1.8;font-weight:300;">• LlamaIndex RAG pipeline<br>• Pinecone vector database<br>• GPT-4o mini (OpenAI)<br>• text-embedding-ada-002<br>• Gradio UI + HuggingFace</p>
                    </div>
                </div>
            </div>
            """)

    # ══════════════════════════════════════════
    #  FOOTER
    # ══════════════════════════════════════════
    gr.HTML("""
    <div class="dora-footer">
        <div class="footer-grid">

            <!-- COL 1: Developer -->
            <div>
                <p class="footer-col-title">Developer</p>
                <p class="footer-about-name">Nipun Kavinda</p>
                <p class="footer-about-role">AI & Data Science Engineer</p>
                <p class="footer-about-desc">
                    Building intelligent enterprise solutions at the intersection of LLMs, RAG pipelines, and production-grade AI systems.
                    Passionate about making AI practical and accessible.
                </p>
                <div style="margin-top:14px;display:flex;gap:8px;flex-wrap:wrap;">
                    <span class="footer-tech-pill">🐍 Python</span>
                    <span class="footer-tech-pill">🦙 LlamaIndex</span>
                    <span class="footer-tech-pill">🌲 Pinecone</span>
                    <span class="footer-tech-pill">🤖 OpenAI</span>
                </div>
            </div>

            <!-- COL 2: Project -->
            <div>
                <p class="footer-col-title">Project</p>
                <a class="footer-link" href="#">
                    <span class="link-icon">📁</span> DORA v1.0 — HR Chatbot
                </a>
                <a class="footer-link" href="#">
                    <span class="link-icon">🔗</span> HuggingFace Space
                </a>
                <a class="footer-link" href="#">
                    <span class="link-icon">📄</span> DDS HR Handbook
                </a>
                <a class="footer-link" href="mailto:connect@decodingdatascience.com">
                    <span class="link-icon">✉️</span> connect@decodingdatascience.com
                </a>
            </div>

            <!-- COL 3: Organization -->
            <div>
                <p class="footer-col-title">Organization</p>
                <p style="font-family:Sora,sans-serif;font-size:14px;font-weight:600;color:#a8ceff;margin-bottom:4px;">Decoding Data Science</p>
                <p style="font-size:12px;color:rgba(140,180,230,0.45);margin-bottom:12px;font-weight:300;">Enterprise AI Division</p>
                <a class="footer-link" href="https://decodingdatascience.com" target="_blank">
                    <span class="link-icon">🌐</span> decodingdatascience.com
                </a>
                <a class="footer-link" href="mailto:connect@decodingdatascience.com">
                    <span class="link-icon">📧</span> HR Support Email
                </a>
                <div style="margin-top:14px;padding:10px 13px;background:rgba(34,197,94,0.07);border:1px solid rgba(34,197,94,0.15);border-radius:9px;font-size:11.5px;color:rgba(110,231,183,0.7);">
                    ● &nbsp;System Operational &nbsp;|&nbsp; v1.0.0
                </div>
            </div>
        </div>

        <div class="footer-bottom">
            <span class="footer-copyright">© 2025 Nipun Kavinda · Built for Decoding Data Science Internal HR Support</span>
            <span class="footer-dds-brand">DDS · DORA · Enterprise AI</span>
        </div>
    </div>
    """)

demo.launch()