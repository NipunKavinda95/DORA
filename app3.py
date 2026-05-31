# If needed in Colab, install first:
# !pip install -U gradio pinecone llama-index llama-index-vector-stores-pinecone llama-index-readers-file pypdf
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
# --- Imports ---
import logging
import sys
import gradio as gr
import os 

from pinecone import Pinecone, ServerlessSpec
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext , Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.readers.file import PDFReader
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
# --- Logging ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


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


# --- Load API Key from Hugging face environment ---

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


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
footer {display:none !important;}

.main-container {
    max-width: 1200px;
    margin: auto;
}

.header-card {
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 15px;
}

.logo {
    width: 70px;
    margin-bottom: 10px;
}

.title {
    font-size: 30px;
    font-weight: 700;
}

.subtitle {
    font-size: 15px;
    opacity: 0.8;
}

.section-card {
    padding: 18px;
    border-radius: 14px;
    margin-top: 10px;
}

.ask-box textarea {
    border-radius: 12px !important;
}

.gr-button {
    border-radius: 10px !important;
}
"""

# Use modern theme (auto light/dark based on system preference)
theme = gr.themes.Soft()

with gr.Blocks(css=custom_css, theme=theme) as demo:

    with gr.Column(elem_classes="main-container"):

        # ================= HEADER =================
        gr.HTML("""
        <div class="header-card">
            <img class="logo" src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png">
            <div class="title">DORA – DDS Organizational Resources Assistant</div>
            <div class="subtitle">AI-powered HR Policy Support System</div>
        </div>
        """)

        # ================= TABS =================
        with gr.Tabs():

            # ================= CHAT TAB =================
            with gr.Tab("💬 Chat Assistant"):

                with gr.Row():

                    # LEFT PANEL (Info Panel)
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        ### 📌 Supported Topics
                        - Leave Policies  
                        - Work From Home Rules  
                        - Attendance Guidelines  
                        - Employee Benefits  
                        - HR Procedures  

                        ### ⚠️ Note
                        This assistant only answers DDS HR handbook questions.
                        """)

                    # RIGHT PANEL (Chat UI)
                    with gr.Column(scale=2):

                        question = gr.Textbox(
                            label="Ask Your Question",
                            placeholder="Type your HR question here...",
                            lines=2,
                            elem_classes="ask-box"
                        )

                        answer = gr.Textbox(
                            label="DDS HR Response",
                            lines=10
                        )

                        with gr.Row():
                            ask_btn = gr.Button("🚀 Ask", variant="primary")
                            clear_btn = gr.Button("🧹 Clear")

                        gr.Examples(
                            examples=[
                                ["What is the annual leave policy?"],
                                ["How do I apply for medical leave?"],
                                ["What are the working hours?"],
                                ["What is the probation period?"]
                            ],
                            inputs=question
                        )

                        ask_btn.click(
                            fn=query_doc,
                            inputs=question,
                            outputs=answer
                        )

                        question.submit(
                            fn=query_doc,
                            inputs=question,
                            outputs=answer
                        )

                        clear_btn.click(
                            lambda: ("", ""),
                            outputs=[question, answer]
                        )

            # ================= FAQ TAB =================
            with gr.Tab("❓ FAQs"):

                gr.Markdown("### Frequently Asked Questions")

                with gr.Accordion("What can this chatbot answer?", open=True):
                    gr.Markdown("It only answers DDS HR handbook-related questions such as leave, attendance, and benefits.")

                with gr.Accordion("Can it provide salary details?", open=False):
                    gr.Markdown("No. Salary and confidential data are not available.")

                with gr.Accordion("What if I ask something outside HR?", open=False):
                    gr.Markdown("It will politely refuse and redirect you to HR email support.")

                with gr.Accordion("Who do I contact for HR issues?", open=False):
                    gr.Markdown("Please contact: connect@decodingdatascience.com")

            # ================= ABOUT TAB =================
            with gr.Tab("ℹ️ About"):

                gr.Markdown("""
                ## DORA Assistant
                **DDS Organizational Resources Assistant (DORA)** is an AI-powered HR chatbot designed to help employees quickly access HR policies.

                ### Features
                - Instant HR policy answers  
                - Handbook-based responses only  
                - Secure and restricted knowledge scope  
                - Enterprise-grade compliance  

                ### Limitations
                - No access to salary or confidential data  
                - No general company or technical knowledge outside HR  

                ---
                Built for DDS Internal HR Support System
                """)

demo.launch()