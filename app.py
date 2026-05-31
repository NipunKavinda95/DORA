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

def query_doc(prompt):
    try:
        response = query_engine.query(prompt)
        return str(response)
    except Exception as e:
        return f"Error: {str(e)}"


css = """
/* Page background */
body, .gradio-container, gradio-app, .app {
    background: #0a0b10 !important;
}

/* Header area */
.gradio-container h1 {
    font-size: 25px !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #897ef9, #c4bbff) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    text-align: center !important;
    margin-bottom: 2px !important;
}
.gradio-container p {
    color: rgba(255,255,255,0.3) !important;
    font-size: 16px !important;
    text-align: center !important;
}

/* Chat window */
#chatbot {
    background: #0a0b10 !important;
    border: none !important;
    flex: 1 !important;
    min-height: 420px !important;
}
#chatbot .wrap { background: transparent !important; }

/* User bubble */
#chatbot [data-testid="user"] > div,
.message.user > div {
    background: #4f3ef5 !important;
    color: #fff !important;
    border-radius: 18px 4px 18px 18px !important;
    border: none !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
    max-width: 75% !important;
    margin-left: auto !important;
}
/* Force all text inside user bubble white */
#chatbot [data-testid="user"] > div *,
#chatbot [data-testid="user"] p,
#chatbot [data-testid="user"] span,
#chatbot [data-testid="user"] li,
.message.user > div *,
.message.user p {
    color: #fff !important;
}

/* Bot bubble */
#chatbot [data-testid="bot"] > div,
.message.bot > div {
    background: #141520 !important;
    color: rgba(255,255,255,0.82) !important;
    border-radius: 4px 18px 18px 18px !important;
    border: 0.5px solid rgba(255,255,255,0.07) !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
    max-width: 75% !important;
}
/* Force all text inside bot bubble to be white */
#chatbot [data-testid="bot"] > div *,
#chatbot [data-testid="bot"] p,
#chatbot [data-testid="bot"] span,
#chatbot [data-testid="bot"] li,
#chatbot [data-testid="bot"] ol,
#chatbot [data-testid="bot"] ul,
.message.bot > div *,
.message.bot p {
    color: rgba(255,255,255,0.82) !important;
}

/* Avatar icons */
.avatar-container { display: none !important; }

/* Input row at bottom */
.input-row, [data-testid="textbox"] {
    background: #12131e !important;
    border: 0.5px solid rgba(255,255,255,0.1) !important;
    border-radius: 14px !important;
    padding: 4px 8px !important;
}
textarea.scroll-hide {
    background: transparent !important;
    border: none !important;
    color: rgba(255,255,255,0.82) !important;
    font-size: 14px !important;
    padding: 10px 12px !important;
    min-height: 44px !important;
    max-height: 120px !important;
}
textarea.scroll-hide::placeholder { color: rgba(255,255,255,0.2) !important; }
textarea.scroll-hide:focus { outline: none !important; box-shadow: none !important; }

/* Send button */
#submit-btn, button[aria-label="Submit"] {
    background: #4f3ef5 !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    width: 40px !important;
    height: 40px !important;
}
#submit-btn:hover { background: #6054f6 !important; }

/* Clear/Retry buttons */
.btn-base, button[aria-label="Clear"], button[aria-label="Retry"] {
    background: transparent !important;
    border: 0.5px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    color: rgba(255,255,255,0.3) !important;
    font-size: 12px !important;
}
.btn-base:hover {
    background: rgba(255,255,255,0.04) !important;
    color: rgba(255,255,255,0.6) !important;
}

/* Bottom bar container */
.bottom-btns { background: transparent !important; border: none !important; }

footer, .built-with { display: none !important; }
"""

demo = gr.ChatInterface(
    fn=query_doc,
    title="DORA — DDS Organizational Resource Assistant",
    description="Decoding Data Science · Ask about leave, policies, benefits, and more",
    chatbot=gr.Chatbot(height=460),
    textbox=gr.Textbox(
        placeholder="Type your HR question here..."
    examples=[
        "What are the standard working hours in Dubai?",
        "How do I apply for annual leave?",
        "What is the remote work policy?",
        "Tell me about the employee benefits package",
    ],
    css=css,
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.gray,
    ).set(
        body_background_fill="#0a0b10",
        body_background_fill_dark="#0a0b10",
        block_background_fill="transparent",
        block_background_fill_dark="transparent",
        block_border_width="0px",
        input_background_fill="#12131e",
        input_background_fill_dark="#12131e",

    ),
)

if __name__ == "__main__":
    demo.launch()