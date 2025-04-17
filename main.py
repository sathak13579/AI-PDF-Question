from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.pgvector import PgVector
from phi.knowledge.pdf import PDFReader
from phi.agent import Agent
from phi.model.mistral import MistralChat
import os
from dotenv import load_dotenv
from markdown import markdown
from flask import Flask, render_template, request, redirect, url_for, flash
import tempfile
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, Text, DateTime, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from phi.embedder.mistral import MistralEmbedder
from phi.storage.agent.sqlite import SqlAgentStorage
import uuid
import json
import hashlib
from textwrap import dedent

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for flashing messages

# Add markdown filter for templates
app.jinja_env.filters['markdown'] = markdown

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GROG_API_KEY = os.getenv("GROG_API_KEY")

db_url = os.getenv("DB_URL")

# Initialize database and create table if it doesn't exist
def init_db():
    engine = create_engine(db_url)
    metadata = MetaData()
    
    # Define the table using SQLAlchemy ORM
    pdf_documents = Table(
        'pdf_documents', 
        metadata,
        Column("id", String, primary_key=True),
        Column("name", String),
        Column("meta_data", JSONB, server_default=text("'{}'::jsonb")),
        Column("filters", JSONB, server_default=text("'{}'::jsonb"), nullable=True),
        Column("content", Text),
        # Column("embedding", Vector(1536)), #Open ai embedding size
        Column("embedding", Vector(1024)), #Mistral embedding size
        Column("usage", JSONB),
        Column("created_at", DateTime(timezone=True), server_default=func.now()),
        Column("updated_at", DateTime(timezone=True), onupdate=func.now()),
        Column("content_hash", String),
        extend_existing=True,
        schema='ai'
    )
    
    with engine.connect() as conn:
        # Enable vector extension
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        
        # Create schema if it doesn't exist
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS ai"))
        
        # Create the table with proper structure
        metadata.create_all(engine)
        conn.commit()

# Initialize the database on startup
init_db()

def create_agent(pdf_path, existing_questions=None):

    # storage = SqlAgentStorage(
    #     # store sessions in the ai.sessions table
    #     table_name="agent_sessions",
    #     # db_file: Sqlite database file
    #     db_file="tmp/data.db",
    # )

    knowledge_base = PDFKnowledgeBase(
        path=pdf_path,
        vector_db=PgVector(
            table_name="pdf_documents",
            db_url=db_url,
            embedder=MistralEmbedder(), #uncomment this for open ai and change  embedding size
        ),
        reader=PDFReader(chunk=True),
    )

    expected_output = dedent("""\
    1.  **Case:** A software development team, following the iterative model described in the document, discovers a major flaw in the core architecture during the third iteration. This flaw impacts features developed in previous iterations.
        **Question:** According to the principles discussed for handling setbacks in iterative development, what is the most appropriate immediate action for the team?
        **Options:**
        A. Continue the current iteration as planned and fix the flaw in a later iteration.
        B. Halt the current iteration, analyze the flaw's impact, adjust the plan, and potentially revisit previous work before proceeding.
        C. Discard all previous work and restart the project from the first iteration with a corrected architecture.
        D. Assign the flaw fixing to a specialized sub-team without altering the main team's iteration plan.
        **Answer:** Correct Answer: B

    2.  **Case:** The document outlines several project estimation techniques. A project manager needs to estimate the effort for a novel project with many unknown factors and requires input from multiple experts.
        **Question:** Which estimation technique described in the text would be most suitable for achieving a consensus estimate in this situation?
        **Options:**
        A. Analogous Estimating - Using historical data from similar projects.
        B. Parametric Estimating - Using statistical relationships between historical data and variables.
        C. Delphi Technique - Iteratively and anonymously collecting expert opinions until consensus emerges.
        D. Bottom-Up Estimating - Estimating individual work components and summing them up.
        **Answer:** Correct Answer: C

    3.  **Case:** [Insert another plausible scenario based on the document's content here...]
        **Question:** [Insert a relevant question applying document concepts to the scenario here...]
        **Options:**
        A. [Plausible Option A]
        B. [Plausible Option B - The Correct Answer]
        C. [Plausible Option C]
        D. [Plausible Option D]
        **Answer:** Correct Answer: B
    """)

    if existing_questions:
        knowledge_base.load_text('These are the questions that you have already generated, Do not generate the same or similar questions: ' + '\n' + existing_questions)


    knowledge_base.load(recreate=False)
    agent_instructions = [
    "1. You are an AI assistant specializing in creating educational assessment materials.",
    "2. You will be provided with text content extracted from a PDF document (referred to as 'the document' or 'the text' internally) as your primary context.",
    "3. Your main objective is to generate meaningful CASE-BASED questions based SOLELY on the provided text content.",
    "4. The questions must be suitable for university-level students, requiring critical thinking, application of knowledge, analysis, and potentially evaluation or synthesis, not just simple recall.",
    "5. Define a 'Case-Based Question': It presents a specific scenario, situation, problem, or narrative (the 'case') grounded in the document's content.",
    "6. The question must require the student to APPLY knowledge from the document to analyze the case, solve a problem, make a decision, predict an outcome, or evaluate the situation described in the case.",
    "7. Focus on questions like 'Given scenario A, how does concept X apply?' or 'Analyze situation B using the framework from the text,' NOT simple recall like 'What is X?'.",
    "8. Ensure EVERY question is strictly case-based, presenting a scenario demanding application of the text's content.",
    "9. Each question and its scenario must be directly derivable from and answerable using ONLY the information within the provided document text. Do not introduce external information.",
    "10. Questions should reflect university-level complexity, challenging students to think critically and apply concepts in nuanced ways.",
    "11. Ensure the set of questions covers diverse sections, topics, theories, or data points presented throughout the document.",
    "12. Word all questions clearly and unambiguously.",
    "13. Scenarios, while potentially hypothetical, must be plausible within the context of the document's subject matter.",
    "14. IMPORTANT: Do NOT preface the questions with phrases like 'Based on the lecture notes,', 'According to the text,', 'Using the information provided,', or any similar reference to the source document. The questions should stand alone, assuming the context of the document is implicit.",
    "15. Format the final output as a numbered list using Markdown.", 
    "16. To generate questions: First, identify key concepts/principles/data in the text. Then, construct a relevant scenario for each. Finally, formulate a question requiring application of the text's concepts to that scenario.",
    "17. Do not repeat questions from the existing questions in the storage."
    ]


    return Agent(
        knowledge=knowledge_base,
        search_knowledge=True,
        description="This agent can understand the content of a PDF and generate meaningful questions.",
        instructions=agent_instructions,
        markdown=True,
        show_tool_calls=True,
        # storage=storage,
        read_chat_history=True,
        expected_output=expected_output
    )

def format_response_html(response):
    """Extract and format the assistant's response for HTML display."""
    try:
        if hasattr(response, "messages") and isinstance(response.messages, list):
            # Extract only the assistant messages and filter out None values
            assistant_messages = [
                msg.content if msg.content is not None else ""
                for msg in response.messages if msg.role == "assistant"
            ]
            formatted_response = "\n\n".join(assistant_messages).strip() or "⚠️ No assistant response."
            # Convert Markdown to HTML
            formatted_response = markdown(formatted_response)
    
        elif isinstance(response, str):
            formatted_response = markdown(response)  # Convert Markdown to HTML
    
        else:
            formatted_response = "<p>⚠️ Unexpected response format.</p>"
    
        return formatted_response
    
    except Exception as e:
        return f"<p>⚠️ Error formatting response: {e}</p><p>{response}</p>"

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if not file.filename.lower().endswith('.pdf'):
        flash('Please upload a PDF file')
        return redirect(url_for('index'))
    
    try:
        # Save the uploaded file to a temporary location
        temp_dir = tempfile.mkdtemp()
        pdf_path = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(pdf_path)
        
        # Compute a hash for the PDF file
        with open(pdf_path, "rb") as f:
            file_content_bytes = f.read()
        content_hash = hashlib.md5(file_content_bytes).hexdigest()
        
        # Read the content from the PDF so that it gets stored (adjust as needed for your PDFReader)
        pdf_reader = PDFReader(chunk=False)
        document_content = pdf_reader.read(pdf_path)
        
        # Convert the document_content to a plain string if necessary.
        # If it's a list of Document objects, extract the text from the .content attribute.
        if isinstance(document_content, list):
            extracted_text = "\n\n".join(
                [str(doc.content) if doc.content is not None else "" for doc in document_content if hasattr(doc, "content")]
            )

        elif hasattr(document_content, "content"):
            extracted_text = document_content.content
        elif isinstance(document_content, str):
            extracted_text = document_content
        else:
            extracted_text = ""
        
        # Connect to the database and check if questions for this PDF exist
        engine = create_engine(db_url)
        existing_questions = None
        with engine.connect() as conn:
            select_query = text("""
                SELECT usage 
                FROM ai.pdf_documents 
                WHERE content_hash = :hash
                LIMIT 1
            """)
            result = conn.execute(select_query, {"hash": content_hash})
            row = result.fetchone()
            # Expect the "usage" column to be a JSON/dict already if present
            if row and row._mapping['usage'] and 'questions' in row._mapping['usage']:
                existing_questions = row._mapping['usage']['questions']
        
        # if existing_questions:
        #     # Clean up temporary file
        #     os.remove(pdf_path)
        #     os.rmdir(temp_dir)
            
        #     formatted_questions = format_response_html(existing_questions)
        #     flash('Questions retrieved from database.')
        #     return render_template('questions.html', questions=formatted_questions)
        
        # If no existing questions, create agent and generate new ones
        print(pdf_path)
        agent = create_agent(pdf_path, existing_questions)
        instruction = "Generate  10 Case-Based MCQ questions from the knowledge base. Format them in markdown with proper numbering."
        
        response = agent.run(instruction)
        
        # Extract the raw question text from the response
        if isinstance(response, str):
            raw_questions = response
        elif hasattr(response, "messages") and isinstance(response.messages, list):
            raw_questions = "\n\n".join(
            [str(msg.content) for msg in response.messages if msg.role == "assistant" and msg.content is not None]
            )
        else:
            raw_questions = str(response)
        
        # Save the generated questions along with the document content and content hash in the database
        new_id = str(uuid.uuid4())
        with engine.connect() as conn:
            insert_query = text("""
                INSERT INTO ai.pdf_documents (id, name, usage, content_hash, content)
                VALUES (:id, :name, :usage, :hash, :content)
            """)
            conn.execute(insert_query, {
                "id": new_id,
                "name": file.filename,
                "usage": json.dumps({"questions": raw_questions}),
                "hash": content_hash,
                "content": extracted_text  # now a plain string
            })
            conn.commit()
        
        # Clean up temporary file
        os.remove(pdf_path)
        os.rmdir(temp_dir)
        
        formatted_questions = format_response_html(raw_questions)
        return render_template('questions.html', questions=formatted_questions)
    
    except Exception as e:
        flash(f'Error processing file: {str(e)}')
        return redirect(url_for('index'))

@app.route('/view_questions', methods=['GET'])
def view_questions():
    try:
        # Connect to the database and fetch all documents with questions
        engine = create_engine(db_url)
        questions_data = []
        
        with engine.connect() as conn:
            select_query = text("""
                SELECT name, usage, content_hash 
                FROM ai.pdf_documents 
                WHERE usage->'questions' IS NOT NULL
                ORDER BY created_at DESC
            """)
            result = conn.execute(select_query)
            
            for row in result:
                if row._mapping['usage'] and 'questions' in row._mapping['usage']:
                    questions_data.append({
                        'pdf_name': row._mapping['name'],
                        'questions': row._mapping['usage']['questions'],
                        'content_hash': row._mapping['content_hash']
                    })
        
        return render_template('view_questions.html', questions_data=questions_data)
    
    except Exception as e:
        flash(f'Error retrieving questions: {str(e)}')
        return redirect(url_for('index'))

# def save_questions_to_db(questions, file):
#     if isinstance(questions, str):
#         raw_questions = questions
#     elif hasattr(questions, "messages") and isinstance(questions.messages, list):
#         raw_questions = "\n\n".join([msg.content for msg in questions.messages if msg.role == "assistant"])
#     else:
#         raw_questions = str(questions)
        
#     # Save the generated questions to the database
#     new_id = str(uuid.uuid4())
#     engine = create_engine(db_url)
#     with engine.connect() as conn:
#         insert_query = text("""
#             INSERT INTO ai.pdf_documents (id, name, usage)
#             VALUES (:id, :name, :usage)
#             """)
#         conn.execute(insert_query, {
#             "id": new_id,
#             "name": file.filename,
#             "usage": json.dumps({"questions": raw_questions})
#         })
#         conn.commit()

if __name__ == '__main__':
    app.run(debug=True, port=5001)

