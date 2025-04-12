# AI-PDF-Question Generator

A Flask web application that uses Mistral AI to generate university-level, case-based questions from PDF documents.

## Features

- Upload PDF documents through a simple web interface
- Generate case-based questions requiring critical thinking
- Store questions in a PostgreSQL database
- Retrieve previously generated questions for the same document

## Technologies Used

- Flask - Web framework
- MistralAI - Large language model for question generation
- PgVector - PostgreSQL extension for vector similarity search
- Phidata - Framework for AI applications
- SQLAlchemy - ORM for database operations

## Setup

1. Clone the repository:

```bash
git clone https://github.com/YourUsername/AI-PDF-Question.git
cd AI-PDF-Question
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up environment variables in a `.env` file:

```
MISTRAL_API_KEY=your_mistral_api_key
DB_URL=postgresql://username:password@localhost:5432/dbname
```

4. Run the application:

```bash
python main.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Database Setup

The application requires a PostgreSQL database with the pgvector extension installed. Run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS ai;
```

The application will create the necessary tables on startup.

## License

MIT License