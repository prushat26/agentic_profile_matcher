import os
import glob
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import OpenAI
from pypdf import PdfReader
import chromadb

# 1. Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = "gpt-4o-mini"

# 2. Define Extraction Schema using Pydantic
class ResumeSchema(BaseModel):
    candidate_name: str = Field(description="Full name of candidate")
    skills: list[str] = Field(description="List of technical and soft skills")
    years_experience: float = Field(description="Total years of work experience")
    education_level: str = Field(description="Highest degree obtained")
    experience_summary: str = Field(description="Detailed summary of roles, experience, and key projects")

# 3. PDF Text Extractor
def extract_pdf_text(filepath: str) -> str:
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

# 4. Structured Parsing via Native OpenAI Pydantic Integration
def parse_resume(raw_text: str) -> ResumeSchema:
    completion = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Extract candidate details from the resume text into the requested JSON schema."},
            {"role": "user", "content": raw_text}
        ],
        response_format=ResumeSchema,
        temperature=0
    )
    return completion.choices[0].message.parsed

# 5. Build Vector Store (ChromaDB)
def build_vector_db(resumes_dir: str = "resumes"):
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="resumes")
    
    pdf_files = glob.glob(os.path.join(resumes_dir, "*.pdf"))
    if not pdf_files:
        print("⚠️ No PDF files found in the 'resumes' directory. Please add some PDFs!")
        return

    print(f"Processing {len(pdf_files)} resumes...")
    
    for idx, filepath in enumerate(pdf_files):
        print(f"[{idx+1}/{len(pdf_files)}] Ingesting {filepath}...")
        raw_text = extract_pdf_text(filepath)
        parsed = parse_resume(raw_text)
        
        # Prepare Chroma Payload
        metadata = {
            "candidate_name": parsed.candidate_name,
            "years_experience": parsed.years_experience,
            "skills_str": ", ".join(parsed.skills),
            "education_level": parsed.education_level,
            "resume_path": filepath
        }
        
        collection.add(
            documents=[parsed.experience_summary],
            metadatas=[metadata],
            ids=[f"resume_{idx}"]
        )
    print("✅ Indexing Complete!")

if __name__ == "__main__":
    build_vector_db()