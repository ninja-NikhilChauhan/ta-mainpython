import os
import zipfile
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv
import logging
from typing import Dict, List
from pydantic import BaseModel
import json
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tafrontendnext-cvy8q7djc-nikhil-chauhans-projects-f205663d.vercel.app"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHROMA_DB_PATH = os.path.abspath("./chroma_db")
EMBEDDING_MODEL = "text-embedding-3-large"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class ScoringPattern(BaseModel):
    component: str
    max_score: int

class AnalysisRequest(BaseModel):
    project_about: str
    technology: str
    problem_statement: str
    scoring_pattern: List[ScoringPattern]

def get_chroma_client():
    return PersistentClient(path=CHROMA_DB_PATH)

def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)

def setup_chromadb(project_path: str) -> int:
    """Process project files and store in ChromaDB"""
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name="code_analysis",
        embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )
    
    code_files = []
    supported_extensions = ('.html', '.js', '.jsx', '.ts', '.tsx', '.css', '.py', '.java', '.php')
    excluded_dirs = {'node_modules', 'test', '__tests__', 'dist', 'build', '.git', '.github'}
    
    for root, dirs, files in os.walk(project_path):
        dirs[:] = [d for d in dirs if d not in excluded_dirs]

        for file in files:
            if file.lower().endswith(supported_extensions):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        rel_path = os.path.relpath(file_path, project_path)
                        code_files.append({
                            'path': rel_path,
                            'content': content
                        })
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")
                    continue
    
    if not code_files:
        return 0
    
    documents = [f"{f['path']}\n{f['content']}" for f in code_files]
    metadatas = [{"path": f["path"]} for f in code_files]
    ids = [f"id_{i}" for i in range(len(code_files))]
    
    try:
        collection.delete(ids=ids)
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Added {len(code_files)} documents to ChromaDB")
        return len(code_files)
    except Exception as e:
        logger.error(f"Error adding to ChromaDB: {e}")
        return 0

def generate_scoring_prompt_section(scoring_pattern: List[ScoringPattern]) -> str:
    """Generate the scoring section of the prompt"""
    prompt_section = "Scoring Breakdown (Total must sum to 100):\n"
    for item in scoring_pattern:
        prompt_section += f"- {item.component}: {item.max_score} points\n"
    prompt_section += "\nEvaluate each component and deduct points for missing or incomplete features."
    return prompt_section

def count_tokens(text: str) -> int:
    """Estimate token count (approximate)"""
    return int(len(text.split()) * 1.33)  # Rough estimate (1 token ~= 1.33 words)

def analyze_with_ai(project_path: str, analysis_request: AnalysisRequest) -> dict:
    """Analyze the project using OpenAI with two-phase validation"""
    client = get_openai_client()
    collection = get_chroma_client().get_collection("code_analysis")
    
    # Get all files ranked by importance
    all_files = collection.query(
        query_texts=[f"Most important files for {analysis_request.technology} project about {analysis_request.project_about}"],
        n_results=collection.count(),
        include=["documents", "metadatas", "distances"]
    )
    
    # Process files by importance (lower distance = more relevant)
    ranked_files = sorted(
        zip(all_files['documents'][0], all_files['metadatas'][0], all_files['distances'][0]),
        key=lambda x: x[2]  # Sort by distance (relevance)
    )
    
    # Select files until we reach token limit
    selected_files = []
    total_tokens = 0
    max_tokens = 20000  # Adjust as needed
    
    for doc, meta, distance in ranked_files:
        doc_tokens = count_tokens(doc)
        if total_tokens + doc_tokens > max_tokens:
            break
        selected_files.append((doc, meta))
        total_tokens += doc_tokens
    
    if not selected_files:
        return {
            "status": "rejected",
            "reasons": ["No relevant code files found"],
            "error_locations": [],
            "score": 0
        }
    
    context = "\n\n".join([
        f"=== FILE: {meta['path']} ===\n{doc}"
        for doc, meta in selected_files
    ])
    
    # Phase 1: Strict Validation
    phase1_prompt = f"""
    STRICT VALIDATION REQUIREMENTS:
    1. Project must be about: {analysis_request.project_about}
    2. Must use technology: {analysis_request.technology}
    3. Must have no syntax errors or incomplete code or pass:false if any code is missing or incomplete or if the point number 1 or 2 is not met.
    4. Do not give pass: false if the code is not well maintained or not well structured, I want to check if the code is compilable or not. Not looking for well maintained or well structured code.
    5. Do not give pass: false if the code has a minor syntax error or typo, let it pass.
    6. If the code works and is not a syntax error, then pass: true.
    
    CODE CONTEXT:
    {context}
    
    INSTRUCTIONS:
    Analyze the code and answer ONLY in this JSON format:
    {{
        "pass": boolean,
        "reasons": list[str],
        "error_locations": list[str]  // file:line if available
    }}
    
    - "pass" should be true ONLY if ALL requirements are met
    - "reasons" should explain each failure
    - Be extremely strict - any doubt means failure
    """
    
    phase1_response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": phase1_prompt}],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    
    validation = json.loads(phase1_response.choices[0].message.content)
    
    if not validation["pass"]:
        return {
            "status": "rejected",
            "reasons": validation["reasons"],
            "error_locations": validation.get("error_locations", []),
            "score": 0
        }
    
    # Phase 2: Detailed Evaluation
    scoring_prompt = generate_scoring_prompt_section(analysis_request.scoring_pattern)
    
    phase2_prompt = f"""
    DETAILED PROJECT EVALUATION:
    - Problem Statement: {analysis_request.problem_statement}
    - Scoring Criteria: {scoring_prompt}
    
    CODE CONTEXT:
    {context}
    
    INSTRUCTIONS:
    1. Evaluate each scoring component strictly
    2. Deduct points for any incomplete features
    3. If doesn_it_exist is false, return 0 score
    4. Provide specific feedback for each component
    5. Calculate total score (0-100), which should also be equal to the sum of all individual component scores
    6. VERY IMPORTANT : The problem statement might contain some html tags or other tags, so please ignore them and only focus on the text. 
    7. VERY IMPORTANT : Ignore the parts in the problem statement where it asks to consult an image, video or any other external source like links.
    
    RESPONSE FORMAT (JSON):
    {{
        "does_it_exist?": boolean,
        "score": int,
        "component_evaluations": [
            {{
                "component": str,
                "score": int,
                "feedback": str,
                "suggestions": str
            }}
        ],
        "overall_feedback": str
    }}
    """
    
    phase2_response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": phase2_prompt}],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    
    evaluation = json.loads(phase2_response.choices[0].message.content)
    
    return {
        "status": "evaluated",
        "score": evaluation["score"],
        "component_evaluations": evaluation["component_evaluations"],
        "overall_feedback": evaluation["overall_feedback"]
    }

def cleanup_chromadb():
    """Clean up ChromaDB after analysis"""
    try:
        client = get_chroma_client()
        client.delete_collection("code_analysis")
        logger.info("Cleaned up ChromaDB collection")
    except Exception as e:
        logger.error(f"Error cleaning up ChromaDB: {e}")

@app.post("/analyze-project")
async def analyze_project(
    zip_file: UploadFile = File(...),
    project_about: str = Form(...),
    technology: str = Form(...),
    problem_statement: str = Form(...),
    scoring_pattern: str = Form(...)
):
    try:
        # Parse and validate scoring pattern
        try:
            scoring_data = json.loads(scoring_pattern)
            scoring_objects = [ScoringPattern(**item) for item in scoring_data]
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid scoring pattern format")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error parsing scoring pattern: {str(e)}")

        total_score = sum(item.max_score for item in scoring_objects)
        if total_score != 100:
            raise HTTPException(status_code=400, detail="Scoring pattern must sum to 100")

        # Construct analysis request
        analysis_request = AnalysisRequest(
            project_about=project_about,
            technology=technology,
            problem_statement=problem_statement,
            scoring_pattern=scoring_objects
        )

        # Prepare temp directory
        temp_dir = "temp_project"
        os.makedirs(temp_dir, exist_ok=True)
        zip_path = os.path.join(temp_dir, zip_file.filename)

        # Save zip file
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(zip_file.file, buffer)

        # Extract zip contents safely
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                if "__MACOSX" in member:
                    continue
                zip_ref.extract(member, temp_dir)

        # Get valid project directory
        contents = [item for item in os.listdir(temp_dir) if item != "__MACOSX" and item != zip_file.filename]
        project_dir = temp_dir
        if len(contents) == 1 and os.path.isdir(os.path.join(temp_dir, contents[0])):
            project_dir = os.path.join(temp_dir, contents[0])

        # Setup and analyze
        file_count = setup_chromadb(project_dir)
        if file_count == 0:
            raise HTTPException(status_code=400, detail="No relevant code files found")

        analysis = analyze_with_ai(project_dir, analysis_request)

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        cleanup_chromadb()

        return JSONResponse({
            "status": "success",
            "message": f"Processed {file_count} files",
            "analysis": analysis,
            "evaluation_criteria": {
                "project_about": project_about,
                "technology": technology,
                "problem_statement": problem_statement,
                "scoring_pattern": [item.dict() for item in scoring_objects]
            }
        })

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in analyze_project: {str(e)}")
        shutil.rmtree("temp_project", ignore_errors=True)
        cleanup_chromadb()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)