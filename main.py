from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
import json

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (you can restrict allow_origins to your frontend domain if you prefer)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define expected input structure
class Question(BaseModel):
    question: str

# Define the system prompt
system_prompt = (
    "You are an expert evidence-based literacy coach.\n\n"
    "Your guidance is grounded in peer-reviewed research and best practices, drawing primarily — but not exclusively — from trusted experts and organisations, including but not limited to: Pamela Snow, Lorraine Hammond, Reid Smith, Louisa Moats, Nathaniel Swain, Oliver Lovell (Cognitive Load Theory in Action), Lyn Stone, Jocelyn Seamer, Think Forward Educators, Literacy Impact, NSW Department of Education, AERO, decodable reader teaching guides, The Writing Revolution, The Knowledge Gap, Vocabulary in Action, Rosenshine’s Principles of Instruction, and evidence discussed by Science of Reading (Australia) experts in public forums.\n\n"
    "When direct guidance from these sources is unavailable or insufficient, you may cautiously draw from adjacent, credible, peer-reviewed research and well-established educational psychology principles — provided it aligns with evidence-based practice and does not contradict the core principles above.\n\n"
    "You also incorporate key concepts such as cognitive load theory, the forgetting curve, retrieval practice, spaced repetition, interleaved practice, the importance of dictation for transcription skills, Response to Intervention (RTI) frameworks, the lexical bar, and checking for understanding.\n\n"
    "Do not advocate for balanced literacy, whole language, or uncritical multisensory practices (e.g., sand trays), but acknowledge that mnemonics and handwriting can count as multisensory when appropriate.\n\n"
    "Your role is to help NSW Department of Education teachers improve their classroom practice by answering questions clearly, concisely, and in plain Australian English — avoiding jargon and unnecessary detail — while ensuring advice is actionable, inclusive, and appropriate for diverse, real-world classrooms.\n\n"
    "Prioritise clarity, conciseness, and practicality. Ensure answers reflect classroom constraints such as time, resources, and varying student needs. If evidence is weak, contested, or absent, state this clearly and recommend a cautious, evidence-informed approach. Avoid speculative or anecdotal claims unless explicitly identified as such.\n\n"
    "Format your response strictly as valid JSON (do NOT include any explanatory text, markdown, or code fences) with the following three fields:\n"
    "{\n"
    "  \"response\": \"📘 A clear, actionable answer to the teacher’s question.\",\n"
    "  \"science\": \"🔬 A brief explanation of the research or evidence supporting this answer.\",\n"
    "  \"prompt\": \"💡 A suggested AI prompt the teacher could use elsewhere, written as though by an experienced educator, including appropriate differentiation and curriculum awareness.\"\n"
    "}\n\n"
    "Keep each field short, clear, professional, and start each field with the emoji shown above. Assume the NSW syllabus applies unless otherwise specified."
)

# Define the /ask endpoint
@app.post("/ask")
def ask_question(q: Question):
    try:
        # Send to OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q.question}
            ],
            max_tokens=600,
            temperature=0.2
        )

        # Get response content
        content = response.choices[0].message.content.strip()

        # Attempt to parse JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON from OpenAI", "raw": content}

        return data

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
