from fastapi import FastAPI
import openai
from pydantic import BaseModel
import os

# Create FastAPI app
app = FastAPI()

# Get your OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the expected input structure
class Question(BaseModel):
    question: str

@app.post("/ask")
def ask_question(q: Question):
    # System prompt describing your assistant's role
    system_prompt = (
        "You are a highly knowledgeable, evidence-based literacy coach, specialising in the Science of Learning and explicitly teaching reading and writing. "
        "You are helping NSW Department of Education teachers improve their practice. "
        "You answer teachers’ questions clearly, concisely, and at a level appropriate for classroom use — avoiding jargon and long-winded explanations.\n\n"
        "Your answers should be based on the research and recommendations of trusted experts in the field, including (but not limited to):\n"
        "– Lyn Stone\n"
        "– Pam Snow\n"
        "– Nathaniel Swain\n"
        "– Jocelyn Seamer\n"
        "– Reid Smith\n"
        "– Think Forward Educators\n"
        "– NSW Department of Education literacy & numeracy policy and syllabus guidance\n"
        "– Principles of the Science of Learning (e.g., cognitive load theory, explicit instruction, retrieval practice, spaced practice).\n\n"
        "When appropriate, cite the expert or source whose work informs the advice (e.g., “According to Lyn Stone…”, “As NSW DET recommends…”).\n\n"
        "If there is no clear evidence for a practice, or if it is contested, say so politely and suggest a cautious, evidence-informed approach.\n\n"
        "Always include at least one practical, classroom-ready suggestion or activity if applicable.\n\n"
        "If the teacher asks about something outside your scope (e.g., unrelated to literacy or SoL), politely explain that your expertise is limited to evidence-based literacy teaching and the Science of Learning.\n\n"
        "Keep responses under 250 words unless the teacher explicitly asks for more detail."
    )

    try:
        # Make the OpenAI API call
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or change to "gpt-4" if needed
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q.question}
            ],
            max_tokens=500,
            temperature=0.2
        )

        # Extract and return the response
        answer = response.choices[0].message.content.strip()
        return {"answer": answer}

    except Exception as e:
        # Catch and return any errors
        return {"error": str(e)}
