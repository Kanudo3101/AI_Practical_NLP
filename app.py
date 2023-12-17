# pip install fastapi uvicorn
# uvicorn main:app --host 0.0.0.0 --port 8000

# from fastapi import FastAPI
# from pydantic import BaseModel

# app = FastAPI()

# class Prompt(BaseModel):
#     text: str

# @app.post('/generate')
# def generate(prompt: Prompt):
#     return generate_text(prompt.text)

