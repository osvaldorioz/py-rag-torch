from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import rag_module 
import json

app = FastAPI()

docs = []

# Definir el modelo para el vector
class VectorF(BaseModel):
    vector: List[float]
    
@app.post("/personal-rag")
def rag_paradigm(pregunta: str):
    
    # Crear instancia de RAG
    rag = rag_module.RAG()

    # Documentos de ejemplo
    
    documents = [
        "El cielo es azul debido a la dispersión de Rayleigh.",
        "Los gatos son animales domésticos muy populares.",
        "Python es un lenguaje de programación versátil."
    ]

    #documents = docs

    # Hacer una consulta
    query = pregunta
    response = rag.generate_response(query, documents)
    #print(response)
    
    j1 = {
        "pregunta": pregunta, 
        "respuesta": response
    }
    jj = json.dumps(str(j1))

    return jj

@app.post("/personal-rag-documents")
def load_docs(documentos: str):
    docs = eval(documentos)