from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from app.services.chat_service import handle_chat_message

# (NOVO) Imports para servir arquivos
from fastapi.staticfiles import StaticFiles
import os

# (NOVO) Define o diretório de estáticos e o cria
STATIC_DIR = "app/static"
os.makedirs(STATIC_DIR, exist_ok=True)

# --- Modelos Pydantic para Validação ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage]

class ChatResponse(BaseModel):
    reply: str

# --- Criação da Aplicação FastAPI ---
app = FastAPI(
    title="API de Chatbot - Manutenção Preditiva",
    description="Backend para o chatbot com Gemini e ferramentas de ML",
    version="1.0.0"
)

# --- Configuração do CORS ---
# Permite que o frontend (ex: http://localhost:3000) acesse esta API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Em produção, restrinja para o seu domínio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Endpoint de "Saúde" ---
@app.get("/")
def read_root():
    return {"status": "ok", "message": "API do Chatbot de Manutenção Preditiva está online."}

# --- Endpoint Principal do Chat ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Recebe uma mensagem e o histórico, processa com o Gemini (incluindo 
    chamadas de função) e retorna a resposta final do assistente.
    """
    try:
        # Converte o histórico de Pydantic para dicts simples
        history_dicts = [msg.model_dump() for msg in request.history]
        
        reply_text = await handle_chat_message(request.message, history_dicts)
        
        return ChatResponse(reply=reply_text)
        
    except Exception as e:
        print(f"Erro no endpoint /chat: {e}")
        return ChatResponse(reply=f"Erro interno no servidor: {str(e)}")

# --- Ponto de entrada para Uvicorn (opcional, mas bom para debug) ---
if __name__ == "__main__":
    import uvicorn
    # Isso permite executar 'python app/main.py'
    # Mas o ideal é 'uvicorn app.main:app --reload'
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)