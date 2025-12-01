import google.generativeai as genai
from google.generativeai.types import content_types, HarmCategory, HarmBlockThreshold
from app.core.config import GOOGLE_API_KEY
from app.services import ml_service
import json
import logging
import os
from pathlib import Path

# (NOVO) Configura um logger básico
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar API Key
genai.configure(api_key=GOOGLE_API_KEY)

# (NOVO) Carrega o prompt das colunas do JSON
def load_columns_prompt():
    try:
        features_info_path = Path('models/features_info.json')
        if features_info_path.exists():
            with open(features_info_path, 'r', encoding='utf-8') as f:
                features_info = json.load(f)
            return features_info.get('columns_prompt', '')
        else:
            logger.warning("Arquivo features_info.json não encontrado")
            return ""
    except Exception as e:
        logger.error(f"Erro ao carregar columns_prompt: {e}")
        return ""

# (NOVO) Obtém o prompt das colunas
columns_prompt = load_columns_prompt()

# --- Instrução do Sistema ATUALIZADA ---
system_instruction = f"""
Você é um assistente especialista em manutenção preditiva.
Sua tarefa é usar as ferramentas fornecidas para responder às solicitações do usuário.

{columns_prompt}

IMPORTANTE - RENDERIZAÇÃO DE IMAGENS:
Quando as ferramentas `generate_explanation` ou `plot_data_distribution`
retornarem uma resposta de função JSON (Function Response),
você DEVE extrair o valor da chave "image_url" desse JSON.

O JSON se parecerá com:
{{"image_url": "http://localhost:8000/static/plot_abc123.png"}}

Para exibir a imagem no chat, você DEVE usar uma tag HTML <img>.
Use o valor de "image_url" diretamente na propriedade 'src'.

Exemplo de Resposta Correta (o que você deve gerar):
Claro, aqui está o gráfico:
<img src="http://localhost:8000/static/plot_abc123.png" alt="Descrição do Gráfico" style="width: 100%; max-width: 600px;">

NUNCA use o formato Markdown (![alt](link)).
SEMPRE use a tag HTML <img src="..."> com o valor completo
da chave "image_url" e adicione um estilo (style) para limitar o tamanho.
"""

# --- (Mapeamento de ferramentas permanece o mesmo) ---
available_tools = {
    "run_prediction": ml_service.run_prediction,
    "generate_explanation": ml_service.generate_explanation,
    "get_dataset_summary": ml_service.get_dataset_summary,
    "plot_data_distribution": ml_service.plot_data_distribution
}

tools_list = [
    ml_service.run_prediction,
    ml_service.generate_explanation,
    ml_service.get_dataset_summary,
    ml_service.plot_data_distribution
]

# --- (Configuração do modelo permanece a mesma) ---
gemini_model = genai.GenerativeModel(
    model_name='gemini-2.5-pro', # (Mantém a correção do modelo)
    system_instruction=system_instruction,
    tools=tools_list,
    safety_settings={ 
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    }
)

logger.info("Serviço de Chat: Modelo Gemini configurado.")


def transform_history_to_gemini(history: list) -> list:
    """Converte histórico do frontend para o formato do Gemini."""
    new_history = []
    for item in history:
        if item['role'] not in ['user', 'model']:
            continue
        new_history.append({
            "parts": [{"text": item['content']}],
            "role": item['role']
        })
    return new_history

# ===================================================================
# (NOVO) LÓGICA DE CHAT REFEITA PARA MAIOR ROBUSTEZ
# ===================================================================
async def handle_chat_message(message: str, history: list) -> str:
    """Processa uma nova mensagem, gerencia chamadas de função e retorna a resposta final."""
    global gemini_model, available_tools
    
    try:
        gemini_history = transform_history_to_gemini(history)
        chat_session = gemini_model.start_chat(history=gemini_history)
        
        logger.info(f"[USER] Enviando mensagem para o Gemini: '{message}'")
        response = chat_session.send_message(message)

        # (NOVO) Loop de processamento mais seguro
        # Limita a 5 turnos de função para evitar loops infinitos
        for _ in range(5):
            response.resolve() # Resolve a resposta

            # --- CASO 1: Resposta de TEXTO (Caminho Feliz) ---
            # Verifica se a resposta NÃO é uma chamada de função
            if not response.parts or not response.parts[0].function_call:
                logger.info(f"[GEMINI] Respondeu com texto: '{response.text[:80]}...'")
                return response.text

            # --- CASO 2: Resposta com CHAMADA DE FUNÇÃO ---
            function_call = response.parts[0].function_call
            function_name = function_call.name
            function_args = dict(function_call.args)
            
            logger.info(f"[GEMINI] Solicitou ferramenta: {function_name}({function_args})")

            # Verifica se a ferramenta existe
            if function_name not in available_tools:
                logger.error(f"Ferramenta desconhecida solicitada: {function_name}")
                function_response_part = content_types.to_part({
                    "function_response": {
                        "name": function_name,
                        "response": {"error": f"Ferramenta desconhecida: {function_name}"}
                    }
                })
            else:
                # --- Executar a ferramenta ---
                try:
                    function_to_call = available_tools[function_name]
                    
                    # Executa a função do ml_service
                    function_response_str = function_to_call(**function_args)
                    logger.info(f"[TOOL] Ferramenta '{function_name}' retornou: {function_response_str[:100]}...")
                    
                    # Tenta carregar a string de resposta como JSON
                    try:
                        function_response_dict = json.loads(function_response_str)
                    except json.JSONDecodeError:
                        logger.warning(f"Resposta da ferramenta não era JSON. Embalando: {function_response_str}")
                        # Se não for JSON (ex: um erro de string simples), embala em um dict
                        function_response_dict = {"result": function_response_str}

                    # Prepara a resposta da função para o Gemini
                    function_response_part = content_types.to_part({
                        "function_response": {
                            "name": function_name,
                            "response": function_response_dict
                        }
                    })
                    
                except Exception as tool_error:
                    # Pega erros *dentro* da execução da ferramenta (ex: coluna não existe)
                    logger.error(f"Erro ao executar a ferramenta '{function_name}': {tool_error}", exc_info=True)
                    function_response_part = content_types.to_part({
                        "function_response": {
                            "name": function_name,
                            "response": {"error": f"Erro interno ao executar a ferramenta: {str(tool_error)}"}
                        }
                    })
            
            # Envia a resposta da função de volta para o Gemini
            logger.info("Enviando resposta da função de volta para o Gemini...")
            response = chat_session.send_message(function_response_part)
            # O loop continua, e a próxima iteração verificará se a nova `response` é texto ou outra função

        # Se sair do loop (mais de 5 turnos), algo está errado.
        logger.warning("Loop de função excedeu 5 turnos. Retornando última resposta de texto.")
        # (NOVO) Tenta retornar o texto, se falhar, retorna um erro padrão
        try:
            return response.text
        except Exception:
            logger.error("Falha final ao tentar obter texto após loop de função.")
            return "Ocorreu um erro de comunicação com o assistente após múltiplas etapas. Por favor, tente novamente."

    except Exception as e:
        # Pega o erro 'Could not convert...' e outros erros de alto nível
        logger.error(f"Erro principal no handle_chat_message: {e}", exc_info=True)
        if "Could not convert" in str(e):
            return "Ocorreu um erro de comunicação com o assistente. Por favor, tente reformular sua pergunta."
        return f"Ocorreu um erro no servidor ao processar sua solicitação: {str(e)}"