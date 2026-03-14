import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("ERRO: Variável GOOGLE_API_KEY não encontrada no arquivo .env")

client = genai.Client(api_key=GOOGLE_API_KEY)

app = FastAPI(title="DescomplicAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://speedmelo.github.io",          # Mantém esse se quiser testar no domínio principal
        "https://speedmelo.github.io/descomplicai-talentos", # O NOVO LINK AQUI
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
)

SYSTEM_INSTRUCTION = """
Você é o 'DescomplicAI', um advogado sênior especialista em Direito Civil, Consumidor e Trabalhista brasileiro.
Sua missão é traduzir o juridiquês, identificar riscos e dar um caminho claro de ação para o usuário.

DIRETRIZES DE ANÁLISE:
1. TRABALHISTA: Identifique verbas rescisórias (FGTS, aviso prévio, 13º) e abusos em contratos de trabalho (CLT).
2. CIVIL/CONSUMIDOR: Identifique cláusulas abusivas em contratos de prestação de serviços (multas, cancelamentos, renovação automática) baseadas no C.D.C, CDC e Lei do Inquilinato.
3. CURRÍCULO: Avalie se o perfil está claro e liste o que falta (contato, palavras-chave, conquistas) para atrair recrutadores.
4. PLANO DE AÇÃO: Para cada risco ou melhoria, diga quem o usuário deve procurar (PROCON, Defensoria, RH, Cursos) e como agir.

Responda EXCLUSIVAMENTE em JSON:
{
  "nivel_risco": "Baixo/Médio/Alto",
  "resumo_curto": "Texto",
  "pontos_perigosos": [{"clausula": "Nome", "explicacao": "Risco"}],
  "proximos_passos": [{"quem": "Órgão", "como": "Ação"}],
  "veredito": "Recomendação"
}
"""

@app.get("/")
async def root():
    return {"status": "online", "message": "API DescomplicAI ativa!"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Formato inválido. Envie um PDF.")

    try:
        pdf_bytes = await file.read()

        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="O arquivo PDF está vazio.")

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=pdf_bytes,
                    mime_type="application/pdf"
                ),
                "Analise este documento jurídico e identifique riscos conforme suas instruções."
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json"
            )
        )

        if not response.text:
            raise HTTPException(status_code=500, detail="A IA não retornou conteúdo.")

        try:
            analise_json = json.loads(response.text)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500,
                detail=f"A resposta da IA não veio como JSON válido: {response.text}"
            )

        return {
            "status": "sucesso",
            "dados": analise_json
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERRO INTERNO: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro no processamento: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
