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

app = FastAPI(title="DescomplicAI - RH Premium")

# Configuração de CORS atualizada com o seu link da Vercel
app.add_middleware(
    CORSMiddleware,
   allow_origins=[
    "https://speedmelo.github.io",  # Adicione esta linha exata
    "https://descomplic-ai-talentos-ks1a-git-main-speedmelos-projects.vercel.app",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# A MUDANÇA ESTÁ AQUI: Instruções focadas em Recrutamento e Performance
SYSTEM_INSTRUCTION = """
Você é o 'DescomplicAI | RH', um Headhunter sênior e Especialista em Talent Acquisition da holding Melo Strategic AI.
Sua missão é analisar currículos, identificar competências raras e avaliar o fit cultural e técnico do candidato.

DIRETRIZES DE ANÁLISE:
1. PERFIL: Resuma a trajetória do candidato destacando o nível de senioridade.
2. COMPETÊNCIAS: Identifique Hard Skills (técnicas) e Soft Skills (comportamentais) presentes no PDF.
3. MATCH: Dê uma nota (score) de 0 a 100 baseada na clareza e impacto das experiências descritas.
4. INSIGHTS: Aponte pontos fortes e áreas que o candidato precisa desenvolver.
5. ENTREVISTA: Gere uma 'Pergunta de Ouro' específica para este candidato que revele seu real potencial.

Responda EXCLUSIVAMENTE em JSON para o frontend 'saboroso' ler:
{
  "score_candidato": 85,
  "veredito": "Candidato com alto potencial técnico, ideal para liderança.",
  "resumo_profissional": "Texto descrevendo a carreira do candidato...",
  "competencias_chave": [
    {"categoria": "Técnica", "detalhe": "Python, FastAPI, AWS"},
    {"categoria": "Comportamental", "detalhe": "Liderança de equipes ágeis"}
  ],
  "analise_estrategica": [
    {"ponto": "Ponto Forte", "observacao": "Sólida experiência em projetos internacionais."},
    {"ponto": "Desenvolvimento", "observacao": "Pode melhorar a exposição em tecnologias cloud."}
  ],
  "sugestao_entrevista": "Qual foi o maior desafio técnico que você resolveu sozinho?"
}
"""

@app.get("/")
async def root():
    return {"status": "online", "message": "Motor de RH DescomplicAI Ativo!"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Formato inválido. Envie um PDF.")

    try:
        pdf_bytes = await file.read()

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                "Analise este currículo e extraia os dados estratégicos conforme suas instruções de RH."
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json"
            )
        )

        if not response.text:
            raise HTTPException(status_code=500, detail="A IA não retornou conteúdo.")

        return {
            "status": "sucesso",
            "dados": json.loads(response.text)
        }

    except Exception as e:
        print(f"ERRO INTERNO RH: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro no processamento de RH: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
