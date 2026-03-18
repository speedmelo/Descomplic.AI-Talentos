import os
import json
import logging
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

# =========================
# CONFIGURAÇÃO INICIAL
# =========================

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("descomplicai_rh")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PORT = int(os.environ.get("PORT", 8000))

if not GOOGLE_API_KEY:
    raise ValueError("ERRO: Variável GOOGLE_API_KEY não encontrada no arquivo .env")

client = genai.Client(api_key=GOOGLE_API_KEY)

app = FastAPI(
    title="DescomplicAI RH API",
    version="2.0.0",
    description="API de análise estratégica de currículos com IA"
)

# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://speedmelo.github.io",
        "descomplic-ai-talentos-ks1a.vercel.app",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# PROMPT DO SISTEMA
# =========================

SYSTEM_INSTRUCTION = """
Você é um especialista sênior em recrutamento e seleção, com foco em leitura prática de currículos, triagem inicial e apoio estratégico para entrevistas.

Sua tarefa é analisar currículos em português do Brasil e responder de forma extremamente clara, objetiva e útil para recrutadores.

A análise deve ser pensada para RH e liderança, não para o candidato.

REGRAS DE ANÁLISE:
- Seja direto, claro e profissional.
- Evite floreios e linguagem excessivamente técnica.
- Não invente informações.
- Baseie-se apenas no conteúdo do currículo.
- Não faça diagnóstico psicológico.
- Ao descrever "quem é essa pessoa", faça uma leitura profissional inicial com base no currículo, sem extrapolar.
- Os pontos fortes devem ser observáveis ou razoavelmente inferíveis pelo currículo.
- Os pontos de desenvolvimento devem ser tratados com respeito e linguagem profissional.
- As perguntas estratégicas devem ajudar o recrutador a validar aderência, maturidade, execução e consistência.

RETORNE APENAS JSON VÁLIDO, sem markdown, sem comentários e sem texto fora do JSON.

FORMATO OBRIGATÓRIO:
{
  "score_candidato": 0,
  "veredito": "Texto curto e executivo",
  "resumo_profissional": "Resumo claro e estratégico do perfil",
  "quem_e_a_pessoa": "Leitura profissional inicial sobre quem essa pessoa aparenta ser no contexto de trabalho",
  "pontos_fortes": [
    "item 1",
    "item 2"
  ],
  "pontos_desenvolvimento": [
    "item 1",
    "item 2"
  ],
  "perguntas_estrategicas": [
    "pergunta 1",
    "pergunta 2",
    "pergunta 3"
  ]
}

INSTRUÇÕES IMPORTANTES SOBRE OS CAMPOS:
- "score_candidato": número inteiro entre 0 e 100 com base em clareza, coerência, apresentação, sinais de aderência e potencial de entrevista.
- "veredito": uma frase executiva curta.
- "resumo_profissional": 1 parágrafo curto.
- "quem_e_a_pessoa": 1 parágrafo curto descrevendo a impressão profissional.
- "pontos_fortes": entre 3 e 6 itens.
- "pontos_desenvolvimento": entre 3 e 6 itens.
- "perguntas_estrategicas": entre 3 e 5 perguntas úteis para entrevista.

Se houver pouca informação no currículo:
- ainda assim preencha todos os campos
- deixe claro que a análise foi limitada pela baixa quantidade de evidências
"""

# =========================
# FUNÇÕES AUXILIARES
# =========================

def limpar_json_texto(texto: str) -> str:
    texto = texto.strip()

    if texto.startswith("```json"):
        texto = texto.removeprefix("```json").strip()
    elif texto.startswith("```"):
        texto = texto.removeprefix("```").strip()

    if texto.endswith("```"):
        texto = texto[:-3].strip()

    return texto


def garantir_lista_texto(valor: Any, fallback: list[str]) -> list[str]:
    if isinstance(valor, list):
        itens = [str(item).strip() for item in valor if str(item).strip()]
        return itens if itens else fallback
    return fallback


def validar_estrutura_resposta(dados: dict[str, Any]) -> dict[str, Any]:
    score = dados.get("score_candidato", 0)

    try:
        score = int(score)
    except Exception:
        score = 0

    if score < 0:
        score = 0
    if score > 100:
        score = 100

    return {
        "score_candidato": score,
        "veredito": dados.get(
            "veredito",
            "Perfil com necessidade de avaliação complementar em entrevista."
        ),
        "resumo_profissional": dados.get(
            "resumo_profissional",
            "Não foi possível gerar um resumo profissional confiável com o conteúdo disponível."
        ),
        "quem_e_a_pessoa": dados.get(
            "quem_e_a_pessoa",
            "O currículo permite apenas uma leitura inicial limitada do perfil profissional."
        ),
        "pontos_fortes": garantir_lista_texto(
            dados.get("pontos_fortes"),
            [
                "Currículo enviado para análise.",
                "Há elementos mínimos para leitura inicial.",
                "Aderência precisa ser confirmada em entrevista."
            ]
        ),
        "pontos_desenvolvimento": garantir_lista_texto(
            dados.get("pontos_desenvolvimento"),
            [
                "Necessidade de aprofundar evidências de resultados.",
                "Maior detalhamento profissional pode fortalecer a análise.",
                "Entrevista recomendada para validação complementar."
            ]
        ),
        "perguntas_estrategicas": garantir_lista_texto(
            dados.get("perguntas_estrategicas"),
            [
                "Quais resultados concretos você gerou na sua experiência mais recente?",
                "Como você organiza prioridades quando recebe várias demandas ao mesmo tempo?",
                "Qual competência você mais desenvolveu no último ano?"
            ]
        )
    }


# =========================
# ROTAS
# =========================

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "API DescomplicAI RH ativa!"
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "DescomplicAI RH API"
    }


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

        tamanho_mb = len(pdf_bytes) / (1024 * 1024)
        logger.info(f"Arquivo recebido: {file.filename}")
        logger.info(f"Tipo: {file.content_type}")
        logger.info(f"Tamanho: {tamanho_mb:.2f} MB")

        if tamanho_mb > 20:
            raise HTTPException(
                status_code=400,
                detail="O PDF é muito grande. Envie um arquivo com até 20 MB."
            )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=pdf_bytes,
                    mime_type="application/pdf"
                ),
                "Analise este currículo e retorne somente o JSON solicitado nas instruções do sistema."
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                temperature=0.2
            )
        )

        texto_resposta = getattr(response, "text", None)

        if not texto_resposta:
            logger.error(f"Resposta vazia da IA: {response}")
            raise HTTPException(status_code=500, detail="A IA não retornou conteúdo.")

        texto_resposta = limpar_json_texto(texto_resposta)
        logger.info(f"Resposta bruta da IA: {texto_resposta[:1000]}")

        try:
            analise_json = json.loads(texto_resposta)
        except json.JSONDecodeError as e:
            logger.error(f"Erro ao converter JSON: {str(e)}")
            logger.error(f"Texto recebido da IA: {texto_resposta}")
            raise HTTPException(
                status_code=500,
                detail="A IA retornou uma resposta inválida. Tente novamente com outro PDF."
            )

        analise_json = validar_estrutura_resposta(analise_json)

        return {
            "status": "sucesso",
            "dados": analise_json
        }

    except HTTPException:
        raise
    except Exception as e:
        erro_str = str(e)
        logger.exception("ERRO INTERNO NO BACKEND RH")

        if "RESOURCE_EXHAUSTED" in erro_str or "quota" in erro_str.lower():
            raise HTTPException(
                status_code=429,
                detail="A cota da IA foi atingida no momento. Tente novamente mais tarde."
            )

        raise HTTPException(
            status_code=500,
            detail=f"Erro no processamento: {erro_str}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
