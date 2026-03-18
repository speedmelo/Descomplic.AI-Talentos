import os
import json
import logging
from typing import Any, List

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
logger = logging.getLogger("descomplicai_rh_multi")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PORT = int(os.environ.get("PORT", 8000))

if not GOOGLE_API_KEY:
    raise ValueError("ERRO: Variável GOOGLE_API_KEY não encontrada no arquivo .env")

client = genai.Client(api_key=GOOGLE_API_KEY)

app = FastAPI(
    title="DescomplicAI RH API",
    version="3.0.0",
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
# PROMPTS
# =========================

SYSTEM_INSTRUCTION_SINGLE = """
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

RETORNE APENAS JSON VÁLIDO.

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

INSTRUÇÕES IMPORTANTES:
- "score_candidato": número inteiro entre 0 e 100.
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

SYSTEM_INSTRUCTION_COMPARE = """
Você é um especialista sênior em recrutamento e seleção.

Sua tarefa é comparar candidatos com base em análises individuais já produzidas.

RETORNE APENAS JSON VÁLIDO.

FORMATO OBRIGATÓRIO:
{
  "resumo_comparativo": "Texto curto comparando os candidatos",
  "melhor_aderencia": "nome_arquivo",
  "maior_potencial": "nome_arquivo",
  "mais_experiente": "nome_arquivo",
  "exige_maior_validacao": "nome_arquivo",
  "ranking_final": [
    "nome_arquivo_1",
    "nome_arquivo_2",
    "nome_arquivo_3"
  ]
}

REGRAS:
- Baseie-se apenas nas análises recebidas.
- Seja objetivo.
- Não invente experiência não informada.
- Se houver empate técnico, ainda assim ordene pelo melhor julgamento possível.
- O ranking deve ir do candidato mais aderente ao menos aderente.
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


def validar_comparacao(
    dados: dict[str, Any],
    nomes_arquivos: list[str]
) -> dict[str, Any]:
    ranking = dados.get("ranking_final", [])
    if not isinstance(ranking, list):
        ranking = []

    ranking_limpo = [str(x).strip() for x in ranking if str(x).strip()]
    ranking_filtrado = [x for x in ranking_limpo if x in nomes_arquivos]

    # Completa ranking com nomes faltantes
    for nome in nomes_arquivos:
        if nome not in ranking_filtrado:
            ranking_filtrado.append(nome)

    def pick_nome(chave: str, fallback: str) -> str:
        valor = str(dados.get(chave, "")).strip()
        return valor if valor in nomes_arquivos else fallback

    return {
        "resumo_comparativo": dados.get(
            "resumo_comparativo",
            "Os candidatos apresentam perfis distintos e exigem validação complementar em entrevista."
        ),
        "melhor_aderencia": pick_nome("melhor_aderencia", nomes_arquivos[0]),
        "maior_potencial": pick_nome("maior_potencial", nomes_arquivos[0]),
        "mais_experiente": pick_nome("mais_experiente", nomes_arquivos[0]),
        "exige_maior_validacao": pick_nome("exige_maior_validacao", nomes_arquivos[-1]),
        "ranking_final": ranking_filtrado
    }


async def analisar_curriculo_individual(file: UploadFile) -> dict[str, Any]:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail=f"O arquivo '{file.filename}' não é um PDF válido.")

    pdf_bytes = await file.read()

    if not pdf_bytes:
        raise HTTPException(status_code=400, detail=f"O arquivo '{file.filename}' está vazio.")

    tamanho_mb = len(pdf_bytes) / (1024 * 1024)
    logger.info(f"Arquivo recebido: {file.filename} | {tamanho_mb:.2f} MB")

    if tamanho_mb > 20:
        raise HTTPException(
            status_code=400,
            detail=f"O arquivo '{file.filename}' é muito grande. Limite de 20 MB."
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
            system_instruction=SYSTEM_INSTRUCTION_SINGLE,
            response_mime_type="application/json",
            temperature=0.2
        )
    )

    texto_resposta = getattr(response, "text", None)

    if not texto_resposta:
        raise HTTPException(status_code=500, detail=f"A IA não retornou conteúdo para '{file.filename}'.")

    texto_resposta = limpar_json_texto(texto_resposta)

    try:
        analise_json = json.loads(texto_resposta)
    except json.JSONDecodeError:
        logger.error(f"JSON inválido da IA para arquivo {file.filename}: {texto_resposta}")
        raise HTTPException(
            status_code=500,
            detail=f"A IA retornou uma resposta inválida para '{file.filename}'."
        )

    analise_json = validar_estrutura_resposta(analise_json)
    analise_json["nome_arquivo"] = file.filename or "curriculo.pdf"
    return analise_json


def comparar_candidatos(candidatos: list[dict[str, Any]]) -> dict[str, Any]:
    nomes = [c["nome_arquivo"] for c in candidatos]

    resumo_base = {
        "candidatos": [
            {
                "nome_arquivo": c["nome_arquivo"],
                "score_candidato": c["score_candidato"],
                "veredito": c["veredito"],
                "resumo_profissional": c["resumo_profissional"],
                "quem_e_a_pessoa": c["quem_e_a_pessoa"],
                "pontos_fortes": c["pontos_fortes"],
                "pontos_desenvolvimento": c["pontos_desenvolvimento"],
            }
            for c in candidatos
        ]
    }

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            json.dumps(resumo_base, ensure_ascii=False)
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION_COMPARE,
            response_mime_type="application/json",
            temperature=0.2
        )
    )

    texto_resposta = getattr(response, "text", None)

    if not texto_resposta:
        # fallback simples por score
        ordenados = sorted(candidatos, key=lambda x: x["score_candidato"], reverse=True)
        return {
            "resumo_comparativo": "Os candidatos apresentam níveis diferentes de aderência com base nas análises individuais.",
            "melhor_aderencia": ordenados[0]["nome_arquivo"],
            "maior_potencial": ordenados[0]["nome_arquivo"],
            "mais_experiente": ordenados[0]["nome_arquivo"],
            "exige_maior_validacao": ordenados[-1]["nome_arquivo"],
            "ranking_final": [c["nome_arquivo"] for c in ordenados]
        }

    texto_resposta = limpar_json_texto(texto_resposta)

    try:
        comparacao_json = json.loads(texto_resposta)
    except json.JSONDecodeError:
        logger.error(f"JSON inválido na comparação: {texto_resposta}")
        ordenados = sorted(candidatos, key=lambda x: x["score_candidato"], reverse=True)
        comparacao_json = {
            "resumo_comparativo": "Os candidatos apresentam níveis diferentes de aderência com base nas análises individuais.",
            "melhor_aderencia": ordenados[0]["nome_arquivo"],
            "maior_potencial": ordenados[0]["nome_arquivo"],
            "mais_experiente": ordenados[0]["nome_arquivo"],
            "exige_maior_validacao": ordenados[-1]["nome_arquivo"],
            "ranking_final": [c["nome_arquivo"] for c in ordenados]
        }

    return validar_comparacao(comparacao_json, nomes)


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
    try:
        resultado = await analisar_curriculo_individual(file)
        return {
            "status": "sucesso",
            "dados": resultado
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


@app.post("/analyze-multiple")
async def analyze_multiple(files: List[UploadFile] = File(...)):
    try:
        if not files or len(files) < 2:
            raise HTTPException(
                status_code=400,
                detail="Envie pelo menos 2 currículos em PDF para comparação."
            )

        if len(files) > 3:
            raise HTTPException(
                status_code=400,
                detail="O limite atual é de até 3 currículos por comparação."
            )

        candidatos = []
        for file in files:
            resultado = await analisar_curriculo_individual(file)
            candidatos.append(resultado)

        comparacao = comparar_candidatos(candidatos)

        return {
            "status": "sucesso",
            "candidatos": candidatos,
            "comparacao": comparacao
        }

    except HTTPException:
        raise
    except Exception as e:
        erro_str = str(e)
        logger.exception("ERRO INTERNO NO BACKEND RH MULTI")

        if "RESOURCE_EXHAUSTED" in erro_str or "quota" in erro_str.lower():
            raise HTTPException(
                status_code=429,
                detail="A cota da IA foi atingida no momento. Tente novamente mais tarde."
            )

        raise HTTPException(
            status_code=500,
            detail=f"Erro no processamento múltiplo: {erro_str}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
