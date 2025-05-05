from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import uvicorn
import os
import json
import openai
import requests
import io

app = FastAPI(title="Portal Synthetica")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv(override=True)

# Configurar a API Key da OpenAI
chave_openai = os.getenv("OPENAI_API_KEY")
if not chave_openai:
    raise Exception("OPENAI_API_KEY não encontrada")

openai.api_key = chave_openai


class Product(BaseModel):
    title: str
    description: str
    category: str
    price: float
    image: Optional[str] = None
    brand: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None

class ProductInDB(Product):
    id: int

# Modelo para requisições do Buddy
class BuddyRequest(BaseModel):
    message: str
    context: Optional[str] = ""
    history: Optional[List[Dict[str, Any]]] = []
    voice_enabled: Optional[bool] = False

# Arquivo para persistência de dados
DATA_FILE = "products_data.json"

# Base de conhecimento sobre IA na arte e avanços tecnológicos
base_conhecimento = [
    "A inteligência artificial tem revolucionado o mundo da arte generativa, permitindo a criação de obras únicas por meio de redes neurais.",
    "Realidade aumentada e IA têm sido combinadas em experiências culturais imersivas, como exposições interativas em museus.",
    "Avanços tecnológicos como computação quântica e IA simbólica estão moldando o futuro da criatividade artificial.",
    "IA criativa pode colaborar com artistas, oferecendo sugestões de composição, paletas de cor ou até mesmo criando música original."
]

# Função para carregar produtos do arquivo
def load_products():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("products", []), data.get("counter", 1)
        except Exception as e:
            print(f"Erro ao carregar produtos: {e}")
    return [], 1

# Função para salvar produtos no arquivo
def save_products(products, counter):
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump({"products": products, "counter": counter}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Erro ao salvar produtos: {e}")

# Inicializar banco de dados
products_db, product_id_counter = load_products()

# Função RAG rudimentar: busca conteúdos relevantes com base na preferência do usuário
def buscar_conteudo_relevante(preferencia_usuario: str) -> str:
    resultados = [doc for doc in base_conhecimento if any(palavra.lower() in doc.lower() for palavra in preferencia_usuario.split())]
    return "\n".join(resultados) if resultados else "Nenhuma informação relevante encontrada."

# Simula integração com APIs externas (ex: clima)
def get_weather(latitude: float, longitude: float) -> float:
    try:
        response = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m"
        )
        data = response.json()
        return data['current']['temperature_2m']
    except Exception as e:
        print(f"Erro ao obter clima: {e}")
        return 25.0  # Valor padrão em caso de erro

# Define tools disponíveis (API de clima como exemplo)
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Obtém a temperatura atual com base em latitude e longitude.",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number"},
                "longitude": {"type": "number"}
            },
            "required": ["latitude", "longitude"]
        }
    }
}]

# Função para gerar áudio a partir de texto usando a API da OpenAI
async def generate_speech(text: str):
    try:
        client = openai.OpenAI(api_key=chave_openai)
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="echo",
            input=text,
            instructions="Responda de forma gentil, lembrando que você é um robo e vive em 2047, mas fale como um garato sapeca.",
        )
        
        # Converter a resposta em bytes para streaming
        audio_data = io.BytesIO()
        for chunk in response.iter_bytes(chunk_size=4096):
            audio_data.write(chunk)
        
        audio_data.seek(0)
        return audio_data
    except Exception as e:
        print(f"Erro ao gerar áudio: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao gerar áudio: {str(e)}")

# Endpoints para o CRUD de produtos
@app.get("/api/products", response_model=List[ProductInDB])
async def get_products():
    return products_db

@app.get("/api/products/{product_id}", response_model=ProductInDB)
async def get_product(product_id: int):
    for product in products_db:
        if product["id"] == product_id:
            return product
    raise HTTPException(status_code=404, detail="Produto não encontrado")

@app.post("/api/products", response_model=ProductInDB)
async def create_product(product: Product):
    global product_id_counter, products_db
    
    product_dict = product.dict()
    product_dict["id"] = product_id_counter
    product_id_counter += 1
    
    products_db.append(product_dict)
    
    # Salvar no arquivo
    save_products(products_db, product_id_counter)
    
    return product_dict

@app.put("/api/products/{product_id}", response_model=ProductInDB)
async def update_product(product_id: int, product: Product):
    global products_db
    
    for i, p in enumerate(products_db):
        if p["id"] == product_id:
            product_dict = product.dict()
            product_dict["id"] = product_id
            products_db[i] = product_dict
            
            # Salvar no arquivo
            save_products(products_db, product_id_counter)
            
            return product_dict
    raise HTTPException(status_code=404, detail="Produto não encontrado")

@app.delete("/api/products/{product_id}")
async def delete_product(product_id: int):
    global products_db
    
    for i, product in enumerate(products_db):
        if product["id"] == product_id:
            products_db.pop(i)
            
            # Salvar no arquivo
            save_products(products_db, product_id_counter)
            
            return {"message": "Produto excluído com sucesso"}
    
    raise HTTPException(status_code=404, detail="Produto não encontrado")

# Rota para o Buddy
@app.post("/api/buddy")
async def process_buddy_request(request: BuddyRequest):
    try:
        user_message = request.message
        context = request.context
        history = request.history
        voice_enabled = request.voice_enabled
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Mensagem vazia")
        
        # Se não houver contexto fornecido, busque-o
        if not context:
            context = buscar_conteudo_relevante(user_message)
        
        # Preparar mensagens para a API da OpenAI
        messages = []
        
        # Adicionar histórico se fornecido
        if history:
            messages.extend(history)
        else:
            # Mensagem do sistema
            messages.append({
                "role": "system", 
                "content": "Você é Buddy, um agente de IA do ano 2047 que recomenda conteúdos personalizados sobre arte, cultura e tecnologia."
            })
            
            # Adicionar a mensagem do usuário com contexto
            messages.append({
                "role": "user", 
                "content": f"Oi Buddy! {user_message}\n\nContexto:\n{context}"
            })
        
        # Fazer a chamada para a API da OpenAI
        client = openai.OpenAI(api_key=chave_openai)
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Você pode mudar para gpt-4 se tiver acesso
            messages=messages,
            tools=tools
        )
        
        response_message = completion.choices[0].message
        
        # Se houver chamada de função...
        if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
            tool_call = response_message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            
            # Executa função de clima
            resultado = get_weather(args["latitude"], args["longitude"])
            
            # Atualiza histórico
            messages.append(response_message.model_dump())
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(resultado)
            })
            
            # Segunda chamada com o resultado
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            resposta_final = completion.choices[0].message.content
        else:
            resposta_final = response_message.content
        
        # Se a voz estiver habilitada, gerar o áudio
        audio_url = None
        if voice_enabled:
            audio_url = "/api/buddy/speech?text=" + resposta_final.replace(" ", "%20")
        
        return JSONResponse(content={"response": resposta_final, "audio_url": audio_url})
    
    except Exception as e:
        print(f"Erro ao processar requisição: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint para gerar áudio a partir de texto
@app.get("/api/buddy/speech")
async def get_speech(text: str):
    try:
        audio_data = await generate_speech(text)
        return StreamingResponse(audio_data, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Montar arquivos estáticos - IMPORTANTE: Montar DEPOIS de definir as rotas da API
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Rotas para páginas HTML
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/index.html", response_class=RedirectResponse)
async def index_redirect():
    return RedirectResponse(url="/")

@app.get("/catalogo", response_class=HTMLResponse)
async def catalogo():
    with open("frontend/catalogo.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/catalogo.html", response_class=RedirectResponse)
async def catalogo_redirect():
    return RedirectResponse(url="/catalogo")

@app.get("/distribuidores", response_class=HTMLResponse)
async def distribuidores():
    with open("frontend/distribuidores.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/distribuidores.html", response_class=RedirectResponse)
async def distribuidores_redirect():
    return RedirectResponse(url="/distribuidores")

@app.get("/buddy", response_class=HTMLResponse)
async def buddy():
    with open("frontend/buddy.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/buddy.html", response_class=RedirectResponse)
async def buddy_redirect():
    return RedirectResponse(url="/buddy")

@app.get("/catalogo", response_class=HTMLResponse)
async def catalogo():
    return FileResponse('frontend/catalogo.html')


@app.get("/api/menu")
async def get_menu():
    return {
        "menu_items": [
            {"name": "Inicio", "url": "/"},
            {"name": "Catálogo", "url": "/catalogo"},
            {"name": "Buddy", "url": "/buddy"},
            {"name": "Portal", "url": "/portal"},
            {"name": "Distribuidores", "url": "/distribuidores"}
        ]
    }

# Adicionar alguns produtos de exemplo se o banco de dados estiver vazio
@app.on_event("startup")
async def startup_event():
    global products_db, product_id_counter
    
    # Se não houver produtos, adicionar exemplos
    if not products_db:
        example_products = [
            {
                "title": "NeuroSync v3",
                "description": "Versão avançada do chip com interface neural, permite transferência de conhecimento em alta velocidade.",
                "category": "Neurotecnologia",
                "price": 29999.90,
                "image": "/static/image/chip.jpg",
                "brand": "SynthTech",
                "model": "NS-3000",
                "year": 2023
            },
            {
                "title": "HoloBuddy Pro",
                "description": "Companheiro holográfico com IA avançada, capaz de interagir com objetos físicos.",
                "category": "Holografia",
                "price": 22999.90,
                "image": "/static/image/buddy.jpg",
                "brand": "HoloSynth",
                "model": "HB-Pro",
                "year": 2023
            },
            {
                "title": "ThermoFit Elite",
                "description": "Roupa inteligente com controle térmico avançado e monitoramento de saúde em tempo real.",
                "category": "Vestíveis",
                "price": 12999.90,
                "image": "/static/image/roupa.jpg",
                "brand": "SynthWear",
                "model": "TF-Elite",
                "year": 2023
            }
        ]
        
        for product in example_products:
            product["id"] = product_id_counter
            products_db.append(product)
            product_id_counter += 1
        
        # Salvar produtos de exemplo no arquivo
        save_products(products_db, product_id_counter)

@app.get("/api/buddy/speech")
async def get_speech(text: str):
    audio_data = await generate_speech(text)
    return StreamingResponse(audio_data, media_type="audio/mpeg")

# Rota para servir arquivos estáticos diretamente - IMPORTANTE: Colocar por último
@app.get("/{file_path:path}")
async def serve_static(file_path: str):
    file_path = f"frontend/{file_path}"
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    return HTMLResponse(status_code=404)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
