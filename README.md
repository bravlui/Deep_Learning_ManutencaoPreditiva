# Chatbot de Manutenção Preditiva 🤖

Um projeto full-stack para análise de manutenção preditiva com chatbot inteligente. Combina um **backend em FastAPI**, um **frontend em Next.js** e um **pipeline de treinamento de ML**.

## 📋 Pré-requisitos

### Requisitos Globais
- **Python 3.9+** (para backend e treinamento)
- **Node.js 18+** (para frontend)
- **pip** (gerenciador de pacotes Python)
- **npm** ou **yarn** (gerenciador de pacotes Node.js)
- **Git** (opcional, para controle de versão)

### Chaves de API
- **Google API Key** (para Gemini AI) - [Obter aqui](https://makersuite.google.com/app/apikey)

---

## 🚀 Configuração Rápida (3 passos)

### Passo 1: Configurar o Backend

```bash
# 1.1 Navegue até o diretório do backend
cd backend

# 1.2 Crie um ambiente virtual (opcional, mas recomendado)
python -m venv venv

# 1.3 Ative o ambiente virtual
# No Windows:
venv\Scripts\activate
# No macOS/Linux:
source venv/bin/activate

# 1.4 Instale as dependências
pip install -r requirements.txt

# 1.5 Configure as variáveis de ambiente
# Crie um arquivo `.env` na pasta `backend/` com:
echo GOOGLE_API_KEY=sua_chave_aqui > .env
```

### Passo 2: Treinar os Modelos de ML

```bash
# 2.1 Navegue até a pasta de treinamento
cd ../train

# 2.2 Execute o script de treinamento
python train.py
```

Este script irá:
- Carregar o dataset `predictive_maintenance.csv`
- Treinar modelos de classificação e regressão
- Salvar os modelos em `backend/models/`
- Criar dataset limpo em `backend/data/`

### Passo 3: Configurar e Rodar o Frontend

```bash
# 3.1 Navegue até o diretório do frontend
cd ../frontend

# 3.2 Instale as dependências
npm install
# ou
yarn install

# 3.3 Inicie o servidor de desenvolvimento
npm run dev
# ou
yarn dev
```

---

## 🛠️ Instruções Detalhadas

### Backend (FastAPI)

#### 📁 Estrutura de Pastas

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # Arquivo principal da aplicação
│   ├── core/
│   │   ├── config.py        # Configurações (Google API Key)
│   │   └── __init__.py
│   ├── services/
│   │   ├── chat_service.py  # Lógica do chatbot
│   │   ├── ml_service.py    # Integração com modelos ML
│   │   └── __init__.py
│   ├── utils/
│   │   ├── plotting.py      # Gráficos e visualizações
│   │   └── __init__.py
│   └── static/              # Arquivos estáticos (gráficos, etc)
├── data/
│   └── predictive_maintenance_cleaned.csv  # Dataset limpo
├── models/
│   ├── best_classifier_model.pkl          # Modelo de classificação
│   ├── best_regressor_model.pkl           # Modelo de regressão
│   ├── classifier_importances.pkl         # Importância das features
│   ├── regressor_importances.pkl
│   ├── type_label_encoder.pkl
│   └── features_info.json                 # Metadados das features
├── .env                     # Variáveis de ambiente
└── requirements.txt         # Dependências Python
```

#### 🔧 Configuração do Backend

**1. Criar arquivo `.env`:**

```bash
cd backend
cat > .env << EOF
GOOGLE_API_KEY=sua_chave_api_google_aqui
EOF
```

**2. Instalar dependências:**

```bash
pip install -r requirements.txt
```

**3. Iniciar o servidor:**

```bash
# Opção 1: Com hot-reload (desenvolvimento)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Opção 2: Modo produção
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

O servidor estará disponível em: **http://localhost:8000**

#### 📚 Endpoints da API

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `GET` | `/` | Verifica status da API |
| `POST` | `/chat` | Envia mensagem ao chatbot |

**Exemplo de requisição POST /chat:**

```json
{
  "message": "Qual é a temperatura média do processo?",
  "history": [
    {
      "role": "user",
      "content": "Olá"
    },
    {
      "role": "assistant",
      "content": "Olá! Como posso ajudá-lo com a manutenção preditiva?"
    }
  ]
}
```

---

### Frontend (Next.js)

#### 📁 Estrutura de Pastas

```
frontend/
├── app/
│   ├── layout.tsx           # Layout principal
│   ├── page.tsx             # Página inicial
│   ├── globals.css          # Estilos globais
│   └── favicon.ico
├── public/                  # Arquivos estáticos públicos
├── package.json             # Dependências Node.js
├── next.config.ts           # Configuração do Next.js
├── tsconfig.json            # Configuração do TypeScript
├── tailwind.config.ts       # Configuração do Tailwind CSS
└── eslint.config.mjs        # Configuração do ESLint
```

#### 🔧 Configuração do Frontend

**1. Instalar dependências:**

```bash
cd frontend
npm install
```

**2. Iniciar servidor de desenvolvimento:**

```bash
npm run dev
```

O aplicativo estará disponível em: **http://localhost:3000**

#### 📦 Scripts Disponíveis

```bash
npm run dev      # Inicia servidor de desenvolvimento
npm run build    # Cria build otimizado para produção
npm start        # Inicia servidor em modo produção
npm run lint     # Valida código com ESLint
```

#### 🎨 Tecnologias

- **Framework**: Next.js 16
- **UI Components**: Lucide React (ícones)
- **Styling**: Tailwind CSS
- **Linguagem**: TypeScript/TSX
- **Linter**: ESLint

---

### Treinamento de Modelos (Train)

#### 📁 Estrutura de Pastas

```
train/
├── train.py                          # Script de treinamento
└── predictive_maintenance.csv        # Dataset original
```

#### 🔧 Execução do Treinamento

**1. Verificar dependências:**

Certifique-se de que `train.py` tem acesso ao dataset `predictive_maintenance.csv` na mesma pasta.

**2. Executar treinamento:**

```bash
cd train
python train.py
```

#### 📊 O que o Script Faz

| Etapa | Descrição | Saída |
|-------|-----------|-------|
| 1. Carregamento | Lê e limpa o dataset | `data/predictive_maintenance_cleaned.csv` |
| 2. Classificação | Treina modelos para prever falhas | `models/best_classifier_model.pkl` |
| 3. Regressão | Treina modelos para prever desgaste | `models/best_regressor_model.pkl` |
| 4. XAI | Extrai importância das features | `models/*_importances.pkl` |
| 5. Metadados | Gera info sobre features e aliases | `models/features_info.json` |

#### 🤖 Modelos Treinados

**Classificação (Previsão de Falha):**
- Logistic Regression
- k-Nearest Neighbors (kNN)
- Random Forest
- XGBoost
- LightGBM
- ✅ Melhor modelo selecionado automaticamente

**Regressão (Previsão de Desgaste):**
- Random Forest
- XGBoost
- LightGBM
- ✅ Melhor modelo selecionado automaticamente

#### 📈 Métricas

- **Classificação**: F1-Score (macro)
- **Regressão**: RMSE (Root Mean Squared Error)

---

## 🔄 Fluxo de Execução Completo

```
1. PREPARAÇÃO
   ├── python -m venv venv          (criar ambiente virtual)
   ├── venv\Scripts\activate        (ativar)
   └── pip install -r requirements.txt (instalar deps)

2. TREINAMENTO
   ├── cd train
   ├── python train.py              (treinar modelos)
   └── Saída: models/ e data/

3. BACKEND
   ├── cd ../backend
   ├── echo GOOGLE_API_KEY=... > .env
   └── uvicorn app.main:app --reload

4. FRONTEND
   ├── cd ../frontend
   ├── npm install
   └── npm run dev

5. ACESSO
   ├── Frontend: http://localhost:3000
   ├── Backend API: http://localhost:8000
   └── Docs da API: http://localhost:8000/docs
```

---

## 🌐 Variáveis de Ambiente

### Backend (.env)

```env
# Obrigatório
GOOGLE_API_KEY=seu_api_key_aqui

# Opcional (valores padrão se não especificados)
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
CORS_ORIGINS=*
```

### Frontend

O frontend se conecta ao backend em `http://localhost:8000` por padrão. Se precisar mudar, edite o URL da API em `app/page.tsx`.

---

## 🐛 Troubleshooting

### Backend

| Problema | Solução |
|----------|---------|
| `ModuleNotFoundError: No module named 'fastapi'` | Execute `pip install -r requirements.txt` |
| `ValueError: GOOGLE_API_KEY não definida` | Crie arquivo `.env` com sua chave de API |
| `Port 8000 already in use` | Mude a porta: `uvicorn app.main:app --port 8001` |
| `CORS error ao conectar frontend` | Verifique configuração de CORS em `app/main.py` |

### Frontend

| Problema | Solução |
|----------|---------|
| `npm ERR! 404 Not Found` | Execute `npm install` novamente |
| `Port 3000 already in use` | Execute `npm run dev -- -p 3001` |
| `API connection failed` | Verifique se backend está rodando em `http://localhost:8000` |

### Treinamento

| Problema | Solução |
|----------|---------|
| `FileNotFoundError: predictive_maintenance.csv` | Coloque o arquivo na pasta `train/` |
| `ImportError: No module named 'sklearn'` | Execute `pip install -r requirements.txt` no backend |

---

## 📝 Fluxo de Uso

1. **Usuário acessa** `http://localhost:3000`
2. **Frontend renderiza** página do chatbot
3. **Usuário digita** mensagem (ex: "Qual é a temperatura média?")
4. **Frontend envia** para `POST /chat` no backend
5. **Backend processa** com Gemini AI + ML models
6. **Backend retorna** resposta com análises
7. **Frontend exibe** resultado ao usuário

---

## 🚀 Deploy

### Backend (Heroku, Railway, Render, etc)

```bash
# Exemplo: Railway
railway link
railway up
```

### Frontend (Vercel, Netlify, etc)

```bash
# Exemplo: Vercel
npm install -g vercel
vercel
```

---

## 📖 Documentação Adicional

- **FastAPI**: https://fastapi.tiangolo.com/
- **Next.js**: https://nextjs.org/docs
- **Tailwind CSS**: https://tailwindcss.com/docs
- **Gemini AI**: https://ai.google.dev/

---

## 👥 Contribuição

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

---

## 📄 Licença

Este projeto está sob a licença MIT.

---

## 📞 Suporte

Para dúvidas ou problemas, abra uma issue no repositório.

---

**Desenvolvido com ❤️ para análise de manutenção preditiva**
