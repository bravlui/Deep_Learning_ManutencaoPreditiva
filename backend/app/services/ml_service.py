import joblib
import pandas as pd
import numpy as np
import json
from app.utils.plotting import create_feature_importance_plot, create_data_distribution_plot

BACKEND_BASE_URL = "http://localhost:8000"

# --- Carregamento de Artefatos na Inicialização ---
try:
    # Carrega modelos
    model_classifier = joblib.load('models/best_classifier_model.pkl')
    model_regressor = joblib.load('models/best_regressor_model.pkl')
    
    # Carrega XAI
    importances_classifier = joblib.load('models/classifier_importances.pkl')
    importances_regressor = joblib.load('models/regressor_importances.pkl')
    
    # Carrega infos de features
    with open('models/features_info.json', 'r') as f:
        features_info = json.load(f)
        
    CLASS_FEATURES_CLEANED = features_info['classification_features_cleaned']
    REG_FEATURES_CLEANED = features_info['regression_features_cleaned']
    COLUMN_ALIASES = features_info['column_aliases']
    COLUMN_ALIASES['machine failure'] = 'Target'
    
    # Carrega o Label Encoder
    le_type = joblib.load('models/type_label_encoder.pkl')

    # Carrega o DataFrame para análise
    df_for_analysis = pd.read_csv('data/predictive_maintenance_cleaned.csv')
    
    print("Serviço de ML: Modelos, XAI e Dados carregados com sucesso.")

except Exception as e:
    print(f"ERRO CRÍTICO ao carregar modelos: {e}")
    print("Certifique-se de executar o script 'train.py' primeiro.")
    model_classifier = None
    # ... (adicionar tratamento de erro, se necessário)

# ===================================================================
# FERRAMENTAS PARA O GEMINI (Refatoração do Bloco 5)
# ===================================================================

def run_prediction(type_machine: str, air_temp_k: float, process_temp_k: float, rotation_rpm: float, torque_nm: float, tool_wear_min: float) -> str:
    """
    Executa a previsão de falha (classificação) e desgaste (regressão).
    O 'type_machine' deve ser 'L', 'M' ou 'H'.
    """
    if not all([model_classifier, model_regressor, le_type]):
         return json.dumps({"error": "Modelos de ML não estão carregados no servidor."})
    try:
        # Codificar 'type_machine'
        try:
            type_machine_encoded = le_type.transform([type_machine])[0]
        except:
            return json.dumps({"error": f"Tipo de máquina '{type_machine}' inválido. Use 'L', 'M' ou 'H'."})

        # Preparar dados para classificação
        class_data = [[
            type_machine_encoded, air_temp_k, process_temp_k,
            rotation_rpm, torque_nm, tool_wear_min
        ]]
        class_data_df = pd.DataFrame(class_data, columns=CLASS_FEATURES_CLEANED)
        
        # Previsão de Classificação
        prob_falha = model_classifier.predict_proba(class_data_df)[0][1]
        
        # Preparar dados para regressão
        reg_data = [[
            type_machine_encoded, air_temp_k, process_temp_k,
            rotation_rpm, torque_nm
        ]]
        reg_data_df = pd.DataFrame(reg_data, columns=REG_FEATURES_CLEANED)
        
        # Previsão de Regressão
        desgaste_previsto = model_regressor.predict(reg_data_df)[0]
        
        limite_desgaste = 240 # (Definido no seu código original)
        rul_estimado = max(0, limite_desgaste - desgaste_previsto)
        
        results = {
            "probability_of_failure": float(prob_falha),
            "predicted_tool_wear_min": float(desgaste_previsto),
            "estimated_rul_min": float(rul_estimado),
            "rul_limit_threshold": limite_desgaste
        }
        return json.dumps(results)
        
    except Exception as e:
        return json.dumps({"error": f"Erro durante a previsão: {str(e)}"})

def generate_explanation(model_to_explain: str) -> str:
    """
    Gera um gráfico XAI, salva em disco e retorna um JSON com a URL pública da imagem.
    """
    # (MODIFICADO)
    if model_to_explain.lower() == 'classification':
        filename = create_feature_importance_plot(
            importances_dict=importances_classifier,
            title="Previsão de Falha (Classificação)"
        )
        image_url = f"{BACKEND_BASE_URL}/static/{filename}"
        return json.dumps({"image_url": image_url})

    elif model_to_explain.lower() == 'regression':
        filename = create_feature_importance_plot(
            importances_dict=importances_regressor,
            title="Previsão de Desgaste (Regressão)"
        )
        image_url = f"{BACKEND_BASE_URL}/static/{filename}"
        return json.dumps({"image_url": image_url})
    else:
        return json.dumps({"error": "Modelo desconhecido. Use 'classification' ou 'regression'."})

def get_dataset_summary() -> str:
    """Retorna um sumário estatístico do dataset de manutenção."""
    if df_for_analysis is None:
        return json.dumps({"error": "DataFrame 'df_for_analysis' não foi carregado."})
    try:
        numeric_df = df_for_analysis.select_dtypes(include=np.number)
        description = numeric_df.describe().to_dict()
        type_counts = df_for_analysis['Type'].value_counts().to_dict()
        failure_counts = df_for_analysis['Target'].value_counts().to_dict()
        summary = {
            "total_records": int(len(df_for_analysis)),
            "machine_type_counts": type_counts,
            "failure_counts (0=No, 1=Yes)": failure_counts,
            "numeric_statistics": description
        }
        return json.dumps(summary, default=str)
    except Exception as e:
        return json.dumps({"error": f"Erro ao gerar sumário: {str(e)}"})

def plot_data_distribution(column_name: str, hue_column: str = None) -> str:
    """
    Gera um gráfico de distribuição para uma coluna do dataset.
    Retorna um JSON com o Data URI da imagem em Base64.
    """
    if df_for_analysis is None:
        return json.dumps({"error": "DataFrame 'df_for_analysis' não foi carregado."})

    # Mapeamento de Sinônimos
    def get_real_column(name):
        if not name: return None
        name_lower = name.lower().strip()
        if name_lower in COLUMN_ALIASES:
            return COLUMN_ALIASES[name_lower]
        # Se não for um alias, verifica se o nome original existe
        if name in df_for_analysis.columns:
            return name
        return None # Nome inválido

    real_column_name = get_real_column(column_name)
    real_hue_column = get_real_column(hue_column)

    if not real_column_name:
        return json.dumps({"error": f"Coluna '{column_name}' não encontrada."})
    if hue_column and not real_hue_column:
         return json.dumps({"error": f"Coluna 'hue' '{hue_column}' não encontrada."})

    # (MODIFICADO)
    try:
        filename = create_data_distribution_plot(df_for_analysis, real_column_name, real_hue_column)
        if not filename:
             raise Exception("Plotting function returned no filename.")
             
        image_url = f"{BACKEND_BASE_URL}/static/{filename}"
        return json.dumps({"image_url": image_url})
    except Exception as e:
        return json.dumps({"error": f"Erro ao gerar gráfico: {str(e)}"})