# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_squared_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import json 

# Configurações
warnings.filterwarnings("ignore")
RANDOM_SEED = 42

# ===================================================================
# 2. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ===================================================================

caminho_do_arquivo = 'predictive_maintenance.csv'

def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Erro: Arquivo {filepath} não encontrado.")
        return None, None, None, None, None

    df = pd.read_csv(filepath)
    df = df.drop(columns=['UDI', 'Product ID'], errors='ignore')

    # Salva uma cópia limpa para o backend usar
    # Criamos uma pasta 'data' para organizar
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/predictive_maintenance_cleaned.csv', index=False)
    print("DataFrame limpo salvo em 'data/predictive_maintenance_cleaned.csv'")

    df_ml = df.copy()
    le = LabelEncoder()
    df_ml['Type'] = le.fit_transform(df_ml['Type'])
    
    # Salvar o LabelEncoder para o backend
    os.makedirs('models', exist_ok=True)
    joblib.dump(le, 'models/type_label_encoder.pkl')
    print("LabelEncoder 'Type' salvo em 'models/type_label_encoder.pkl'")


    features_classification = [
        'Type', 'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
    ]
    target_classification = 'Target'

    features_regression = [
        'Type', 'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]'
    ]
    target_regression = 'Tool wear [min]'

    X_class = df_ml[features_classification]
    y_class = df_ml[target_classification]
    X_reg = df_ml[features_regression]
    y_reg = df_ml[target_regression]

    print("Dados carregados e features definidas.")
    return X_class, y_class, X_reg, y_reg, df.columns

# ===================================================================
# 3. PIPELINE DE ML - CLASSIFICAÇÃO
# ===================================================================
def train_classification_models(X, y):
    print("\n--- Iniciando Pipeline de Classificação (Previsão de Falha) ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    X_train.columns = X_train.columns.str.replace(r'\[|\]|<', '', regex=True)
    X_test.columns = X_test.columns.str.replace(r'\[|\]|<', '', regex=True)
    
    # Renomear colunas para consistência
    feature_names_cleaned = X_train.columns
    
    model_configs = {
        'LogisticRegression': {'model': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(random_state=RANDOM_SEED))])},
        'kNN': {'model': Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier(n_neighbors=5))])},
        'RandomForest': {'model': RandomForestClassifier(random_state=RANDOM_SEED, n_estimators=100)},
        'XGBoost': {'model': XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss')},
        'LightGBM': {'model': LGBMClassifier(random_state=RANDOM_SEED, verbose=-1)}
    }
    
    results, best_model, best_f1, best_model_name = [], None, -1.0, ""

    for name, config in model_configs.items():
        print(f"Treinando {name}...")
        model = config['model']
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        if f1 > best_f1:
            best_f1, best_model, best_model_name = f1, model, name

    print(f"\nMelhor modelo de classificação: {best_model_name} (F1: {best_f1:.4f})")
    joblib.dump(best_model, 'models/best_classifier_model.pkl')
    print("Modelo de classificação salvo em 'models/best_classifier_model.pkl'")

    try:
        if hasattr(best_model, 'steps'): model_step = best_model.steps[-1][1]
        else: model_step = best_model

        if hasattr(model_step, 'feature_importances_'): importances = model_step.feature_importances_
        elif hasattr(model_step, 'coef_'): importances = np.abs(model_step.coef_[0])
        else: importances = np.zeros(len(feature_names_cleaned))
        
        importances_dict = dict(zip(feature_names_cleaned, importances))
        joblib.dump(importances_dict, 'models/classifier_importances.pkl')
        print("Importância das features (XAI) salva em 'models/classifier_importances.pkl'")
    except Exception as e:
        print(f"Erro ao salvar XAI de classificação: {e}")
    
    return feature_names_cleaned

# ===================================================================
# 4. PIPELINE DE ML - REGRESSÃO
# ===================================================================
def train_regression_models(X, y):
    print("\n--- Iniciando Pipeline de Regressão (Previsão de Desgaste) ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    X_train.columns = X_train.columns.str.replace(r'\[|\]|<', '', regex=True)
    X_test.columns = X_test.columns.str.replace(r'\[|\]|<', '', regex=True)
    
    # Renomear colunas para consistência
    feature_names_cleaned = X_train.columns

    model_configs = {
        'RandomForest': {'model': RandomForestRegressor(random_state=RANDOM_SEED, n_estimators=100)},
        'XGBoost': {'model': XGBRegressor(random_state=RANDOM_SEED)},
        'LightGBM': {'model': LGBMRegressor(random_state=RANDOM_SEED, verbose=-1)}
    }
    
    results, best_model, best_rmse, best_model_name = [], None, float('inf'), ""

    for name, config in model_configs.items():
        print(f"Treinando {name}...")
        model = config['model']
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        if rmse < best_rmse:
            best_rmse, best_model, best_model_name = rmse, model, name
            
    print(f"\nMelhor modelo de regressão: {best_model_name} (RMSE: {best_rmse:.4f})")
    joblib.dump(best_model, 'models/best_regressor_model.pkl')
    print("Modelo de regressão salvo em 'models/best_regressor_model.pkl'")

    try:
        importances = best_model.feature_importances_
        importances_dict = dict(zip(feature_names_cleaned, importances))
        joblib.dump(importances_dict, 'models/regressor_importances.pkl')
        print("Importância das features (XAI) salva em 'models/regressor_importances.pkl'")
    except Exception as e:
        print(f"Erro ao salvar XAI de regressão: {e}")
        
    return feature_names_cleaned

# --- Execução Principal ---
if __name__ == "__main__":
    X_class, y_class, X_reg, y_reg, original_cols = load_data(filepath=caminho_do_arquivo)
    
    if X_class is not None:
        # Captura os nomes limpos das features
        clf_features_cleaned = train_classification_models(X_class, y_class)
        reg_features_cleaned = train_regression_models(X_reg, y_reg)
        
        # --- (NOVA LÓGICA DE ALIAS) ---
        
        # 1. Define um mapeamento mestre de "Nome Original" para "Lista de Variações"
        mapping_to_original = {
            "Type": ["type", "tipo", "tipo maquina"],
            "Air temperature [K]": [
                "air temperature [k]",
                "air temperature k",
                "air_temp_k",
                "temperatura ar"
            ],
            "Process temperature [K]": [
                "process temperature [k]",
                "process temperature k",
                "process_temp_k",
                "temperatura processo"
            ],
            "Rotational speed [rpm]": [
                "rotational speed [rpm]",
                "rotational speed rpm",
                "rotation_rpm",
                "velocidade",
                "rpm"
            ],
            "Torque [Nm]": [
                "torque [nm]",
                "torque nm",
                "torque_nm",
                "torque"
            ],
            "Tool wear [min]": [
                "tool wear [min]",
                "tool wear min",
                "tool_wear_min",
                "desgaste",
                "desgaste ferramenta"
            ],
            "Target": [
                "target",
                "falha",
                "machine failure"
            ],
            "Failure Type": [
                "failure type",
                "failure_type",
                "tipo de falha"
            ]
        }

        # 2. Constrói o dicionário 'column_aliases' final
        column_aliases = {}
        for original_name, alias_list in mapping_to_original.items():
            if original_name in original_cols:
                for alias in alias_list:
                    column_aliases[alias.lower()] = original_name

        # --- GERAÇÃO DO PROMPT DAS COLUNAS ---
        columns_prompt = f"""
CONTEXTO DO DATASET - COLUNAS DISPONÍVEIS:

O dataset de manutenção preditiva contém as seguintes colunas oficiais (use sempre estes nomes exatos):

{', '.join(original_cols)}

DESCRIÇÃO DAS COLUNAS PRINCIPAIS:
- Type: Tipo de máquina (L, M, H)
- Air temperature [K]: Temperatura do ar em Kelvin
- Process temperature [K]: Temperatura do processo em Kelvin  
- Rotational speed [rpm]: Velocidade rotacional em RPM
- Torque [Nm]: Torque em Newton-metro
- Tool wear [min]: Desgaste da ferramenta em minutos
- Target: Indica se houve falha (1) ou não (0)
- Failure Type: Tipo específico de falha (se aplicável)

ALIAS RECONHECIDOS (mapeamento automático):
O sistema reconhece automaticamente os seguintes sinônimos para cada coluna:
{json.dumps(column_aliases, indent=2, ensure_ascii=False)}

REGRA CRÍTICA: 
Ao receber uma consulta do usuário, PRIMEIRO identifique quais colunas estão sendo mencionadas 
usando o mapeamento de aliases acima. SEMPRE use os nomes oficiais das colunas ao chamar 
qualquer função do sistema. Se o usuário usar um sinônimo, converta automaticamente para 
o nome oficial antes de prosseguir.

EXEMPLOS DE CONVERSÃO:
- "temperatura ar" → "Air temperature [K]"
- "rpm" → "Rotational speed [rpm]" 
- "desgaste" → "Tool wear [min]"
- "falha" → "Target"
"""

        # Salva os nomes das features
        features_info = {
            'classification_features': list(X_class.columns),
            'regression_features': list(X_reg.columns),
            'classification_features_cleaned': list(clf_features_cleaned),
            'regression_features_cleaned': list(reg_features_cleaned),
            'original_columns': list(original_cols),
            'column_aliases': column_aliases,
            'columns_prompt': columns_prompt  # NOVO: prompt completo sobre colunas
        }
        
        with open('models/features_info.json', 'w', encoding='utf-8') as f:
            json.dump(features_info, f, indent=4, ensure_ascii=False)
            
        print("Informações de features (com aliases expandidos) salvas em 'models/features_info.json'")
        
    print("\nTreinamento concluído. Artefatos salvos nas pastas 'models/' e 'data/'.")