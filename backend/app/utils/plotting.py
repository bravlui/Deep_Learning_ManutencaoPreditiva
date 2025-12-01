import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import base64

import os
import uuid

# Configurações de plotagem
sns.set_theme(style="whitegrid")

# (NOVO) Define o diretório onde as imagens serão salvas
STATIC_DIR = "app/static"
# Garante que o diretório existe (o main.py já faz, mas é bom ter redundância)
os.makedirs(STATIC_DIR, exist_ok=True)

def create_feature_importance_plot(importances_dict: dict, title: str) -> str:
    """Gera um gráfico de importância e retorna como Base64 Data URI."""
    if not importances_dict:
        return ""
    
    sorted_features = sorted(importances_dict.items(), key=lambda item: item[1], reverse=True)
    feature_names = [item[0] for item in sorted_features]
    importance_values = [item[1] for item in sorted_features]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance_values, y=feature_names, palette="viridis")
    plt.title(f'XAI: Importância das Features - {title}', fontsize=16)
    plt.xlabel('Importância', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    # (NOVO) Gera um nome de arquivo único
    filename = f"plot_xai_{uuid.uuid4()}.png"
    save_path = os.path.join(STATIC_DIR, filename)
    
    # (MODIFICADO) Salva o gráfico no arquivo
    plt.savefig(save_path, format='png', bbox_inches='tight', dpi=96) # Mantém o dpi=96!
    plt.close() # Fecha a figura para liberar memória
    
    # (MODIFICADO) Retorna apenas o nome do arquivo
    return filename

def create_data_distribution_plot(df: pd.DataFrame, column_name: str, hue_column: str = None) -> str:
    """Gera um gráfico de distribuição e retorna como Base64 Data URI."""
    try:
        plt.figure(figsize=(10, 6))

        if pd.api.types.is_numeric_dtype(df[column_name]) and df[column_name].nunique() > 20:
            sns.histplot(data=df, x=column_name, hue=hue_column, kde=True, palette="viridis", multiple="stack" if hue_column else "layer")
            plt.title(f'Distribuição de {column_name}', fontsize=16)
            plt.ylabel('Densidade/Contagem', fontsize=12)
        else:
            sns.countplot(data=df, x=column_name, hue=hue_column, palette="viridis")
            plt.title(f'Contagem de {column_name}', fontsize=16)
            plt.ylabel('Contagem', fontsize=12)

        plt.xlabel(column_name, fontsize=12)
        plt.tight_layout()
        
        # (NOVO) Gera um nome de arquivo único
        filename = f"plot_dist_{uuid.uuid4()}.png"
        save_path = os.path.join(STATIC_DIR, filename)

        # (MODIFICADO) Salva o gráfico no arquivo
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=96)
        plt.close() # Fecha a figura para liberar memória
        
        # (MODIFICADO) Retorna apenas o nome do arquivo
        return filename
        
    except Exception as e:
        plt.close()
        print(f"Erro ao gerar gráfico: {e}")
        return ""