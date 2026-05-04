import pandas as pd
import numpy as np

# Load dataset with decimal=',' as established before
df = pd.read_csv('Dados_Projeto_Imagem_Ultrassom.xlsx - Planilha1.csv', decimal=',')

# Identify PESO column index
try:
    peso_idx = df.columns.get_loc('PESO')
    columns_after_peso = df.columns[peso_idx + 1:]
    
    # Basic info
    print(f"Columns after PESO: {list(columns_after_peso)}")
    
    # Select only columns after PESO and not Unnamed
    cols_to_check = [c for c in columns_after_peso if not c.startswith('Unnamed')]
    
    subset = df[cols_to_check].copy()
    
    print("\nMissing values per column:")
    print(subset.isnull().sum())
    
    print("\nData types:")
    print(subset.dtypes)
    
    # Check if they are numeric
    for col in subset.columns:
        if not np.issubdtype(subset[col].dtype, np.number):
            print(f"\nColumn {col} is NOT numeric. Attempting to convert...")
            subset[col] = pd.to_numeric(subset[col], errors='coerce')
            print(f"New nulls in {col}: {subset[col].isnull().sum()}")

except Exception as e:
    print(f"Error: {e}")
