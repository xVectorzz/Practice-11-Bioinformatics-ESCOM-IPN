import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.impute import SimpleImputer
from pysr import PySRRegressor

def clean_smiles(smiles_str):
    if not isinstance(smiles_str, str):
        return None
    s = smiles_str.split(',')[0]
    s = s.split('|')[0]
    return s.strip()

def get_descriptors(smiles_list):
    data = []
    valid_indices = []
    descriptor_funcs = Descriptors._descList
    desc_names = [x[0] for x in descriptor_funcs]
    
    print(f"Calculating descriptors for {len(smiles_list)} molecules...")
    
    for i, smiles in enumerate(smiles_list):
        if i % 100 == 0: print(f"Processing {i}/{len(smiles_list)}...", end='\r')
        
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            try:
                desc_vals = [func(mol) for _, func in descriptor_funcs]
                data.append(desc_vals)
                valid_indices.append(i)
            except Exception:
                continue
        else:
            continue
            
    return np.array(data), desc_names, valid_indices

def main():
    file_train = 'DOWNLOAD-hr6KC1YSCwQCRLlvoJVxzPSj0LO_NXuYV9VbHLdMK50_eq_.csv'
    file_target = 'Launched 30may25.txt'
    
    print("Loading data...")
    df_train = pd.read_csv(file_train, sep=';')
    df_target = pd.read_csv(file_target, sep='\t')
    
    df_train = df_train.dropna(subset=['pChEMBL Value', 'Smiles']).copy()
    
    df_target['cleaned_smiles'] = df_target['SMILES'].apply(clean_smiles)
    df_target = df_target.dropna(subset=['cleaned_smiles'])
    
    print("\n--- Generating Descriptors (Train) ---")
    X_train_raw, desc_names, idx_train = get_descriptors(df_train['Smiles'].tolist())
    y_train = df_train['pChEMBL Value'].iloc[idx_train].values
    
    print("\n--- Generating Descriptors (Target FDA) ---")
    X_target_raw, _, idx_target = get_descriptors(df_target['cleaned_smiles'].tolist())
    df_target_valid = df_target.iloc[idx_target].copy()
    
    X_train_raw[np.isinf(X_train_raw)] = np.nan
    X_target_raw[np.isinf(X_target_raw)] = np.nan
    
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train_raw)
    X_target = imputer.transform(X_target_raw)
    
    print("\n--- Training PySR Model ---")
    model = PySRRegressor(
        niterations=40,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["cos", "exp", "sin", "log", "sqrt"],
        model_selection="best",
        verbosity=1
    )
    
    model.fit(X_train, y_train)
    
    print("\nBest equation found:")
    print(model.sympy())
    
    with open("pysr_equation.txt", "w") as f:
        f.write(str(model.sympy()))
        
    print("\n--- Predicting on FDA molecules ---")
    y_pred = model.predict(X_target)

    df_target_valid['Predicted_pChEMBL'] = y_pred
    
    results = df_target_valid.sort_values(by='Predicted_pChEMBL', ascending=False)
    
    print("\nTop 10 Repurposing Candidates:")
    print(results[['Name', 'Predicted_pChEMBL']].head(10))
    
    results.to_csv('fda_repurposing_candidates.csv', index=False)
    print("\nResults saved to 'fda_repurposing_candidates.csv'")

if __name__ == "__main__":
    main()