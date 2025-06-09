import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pubchempy as pcp

# ----------- Utility Functions -----------

def fetch_properties(smiles):
    comp = pcp.get_compounds(smiles, 'smiles')
    if not comp:
        return None
    c = comp[0]
    props = {
        'MolWt': c.molecular_weight,
        'LogP': c.xlogp or 0.0,
        'TPSA': c.tpsa or 0.0,
        'H_donors': c.h_bond_donor_count or 0,
        'H_acceptors': c.h_bond_acceptor_count or 0
    }
    return props

def simulate_activity(props, concentrations):
    results = {a: [] for a in ['DPPH','ABTS','ORAC','FRAP','TEAC','SOD','Catalase','GPx','GR']}
    for conc in concentrations:
        factor = 1 - 1/(1 + conc)  # Dose-response logic
        results['DPPH'].append(round((100 - props['LogP']*5) * factor, 3))
        results['ABTS'].append(round((props['TPSA']/2) * factor, 3))
        results['ORAC'].append(round((props['MolWt']/10) * factor, 3))
        results['FRAP'].append(round((props['H_donors']*15) * factor, 3))
        results['TEAC'].append(round((props['H_acceptors']*12) * factor, 3))
        results['SOD'].append(round((props['TPSA']/3) * factor, 3))
        results['Catalase'].append(round((100 - props['MolWt']/5) * factor, 3))
        results['GPx'].append(round((props['LogP']*7) * factor, 3))
        results['GR'].append(round(((props['H_donors']+props['H_acceptors'])*10) * factor, 3))
    return results

# ----------- Streamlit App -----------

st.set_page_config(page_title="SMILES → Antioxidant Simulator", layout="centered")
st.title("🧪 SMILES‑Based Antioxidant Activity Simulator")

smiles = st.text_input("Enter a SMILES string (e.g. C1=CC=C(C=C1)O):", value="C1=CC=C(C=C1)O")

if st.button("Simulate Antioxidant Activity"):
    props = fetch_properties(smiles)
    
    if not props:
        st.error("❌ Could not retrieve molecular info from PubChem.")
    else:
        st.success("✅ Molecular properties retrieved from PubChem:")
        st.json(props)

        concentrations = np.logspace(-3, 1, 10)  # 0.001 to 10 mg/mL
        sim_data = simulate_activity(props, concentrations)

        df = pd.DataFrame(sim_data, index=np.round(concentrations, 4))
        df.index.name = "Concentration (mg/mL)"

        st.subheader("📋 Assay Simulation Table")
        st.dataframe(df)

        st.subheader("📊 Dose-Dependent Antioxidant Activity")
        fig, ax = plt.subplots(3, 3, figsize=(15, 12))
        assays = list(df.columns)

        for idx, assay in enumerate(assays):
            row, col = divmod(idx, 3)
            ax[row][col].plot(df.index, df[assay], marker='o')
            ax[row][col].set_title(assay)
            ax[row][col].set_xscale('log')
            ax[row][col].set_xlabel("mg/mL")
            ax[row][col].set_ylabel("Activity")
            max_idx = df[assay].idxmax()
            ax[row][col].axvline(max_idx, color='red', linestyle='--')
            ax[row][col].annotate(f"Peak @ {max_idx}", xy=(max_idx, df[assay].max()), 
                                  xytext=(5,5), textcoords='offset points', color='red')
        
        plt.tight_layout()
        st.pyplot(fig)
