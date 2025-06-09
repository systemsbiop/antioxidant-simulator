import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pubchempy as pcp

# ----------- Utility Functions -----------

def fetch_properties(smiles):
    try:
        comp = pcp.get_compounds(smiles, 'smiles')
        if not comp:
            return None
        c = comp[0]
        props = {
            'MolWt': float(c.molecular_weight or 0.0),
            'LogP': float(c.xlogp or 0.0),
            'TPSA': float(c.tpsa or 0.0),
            'H_donors': int(c.h_bond_donor_count or 0),
            'H_acceptors': int(c.h_bond_acceptor_count or 0)
        }
        return props
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def simulate_activity(props, concentrations):
    results = {a: [] for a in ['DPPH','ABTS','ORAC','FRAP','TEAC','SOD','Catalase','GPx','GR']}
    for conc in concentrations:
        factor = 1 - 1 / (1 + conc)  # Dose-response factor
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

st.set_page_config(page_title="Antioxidant Assay Simulator", layout="centered")
st.title("🧪 SMILES-Based Antioxidant Activity Simulator")

smiles = st.text_input("🔬 Enter a SMILES string:", value="C1=CC=C(C=C1)O")

if st.button("▶️ Run Simulation"):
    props = fetch_properties(smiles)

    if not props:
        st.error("❌ Could not retrieve compound details. Try another SMILES.")
    elif any(v is None for v in props.values()):
        st.error("❗ Some properties are missing or invalid. Please use a valid chemical structure.")
    else:
        st.success("✅ Fetched molecular properties:")
        st.json(props)

        concentrations = np.logspace(-3, 1, 10)  # 0.001 to 10 mg/mL
        sim_data = simulate_activity(props, concentrations)

        df = pd.DataFrame(sim_data, index=np.round(concentrations, 4))
        df.index.name = "Concentration (mg/mL)"

        st.subheader("📋 Antioxidant Assay Table (Dose-Dependent)")
        st.dataframe(df)

        st.subheader("📈 Dose-Response Graphs for Each Assay")
        fig, ax = plt.subplots(3, 3, figsize=(15, 12))
        assays = df.columns

        for idx, assay in enumerate(assays):
            row, col = divmod(idx, 3)
            ax[row][col].plot(df.index, df[assay], marker='o', color='darkgreen')
            ax[row][col].set_title(f"{assay} Assay")
            ax[row][col].set_xlabel("Concentration (mg/mL)")
            ax[row][col].set_ylabel("Activity")
            ax[row][col].set_xscale("log")
            max_conc = df[assay].idxmax()
            max_val = df[assay].max()
            ax[row][col].axvline(max_conc, color='red', linestyle='--')
            ax[row][col].annotate(f"Peak @ {max_conc}", xy=(max_conc, max_val),
                                  xytext=(5,5), textcoords='offset points', color='red')
        
        plt.tight_layout()
        st.pyplot(fig)
