import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import fetch_properties, simulate_activity

st.set_page_config(page_title="SMILES → Antioxidant Simulator", layout="centered")
st.title("🧪 SMILES‑based Antioxidant Simulator")

smiles = st.text_input("Enter SMILES string:", value="C1=CC=C(C=C1)O")
if st.button("Simulate"):
    props = fetch_properties(smiles)
    if not props:
        st.error("❌ Could not fetch from PubChem.")
    else:
        st.success("✅ Retrieved molecular properties:")
        st.json(props)
        
        concs = np.logspace(-3, 1, num=10)  # 0.001 to 10 mg/mL
        sim = simulate_activity(props, concs)
        
        # Create a melt table
        df = pd.DataFrame(sim, index=np.round(concs, 4))
        df.index.name = "Concentration (mg/mL)"
        st.subheader("📋 Dose‑Response Table")
        st.dataframe(df)

        st.subheader("📊 Dose‑Response Graphs")
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
            ax[row][col].annotate(f"Peak @ {max_idx}", xy=(max_idx, df[assay].max()), xytext=(5,5),
                                  textcoords='offset points', color='red')
        plt.tight_layout()
        st.pyplot(fig)
