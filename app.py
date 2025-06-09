import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Antioxidant Assay Simulator", layout="centered")

st.title("🧪 Antioxidant Activity Simulator (No RDKit)")
st.markdown("Manually enter chemical properties to simulate antioxidant assay results.")

# Input features (simulate chemical properties)
molwt = st.number_input("Molecular Weight", min_value=50.0, max_value=1000.0, value=150.0)
logp = st.number_input("LogP (Partition Coefficient)", min_value=-5.0, max_value=10.0, value=2.0)
tpsa = st.number_input("Topological Polar Surface Area (TPSA)", min_value=0.0, max_value=300.0, value=75.0)
donors = st.number_input("Number of H-Donors", min_value=0, max_value=10, value=2)
acceptors = st.number_input("Number of H-Acceptors", min_value=0, max_value=15, value=3)

if st.button("🧪 Simulate Activity"):
    # Simulated assay results (rule-based)
    results = {
        'DPPH': round(100 - logp * 5, 2),
        'ABTS': round(tpsa / 2, 2),
        'ORAC': round(molwt / 10, 2),
        'FRAP': round(donors * 15, 2),
        'TEAC': round(acceptors * 12, 2),
        'SOD': round(tpsa / 3, 2),
        'Catalase': round(100 - molwt / 5, 2),
        'GPx': round(logp * 7, 2),
        'GR': round((donors + acceptors) * 10, 2)
    }

    df = pd.DataFrame(results.items(), columns=["Assay", "Activity Score"])
    st.subheader("📋 Assay Simulation Results")
    st.dataframe(df)

    st.subheader("📊 Activity Plot")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="Assay", y="Activity Score", palette="coolwarm")
    plt.xticks(rotation=45)
    st.pyplot(fig)
