# streamlit_app.py
# Dashboard Streamlit cho mini project Data Mining
# Đọc các file output từ run_full_pipeline.py và hiển thị trực quan

import streamlit as st
import pandas as pd
import json
import os

DATA_DIR = "data/processed"

# -----------------------------
# Load dữ liệu
# -----------------------------
def load_data():
    clusters = pd.read_csv(os.path.join(DATA_DIR, "clusters_rule_rfm_weighted.csv"))
    rules = pd.read_csv(os.path.join(DATA_DIR, "rules_apriori.csv"))
    rfm = pd.read_csv(os.path.join(DATA_DIR, "rfm.csv")) if os.path.exists(os.path.join(DATA_DIR, "rfm.csv")) else None
    comparison = pd.read_csv(os.path.join(DATA_DIR, "comparison_summary.csv")) if os.path.exists(os.path.join(DATA_DIR, "comparison_summary.csv")) else None

    top_rules_per_cluster = {}
    trpc_path = os.path.join(DATA_DIR, "top_rules_per_cluster.json")
    if os.path.exists(trpc_path):
        with open(trpc_path, "r", encoding="utf-8") as f:
            top_rules_per_cluster = json.load(f)

    return clusters, rules, rfm, comparison, top_rules_per_cluster

# -----------------------------
# Giao diện chính
# -----------------------------
def main():
    st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")
    st.title("📊 Customer Segmentation — Rules & RFM")

    clusters, rules, rfm, comparison, top_rules_per_cluster = load_data()

    # Sidebar: chọn cụm
    cluster_ids = sorted(clusters['Cluster'].unique())
    selected_cluster = st.sidebar.selectbox("Chọn cụm khách hàng", cluster_ids)
    sort_by = st.sidebar.selectbox("Sắp xếp luật theo", ["lift","confidence"])
    top_n = st.sidebar.slider("Top N rules", 5, 50, 10)

    # -----------------------------
    # Thống kê cụm
    # -----------------------------
    cust_in_cluster = clusters[clusters['Cluster'] == selected_cluster]['CustomerID']
    st.subheader(f"Cluster {selected_cluster} — {len(cust_in_cluster)} khách hàng")

    if rfm is not None:
        rfm_cluster = rfm[rfm['CustomerID'].isin(cust_in_cluster)]
        st.write("📈 RFM summary (mean/median):")
        st.write(rfm_cluster[['Recency','Frequency','Monetary']].agg(['mean','median']))

    # -----------------------------
    # Top rules (minh chứng chất lượng luật)
    # -----------------------------
    rules['antecedents'] = rules['antecedents'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    rules['consequents'] = rules['consequents'].apply(lambda x: eval(x) if isinstance(x, str) else x)

    st.subheader("🔑 Top rules (toàn bộ tập, sắp xếp để tham khảo)")
    st.write(rules.sort_values(by=sort_by, ascending=False).head(top_n)[['antecedents','consequents','support','confidence','lift']])

    # -----------------------------
    # Gợi ý bundle/cross-sell theo cụm
    # -----------------------------
    st.subheader("💡 Gợi ý bundle/cross-sell theo cụm")
    if str(selected_cluster) in top_rules_per_cluster:
        st.write(pd.DataFrame.from_dict(top_rules_per_cluster[str(selected_cluster)], orient="index", columns=["score"]).head(10))
    else:
        st.info("Chưa có dữ liệu top_rules_per_cluster cho cụm này.")

    # -----------------------------
    # Hiển thị PCA 2D
    # -----------------------------
    pca_path = os.path.join(DATA_DIR, "pca_rule_rfm_weighted.png")
    if os.path.exists(pca_path):
        st.subheader("🌀 PCA 2D — mức độ tách cụm")
        st.image(pca_path, caption="Rule+RFM Weighted — PCA 2D", use_column_width=True)

    # -----------------------------
    # So sánh biến thể đặc trưng
    # -----------------------------
    if comparison is not None:
        st.subheader("⚖️ So sánh các biến thể đặc trưng")
        st.write(comparison)

    # -----------------------------
    # Template đặt tên cụm & chiến lược
    # -----------------------------
    template_path = os.path.join(DATA_DIR, "cluster_naming_strategy_template.csv")
    if os.path.exists(template_path):
        st.subheader("🏷️ Template đặt tên cụm & chiến lược")
        st.write(pd.read_csv(template_path))
if __name__ == "__main__":
    main()