# run_full_pipeline.py
# Pipeline: luật kết hợp → đặc trưng hành vi mua kèm → phân cụm → diễn giải → đề xuất chiến lược marketing
# Đáp ứng đủ 7 yêu cầu đề tài, có in kết quả ra màn hình để xem trong VS Code terminal

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

# -----------------------------
# Cấu hình
# -----------------------------
CONFIG = {
    "DATA_PATH": "data/raw/online_retail.csv",
    "COUNTRY": "United Kingdom",
    "OUTPUT_DIR": "data/processed",

    "INVOICE_COL": "InvoiceNo",
    "ITEM_COL": "Description",
    "CUSTOMER_COL": "CustomerID",
    "QUANTITY_COL": "Quantity",
    "DATE_COL": "InvoiceDate",
    "UNIT_PRICE_COL": "UnitPrice",

    "MIN_SUPPORT": 0.01,
    "MAX_LEN": 3,
    "METRIC": "lift",
    "MIN_THRESHOLD": 1.0,
    "FILTER_MIN_SUPPORT": 0.01,
    "FILTER_MIN_CONF": 0.3,
    "FILTER_MIN_LIFT": 1.2,

    "TOP_K_RULES_SMALL": 50,
    "TOP_K_RULES_LARGE": 200,
    "SORT_RULES_BY": "lift",

    "USE_RFM": True,
    "RFM_SCALE": True,
    "RULE_SCALE": False,
    "WEIGHTING": "lift_conf",

    "K_MIN": 2,
    "K_MAX": 10,
    "RANDOM_STATE": 42
}

# -----------------------------
# Các hàm tiện ích
# -----------------------------
def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

# -----------------------------
# 1) Load & clean
# -----------------------------
def load_and_clean(config):
    df = pd.read_csv(config["DATA_PATH"], encoding="ISO-8859-1")
    df[config["DATE_COL"]] = pd.to_datetime(df[config["DATE_COL"]], errors="coerce")
    df = df[df["Country"] == config["COUNTRY"]]
    df = df.dropna(subset=[config["CUSTOMER_COL"], config["ITEM_COL"], config["INVOICE_COL"]])
    df = df[df[config["QUANTITY_COL"]] > 0]
    save_csv(df, os.path.join(config["OUTPUT_DIR"], "cleaned_data.csv"))
    print("✅ Cleaned data:", df.shape)
    return df

# -----------------------------
# 2) Basket
# -----------------------------
def build_basket(df, config):
    basket = (df.groupby([config["INVOICE_COL"], config["ITEM_COL"]])[config["QUANTITY_COL"]]
              .sum().unstack().fillna(0))
    basket_bool = basket.applymap(lambda x: 1 if x > 0 else 0).astype(bool)
    basket_bool.to_parquet(os.path.join(config["OUTPUT_DIR"], "basket_bool.parquet"))
    print("✅ Basket bool:", basket_bool.shape)
    return basket_bool

# -----------------------------
# 3) Khai phá luật
# -----------------------------
def mine_rules(basket_bool, config, method="apriori"):
    if method == "apriori":
        freq = apriori(basket_bool, min_support=config["MIN_SUPPORT"], use_colnames=True, max_len=config["MAX_LEN"])
    else:
        freq = fpgrowth(basket_bool, min_support=config["MIN_SUPPORT"], use_colnames=True, max_len=config["MAX_LEN"])
    rules = association_rules(freq, metric=config["METRIC"], min_threshold=config["MIN_THRESHOLD"])
    rules = rules[(rules["support"] >= config["FILTER_MIN_SUPPORT"]) &
                  (rules["confidence"] >= config["FILTER_MIN_CONF"]) &
                  (rules["lift"] >= config["FILTER_MIN_LIFT"])]
    rules["antecedents"] = rules["antecedents"].apply(lambda s: list(s))
    rules["consequents"] = rules["consequents"].apply(lambda s: list(s))
    save_csv(rules, os.path.join(config["OUTPUT_DIR"], f"rules_{method}.csv"))
    print(f"✅ {method} rules:", rules.shape)
    return rules

# -----------------------------
# 4) Chọn luật Top-K
# -----------------------------
def select_top_rules(rules, config, top_k):
    rules_sorted = rules.sort_values(by=config["SORT_RULES_BY"], ascending=False).head(top_k)
    showcase = rules_sorted[["antecedents","consequents","support","confidence","lift"]].head(10)
    save_csv(showcase, os.path.join(config["OUTPUT_DIR"], "top10_rules.csv"))
    print("📊 Top-10 rules showcase:\n", showcase)
    return rules_sorted

# -----------------------------
# 5) Feature engineering
# -----------------------------
def build_features(rules, df, config):
    cust_items = df.groupby(config["CUSTOMER_COL"])[config["ITEM_COL"]].apply(set).to_dict()
    rule_cols, X_bin_rows = [], []
    cids = list(cust_items.keys())
    for _, r in rules.iterrows():
        ant = set(r["antecedents"])
        col = "rule_" + "_".join(sorted(ant))
        rule_cols.append(col)
        X_bin_rows.append([1 if ant.issubset(cust_items[c]) else 0 for c in cids])
    X_bin = pd.DataFrame(np.array(X_bin_rows).T, columns=rule_cols, index=cids)
    rules["weight"] = rules["lift"]*rules["confidence"] if config["WEIGHTING"]=="lift_conf" else rules["lift"]
    X_w = X_bin.copy()
    for i,col in enumerate(X_w.columns):
        X_w[col] = X_w[col]*rules.iloc[i]["weight"]
    if config["RULE_SCALE"]:
        X_w = pd.DataFrame(StandardScaler().fit_transform(X_w), columns=X_w.columns, index=X_w.index)
    df["Revenue"] = df[config["UNIT_PRICE_COL"]]*df[config["QUANTITY_COL"]]
    snap = df[config["DATE_COL"]].max()+timedelta(days=1)
    rfm = df.groupby(config["CUSTOMER_COL"]).agg({
        config["DATE_COL"]: lambda x: (snap-x.max()).days,
        config["INVOICE_COL"]: "nunique",
        "Revenue": "sum"
    }).rename(columns={config["DATE_COL"]:"Recency",config["INVOICE_COL"]:"Frequency","Revenue":"Monetary"})
    rfm_scaled = pd.DataFrame(StandardScaler().fit_transform(rfm), columns=rfm.columns, index=rfm.index) if config["RFM_SCALE"] else rfm
    save_csv(rfm.reset_index(), os.path.join(config["OUTPUT_DIR"], "rfm.csv"))
    print("✅ Features built: Binary", X_bin.shape, "| Weighted", X_w.shape, "| RFM", rfm_scaled.shape)
    return X_bin, X_w, rfm_scaled

# -----------------------------
# 6) Khảo sát K & huấn luyện
# -----------------------------
def evaluate_k_range(X, k_min, k_max, seed):
    results = []
    for k in range(k_min, k_max+1):
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        results.append({"K":k,"Silhouette":sil})
    return pd.DataFrame(results)

def train_kmeans(X, index, config, name):
    eval_df = evaluate_k_range(X, config["K_MIN"], config["K_MAX"], config["RANDOM_STATE"])
    best_k = int(eval_df.loc[eval_df["Silhouette"].idxmax(),"K"])
    km = KMeans(n_clusters=best_k, random_state=config["RANDOM_STATE"], n_init="auto")
    labels = km.fit_predict(X)
    clusters = pd.DataFrame({"CustomerID":index,"Cluster":labels})
    save_csv(clusters, os.path.join(config["OUTPUT_DIR"], f"clusters_{name}.csv"))
    print("📊 Silhouette scores:\n", eval_df)
    print(f"✅ Best K={best_k}, silhouette={eval_df['Silhouette'].max():.3f}")
    return clusters

# -----------------------------
# 7) Profiling & diễn giải
# -----------------------------
def profiling(clusters, X_rules, rfm_scaled, config):
    # Hợp nhất dữ liệu theo CustomerID
    profile = clusters.set_index("CustomerID").join(rfm_scaled, how="left").join(X_rules, how="left")

    # 1) Số lượng khách hàng theo cụm
    counts = profile.groupby("Cluster").size().reset_index(name="n_customers")
    save_csv(counts, os.path.join(config["OUTPUT_DIR"], "cluster_counts.csv"))
    print("📊 Cluster counts:")
    print(counts.to_string(index=False))

    # 2) RFM mean/median theo cụm (nếu có RFM)
    has_rfm = {"Recency", "Frequency", "Monetary"}.issubset(set(profile.columns))
    if has_rfm:
        rfm_stats = profile.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].agg(["mean", "median"])
        rfm_stats.columns = ["_".join(col) for col in rfm_stats.columns]  # flatten multi-index
        rfm_stats_out = rfm_stats.reset_index()
        save_csv(rfm_stats_out, os.path.join(config["OUTPUT_DIR"], "cluster_rfm_stats.csv"))
        print("\n📈 RFM stats (mean/median) by cluster:")
        print(rfm_stats_out.to_string(index=False))
    else:
        print("\nℹ️ Không tìm thấy cột RFM trong profile — bỏ qua phần thống kê RFM.")

    # 3) Top rule-features kích hoạt nhiều nhất theo cụm
    rule_cols = [c for c in profile.columns if c.startswith("rule_")]
    top_rules = {}
    print("\n🔑 Top 10 rule-features theo từng cụm (giá trị trung bình kích hoạt):")
    for cl in sorted(profile["Cluster"].unique()):
        sub = profile[profile["Cluster"] == cl][rule_cols]
        # mean theo cột → mức độ kích hoạt trung bình của từng rule-feature trong cụm
        scores = sub.mean().sort_values(ascending=False).head(10)
        top_rules[int(cl)] = scores.to_dict()
        print(f"\nCluster {cl}:")
        print(scores.to_string())

    with open(os.path.join(config["OUTPUT_DIR"], "top_rules_per_cluster.json"), "w", encoding="utf-8") as f:
        json.dump(top_rules, f, ensure_ascii=False, indent=2)

    # 4) Template đặt tên cụm & chiến lược
    template = pd.DataFrame({
        "Cluster": sorted(profile["Cluster"].unique()),
        "Name_EN": ["" for _ in sorted(profile["Cluster"].unique())],
        "Name_VN": ["" for _ in sorted(profile["Cluster"].unique())],
        "Persona_1_sentence": ["" for _ in sorted(profile["Cluster"].unique())],
        "Strategy": ["(bundle/cross-sell/upsell/VIP/reactivation)" for _ in sorted(profile["Cluster"].unique())]
    })
    save_csv(template, os.path.join(config["OUTPUT_DIR"], "cluster_naming_strategy_template.csv"))
    print("\n🏷️ Đã tạo template đặt tên cụm & chiến lược: cluster_naming_strategy_template.csv")