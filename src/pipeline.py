import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import os

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = "data/flipkard.csv"
OUTPUT_DIR = "outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Load ───────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load]  {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"[load]  Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return df


# ── 2. Feature Engineering ────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["listing_date"]        = pd.to_datetime(df["listing_date"])
    df["listing_year"]        = df["listing_date"].dt.year
    df["listing_month"]       = df["listing_date"].dt.month
    df["listing_age_days"]    = (pd.Timestamp("2026-01-01") - df["listing_date"]).dt.days
    df["engagement"]          = df["rating"] * np.log1p(df["review_count"])
    df["stock_sold_ratio"]    = df["units_sold"] / (df["stock_available"] + df["units_sold"] + 1)
    df["payment_modes_count"] = df["payment_modes"].str.split(",").apply(len)
    df["size"]                = df["size"].fillna("Unknown")
    df["revenue"]             = df["final_price"] * df["units_sold"]
    print("[feat]  7 features engineered")
    return df


# ── 3. Leakage Check ──────────────────────────────────────────────────────────
def check_leakage(df: pd.DataFrame) -> None:
    """
    final_price = price * (1 - discount_percent / 100)
    Including `price` or `discount_amount` as features would be data leakage
    because the target is deterministically derivable from them.
    """
    calc = df["price"] * (1 - df["discount_percent"] / 100)
    leakage = np.allclose(calc, df["final_price"], atol=1)
    print(f"[leak]  Leakage detected via price+discount: {leakage} → excluded from features")


# ── 4. Prepare X / y ─────────────────────────────────────────────────────────
NUM_FEATURES = [
    "discount_percent", "rating", "review_count", "stock_available",
    "units_sold", "delivery_days", "weight_g", "warranty_months",
    "return_policy_days", "shipping_weight_g", "product_score", "seller_rating",
    "listing_year", "listing_month", "listing_age_days",
    "engagement", "stock_sold_ratio", "payment_modes_count",
]
CAT_FEATURES = ["category", "brand", "seller_city", "size"]
TARGET       = "final_price"


def prepare_splits(df: pd.DataFrame):
    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[split] Train: {len(Xtr):,}  Test: {len(Xte):,}")
    return Xtr, Xte, ytr, yte


# ── 5. Preprocessor ───────────────────────────────────────────────────────────
def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ])


# ── 6 & 7. Train + Evaluate ───────────────────────────────────────────────────
def train_and_evaluate(Xtr, Xte, ytr, yte) -> dict:
    prep = build_preprocessor()
    models = {
        "Ridge Regression":  Ridge(alpha=100),
        "Random Forest":     RandomForestRegressor(n_estimators=80, max_depth=12,
                                                    n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=80, max_depth=5,
                                                        learning_rate=0.1, random_state=42),
    }
    results = {}
    for name, model in models.items():
        pipe = Pipeline([("prep", prep), ("model", model)])
        pipe.fit(Xtr, ytr)
        yp   = pipe.predict(Xte)
        r2   = r2_score(yte, yp)
        mae  = mean_absolute_error(yte, yp)
        rmse = np.sqrt(mean_squared_error(yte, yp))
        results[name] = {"r2": r2, "mae": mae, "rmse": rmse, "pipe": pipe, "preds": yp}
        print(f"[eval]  {name:<25}  R²={r2:.4f}  MAE=₹{mae:,.0f}  RMSE=₹{rmse:,.0f}")
    best = max(results, key=lambda k: results[k]["r2"])
    print(f"[best]  {best}  R²={results[best]['r2']:.4f}")
    return results, best


# ── 8. Dashboard ──────────────────────────────────────────────────────────────
def plot_dashboard(df: pd.DataFrame, results: dict, best: str, yte) -> None:
    BLUE, ACCENT, MID = "#0f3460", "#e94560", "#2a6fa8"
    LIGHT, GOLD, GREEN = "#f7f9fc", "#f5a623", "#27ae60"

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("white")

    # 1 — Listings per year
    ax1 = fig.add_subplot(3, 3, 1)
    yr = df.groupby("year").size()
    colors = [BLUE] * (len(yr) - 1) + [ACCENT]
    ax1.bar(yr.index, yr.values, color=colors, edgecolor="white", zorder=3)
    ax1.set_title("Product Listings by Year", fontweight="bold", color=BLUE, fontsize=11)
    ax1.set_ylabel("No. of Products"); ax1.set_facecolor(LIGHT)
    ax1.grid(axis="y", alpha=0.3, zorder=0)

    # 2 — Revenue by category
    ax2 = fig.add_subplot(3, 3, 2)
    cat_rev = df.groupby("category")["revenue"].sum().sort_values()
    ax2.barh(cat_rev.index, cat_rev.values / 1e9,
             color=[BLUE if v != cat_rev.max() else ACCENT for v in cat_rev.values],
             edgecolor="white")
    ax2.set_title("Total Revenue by Category (₹B)", fontweight="bold", color=BLUE, fontsize=11)
    ax2.set_facecolor(LIGHT)

    # 3 — Discount distribution
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.hist(df["discount_percent"], bins=40, color=BLUE, edgecolor="white", alpha=0.85, zorder=3)
    ax3.axvline(df["discount_percent"].mean(), color=ACCENT, lw=2, ls="--",
                label=f"Mean: {df['discount_percent'].mean():.1f}%")
    ax3.set_title("Discount % Distribution", fontweight="bold", color=BLUE, fontsize=11)
    ax3.legend(fontsize=9); ax3.set_facecolor(LIGHT)

    # 4 — Rating bands
    ax4 = fig.add_subplot(3, 3, 4)
    labels4 = ["1–2", "2–3", "3–4", "4–5"]
    df["rating_bin"] = pd.cut(df["rating"], bins=[1,2,3,4,5], labels=labels4)
    rb = df["rating_bin"].value_counts().reindex(labels4)
    ax4.bar(rb.index, rb.values, color=[ACCENT, GOLD, MID, GREEN], edgecolor="white", zorder=3)
    ax4.set_title("Rating Distribution", fontweight="bold", color=BLUE, fontsize=11)
    ax4.set_facecolor(LIGHT); ax4.grid(axis="y", alpha=0.3, zorder=0)

    # 5 — Top 10 sellers
    ax5 = fig.add_subplot(3, 3, 5)
    ts = df.groupby("seller")["revenue"].sum().sort_values(ascending=True).tail(10)
    ax5.barh(ts.index, ts.values / 1e9,
             color=[BLUE]*9 + [ACCENT], edgecolor="white")
    ax5.set_title("Top 10 Sellers by Revenue", fontweight="bold", color=BLUE, fontsize=11)
    ax5.set_facecolor(LIGHT)

    # 6 — Price vs units sold
    ax6 = fig.add_subplot(3, 3, 6)
    palette = {"Electronics": BLUE, "Fashion": ACCENT, "Mobiles": MID,
               "Beauty": GOLD, "Toys": GREEN, "Sports": "#9b59b6",
               "Appliances": "#e67e22", "Home & Kitchen": "#1abc9c"}
    for cat in df["category"].unique():
        s = df[df["category"] == cat].sample(150, random_state=42)
        ax6.scatter(s["final_price"], s["units_sold"], alpha=0.4, s=12,
                    label=cat, color=palette.get(cat, BLUE))
    ax6.set_title("Price vs Units Sold by Category", fontweight="bold", color=BLUE, fontsize=11)
    ax6.legend(fontsize=7, markerscale=2, ncol=2); ax6.set_facecolor(LIGHT)

    # 7 — Seller rating by city
    ax7 = fig.add_subplot(3, 3, 7)
    cr = df.groupby("seller_city")["seller_rating"].mean().sort_values(ascending=False).head(8)
    ax7.bar(cr.index, cr.values, color=MID, edgecolor="white", zorder=3)
    ax7.set_ylim(3.9, 4.1)
    ax7.set_title("Avg Seller Rating by City", fontweight="bold", color=BLUE, fontsize=11)
    ax7.tick_params(axis="x", rotation=35, labelsize=8)
    ax7.set_facecolor(LIGHT); ax7.grid(axis="y", alpha=0.3, zorder=0)

    # 8 — Actual vs Predicted
    ax8 = fig.add_subplot(3, 3, 8)
    idx = np.random.choice(len(yte), 500, replace=False)
    preds = results[best]["preds"]
    ax8.scatter(yte.values[idx], preds[idx], alpha=0.35, color=BLUE, s=12, zorder=3)
    lim = [yte.min(), yte.max()]
    ax8.plot(lim, lim, "r--", lw=1.5, label="Perfect fit")
    ax8.set_title(f"Actual vs Predicted ({best})\nR²={results[best]['r2']:.3f}  MAE=₹{results[best]['mae']:,.0f}",
                  fontweight="bold", color=BLUE, fontsize=10)
    ax8.legend(fontsize=8); ax8.set_facecolor(LIGHT)

    # 9 — Returnable vs Non-returnable
    ax9 = fig.add_subplot(3, 3, 9)
    rg = df.groupby("is_returnable")["final_price"].describe()[["mean", "25%", "75%"]]
    means = [rg.loc[False, "mean"], rg.loc[True, "mean"]]
    q25   = [rg.loc[False, "25%"],  rg.loc[True, "25%"]]
    q75   = [rg.loc[False, "75%"],  rg.loc[True, "75%"]]
    bars9 = ax9.bar(["Non-Returnable", "Returnable"], means,
                    color=[ACCENT, BLUE], edgecolor="white", width=0.5, zorder=3)
    ax9.errorbar(["Non-Returnable", "Returnable"], means,
                 yerr=[[m-q for m,q in zip(means,q25)], [q-m for m,q in zip(means,q75)]],
                 fmt="none", color="black", capsize=6, lw=1.5, zorder=4)
    ax9.set_title("Avg Price: Returnable vs Not", fontweight="bold", color=BLUE, fontsize=11)
    ax9.set_facecolor(LIGHT); ax9.grid(axis="y", alpha=0.3, zorder=0)

    fig.suptitle("Flipkart E-Commerce: EDA & ML Price Prediction Dashboard",
                 fontsize=16, fontweight="bold", color=BLUE, y=1.01)
    plt.tight_layout(pad=2.5)
    path = os.path.join(OUTPUT_DIR, "dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[plot]  Dashboard saved → {path}")


# ── 9. Save Model ─────────────────────────────────────────────────────────────
def save_model(results: dict, best: str) -> None:
    path = os.path.join(OUTPUT_DIR, "best_model.pkl")
    joblib.dump(results[best]["pipe"], path)
    print(f"[save]  Model saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df              = load_data(DATA_PATH)
    df              = engineer_features(df)
    check_leakage(df)
    Xtr, Xte, ytr, yte = prepare_splits(df)
    results, best   = train_and_evaluate(Xtr, Xte, ytr, yte)
    plot_dashboard(df, results, best, yte)
    save_model(results, best)
    print("\n✅ Pipeline complete.")
