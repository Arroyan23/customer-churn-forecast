import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "train.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = "train.csv"  # fallback kalau __file__ tidak reliable (mis. sedang di-exec)

MODEL_CACHE_PATH = os.path.join(os.path.dirname(DATA_PATH), "churn_pipeline_cache.pkl")

BLANK = "— Tidak diisi —"
COLOR_STAY = "#2A9D8F"
COLOR_CHURN = "#E76F51"
COLOR_MED = "#E9C46A"

st.set_page_config(
    page_title="Churn Radar — Prediksi Churn Pelanggan",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    .churn-hero {
        background: linear-gradient(120deg, #0F4C5C 0%, #1B7A6E 100%);
        padding: 1.6rem 2rem; border-radius: 16px; color: white;
        margin-bottom: 1.4rem;
    }
    .churn-hero h1 { margin: 0; font-size: 1.7rem; }
    .churn-hero p { margin: 0.35rem 0 0 0; opacity: 0.9; font-size: 0.95rem; }
    div[data-testid="stMetric"] {
        background: #F7F9F9; border: 1px solid #E7ECEB;
        border-radius: 12px; padding: 0.8rem 1rem;
    }
    .verdict-badge {
        font-size: 1.3rem; font-weight: 700; margin-bottom: 0.3rem;
    }
    .note-box {
        background: #F7F9F9; border-left: 4px solid #1B7A6E;
        padding: 0.7rem 1rem; border-radius: 8px; font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================================
# DATA & MODEL (di-cache biar tidak diulang tiap interaksi)
# =========================================================================
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource(show_spinner="🔧 Menyiapkan model (training pertama kali bisa 1-3 menit)...")
def get_model_bundle(path: str, cache_path: str) -> dict:
    # Coba pakai model yang sudah pernah dilatih & disimpan di disk
    if os.path.exists(cache_path):
        try:
            return joblib.load(cache_path)
        except Exception:
            pass

    df = load_data(path)

    # --- replikasi persis pipeline dari model.ipynb ---
    X = df.drop(columns=["Churn"])
    le = LabelEncoder()
    y = le.fit_transform(df["Churn"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numerical_cols = list(X_train.select_dtypes(include=["int64", "float64"]).columns)
    categorical_cols = list(X_train.select_dtypes(include=["object"]).columns)

    imputer_numerical = SimpleImputer(strategy="mean")
    imputer_categorical = SimpleImputer(strategy="most_frequent")
    oh_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    onehot_transformer = Pipeline(
        steps=[("imputer", imputer_categorical), ("encoder", oh_encoder)]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numerical_cols", imputer_numerical, numerical_cols),
            ("categorical_cols", onehot_transformer, categorical_cols),
        ]
    )

    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    model_xgb = XGBClassifier(
        n_estimators=200,
        scale_pos_weight=ratio,
        learning_rate=0.05,
        eval_metric="logloss",
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessing), ("model", model_xgb)])
    pipeline.fit(X_train, y_train)

    acc = accuracy_score(y_test, pipeline.predict(X_test))
    yes_index = list(le.classes_).index("Yes")

    bundle = {
        "pipeline": pipeline,
        "yes_index": yes_index,
        "accuracy": acc,
        "n_rows": len(df),
        "next_id": int(df["id"].max()) + 1,
    }

    try:
        joblib.dump(bundle, cache_path)
    except Exception:
        pass

    return bundle


@st.cache_data(show_spinner=False)
def compute_eda(path: str) -> dict:
    df = load_data(path)

    # 1) Distribusi churn (imbalance)
    churn_counts = df["Churn"].value_counts()
    churn_pct = (churn_counts / churn_counts.sum() * 100).round(1)

    # 2) Distribusi MonthlyCharges vs Churn (dibin manual biar ringan di browser)
    bins = np.linspace(df["MonthlyCharges"].min(), df["MonthlyCharges"].max(), 26)
    hist_no, edges = np.histogram(df.loc[df["Churn"] == "No", "MonthlyCharges"], bins=bins)
    hist_yes, _ = np.histogram(df.loc[df["Churn"] == "Yes", "MonthlyCharges"], bins=bins)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    # 3) Risk segmentation: Avg_Monthly_Paid = TotalCharges / tenure (insight utama EDA)
    dfc = df[df["tenure"] > 0].copy()
    dfc["Avg_Monthly_Paid"] = dfc["TotalCharges"] / dfc["tenure"]
    dfc["Risk_Group"] = pd.cut(
        dfc["Avg_Monthly_Paid"],
        bins=[0, 30, 90, dfc["Avg_Monthly_Paid"].max()],
        labels=["Risiko Rendah", "Risiko Sedang", "Risiko Tinggi"],
    )
    risk_churn = (
        dfc.groupby("Risk_Group", observed=True)["Churn"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .reindex(["Risiko Rendah", "Risiko Sedang", "Risiko Tinggi"])
    )

    return {
        "churn_counts": churn_counts,
        "churn_pct": churn_pct,
        "mc_bin_centers": bin_centers,
        "mc_hist_no": hist_no,
        "mc_hist_yes": hist_yes,
        "risk_churn": risk_churn,
    }


# =========================================================================
# GUARD: pastikan train.csv ada
# =========================================================================
if not os.path.exists(DATA_PATH):
    st.error(
        "File `train.csv` tidak ditemukan.\n\n"
        "Taruh `app.py` ini di folder yang sama dengan `train.csv` "
        "(folder repo `customer-churn-forecast`), lalu jalankan ulang."
    )
    st.stop()

model_bundle = get_model_bundle(DATA_PATH, MODEL_CACHE_PATH)
eda = compute_eda(DATA_PATH)


# =========================================================================
# SIDEBAR — identitas app
# =========================================================================
with st.sidebar:
    st.markdown("## 📉 Churn Radar")
    st.caption(
        "Alat bantu prediksi risiko pelanggan berhenti berlangganan (churn), "
        "dilatih dari data histori pelanggan telco."
    )
    st.divider()
    st.metric("Akurasi model (holdout 20%)", f"{model_bundle['accuracy'] * 100:.1f}%")
    st.metric("Jumlah data latih", f"{model_bundle['n_rows']:,}".replace(",", "."))
    st.caption("Model: XGBoost Classifier · Preprocessing: One-Hot Encoding + Imputer")
    st.divider()
    st.caption(
        "Dibuat dari `model.ipynb` & `data-viz.ipynb` — proyek "
        "[customer-churn-forecast](https://github.com/Arroyan23/customer-churn-forecast)."
    )


# =========================================================================
# HERO HEADER
# =========================================================================
st.markdown(
    """
    <div class="churn-hero">
        <h1>📉 Churn Radar</h1>
        <p>Prediksi apakah seorang pelanggan berpotensi berhenti berlangganan (churn),
        lengkap dengan konteks data historis di baliknya.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_insight = st.tabs(["🔮 Prediksi Pelanggan", "📊 Insight dari Data"])


# =========================================================================
# TAB 1 — PREDIKSI
# =========================================================================
with tab_predict:
    st.subheader("Masukkan Profil Pelanggan")
    st.caption(
        "Isi bagian **Data Utama** (wajib). Bagian **Data Tambahan** boleh "
        "dikosongkan — sistem otomatis memakai nilai paling umum dari data historis."
    )

    st.markdown("#### 🔒 Data Utama (wajib diisi)")
    c1, c2, c3 = st.columns(3)
    with c1:
        tenure = st.number_input(
            "Lama berlangganan (bulan)", min_value=0, max_value=100, value=12, step=1,
            help="Sudah berapa bulan pelanggan ini aktif berlangganan.",
        )
    with c2:
        monthly_charges = st.number_input(
            "Tagihan bulanan (USD)", min_value=0.0, max_value=200.0, value=70.0, step=0.5,
        )
    with c3:
        contract = st.selectbox("Jenis kontrak", ["Month-to-month", "One year", "Two year"])

    c4, c5 = st.columns(2)
    with c4:
        internet_service = st.selectbox("Layanan internet", ["DSL", "Fiber optic", "No"])
    with c5:
        default_total = round(tenure * monthly_charges, 2)
        total_charges = st.number_input(
            "Total tagihan sejak awal berlangganan (USD)",
            min_value=0.0, value=float(default_total), step=1.0,
            help="Default dihitung otomatis dari lama berlangganan × tagihan bulanan, boleh disesuaikan.",
        )

    with st.expander("✳️ Data Tambahan (opsional — boleh dikosongkan)", expanded=False):
        st.caption("Kosongkan field yang tidak lu ketahui. Sistem akan mengisinya otomatis.")

        d1, d2, d3 = st.columns(3)
        with d1:
            gender_in = st.selectbox("Jenis kelamin", [BLANK, "Perempuan", "Laki-laki"])
            senior_in = st.selectbox("Warga senior (lansia)?", [BLANK, "Tidak", "Ya"])
        with d2:
            partner_in = st.selectbox("Punya pasangan?", [BLANK, "Tidak", "Ya"])
            dependents_in = st.selectbox("Punya tanggungan?", [BLANK, "Tidak", "Ya"])
        with d3:
            paperless_in = st.selectbox("Tagihan tanpa kertas?", [BLANK, "Tidak", "Ya"])
            payment_in = st.selectbox(
                "Metode pembayaran",
                [BLANK, "Electronic check", "Mailed check",
                 "Bank transfer (automatic)", "Credit card (automatic)"],
            )

        st.markdown("###### Layanan telepon")
        phone_in = st.selectbox("Punya layanan telepon?", [BLANK, "Tidak", "Ya"], key="phone_in")
        if phone_in == "Tidak":
            st.caption("📵 Layanan telepon nonaktif → 'Multiple lines' otomatis diset 'No phone service'.")
            multiple_lines_val = "No phone service"
        else:
            multiple_lines_in = st.selectbox("Punya multiple lines?", [BLANK, "Tidak", "Ya"])
            multiple_lines_val = {"Ya": "Yes", "Tidak": "No"}.get(multiple_lines_in, np.nan)

        st.markdown("###### Layanan tambahan (mengikuti status internet)")
        if internet_service == "No":
            st.caption(
                "🌐 Tidak berlangganan internet → semua layanan tambahan di bawah "
                "otomatis diset 'No internet service'."
            )
            online_security_val = online_backup_val = device_protection_val = "No internet service"
            tech_support_val = streaming_tv_val = streaming_movies_val = "No internet service"
        else:
            e1, e2, e3 = st.columns(3)
            with e1:
                online_security_in = st.selectbox("Online Security", [BLANK, "Tidak", "Ya"])
                online_backup_in = st.selectbox("Online Backup", [BLANK, "Tidak", "Ya"])
            with e2:
                device_protection_in = st.selectbox("Device Protection", [BLANK, "Tidak", "Ya"])
                tech_support_in = st.selectbox("Tech Support", [BLANK, "Tidak", "Ya"])
            with e3:
                streaming_tv_in = st.selectbox("Streaming TV", [BLANK, "Tidak", "Ya"])
                streaming_movies_in = st.selectbox("Streaming Movies", [BLANK, "Tidak", "Ya"])

            _map = {"Ya": "Yes", "Tidak": "No"}
            online_security_val = _map.get(online_security_in, np.nan)
            online_backup_val = _map.get(online_backup_in, np.nan)
            device_protection_val = _map.get(device_protection_in, np.nan)
            tech_support_val = _map.get(tech_support_in, np.nan)
            streaming_tv_val = _map.get(streaming_tv_in, np.nan)
            streaming_movies_val = _map.get(streaming_movies_in, np.nan)

    predict_clicked = st.button("🔮 Prediksi Sekarang", type="primary", use_container_width=True)

    if predict_clicked:
        yn = {"Ya": "Yes", "Tidak": "No"}
        row = {
            "id": model_bundle["next_id"],
            "gender": {"Perempuan": "Female", "Laki-laki": "Male"}.get(gender_in, np.nan),
            "SeniorCitizen": {"Ya": 1, "Tidak": 0}.get(senior_in, np.nan),
            "Partner": yn.get(partner_in, np.nan),
            "Dependents": yn.get(dependents_in, np.nan),
            "tenure": tenure,
            "PhoneService": yn.get(phone_in, np.nan),
            "MultipleLines": multiple_lines_val,
            "InternetService": internet_service,
            "OnlineSecurity": online_security_val,
            "OnlineBackup": online_backup_val,
            "DeviceProtection": device_protection_val,
            "TechSupport": tech_support_val,
            "StreamingTV": streaming_tv_val,
            "StreamingMovies": streaming_movies_val,
            "Contract": contract,
            "PaperlessBilling": yn.get(paperless_in, np.nan),
            "PaymentMethod": payment_in if payment_in != BLANK else np.nan,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }
        input_df = pd.DataFrame([row])

        proba = model_bundle["pipeline"].predict_proba(input_df)[0]
        churn_proba = float(proba[model_bundle["yes_index"]] * 100)

        avg_paid = total_charges / max(tenure, 1)
        if avg_paid <= 30:
            risk_bucket = "Risiko Rendah"
        elif avg_paid <= 90:
            risk_bucket = "Risiko Sedang"
        else:
            risk_bucket = "Risiko Tinggi"
        hist_rate = eda["risk_churn"].get(risk_bucket, np.nan)

        notes = [
            f"Rasio tagihan/tenure pelanggan ini (~${avg_paid:.1f} per bulan) masuk kelompok "
            f"**{risk_bucket}**. Secara historis, sekitar **{hist_rate:.1f}%** pelanggan di "
            f"kelompok ini akhirnya churn."
        ]
        if contract == "Month-to-month":
            notes.append(
                "📄 Kontrak **bulanan (month-to-month)** secara historis punya tingkat churn "
                "yang jauh lebih tinggi dibanding kontrak tahunan."
            )
        elif contract == "Two year":
            notes.append("📄 Kontrak **dua tahun** biasanya jadi salah satu sinyal loyalitas terkuat.")
        if tenure <= 3:
            notes.append("🕒 Pelanggan masih sangat baru (≤ 3 bulan) — periode ini biasanya paling rawan churn.")

        if churn_proba >= 60:
            badge, color = "🔴 Risiko Tinggi Churn", COLOR_CHURN
            verdict = ("Profil pelanggan ini cukup mirip dengan pelanggan yang akhirnya berhenti "
                       "berlangganan. Pertimbangkan langkah retensi seperti penawaran diskon atau "
                       "upgrade kontrak.")
        elif churn_proba >= 30:
            badge, color = "🟡 Risiko Sedang", COLOR_MED
            verdict = "Ada beberapa sinyal risiko, tapi belum dominan. Perlu dipantau berkala."
        else:
            badge, color = "🟢 Risiko Rendah", COLOR_STAY
            verdict = "Profil pelanggan ini cenderung mirip dengan pelanggan yang bertahan."

        st.session_state["churn_result"] = {
            "proba": churn_proba, "badge": badge, "color": color,
            "verdict": verdict, "notes": notes,
        }

    if "churn_result" in st.session_state:
        res = st.session_state["churn_result"]
        st.divider()
        st.markdown("#### 🎯 Hasil Prediksi")
        rc1, rc2 = st.columns([1, 1.3])
        with rc1:
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=res["proba"],
                    number={"suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": res["color"]},
                        "steps": [
                            {"range": [0, 30], "color": "#EAF4F2"},
                            {"range": [30, 60], "color": "#FBF3DD"},
                            {"range": [60, 100], "color": "#FBE6E1"},
                        ],
                    },
                )
            )
            fig_gauge.update_layout(height=250, margin=dict(t=30, b=10, l=25, r=25))
            st.plotly_chart(fig_gauge, use_container_width=True)
        with rc2:
            st.markdown(f'<div class="verdict-badge">{res["badge"]}</div>', unsafe_allow_html=True)
            st.write(res["verdict"])
            st.markdown("**🔎 Konteks dari histori data:**")
            for note in res["notes"]:
                st.markdown(f'<div class="note-box">{note}</div>', unsafe_allow_html=True)
    else:
        st.info("Isi data pelanggan di atas, lalu klik **Prediksi Sekarang** untuk melihat hasilnya.")


# =========================================================================
# TAB 2 — INSIGHT DATA
# =========================================================================
with tab_insight:
    st.subheader("Insight dari Data Historis")
    st.caption(
        f"Ringkasan pola dari {model_bundle['n_rows']:,} data pelanggan yang dipakai untuk "
        "melatih model (analisis lengkap ada di `data-viz.ipynb`).".replace(",", ".")
    )

    k1, k2, k3 = st.columns(3)
    k1.metric("Total pelanggan", f"{model_bundle['n_rows']:,}".replace(",", "."))
    k2.metric("Tingkat churn keseluruhan", f"{eda['churn_pct'].get('Yes', 0):.1f}%")
    k3.metric("Akurasi model (holdout)", f"{model_bundle['accuracy'] * 100:.1f}%")

    st.divider()

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Proporsi pelanggan: bertahan vs churn**")
        fig1 = go.Figure(
            data=[go.Pie(
                labels=eda["churn_counts"].index,
                values=eda["churn_counts"].values,
                hole=0.55,
                marker=dict(colors=[COLOR_STAY, COLOR_CHURN]),
            )]
        )
        fig1.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
        st.plotly_chart(fig1, use_container_width=True)
        st.caption(
            "Dataset ini timpang (imbalanced) — sekitar 3 dari 4 pelanggan tetap bertahan. "
            "Ini alasan model memakai `scale_pos_weight` supaya tidak bias menebak "
            "'tidak churn' terus-menerus."
        )

    with g2:
        st.markdown("**Sebaran tagihan bulanan: bertahan vs churn**")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=eda["mc_bin_centers"], y=eda["mc_hist_no"],
            name="Bertahan", marker_color=COLOR_STAY, opacity=0.75,
        ))
        fig2.add_trace(go.Bar(
            x=eda["mc_bin_centers"], y=eda["mc_hist_yes"],
            name="Churn", marker_color=COLOR_CHURN, opacity=0.75,
        ))
        fig2.update_layout(
            barmode="overlay", xaxis_title="Tagihan bulanan (USD)",
            yaxis_title="Jumlah pelanggan",
            margin=dict(t=10, b=10, l=10, r=10), height=320,
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            "Pelanggan yang churn cenderung menumpuk di tagihan bulanan yang lebih tinggi, "
            "sementara pelanggan setia condong ke tagihan yang lebih rendah dan stabil."
        )

    st.markdown("**Tingkat churn berdasarkan kelompok rasio tagihan/tenure (insight utama)**")
    fig3 = go.Figure(
        data=[go.Bar(
            x=eda["risk_churn"].index,
            y=eda["risk_churn"].values,
            marker_color=[COLOR_STAY, COLOR_MED, COLOR_CHURN],
            text=[f"{v:.1f}%" for v in eda["risk_churn"].values],
            textposition="outside",
        )]
    )
    fig3.update_layout(
        yaxis_title="Tingkat churn (%)",
        xaxis_title="Kelompok risiko (TotalCharges ÷ tenure)",
        margin=dict(t=10, b=30, l=10, r=10), height=340,
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        "Semakin tinggi rasio rata-rata tagihan bulanan terhadap lama berlangganan, semakin "
        "besar peluang pelanggan churn — pola inilah yang jadi dasar fitur `Avg_Monthly_Paid` "
        "di notebook EDA, dan dipakai juga sebagai konteks di tab Prediksi."
    )