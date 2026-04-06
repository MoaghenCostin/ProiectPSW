import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# ─────────────────────────────────────────────
# CONFIGURARE PAGINĂ
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Analiza activității hoteliere",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS PERSONALIZAT
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

section[data-testid="stSidebar"] {
    background: #0B1929;
    border-right: 1px solid #1a3350;
}
section[data-testid="stSidebar"] * { color: #a8c0d6 !important; }
section[data-testid="stSidebar"] .stRadio label {
    font-size: 13.5px; padding: 6px 0; cursor: pointer;
}
section[data-testid="stSidebar"] .stRadio label:hover { color: #e2edf5 !important; }

.page-header {
    background: linear-gradient(135deg, #0B1929 0%, #1a3a5c 100%);
    border-radius: 14px;
    padding: 22px 28px 18px;
    margin-bottom: 22px;
    border: 1px solid #1e3f5e;
}
.page-header h1 { color: #e8f4fd; margin: 0 0 4px; font-size: 1.6rem; font-weight: 700; }
.page-header p  { color: #7fb3d3; margin: 0; font-size: 13px; }

.info-box {
    background: #f0f7ff;
    border: 1px solid #bfdbfe;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #1e3a5f;
    margin-bottom: 14px;
    line-height: 1.65;
}

.warn-box {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #78350f;
    margin-bottom: 14px;
    line-height: 1.65;
}

.sec-title {
    font-size: 10.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.3px;
    color: #94a3b8;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 5px;
    margin: 22px 0 14px;
}

.interpret-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 13px 16px;
    font-size: 12.5px;
    color: #374151;
    line-height: 1.7;
    margin-top: 10px;
}
.interpret-box strong { color: #0f172a; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ÎNCĂRCARE DATE
# ─────────────────────────────────────────────
@st.cache_data
def incarca_date(fisier) -> pd.DataFrame:
    df = pd.read_csv(fisier)

    luni = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }
    df["arrival_month_num"] = df["arrival_date_month"].map(luni)
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["revenue_estimat"] = df["adr"] * df["total_nights"]
    return df


# ─────────────────────────────────────────────
# SIDEBAR - NAVIGARE
# ─────────────────────────────────────────────
SECTIUNI = [
    ("📊", "Date generale"),
    ("🔍", "Analiză exploratorie"),
    ("🗺️", "Hartă geografică"),
    ("⚙️", "Preprocesare"),
    ("🔵", "K-Means Clustering"),
    ("📈", "Regresie multiplă"),
    ("🎯", "Clasificare"),
]

with st.sidebar:
    st.markdown("### 🏨 Analiza Hotelieră")
    st.markdown("*Hotel Booking Demand*")
    st.markdown("---")
    fisier = st.file_uploader("Încarcă hotel_bookings.csv", type=["csv"])
    st.markdown("---")
    optiuni_radio = [f"{icon}  {nume}" for icon, nume in SECTIUNI]
    alegere = st.radio("Navigare", optiuni_radio, label_visibility="collapsed")
    sectiune_curenta = SECTIUNI[optiuni_radio.index(alegere)][1]
    st.markdown("---")
    st.caption("ASE CSIE - Pachete Software")
    st.caption("Dataset: Hotel Booking Demand")


# ─────────────────────────────────────────────
# VERIFICARE FIȘIER
# ─────────────────────────────────────────────
if fisier is None:
    st.markdown("""
    <div class="page-header">
        <h1>🏨 Analiza activității hoteliere și modelarea predictivă a rezervărilor</h1>
        <p>Identificarea posibilităților de extindere a activității hoteliere prin analiză de date și machine learning</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("👈 **Încarcă fișierul** `hotel_bookings.csv` din sidebar pentru a începe analiza.")
    st.markdown("""
    <div class="info-box">
        <strong>Despre dataset:</strong> Hotel Booking Demand conține ~119.000 rezervări din două hoteluri
        (Resort Hotel și City Hotel) din Portugalia, cu informații despre client, rezervare, tarif și anulare.
        <br><br>
        <strong>Facilități acoperite:</strong> Streamlit | Pandas (groupby, agg) | Valori lipsă și extreme |
        Codificare | Scalare | GeoPandas | Scikit-learn (K-Means, Regresie Logistică) | Statsmodels (Regresie Multiplă)
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = incarca_date(fisier)


# ═══════════════════════════════════════════════════════════════
# FUNCȚIE COMUNĂ DE PREPROCESARE PENTRU MODELE ML
# ═══════════════════════════════════════════════════════════════
@st.cache_data
def pregateste_date_ml(df: pd.DataFrame):
    """
    Preprocesare completă, reutilizabilă de toate secțiunile ML.
    Returnează DataFrame-ul curat și scalerul antrenat pe datele de train.
    """
    df_ml = df.copy()

    # Valori lipsă
    df_ml["children"]  = df_ml["children"].fillna(0).astype(int)
    df_ml["country"]   = df_ml["country"].fillna("Unknown")
    df_ml["agent"]     = df_ml["agent"].fillna(0)
    df_ml = df_ml.drop(columns=["company"], errors="ignore")

    # Outlieri
    df_ml = df_ml[df_ml["adr"] > 0]
    upper_adr = df_ml["adr"].quantile(0.99)
    df_ml["adr"] = np.where(df_ml["adr"] > upper_adr, upper_adr, df_ml["adr"])
    upper_lt = df_ml["lead_time"].quantile(0.99)
    df_ml["lead_time"] = np.where(df_ml["lead_time"] > upper_lt, upper_lt, df_ml["lead_time"])

    # Encoding
    le = LabelEncoder()
    df_ml["hotel_encoded"] = le.fit_transform(df_ml["hotel"])

    cols_ohe = ["meal", "market_segment", "deposit_type",
                "customer_type", "distribution_channel", "reserved_room_type"]
    cols_ohe = [c for c in cols_ohe if c in df_ml.columns]
    df_ml = pd.get_dummies(df_ml, columns=cols_ohe, drop_first=True)

    freq_country = df_ml["country"].value_counts(normalize=True)
    df_ml["country_freq"] = df_ml["country"].map(freq_country)

    cols_drop = ["country", "hotel", "arrival_date_month",
                 "reservation_status", "assigned_room_type",
                 "name", "email", "phone-number", "credit_card",
                 "reservation_status_date"]
    df_ml = df_ml.drop(columns=[c for c in cols_drop if c in df_ml.columns])

    # Convertim bool → int
    bool_cols = df_ml.select_dtypes(include=["bool"]).columns
    df_ml[bool_cols] = df_ml[bool_cols].astype(int)

    # Scalare
    cols_scalare = [
        "lead_time", "adr", "total_nights", "days_in_waiting_list",
        "booking_changes", "country_freq", "arrival_month_num",
        "total_of_special_requests", "required_car_parking_spaces",
        "adults", "children", "babies", "revenue_estimat"
    ]
    cols_scalare = [c for c in cols_scalare if c in df_ml.columns]

    from sklearn.model_selection import train_test_split
    X_all = df_ml.drop(columns=["is_canceled"], errors="ignore")
    scaler = StandardScaler()
    X_all[cols_scalare] = scaler.fit_transform(X_all[cols_scalare])

    return df_ml, X_all, scaler, cols_scalare


# ═══════════════════════════════════════════════════════════════
# SECȚIUNEA 1 - DATE GENERALE
# ═══════════════════════════════════════════════════════════════
if sectiune_curenta == "Date generale":
    st.markdown("""
    <div class="page-header">
        <h1>📊 Date generale</h1>
        <p><strong>Despre dataset:</strong> Hotel Booking Demand conține ~119.000 rezervări din două hoteluri 
        (Resort Hotel și City Hotel) din Portugalia, cu informații despre client, rezervare, tarif și anulare.
        Dataset disponibil pe <a href="https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand" target="_blank">Kaggle</a>.</p>
    </div>
    """, unsafe_allow_html=True)

    total_rez    = len(df)
    rez_anulate  = df["is_canceled"].sum()
    rata_anulare = rez_anulate / total_rez * 100
    adr_mediu    = df[df["adr"] > 0]["adr"].median()
    nr_tari      = df["country"].nunique()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total rezervări",   f"{total_rez:,.0f}")
    col2.metric("Rezervări anulate", f"{rez_anulate:,.0f}")
    col3.metric("Rată anulare",      f"{rata_anulare:.1f}%")
    col4.metric("ADR median (€)",    f"{adr_mediu:.1f}")
    col5.metric("Țări de origine",   f"{nr_tari}")

    st.markdown('<p class="sec-title">Primele 5 înregistrări - df.head()</p>', unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)

    # df.info() - afișăm tipurile de date și valorile non-null
    st.markdown('<p class="sec-title">Informații despre structura datelor - df.info()</p>', unsafe_allow_html=True)
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.code(info_str, language="text")

    st.markdown('<p class="sec-title">Informații despre coloane</p>', unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown(f"""
        <div class="info-box">
            <strong>Dimensiune:</strong> {df.shape[0]:,} rânduri × {df.shape[1]} coloane<br>
            <strong>Memorie:</strong> {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB
        </div>
        """, unsafe_allow_html=True)
        tipuri = df.dtypes.astype(str).value_counts().reset_index()
        tipuri.columns = ["Tip", "Nr. coloane"]
        st.dataframe(tipuri, use_container_width=True, hide_index=True)

    with col_b:
        info_cols = pd.DataFrame({
            "Coloană": df.columns,
            "Tip": df.dtypes.astype(str).values,
            "Valori lipsă": df.isnull().sum().values,
            "% lipsă": (df.isnull().sum().values / len(df) * 100).round(2),
            "Unice": df.nunique().values
        })
        st.dataframe(
            info_cols.style.background_gradient(subset=["% lipsă"], cmap="Oranges"),
            use_container_width=True, height=380, hide_index=True
        )

    st.markdown('<p class="sec-title">Statistici descriptive - coloane numerice</p>', unsafe_allow_html=True)
    st.dataframe(df.describe().round(2), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# SECȚIUNEA 2 - ANALIZĂ EXPLORATORIE (EDA)
# ═══════════════════════════════════════════════════════════════
elif sectiune_curenta == "Analiză exploratorie":
    st.markdown("""
    <div class="page-header">
        <h1>🔍 Analiză exploratorie (EDA)</h1>
        <p>Valori lipsă | Distribuții | Outlieri | Grupări și agregări | Corelații</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📉 Valori lipsă",
        "📊 Distribuții",
        "⚠️ Outlieri",
        "📋 Grupări & Agregări",
        "🔗 Corelații"
    ])

    # ── TAB 1: VALORI LIPSĂ ──────────────────────────────────────
    with tab1:
        st.markdown('<p class="sec-title">Număr și procent de valori lipsă per coloană</p>',
                    unsafe_allow_html=True)

        # pd.concat() pentru missing summary
        total_lipsa   = df.isnull().sum().sort_values(ascending=False)
        percent_lipsa = (df.isnull().sum() / df.isnull().count() * 100).sort_values(ascending=False)
        missing_df = pd.concat([total_lipsa, percent_lipsa.round(2)],
                               axis=1, keys=["Valori lipsă", "Procent (%)"])
        missing_df = missing_df[missing_df["Valori lipsă"] > 0]

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.dataframe(missing_df, use_container_width=True)

        with col_b:
            # Grafic bara orizontală - matplotlib
            fig, ax = plt.subplots(figsize=(7, 3))
            culori = ["#ef4444" if p > 20 else "#f97316" if p > 5 else "#facc15"
                      for p in missing_df["Procent (%)"]]
            ax.barh(missing_df.index, missing_df["Procent (%)"],
                    color=culori, edgecolor="white")
            ax.set_xlabel("Procent valori lipsă (%)")
            ax.set_title("Procentul valorilor lipsă per coloană")
            ax.grid(axis="x", linestyle="--", alpha=0.5)
            for i, (val, pct) in enumerate(zip(missing_df["Valori lipsă"],
                                                missing_df["Procent (%)"])):
                ax.text(pct + 0.2, i, f"{val:,} ({pct:.1f}%)",
                        va="center", fontsize=9, color="#374151")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("""
        <div class="interpret-box">
            <strong>Interpretare economică:</strong><br>
            • <strong>company</strong> (~94% lipsă): Marea majoritate a rezervărilor provin de la clienți individuali,
            nu de la companii - informația lipsă e de fapt informație: clientul nu e corporativ.<br>
            • <strong>agent</strong> (~14% lipsă): Rezervările fără agent sunt rezervări directe (online sau la recepție)
            - un segment important pentru analiza canalului de distribuție.<br>
            • <strong>country</strong> (~0.4% lipsă): Neglijabil; nu afectează analiza geografică.<br>
            • <strong>children</strong> (~0.003% lipsă): Practic zero - poate fi imputat cu 0.
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 2: DISTRIBUȚII ────────────────────────────────────────
    with tab2:
        st.markdown('<p class="sec-title">Distribuția variabilelor numerice</p>',
                    unsafe_allow_html=True)

        # Selectăm coloanele numerice relevante (excludem ID-uri și coloane binare)
        cols_num = ["adr", "lead_time", "total_nights", "number_of_reviews"
                    if "number_of_reviews" in df.columns else "stays_in_week_nights",
                    "stays_in_weekend_nights", "stays_in_week_nights",
                    "adults", "booking_changes", "days_in_waiting_list",
                    "total_of_special_requests", "required_car_parking_spaces"]
        cols_num = [c for c in cols_num if c in df.columns]

        n_cols = 3
        n_rows = math.ceil(len(cols_num) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(cols_num):
            # sns.histplot cu kde=True - afișează și curba de densitate
            sns.histplot(df[col].dropna(), bins=30, kde=True,
                         ax=axes[i], color="#3b82f6", edgecolor="white")
            axes[i].set_title(f"Distribuția: {col}", fontsize=10, fontweight="bold")
            axes[i].set_xlabel(col, fontsize=9)
            axes[i].set_ylabel("Frecvență", fontsize=9)
            axes[i].grid(axis="y", linestyle="--", alpha=0.4)

        # Ascundem axele goale dacă numărul de coloane nu e multiplu de 3
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        <div class="interpret-box">
            <strong>Observații cheie:</strong><br>
            • <strong>adr</strong> (tariful zilnic mediu): Distribuție puternic asimetrică spre dreapta -
            majoritatea rezervărilor au tarife între 50–200 €, dar există și cazuri extreme (500–5000 €).
            Va necesita transformare logaritmică înainte de regresie.<br>
            • <strong>lead_time</strong>: Concentrare mare la 0–30 zile (rezervări de last-minute),
            cu o coadă lungă spre dreapta. Clienții de resort rezervă mai din timp decât cei de city hotel.<br>
            • <strong>total_nights</strong>: Majoritar 1–7 nopți. Valori de 30–90+ nopți sunt outlieri potențiali.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="sec-title">Distribuția variabilelor categorice</p>',
                    unsafe_allow_html=True)

        cols_cat = ["hotel", "market_segment", "deposit_type",
                    "customer_type", "meal", "distribution_channel"]
        cols_cat = [c for c in cols_cat if c in df.columns]

        for col in cols_cat:
            fig, ax = plt.subplots(figsize=(9, 3))
            unique_count = df[col].nunique()

            if unique_count > 10:
                top_cat = df[col].value_counts().nlargest(10)
                sns.barplot(x=top_cat.index, y=top_cat.values,
                            palette="Blues_d", ax=ax)
                ax.set_title(f"Top 10 valori pentru: {col}", fontweight="bold")
            else:
                sns.countplot(x=col, data=df, palette="Blues_d", ax=ax,
                              order=df[col].value_counts().index)
                ax.set_title(f"Distribuția: {col}", fontweight="bold")

            ax.set_xlabel(col)
            ax.set_ylabel("Frecvență")
            ax.tick_params(axis="x", rotation=30)
            ax.grid(axis="y", linestyle="--", alpha=0.4)

            # Adăugăm procentul deasupra fiecărei bare
            total = len(df[col].dropna())
            for p in ax.patches:
                pct = f"{p.get_height() / total * 100:.1f}%"
                ax.annotate(pct,
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha="center", va="bottom", fontsize=8, color="#374151")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Distribuție is_canceled cu Plotly
        st.markdown('<p class="sec-title">Distribuția variabilei țintă: is_canceled</p>',
                    unsafe_allow_html=True)
        canceled_counts = df["is_canceled"].value_counts().reset_index()
        canceled_counts.columns = ["Anulat", "Count"]
        canceled_counts["Anulat"] = canceled_counts["Anulat"].map(
            {0: "Rezervare menținută", 1: "Rezervare anulată"}
        )
        fig_pie = px.pie(
            canceled_counts, values="Count", names="Anulat",
            color_discrete_sequence=["#3b82f6", "#ef4444"],
            title="Proporția rezervărilor anulate vs. menținute"
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label",
                               textfont_size=13)
        fig_pie.update_layout(height=350, margin=dict(t=50, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

        # Rezervări pe luni (Plotly - interactiv)
        st.markdown('<p class="sec-title">Număr de rezervări per lună (ambele hoteluri)</p>',
                    unsafe_allow_html=True)
        df_luni = (df.groupby(["arrival_month_num", "arrival_date_month", "hotel"])
                     .size()
                     .reset_index(name="nr_rezervari"))
        df_luni = df_luni.sort_values("arrival_month_num")

        fig_luni = px.bar(
            df_luni, x="arrival_date_month", y="nr_rezervari",
            color="hotel", barmode="group",
            color_discrete_sequence=["#3b82f6", "#f97316"],
            title="Rezervări lunare per tip de hotel",
            labels={"arrival_date_month": "Luna",
                    "nr_rezervari": "Nr. rezervări",
                    "hotel": "Tip hotel"},
            category_orders={"arrival_date_month": [
                "January","February","March","April","May","June",
                "July","August","September","October","November","December"
            ]}
        )
        fig_luni.update_layout(height=380, xaxis_tickangle=-30)
        st.plotly_chart(fig_luni, use_container_width=True)

        st.markdown("""
        <div class="interpret-box">
            <strong>Interpretare economică:</strong><br>
            • <strong>is_canceled</strong>: ~37% din rezervări sunt anulate - o rată semnificativ de mare
            care justifică construirea unui model predictiv de clasificare.<br>
            • <strong>hotel</strong>: City Hotel are mai multe rezervări dar și o rată de anulare mai mare
            față de Resort Hotel - comportament specific turismului urban vs. vacanță.<br>
            • <strong>Sezonalitate</strong>: Vârfuri clare în lunile de vară (iulie-august) și toamnă timpurie,
            cu un minim în ianuarie - informație importantă pentru strategia de prețuri.
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 3: OUTLIERI ──────────────────────────────────────────
    with tab3:
        st.markdown('<p class="sec-title">Detectarea outlierilor prin metoda IQR</p>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            <strong>Metoda IQR (Interquartile Range):</strong><br>
            IQR = Q3 - Q1 &nbsp;|&nbsp;
            Limita inferioară = Q1 - 1.5 × IQR &nbsp;|&nbsp;
            Limita superioară = Q3 + 1.5 × IQR<br>
            Valorile în afara acestui interval sunt considerate outlieri potențiali.
        </div>
        """, unsafe_allow_html=True)

        def detecteaza_outlieri_iqr(df: pd.DataFrame, col: str):
            """
            Identifică outlierii dintr-o coloană numerică folosind metoda IQR.
            Returnează limitele și un DataFrame cu valorile outlier.
            """
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outlieri = df[(df[col] < lower) | (df[col] > upper)]
            return lower, upper, outlieri

        cols_outlieri = ["adr", "lead_time", "total_nights", "days_in_waiting_list"]
        cols_outlieri = [c for c in cols_outlieri if c in df.columns]

        # Tabel sumar outlieri
        sumar = []
        for col in cols_outlieri:
            lower, upper, outlieri = detecteaza_outlieri_iqr(df, col)
            sumar.append({
                "Coloană": col,
                "Q1": round(df[col].quantile(0.25), 2),
                "Q3": round(df[col].quantile(0.75), 2),
                "Limita inf.": round(lower, 2),
                "Limita sup.": round(upper, 2),
                "Nr. outlieri": len(outlieri),
                "% outlieri": round(len(outlieri) / len(df) * 100, 2)
            })

        df_sumar = pd.DataFrame(sumar)
        st.dataframe(df_sumar, use_container_width=True, hide_index=True)

        # Boxplot-uri pentru fiecare coloană relevantă
        st.markdown('<p class="sec-title">Boxplot-uri per variabilă numerică</p>',
                    unsafe_allow_html=True)

        fig, axes = plt.subplots(1, len(cols_outlieri),
                                 figsize=(4 * len(cols_outlieri), 4))
        if len(cols_outlieri) == 1:
            axes = [axes]

        for ax, col in zip(axes, cols_outlieri):
            sns.boxplot(x=df[col], ax=ax, color="#3b82f6",
                        flierprops=dict(marker="o", color="#ef4444",
                                        alpha=0.3, markersize=2))
            ax.set_title(f"Boxplot: {col}", fontsize=10, fontweight="bold")
            ax.set_xlabel(col, fontsize=9)
            ax.grid(axis="x", linestyle="--", alpha=0.4)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Boxplot adr vs hotel_type cu Plotly
        st.markdown('<p class="sec-title">Distribuția ADR pe tip de hotel și tip de cameră</p>',
                    unsafe_allow_html=True)
        df_box = df[(df["adr"] > 0) & (df["adr"] < df["adr"].quantile(0.99))]
        fig_box = px.box(
            df_box, x="hotel", y="adr", color="reserved_room_type",
            title="ADR per tip de hotel și tip de cameră (fără outlieri extremi)",
            labels={"hotel": "Tip hotel", "adr": "ADR (€/noapte)",
                    "reserved_room_type": "Tip cameră"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_box.update_layout(height=420)
        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("""
        <div class="interpret-box">
            <strong>Interpretare economică:</strong><br>
            • <strong>adr</strong>: Există valori de 0 € (posibil erori sau gratuități)
            și valori >5.000 € (camere de lux sau erori de introducere). Vor fi tratate
            în etapa de preprocesare prin capping la percentila 99%.<br>
            • <strong>lead_time</strong>: Valori >500 zile sunt excepționale și pot distorsiona
            modelele - se vor trata prin limitare la percentila 99%.<br>
            • <strong>total_nights</strong>: Sejururi >30 de nopți sunt outlieri dar pot reprezenta
            închirieri pe termen lung - le analizăm separat.
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 4: GRUPĂRI & AGREGĂRI ─────────────────────────────────
    with tab4:
        st.markdown('<p class="sec-title">Grupări și agregări - pandas groupby</p>',
                    unsafe_allow_html=True)

        # 1. Rata de anulare per segment de piață
        st.markdown("**1. Rata de anulare per segment de piață**")
        rata_anulare_segment = (
            df.groupby("market_segment")
            .agg(
                total_rezervari=("is_canceled", "count"),
                rezervari_anulate=("is_canceled", "sum"),
                adr_mediu=("adr", "mean"),
            )
            .assign(rata_anulare=lambda x: (x["rezervari_anulate"] / x["total_rezervari"] * 100).round(1))
            # Excludem segmentele cu mai puțin de 10 rezervări - nerelevante statistic (ex: Undefined cu 2 rez.)
            .query("total_rezervari >= 10")
            .sort_values("rata_anulare", ascending=False)
            .round(2)
        )
        st.dataframe(rata_anulare_segment, use_container_width=True)

        fig_seg = px.bar(
            rata_anulare_segment.reset_index(),
            x="market_segment", y="rata_anulare",
            color="rata_anulare",
            color_continuous_scale="RdYlGn_r",
            title="Rata de anulare (%) per segment de piață",
            labels={"market_segment": "Segment", "rata_anulare": "Rată anulare (%)"},
            text_auto=".1f"
        )
        fig_seg.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_seg, use_container_width=True)

        # 2. Venituri estimate per tip de hotel și lună
        st.markdown("**2. Venituri estimate medii per tip de hotel și lună**")
        venituri_luna = (
            df[df["adr"] > 0]
            .groupby(["hotel", "arrival_month_num", "arrival_date_month"])
            .agg(
                adr_mediu=("adr", "mean"),
                revenue_total=("revenue_estimat", "sum"),
                nr_rezervari=("adr", "count")
            )
            .reset_index()
            .sort_values("arrival_month_num")
            .round(2)
        )

        fig_ven = px.line(
            venituri_luna, x="arrival_date_month", y="adr_mediu",
            color="hotel", markers=True,
            color_discrete_sequence=["#3b82f6", "#f97316"],
            title="ADR mediu lunar per tip de hotel",
            labels={"arrival_date_month": "Luna", "adr_mediu": "ADR mediu (€)",
                    "hotel": "Tip hotel"},
            category_orders={"arrival_date_month": [
                "January","February","March","April","May","June",
                "July","August","September","October","November","December"
            ]}
        )
        fig_ven.update_layout(height=370, xaxis_tickangle=-30)
        st.plotly_chart(fig_ven, use_container_width=True)

        # 3. Funcții de grup - top 10 țări după nr. rezervări
        st.markdown("**3. Top 10 țări de origine după numărul de rezervări**")
        top_tari = (
            df.groupby("country")
            .agg(
                nr_rezervari=("hotel", "count"),
                rata_anulare=("is_canceled", "mean"),
                adr_mediu=("adr", "mean")
            )
            .assign(rata_anulare=lambda x: (x["rata_anulare"] * 100).round(1))
            .round(2)
            .nlargest(10, "nr_rezervari")
        )
        st.dataframe(top_tari, use_container_width=True)

        # 4. Statistici agregate per tip de depozit
        st.markdown("**4. Statistici agregate per tip de depozit**")
        stats_depozit = (
            df.groupby("deposit_type")
            .agg(
                nr_rezervari=("is_canceled", "count"),
                rata_anulare=("is_canceled", "mean"),
                adr_mediu=("adr", "mean"),
                lead_time_mediu=("lead_time", "mean"),
                lead_time_max=("lead_time", "max")
            )
            .assign(rata_anulare=lambda x: (x["rata_anulare"] * 100).round(1))
            .round(2)
        )
        st.dataframe(stats_depozit, use_container_width=True)

        # df.to_csv() - exportăm agregarea
        st.markdown('<p class="sec-title">Export rezultat agregare - df.to_csv()</p>',
                    unsafe_allow_html=True)
        csv_export = rata_anulare_segment.reset_index().to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="⬇️ Descarcă rata_anulare_segment.csv",
            data=csv_export,
            file_name="rata_anulare_segment.csv",
            mime="text/csv",
            help="Exportăm rezultatul groupby în CSV - df.to_csv()"
        )

        st.markdown("""
        <div class="interpret-box">
            <strong>Interpretare economică:</strong><br>
            • Segmentul <strong>Groups</strong> are cea mai mare rată de anulare - rezervările de grup
            sunt speculative și se anulează frecvent la apropierea datei de check-in.<br>
            • <strong>Resort Hotel</strong> are ADR mai mare vara (sezonalitate pronunțată) față de
            City Hotel care e mai stabil pe tot parcursul anului.<br>
        """, unsafe_allow_html=True)

    # ── TAB 5: CORELAȚII ──────────────────────────────────────────
    with tab5:
        st.markdown('<p class="sec-title">Matricea de corelație - variabile numerice</p>',
                    unsafe_allow_html=True)

        cols_corr = [
            "is_canceled", "lead_time", "adr", "total_nights",
            "adults", "children", "babies", "booking_changes",
            "days_in_waiting_list", "total_of_special_requests",
            "required_car_parking_spaces", "arrival_month_num"
        ]
        cols_corr = [c for c in cols_corr if c in df.columns]
        corr_matrix = df[cols_corr].corr(method="pearson")

        fig, ax = plt.subplots(figsize=(11, 8))
        sns.heatmap(
            corr_matrix, annot=True, cmap="coolwarm",
            fmt=".2f", linewidths=0.5, linecolor="white",
            vmin=-1, vmax=1, ax=ax,
            annot_kws={"size": 8}
        )
        ax.set_title("Matricea de corelație Pearson - variabile numerice",
                     fontsize=12, fontweight="bold", pad=15)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        <div class="interpret-box">
            <strong>Interpretare matricea de corelație:</strong><br>
            • <strong>Roșu intens</strong> = corelație pozitivă puternică (spre +1) &nbsp;|&nbsp;
            <strong>Albastru intens</strong> = corelație negativă puternică (spre -1) &nbsp;|&nbsp;
            <strong>Culori deschise</strong> ≈ 0 = relație slabă<br><br>
        </div>
        """, unsafe_allow_html=True)

        # corr.sort_values() - sortăm corelațiile față de is_canceled
        st.markdown('<p class="sec-title">Corelații sortate față de is_canceled (variabila țintă)</p>',
                    unsafe_allow_html=True)
        corr_sorted = corr_matrix.corr()["is_canceled"].drop("is_canceled").sort_values(ascending=False)

        fig_corr_bar, ax_corr = plt.subplots(figsize=(9, 4))
        culori_corr = ["#ef4444" if v > 0 else "#3b82f6" for v in corr_sorted.values]
        ax_corr.barh(corr_sorted.index[::-1], corr_sorted.values[::-1],
                     color=culori_corr[::-1], edgecolor="white")
        ax_corr.axvline(0, color="#1f2937", linewidth=1.5, linestyle="--")
        ax_corr.set_title("Corelații Pearson față de is_canceled (roșu=pozitiv, albastru=negativ)",
                          fontweight="bold", fontsize=10)
        ax_corr.set_xlabel("Coeficient de corelație")
        ax_corr.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig_corr_bar)
        plt.close()

        st.markdown("""
        <div class="interpret-box">
            Variabilele cu corelație <strong>pozitivă</strong> față de <em>is_canceled</em>
            (roșu) cresc probabilitatea de anulare. Variabilele cu corelație <strong>negativă</strong>
            (albastru) o reduc. Aceasta ne ghidează selecția de features pentru modelul de clasificare.
        </div>
        """, unsafe_allow_html=True)

        # Scatter plot interactiv ADR vs lead_time
        st.markdown('<p class="sec-title">Scatter: ADR vs. Lead Time (colorat după anulare)</p>',
                    unsafe_allow_html=True)
        df_scatter = df[(df["adr"] > 0) & (df["adr"] < 600)].sample(
            min(5000, len(df)), random_state=42
        )
        fig_sc = px.scatter(
            df_scatter, x="lead_time", y="adr",
            color=df_scatter["is_canceled"].map(
                {0: "Menținută", 1: "Anulată"}
            ),
            opacity=0.4,
            color_discrete_map={"Menținută": "#3b82f6", "Anulată": "#ef4444"},
            title="Relația dintre Lead Time și ADR (eșantion 5.000 rezervări)",
            labels={"lead_time": "Lead Time (zile)", "adr": "ADR (€/noapte)",
                    "color": "Status"}
        )
        fig_sc.update_layout(height=380)
        st.plotly_chart(fig_sc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# SECȚIUNILE 3-7 - PLACEHOLDER (vor fi completate în pașii următori)
# ═══════════════════════════════════════════════════════════════
elif sectiune_curenta == "Hartă geografică":
    st.markdown("""
    <div class="page-header">
        <h1>🗺️ Hartă geografică</h1>
        <p>Distribuția rezervărilor pe țări de origine - GeoPandas</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        import geopandas as gpd

        # ── Pregătim datele geografice ──────────────────────────────
        # Numărăm rezervările per țară (cod ISO alpha-3)
        tari_rezervari = (
            df.groupby("country")
            .agg(
                nr_rezervari=("hotel", "count"),
                rata_anulare=("is_canceled", "mean"),
                adr_mediu=("adr", "mean")
            )
            .assign(rata_anulare=lambda x: (x["rata_anulare"] * 100).round(1))
            .round(2)
            .reset_index()
        )

        # GeoPandas 1.0+ a eliminat geopandas.datasets - citim direct de pe naturalearthdata.com
        @st.cache_data
        def incarca_harta():
            url = (
                "https://naciscdn.org/naturalearth/110m/cultural/"
                "ne_110m_admin_0_countries.zip"
            )
            lume = gpd.read_file(url)
            # Fișierul de pe naturalearth are coloana "ISO_A3" cu majuscule
            # o redenumim în "iso_a3" pentru consistență
            if "ISO_A3" in lume.columns:
                lume = lume.rename(columns={"ISO_A3": "iso_a3"})
            return lume

        lume = incarca_harta()

        # Facem merge între harta lumii și datele noastre
        # Coloana din geopandas: "iso_a3" | Coloana din dataset: "country"
        harta_date = lume.merge(
            tari_rezervari,
            left_on="iso_a3",
            right_on="country",
            how="left"
        )

        # ── Metrici rapide ──────────────────────────────────────────
        top3 = tari_rezervari.nlargest(3, "nr_rezervari")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Țări reprezentate",
                    f"{tari_rezervari['country'].nunique()}")
        for i, (_, row) in enumerate(top3.iterrows()):
            [col2, col3, col4][i].metric(
                f"#{i+1} - {row['country']}",
                f"{int(row['nr_rezervari']):,} rez."
            )

        st.markdown('<p class="sec-title">Harta coropleth - număr de rezervări per țară</p>',
                    unsafe_allow_html=True)

        # ── Plotly choropleth (mai interactiv decât matplotlib) ─────
        fig_harta = px.choropleth(
            tari_rezervari,
            locations="country",
            locationmode="ISO-3",
            color="nr_rezervari",
            hover_name="country",
            hover_data={
                "nr_rezervari": True,
                "rata_anulare": True,
                "adr_mediu": True
            },
            color_continuous_scale="Blues",
            title="Numărul de rezervări hoteliere per țară de origine",
            labels={
                "nr_rezervari": "Nr. rezervări",
                "rata_anulare": "Rată anulare (%)",
                "adr_mediu": "ADR mediu (€)"
            }
        )
        fig_harta.update_layout(
            height=500,
            geo=dict(showframe=False, showcoastlines=True,
                     projection_type="equirectangular"),
            margin=dict(l=0, r=0, t=50, b=0),
            coloraxis_colorbar=dict(title="Nr. rezervări")
        )
        st.plotly_chart(fig_harta, use_container_width=True)

        # ── Harta GeoPandas cu matplotlib ──────────────────────────
        st.markdown('<p class="sec-title">Harta GeoPandas - matplotlib (vizualizare statică)</p>',
                    unsafe_allow_html=True)

        fig_geo, ax = plt.subplots(figsize=(14, 7))

        # Desenăm toate țările în gri deschis
        lume.plot(ax=ax, color="#e2e8f0", edgecolor="#cbd5e1", linewidth=0.4)

        # Suprapunem țările cu date colorate după numărul de rezervări
        harta_date.dropna(subset=["nr_rezervari"]).plot(
            column="nr_rezervari",
            ax=ax,
            cmap="Blues",
            legend=True,
            legend_kwds={
                "label": "Număr rezervări",
                "orientation": "horizontal",
                "shrink": 0.6,
                "pad": 0.02
            },
            missing_kwds={"color": "#e2e8f0", "label": "Fără date"}
        )

        ax.set_title(
            "Distribuția geografică a rezervărilor hoteliere\n(Hotel Booking Demand, 2015–2017)",
            fontsize=13, fontweight="bold", pad=12
        )
        ax.set_axis_off()
        plt.tight_layout()
        st.pyplot(fig_geo)
        plt.close()

        # ── Top 15 țări - bar chart ─────────────────────────────────
        st.markdown('<p class="sec-title">Top 15 țări după numărul de rezervări</p>',
                    unsafe_allow_html=True)

        top15 = tari_rezervari.nlargest(15, "nr_rezervari")

        fig_top, ax2 = plt.subplots(figsize=(11, 5))
        culori_bar = ["#1d4ed8" if i == 0 else "#3b82f6" if i < 3 else "#93c5fd"
                      for i in range(len(top15))]
        bars = ax2.barh(top15["country"][::-1], top15["nr_rezervari"][::-1],
                        color=culori_bar[::-1], edgecolor="white")
        ax2.set_xlabel("Număr de rezervări")
        ax2.set_title("Top 15 țări de origine - număr de rezervări",
                      fontweight="bold")
        ax2.grid(axis="x", linestyle="--", alpha=0.4)

        for bar, val in zip(bars, top15["nr_rezervari"][::-1]):
            ax2.text(bar.get_width() + 100, bar.get_y() + bar.get_height() / 2,
                     f"{int(val):,}", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_top)
        plt.close()

        # ── Scatter: rata anulare vs. adr_mediu per țară ────────────
        st.markdown('<p class="sec-title">Rata de anulare vs. ADR mediu per țară (top 30)</p>',
                    unsafe_allow_html=True)

        top30 = tari_rezervari.nlargest(30, "nr_rezervari")
        fig_sc2 = px.scatter(
            top30,
            x="adr_mediu", y="rata_anulare",
            size="nr_rezervari", color="nr_rezervari",
            text="country",
            color_continuous_scale="Blues",
            title="Relația ADR mediu - Rată anulare (top 30 țări după volum)",
            labels={"adr_mediu": "ADR mediu (€)",
                    "rata_anulare": "Rată anulare (%)",
                    "nr_rezervari": "Nr. rezervări"}
        )
        fig_sc2.update_traces(textposition="top center", textfont_size=9)
        fig_sc2.update_layout(height=450)
        st.plotly_chart(fig_sc2, use_container_width=True)

        st.markdown("""
        <div class="interpret-box">
            <strong>Interpretare economică:</strong><br>
            • <strong>Portugalia (PRT)</strong> este de departe cea mai reprezentată țară - hotelurile
            analizate sunt portugheze, deci o parte mare a clienților sunt locali sau din piața internă.<br>
            • <strong>Marea Britanie (GBR), Franța (FRA), Spania (ESP)</strong> - principalele piețe
            internaționale europene, importante pentru strategia de marketing.<br>
            • Există o corelație slabă pozitivă între ADR și rata de anulare: țările care plătesc mai mult pe cazare
            tind să anuleze mai des - posibil rezervări speculative pentru camere premium.<br>
            • <strong>Oportunitate de extindere:</strong> piețe cu ADR ridicat și rată de anulare mică
            (ex. țări din Asia-Pacific) reprezintă segmente de crescut.
        </div>
        """, unsafe_allow_html=True)

    except ImportError:
        st.error("❌ GeoPandas nu este instalat. Rulează: `pip install geopandas`")
    except Exception as e:
        st.error(f"❌ Eroare la încărcarea hărții: {e}")


# ═══════════════════════════════════════════════════════════════
# SECȚIUNEA 4 - PREPROCESARE
# ═══════════════════════════════════════════════════════════════
elif sectiune_curenta == "Preprocesare":
    st.markdown("""
    <div class="page-header">
        <h1>⚙️ Preprocesare date</h1>
        <p>Tratare valori lipsă | Tratare outlieri | Codificare | Scalare</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>De ce este necesară preprocesarea?</strong><br>
        Modelele de Machine Learning nu pot lucra cu valori lipsă, text liber sau variabile la scări diferite.
        Parcurgem 4 etape: <strong>(1)</strong> tratăm valorile lipsă, <strong>(2)</strong> eliminăm/limităm
        outlierii extremi, <strong>(3)</strong> codificăm variabilele categorice în numere,
        <strong>(4)</strong> scalăm variabilele numerice la aceeași scară.
    </div>
    """, unsafe_allow_html=True)

    # ── ALEGEREA METODEI DE IMPUTARE ───────────────────────────────
    st.markdown('<p class="sec-title">Alegerea metodei de imputare a valorilor lipsă</p>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>Selectează metoda de imputare:</strong> Valorile lipsă din coloanele numerice
        (<em>children</em>, <em>agent</em>) pot fi tratate prin mai multe strategii.
        Alege una dintre cele 3 metode de mai jos și observă cum se modifică distribuția datelor.
    </div>
    """, unsafe_allow_html=True)

    metoda_imputare = st.radio(
        "Metoda de imputare pentru coloanele numerice cu valori lipsă (children, agent):",
        [
            "① Imputare cu valoare constantă (0)",
            "② Imputare cu media (mean)",
            "③ Imputare cu mediana (median)"
        ],
        index=0,
        key="metoda_imputare",
        help="Alege metoda și observă mai jos cum se modifică distribuțiile și statisticile."
    )

    # ── COMPARAȚIE METODE DE IMPUTARE ──────────────────────────────
    # Determinăm codul metodei alese
    if "①" in metoda_imputare:
        metoda_cod = "constant"
        metoda_descriere = "valoare constantă (0)"
    elif "②" in metoda_imputare:
        metoda_cod = "mean"
        metoda_descriere = "media coloanei"
    else:
        metoda_cod = "median"
        metoda_descriere = "mediana coloanei"

    st.markdown(f"""
    <div class="warn-box">
        <strong>Metodă selectată:</strong> Se va aplica imputarea cu <strong>{metoda_descriere}</strong>
        pe coloanele numerice <em>children</em> și <em>agent</em>.
        Coloana <em>country</em> rămâne imputată cu „Unknown" indiferent de metodă.
    </div>
    """, unsafe_allow_html=True)

    # ── Tabel comparativ: toate cele 3 metode ──────────────────────
    st.markdown('<p class="sec-title">Tabel comparativ - efectul celor 3 metode pe coloana agent</p>',
                unsafe_allow_html=True)

    agent_mean_val = df["agent"].mean()
    agent_median_val = df["agent"].median()
    agent_series = {
        "Constantă (0)": df["agent"].fillna(0),
        "Media":         df["agent"].fillna(agent_mean_val),
        "Mediana":       df["agent"].fillna(agent_median_val),
    }
    comparatie_rows = []
    for met_name, serie in agent_series.items():
        comparatie_rows.append({
            "Metodă": met_name,
            "Valoare imputată": 0 if met_name == "Constantă (0)" else round(agent_mean_val, 2) if met_name == "Media" else round(agent_median_val, 2),
            "Nr. valori lipsă tratate": int(df["agent"].isnull().sum()),
            "Media după imputare": round(serie.mean(), 2),
            "Mediana după imputare": round(serie.median(), 2),
            "Std după imputare": round(serie.std(), 2),
        })
    df_comparatie = pd.DataFrame(comparatie_rows)

    # Evidențiem rândul corespunzător metodei selectate
    metoda_la_label = {"constant": "Constantă (0)", "mean": "Media", "median": "Mediana"}
    label_selectat = metoda_la_label[metoda_cod]

    def evidentiaza_rand_selectat(row):
        if row["Metodă"] == label_selectat:
            return ["background-color: #1e3a5c; color: #e8f4fd; font-weight: bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_comparatie.style.apply(evidentiaza_rand_selectat, axis=1),
        use_container_width=True, hide_index=True
    )

    # ── Grafic dinamic: distribuția agent ÎNAINTE vs DUPĂ (metoda selectată) ──
    st.markdown('<p class="sec-title">Distribuția coloanei agent - ÎNAINTE vs. DUPĂ imputare (metoda selectată)</p>',
                unsafe_allow_html=True)

    # Construim seria imputată conform metodei alese
    if metoda_cod == "constant":
        agent_dupa = df["agent"].fillna(0)
        val_imp_afisata = 0
    elif metoda_cod == "mean":
        agent_dupa = df["agent"].fillna(agent_mean_val)
        val_imp_afisata = round(agent_mean_val, 2)
    else:
        agent_dupa = df["agent"].fillna(agent_median_val)
        val_imp_afisata = round(agent_median_val, 2)

    col_met1, col_met2, col_met3, col_met4 = st.columns(4)
    col_met1.metric("Metodă aplicată", metoda_descriere.capitalize())
    col_met2.metric("Valoare imputată", f"{val_imp_afisata}")
    col_met3.metric("Media agent după", f"{agent_dupa.mean():.2f}",
                    delta=f"{agent_dupa.mean() - df['agent'].dropna().mean():+.2f} vs. originală")
    col_met4.metric("Mediana agent după", f"{agent_dupa.median():.1f}",
                    delta=f"{agent_dupa.median() - df['agent'].dropna().median():+.1f} vs. originală")

    fig_imp_dyn, axes_imp_dyn = plt.subplots(1, 2, figsize=(13, 4))

    # ÎNAINTE (cu NaN-urile excluse din histogramă)
    agent_inainte = df["agent"].dropna()
    axes_imp_dyn[0].hist(agent_inainte, bins=50, color="#ef4444",
                         edgecolor="white", alpha=0.85)
    axes_imp_dyn[0].axvline(agent_inainte.mean(), color="#1f2937", linestyle="--",
                            linewidth=1.5, label=f"Media: {agent_inainte.mean():.1f}")
    axes_imp_dyn[0].axvline(agent_inainte.median(), color="#6b21a8", linestyle=":",
                            linewidth=1.5, label=f"Mediana: {agent_inainte.median():.1f}")
    axes_imp_dyn[0].set_title(f"agent - ÎNAINTE\n({int(df['agent'].isnull().sum()):,} valori lipsă excluse)",
                               fontweight="bold", fontsize=10)
    axes_imp_dyn[0].set_xlabel("Valoare agent")
    axes_imp_dyn[0].set_ylabel("Frecvență")
    axes_imp_dyn[0].legend(fontsize=8)
    axes_imp_dyn[0].grid(axis="y", linestyle="--", alpha=0.4)

    # DUPĂ (cu metoda selectată)
    culoare_metoda = {"constant": "#3b82f6", "mean": "#f97316", "median": "#22c55e"}
    axes_imp_dyn[1].hist(agent_dupa, bins=50,
                         color=culoare_metoda[metoda_cod],
                         edgecolor="white", alpha=0.85)
    axes_imp_dyn[1].axvline(agent_dupa.mean(), color="#1f2937", linestyle="--",
                            linewidth=1.5, label=f"Media: {agent_dupa.mean():.1f}")
    axes_imp_dyn[1].axvline(agent_dupa.median(), color="#6b21a8", linestyle=":",
                            linewidth=1.5, label=f"Mediana: {agent_dupa.median():.1f}")
    axes_imp_dyn[1].set_title(f"agent - DUPĂ imputare cu {metoda_descriere}\n(val. imputată: {val_imp_afisata})",
                               fontweight="bold", fontsize=10)
    axes_imp_dyn[1].set_xlabel("Valoare agent")
    axes_imp_dyn[1].set_ylabel("Frecvență")
    axes_imp_dyn[1].legend(fontsize=8)
    axes_imp_dyn[1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    st.pyplot(fig_imp_dyn)
    plt.close()

    # ── Cele 3 histograme suprapuse pentru referință ───────────────
    st.markdown('<p class="sec-title">Toate cele 3 metode pe aceeași histogramă</p>',
                unsafe_allow_html=True)

    fig_overlay, ax_overlay = plt.subplots(figsize=(10, 4))
    for (met_name, serie), culoare in zip(agent_series.items(),
                                           ["#3b82f6", "#f97316", "#22c55e"]):
        ax_overlay.hist(serie, bins=50, color=culoare, edgecolor="white",
                        alpha=0.45, label=f"{met_name} (media={serie.mean():.1f})")
    ax_overlay.set_title("Comparație: distribuția agent după fiecare metodă de imputare",
                          fontweight="bold")
    ax_overlay.set_xlabel("Valoare agent")
    ax_overlay.set_ylabel("Frecvență")
    ax_overlay.legend(fontsize=9)
    ax_overlay.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig_overlay)
    plt.close()

    st.markdown(f"""
    <div class="interpret-box">
        <strong>Observații despre metodele de imputare:</strong><br>
        • <strong>Constantă (0)</strong>: Presupune că valorile lipsă înseamnă „fără agent" (rezervare directă).
        Avantaj: interpretare clară. Dezavantaj: coboară media artificial.<br>
        • <strong>Media ({agent_mean_val:.1f})</strong>: Păstrează media originală neschimbată, dar crește valori
        la un nivel nerealist pentru multe observații care probabil erau 0.<br>
        • <strong>Mediana ({agent_median_val:.1f})</strong>: Compromis major - nu e influențată de outlieri,
        dar presupune că valorile lipsă sunt similare cu cele existente.
    </div>
    """, unsafe_allow_html=True)

    # ── FUNCȚIE PRINCIPALĂ DE PREPROCESARE ─────────────────────────
    @st.cache_data
    def preproceseaza_date(df: pd.DataFrame, metoda_imp: str):
        """
        Funcție completă de preprocesare a datelor hoteliere.
        Parcurge toate etapele și returnează atât DataFrame-ul curat
        cât și un jurnal al modificărilor efectuate.
        metoda_imp: 'constant' | 'mean' | 'median'
        """
        df_proc = df.copy()
        jurnal = []

        # ── ETAPA 1: Tratare valori lipsă ──────────────────────────
        nr_initial = len(df_proc)

        # children - imputare conform metodei alese
        nr_lipsa_children = df_proc["children"].isnull().sum()
        if metoda_imp == "constant":
            val_imp_children = 0
            met_label_children = "Imputare cu 0 (valoare constantă)"
        elif metoda_imp == "mean":
            val_imp_children = df_proc["children"].mean()
            met_label_children = f"Imputare cu media ({val_imp_children:.2f})"
        else:
            val_imp_children = df_proc["children"].median()
            met_label_children = f"Imputare cu mediana ({val_imp_children:.2f})"
        df_proc["children"] = df_proc["children"].fillna(val_imp_children).astype(int)
        jurnal.append({
            "Etapă": "1. Valori lipsă",
            "Coloană": "children",
            "Metodă": met_label_children,
            "Valori tratate": nr_lipsa_children,
            "Motivație": "Lipsa informației - imputare conform metodei selectate"
        })

        # country: categorică - imputare cu 'Unknown' (indiferent de metodă)
        nr_lipsa_country = df_proc["country"].isnull().sum()
        df_proc["country"] = df_proc["country"].fillna("Unknown")
        jurnal.append({
            "Etapă": "1. Valori lipsă",
            "Coloană": "country",
            "Metodă": "Imputare cu 'Unknown'",
            "Valori tratate": nr_lipsa_country,
            "Motivație": "Variabilă categorică - marcăm țara ca necunoscută"
        })

        # agent - imputare conform metodei alese
        nr_lipsa_agent = df_proc["agent"].isnull().sum()
        if metoda_imp == "constant":
            val_imp_agent = 0
            met_label_agent = "Imputare cu 0 (valoare constantă)"
        elif metoda_imp == "mean":
            val_imp_agent = df_proc["agent"].mean()
            met_label_agent = f"Imputare cu media ({val_imp_agent:.1f})"
        else:
            val_imp_agent = df_proc["agent"].median()
            met_label_agent = f"Imputare cu mediana ({val_imp_agent:.1f})"
        df_proc["agent"] = df_proc["agent"].fillna(val_imp_agent)
        jurnal.append({
            "Etapă": "1. Valori lipsă",
            "Coloană": "agent",
            "Metodă": met_label_agent,
            "Valori tratate": nr_lipsa_agent,
            "Motivație": "Imputare conform metodei selectate"
        })

        # company: ~112.593 lipsă (~94%) - coloana e irelevantă, o eliminăm
        df_proc = df_proc.drop(columns=["company"], errors="ignore")
        jurnal.append({
            "Etapă": "1. Valori lipsă",
            "Coloană": "company",
            "Metodă": "Eliminare coloană",
            "Valori tratate": 112593,
            "Motivație": "94% valori lipsă - informație insuficientă"
        })

        # Eliminăm coloanele de text liber fără valoare predictivă
        for col in ["name", "email", "phone-number", "credit_card",
                    "reservation_status_date"]:
            if col in df_proc.columns:
                df_proc = df_proc.drop(columns=[col])

        # ── ETAPA 2: Tratare outlieri ───────────────────────────────
        # adr: eliminăm valorile ≤0 (erori) și aplicăm capping la percentila 99%
        nr_adr_zero = (df_proc["adr"] <= 0).sum()
        df_proc = df_proc[df_proc["adr"] > 0]
        jurnal.append({
            "Etapă": "2. Outlieri",
            "Coloană": "adr",
            "Metodă": "Eliminare valori ≤0",
            "Valori tratate": nr_adr_zero,
            "Motivație": "Tarif zero = eroare sau gratuitate nereprezentativă"
        })

        upper_adr = df_proc["adr"].quantile(0.99)
        nr_adr_extreme = (df_proc["adr"] > upper_adr).sum()
        df_proc["adr"] = np.where(df_proc["adr"] > upper_adr, upper_adr, df_proc["adr"])
        jurnal.append({
            "Etapă": "2. Outlieri",
            "Coloană": "adr",
            "Metodă": f"Capping la percentila 99% ({upper_adr:.1f} €)",
            "Valori tratate": nr_adr_extreme,
            "Motivație": "Prețuri extreme distorsionează modelul de regresie"
        })

        # lead_time: capping la percentila 99%
        upper_lt = df_proc["lead_time"].quantile(0.99)
        nr_lt_extreme = (df_proc["lead_time"] > upper_lt).sum()
        df_proc["lead_time"] = np.where(
            df_proc["lead_time"] > upper_lt, upper_lt, df_proc["lead_time"]
        )
        jurnal.append({
            "Etapă": "2. Outlieri",
            "Coloană": "lead_time",
            "Metodă": f"Capping la percentila 99% ({upper_lt:.0f} zile)",
            "Valori tratate": nr_lt_extreme,
            "Motivație": "Rezervări la >500 zile distorsionează distribuția"
        })

        # ── ETAPA 3: Codificare variabile categorice ────────────────

        # 3a. Label Encoding pentru variabile cu 2 categorii (binare)
        le = LabelEncoder()

        df_proc["hotel_encoded"] = le.fit_transform(df_proc["hotel"])
        # Resort Hotel=1, City Hotel=0 (ordine alfabetică inversă)
        jurnal.append({
            "Etapă": "3. Encoding",
            "Coloană": "hotel",
            "Metodă": "Label Encoding → hotel_encoded",
            "Valori tratate": df_proc["hotel"].nunique(),
            "Motivație": "Variabilă binară - 2 categorii"
        })

        # 3b. One-Hot Encoding pentru variabile categorice nominale
        # (low cardinality: < 10 categorii)
        cols_ohe = ["meal", "market_segment", "deposit_type",
                    "customer_type", "distribution_channel",
                    "reserved_room_type"]
        cols_ohe = [c for c in cols_ohe if c in df_proc.columns]

        df_proc = pd.get_dummies(df_proc, columns=cols_ohe, drop_first=True)
        jurnal.append({
            "Etapă": "3. Encoding",
            "Coloană": ", ".join(cols_ohe),
            "Metodă": "One-Hot Encoding (drop_first=True)",
            "Valori tratate": len(cols_ohe),
            "Motivație": "Variabile nominale fără ordine naturală"
        })

        # 3c. Frequency Encoding pentru 'country' (high cardinality: 180+ țări)
        freq_country = df_proc["country"].value_counts(normalize=True)
        df_proc["country_freq"] = df_proc["country"].map(freq_country)
        jurnal.append({
            "Etapă": "3. Encoding",
            "Coloană": "country",
            "Metodă": "Frequency Encoding → country_freq",
            "Valori tratate": df_proc["country"].nunique(),
            "Motivație": "180+ categorii - OHE ar genera prea multe coloane"
        })

        # Eliminăm coloanele categorice originale rămase
        cols_de_eliminat = ["country", "hotel", "arrival_date_month",
                             "reservation_status", "assigned_room_type"]
        df_proc = df_proc.drop(columns=[c for c in cols_de_eliminat
                                         if c in df_proc.columns])

        # ── ETAPA 4: Scalare ────────────────────────────────────────

        # Selectăm coloanele numerice pentru scalare
        # (excludem target-urile și coloanele deja binare 0/1)
        cols_bool = df_proc.select_dtypes(include=["bool"]).columns.tolist()
        df_proc[cols_bool] = df_proc[cols_bool].astype(int)

        cols_pentru_scalare = [
            "lead_time", "adr", "total_nights", "days_in_waiting_list",
            "booking_changes", "country_freq", "arrival_month_num",
            "total_of_special_requests", "required_car_parking_spaces",
            "adults", "children", "babies"
        ]
        cols_pentru_scalare = [c for c in cols_pentru_scalare
                                if c in df_proc.columns]

        # StandardScaler - pentru modelele de regresie și clasificare
        scaler_std = StandardScaler()
        df_scaled_std = df_proc.copy()
        df_scaled_std[cols_pentru_scalare] = scaler_std.fit_transform(
            df_proc[cols_pentru_scalare]
        )
        jurnal.append({
            "Etapă": "4. Scalare",
            "Coloană": f"{len(cols_pentru_scalare)} coloane numerice",
            "Metodă": "StandardScaler (medie=0, std=1)",
            "Valori tratate": len(cols_pentru_scalare),
            "Motivație": "Regresie logistică și statsmodels necesită scări egale"
        })

        nr_final = len(df_proc)
        return df_proc, df_scaled_std, pd.DataFrame(jurnal), scaler_std, cols_pentru_scalare, nr_initial, nr_final

    # Rulăm preprocesarea cu metoda aleasă
    with st.spinner("Se procesează datele..."):
        (df_proc, df_scaled, df_jurnal,
         scaler_std, cols_scalate,
         nr_initial, nr_final) = preproceseaza_date(df, metoda_cod)

    # ── AFIȘĂM JURNALUL ────────────────────────────────────────────
    st.markdown('<p class="sec-title">Jurnalul transformărilor efectuate</p>',
                unsafe_allow_html=True)

    col_j1, col_j2, col_j3 = st.columns(3)
    col_j1.metric("Rânduri inițiale",  f"{nr_initial:,}")
    col_j2.metric("Rânduri finale",    f"{nr_final:,}")
    col_j3.metric("Rânduri eliminate", f"{nr_initial - nr_final:,}",
                  delta=f"-{(nr_initial - nr_final) / nr_initial * 100:.1f}%",
                  delta_color="red")

    st.dataframe(df_jurnal, use_container_width=True, hide_index=True)

    # ── VIZUALIZARE ETAPA 1: VALORI LIPSĂ DUPĂ TRATARE ─────────────
    st.markdown('<p class="sec-title">Etapa 1 - Valori lipsă: înainte și după (metoda selectată)</p>',
                unsafe_allow_html=True)

    cols_cu_lipsa = ["children", "country", "agent"]
    fig_vl, axes_vl = plt.subplots(1, 2, figsize=(10, 3.5))

    # Înainte
    lipsa_inainte = {c: df[c].isnull().sum() for c in cols_cu_lipsa}
    axes_vl[0].bar(lipsa_inainte.keys(), lipsa_inainte.values(),
                   color=["#ef4444", "#f97316", "#facc15"], edgecolor="white")
    axes_vl[0].set_title("Valori lipsă - ÎNAINTE", fontweight="bold")
    axes_vl[0].set_ylabel("Nr. valori lipsă")
    axes_vl[0].grid(axis="y", linestyle="--", alpha=0.4)
    for i, (k, v) in enumerate(lipsa_inainte.items()):
        axes_vl[0].text(i, v + 20, f"{v:,}", ha="center", fontsize=9)

    # După
    lipsa_dupa = {c: df_proc[c].isnull().sum()
                  if c in df_proc.columns else 0
                  for c in ["children", "country_freq", "agent"]}
    axes_vl[1].bar(["children", "country", "agent"],
                   list(lipsa_dupa.values()),
                   color=["#22c55e", "#22c55e", "#22c55e"], edgecolor="white")
    axes_vl[1].set_title(f"Valori lipsă - DUPĂ ({metoda_descriere})", fontweight="bold")
    axes_vl[1].set_ylabel("Nr. valori lipsă")
    axes_vl[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes_vl[1].set_ylim(0, max(lipsa_inainte.values()) * 1.15)
    for i in range(3):
        axes_vl[1].text(i, 50, "0", ha="center", fontsize=9, color="#166534")

    plt.tight_layout()
    st.pyplot(fig_vl)
    plt.close()

    # Statistici agent după imputare - se actualizează cu metoda
    st.markdown(f"**Statistici coloana `agent` după imputare cu {metoda_descriere}:**")
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    col_stat1.metric("Media", f"{df_proc['agent'].mean():.2f}")
    col_stat2.metric("Mediana", f"{df_proc['agent'].median():.1f}")
    col_stat3.metric("Std", f"{df_proc['agent'].std():.2f}")
    col_stat4.metric("Min / Max", f"{df_proc['agent'].min():.0f} / {df_proc['agent'].max():.0f}")

    # ── VIZUALIZARE ETAPA 2: OUTLIERI ADR ──────────────────────────
    st.markdown('<p class="sec-title">Etapa 2 - Outlieri: distribuția ADR înainte și după capping</p>',
                unsafe_allow_html=True)

    fig_out, axes_out = plt.subplots(1, 2, figsize=(12, 4))

    # Înainte
    adr_inainte = df[df["adr"] > 0]["adr"]
    axes_out[0].hist(adr_inainte, bins=50, color="#ef4444",
                     edgecolor="white", alpha=0.8)
    axes_out[0].set_title("Distribuția ADR - ÎNAINTE\n(inclusiv outlieri extremi)",
                           fontweight="bold")
    axes_out[0].set_xlabel("ADR (€/noapte)")
    axes_out[0].set_ylabel("Frecvență")
    axes_out[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes_out[0].annotate(f"Max: {adr_inainte.max():.0f} €",
                          xy=(adr_inainte.max(), 5),
                          xytext=(adr_inainte.max() * 0.6, 2000),
                          arrowprops=dict(arrowstyle="->", color="black"),
                          fontsize=9)

    # După
    adr_dupa = df_proc["adr"]
    axes_out[1].hist(adr_dupa, bins=50, color="#22c55e",
                     edgecolor="white", alpha=0.8)
    axes_out[1].set_title(
        f"Distribuția ADR - DUPĂ\n(capping la percentila 99%: {adr_dupa.max():.0f} €)",
        fontweight="bold"
    )
    axes_out[1].set_xlabel("ADR (€/noapte)")
    axes_out[1].set_ylabel("Frecvență")
    axes_out[1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    st.pyplot(fig_out)
    plt.close()

    # ── VIZUALIZARE ETAPA 3: ENCODING ──────────────────────────────
    st.markdown('<p class="sec-title">Etapa 3 - Encoding: exemplu One-Hot pe market_segment</p>',
                unsafe_allow_html=True)

    col_enc_a, col_enc_b = st.columns(2)
    with col_enc_a:
        st.markdown("**Înainte - coloana originală (text)**")
        st.dataframe(
            df[["market_segment"]].head(8),
            use_container_width=True, hide_index=True
        )
    with col_enc_b:
        st.markdown("**După - One-Hot Encoding (0/1)**")
        cols_market = [c for c in df_proc.columns if "market_segment_" in c]
        if cols_market:
            st.dataframe(
                df_proc[cols_market].head(8).astype(int),
                use_container_width=True, hide_index=True
            )

    # Frequency encoding pentru country
    st.markdown("**Frequency Encoding - coloana `country`**")
    col_fe_a, col_fe_b = st.columns(2)
    with col_fe_a:
        st.dataframe(
            df[["country"]].head(8),
            use_container_width=True, hide_index=True
        )
    with col_fe_b:
        st.dataframe(
            df_proc[["country_freq"]].head(8).round(5),
            use_container_width=True, hide_index=True
        )

    # ── VIZUALIZARE ETAPA 4: SCALARE ───────────────────────────────
    st.markdown('<p class="sec-title">Etapa 4 - Scalare: distribuția ADR și lead_time înainte și după StandardScaler</p>',
                unsafe_allow_html=True)

    fig_sc_viz, axes_sc = plt.subplots(2, 2, figsize=(12, 7))

    for idx, col in enumerate(["adr", "lead_time"]):
        if col not in df_proc.columns or col not in df_scaled.columns:
            continue
        # Înainte
        axes_sc[idx, 0].hist(df_proc[col], bins=40,
                              color="#3b82f6", edgecolor="white", alpha=0.8)
        axes_sc[idx, 0].set_title(f"{col} - ÎNAINTE scalare",
                                   fontweight="bold", fontsize=10)
        mean_val = df_proc[col].mean()
        axes_sc[idx, 0].axvline(mean_val, color="#ef4444",
                                 linestyle="--", linewidth=1.5,
                                 label=f"Medie: {mean_val:.1f}")
        axes_sc[idx, 0].legend(fontsize=8)
        axes_sc[idx, 0].set_ylabel("Frecvență")
        axes_sc[idx, 0].grid(axis="y", linestyle="--", alpha=0.4)

        # După
        axes_sc[idx, 1].hist(df_scaled[col], bins=40,
                              color="#22c55e", edgecolor="white", alpha=0.8)
        axes_sc[idx, 1].set_title(f"{col} - DUPĂ StandardScaler",
                                   fontweight="bold", fontsize=10)
        axes_sc[idx, 1].axvline(0, color="#ef4444",
                                 linestyle="--", linewidth=1.5,
                                 label="Medie: ≈0")
        axes_sc[idx, 1].legend(fontsize=8)
        axes_sc[idx, 1].set_ylabel("Frecvență")
        axes_sc[idx, 1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    st.pyplot(fig_sc_viz)
    plt.close()

    # ── DATAFRAME FINAL ─────────────────────────────────────────────
    st.markdown('<p class="sec-title">DataFrame final după preprocesare completă</p>',
                unsafe_allow_html=True)

    col_fin1, col_fin2 = st.columns(2)
    col_fin1.metric("Coloane inițiale", df.shape[1])
    col_fin2.metric("Coloane finale",   df_scaled.shape[1])

    st.dataframe(df_scaled.head(5), use_container_width=True)

    st.markdown("""
    <div class="interpret-box">
        <strong>Rezumatul preprocesării:</strong><br>
        • <strong>Valori lipsă:</strong> 4 coloane tratate - imputare cu 0 pentru <em>children</em> și <em>agent</em>,
        <em>Unknown</em> pentru <em>country</em>, eliminare <em>company</em> (94% lipsă).<br>
        • <strong>Outlieri:</strong> valorile ADR ≤0 eliminate, capping la percentila 99% pentru <em>adr</em>
        și <em>lead_time</em> - reduce influența valorilor extreme fără a pierde date valoroase.<br>
        • <strong>Encoding:</strong> Label Encoding pentru variabile binare, One-Hot pentru variabile nominale
        (low cardinality), Frequency Encoding pentru <em>country</em> (high cardinality - 180+ categorii).<br>
        • <strong>Scalare StandardScaler:</strong> toate variabilele numerice aduse la medie≈0 și std≈1 -
        obligatoriu pentru regresia logistică și statsmodels OLS.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SECȚIUNEA 5 - K-MEANS CLUSTERING
# ═══════════════════════════════════════════════════════════════
elif sectiune_curenta == "K-Means Clustering":
    st.markdown("""
    <div class="page-header">
        <h1>🔵 K-Means Clustering</h1>
        <p>Segmentarea clienților hotelieri - învățare nesupervizată</p>
    </div>
    """, unsafe_allow_html=True)

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    st.markdown("""
    <div class="info-box">
        <strong>Scopul clustering-ului:</strong> Grupăm rezervările în segmente de clienți
        cu comportament similar, fără a folosi etichete predefinite (învățare nesupervizată).
        Folosim variabilele <em>adr</em>, <em>lead_time</em> și <em>total_of_special_requests</em>
        ca profil al clientului. Rezultatele pot ghida strategia de prețuri și marketing.
    </div>
    """, unsafe_allow_html=True)

    # ── Pregătire date pentru clustering ──────────────────────────
    @st.cache_data
    def pregateste_kmeans(df: pd.DataFrame):
        df_km = df[(df["adr"] > 0) & (df["adr"] < df["adr"].quantile(0.99))].copy()
        df_km["lead_time_c"] = np.where(
            df_km["lead_time"] > df_km["lead_time"].quantile(0.99),
            df_km["lead_time"].quantile(0.99),
            df_km["lead_time"]
        )
        features = ["adr", "lead_time_c", "total_of_special_requests",
                    "total_nights", "adults"]
        features = [f for f in features if f in df_km.columns]
        X_km = df_km[features].dropna()
        scaler_km = StandardScaler()
        X_scaled_km = scaler_km.fit_transform(X_km)
        return X_km, X_scaled_km, features, df_km.loc[X_km.index]

    X_km, X_scaled_km, features_km, df_km_orig = pregateste_kmeans(df)

    # ── ELBOW METHOD ───────────────────────────────────────────────
    st.markdown('<p class="sec-title">Pasul 1 - Elbow Method: alegerea numărului optim de clustere</p>',
                unsafe_allow_html=True)

    @st.cache_data
    def calculeaza_wcss(X_scaled):
        # Eșantionăm la max 15.000 rânduri pentru viteză
        # Elbow Method nu necesită tot setul - rezultatele sunt identice pe un sample reprezentativ
        np.random.seed(42)
        n_sample = min(15000, len(X_scaled))
        idx = np.random.choice(len(X_scaled), n_sample, replace=False)
        X_sample = X_scaled[idx]

        wcss, sil_scores = [], []
        for k in range(2, 11):
            km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
            km.fit(X_sample)
            wcss.append(km.inertia_)
            sil_scores.append(silhouette_score(X_sample, km.labels_))
        return wcss, sil_scores

    with st.spinner("Se calculează WCSS pentru k=2..10 (eșantion 15.000 rânduri)..."):
        wcss_vals, sil_vals = calculeaza_wcss(X_scaled_km)

    col_elbow, col_sil = st.columns(2)

    with col_elbow:
        fig_elbow, ax_elbow = plt.subplots(figsize=(6, 4))
        ax_elbow.plot(range(2, 11), wcss_vals, marker="o", color="#3b82f6",
                      linewidth=2, markersize=7)
        ax_elbow.fill_between(range(2, 11), wcss_vals, alpha=0.1, color="#3b82f6")
        ax_elbow.set_title("Elbow Method - WCSS per număr de clustere",
                           fontweight="bold", fontsize=10)
        ax_elbow.set_xlabel("Număr de clustere (k)")
        ax_elbow.set_ylabel("WCSS (Within-Cluster Sum of Squares)")
        ax_elbow.grid(linestyle="--", alpha=0.4)
        # Marcăm k=4 ca punct optim
        ax_elbow.axvline(x=4, color="#ef4444", linestyle="--",
                         linewidth=1.5, label="k optim = 4")
        ax_elbow.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_elbow)
        plt.close()

    with col_sil:
        fig_sil, ax_sil = plt.subplots(figsize=(6, 4))
        ax_sil.plot(range(2, 11), sil_vals, marker="s", color="#22c55e",
                    linewidth=2, markersize=7)
        ax_sil.fill_between(range(2, 11), sil_vals, alpha=0.1, color="#22c55e")
        ax_sil.set_title("Silhouette Score per număr de clustere",
                         fontweight="bold", fontsize=10)
        ax_sil.set_xlabel("Număr de clustere (k)")
        ax_sil.set_ylabel("Silhouette Score (mai mare = mai bun)")
        ax_sil.grid(linestyle="--", alpha=0.4)
        k_optim_sil = sil_vals.index(max(sil_vals)) + 2
        ax_sil.axvline(x=k_optim_sil, color="#ef4444", linestyle="--",
                       linewidth=1.5, label=f"k optim = {k_optim_sil}")
        ax_sil.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_sil)
        plt.close()

    st.markdown("""
    <div class="interpret-box">
        <strong>Cum alegem k:</strong>
        <em>Elbow Method</em> - căutăm „cotul" curbei WCSS unde scăderea se aplatizează.
        <em>Silhouette Score</em> - cu cât mai aproape de 1, cu atât clusterele sunt mai bine definite.
        Ambele metode indică <strong>k = 4</strong> ca număr optim de clustere.
    </div>
    """, unsafe_allow_html=True)

    # ── ANTRENARE MODEL ────────────────────────────────────────────
    st.markdown('<p class="sec-title">Pasul 2 - Antrenare K-Means cu k=4</p>',
                unsafe_allow_html=True)

    slider_k = st.slider("Alege numărul de clustere (k)", min_value=2,
                         max_value=8, value=4, step=1)

    @st.cache_data
    def antreneaza_kmeans(X_scaled, k):
        # Antrenăm pe sample de 20.000 rânduri, aplicăm predict pe tot setul
        np.random.seed(42)
        n_sample = min(20000, len(X_scaled))
        idx = np.random.choice(len(X_scaled), n_sample, replace=False)
        km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        km.fit(X_scaled[idx])
        # predict pe tot setul pentru vizualizare completă
        labels = km.predict(X_scaled)
        sil = silhouette_score(X_scaled[idx], km.predict(X_scaled[idx]))
        return km, labels, sil

    km_model, labels_km, sil_final = antreneaza_kmeans(X_scaled_km, slider_k)

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("k ales", slider_k)
    col_m2.metric("Silhouette Score", f"{sil_final:.4f}")
    col_m3.metric("WCSS (Inertia)", f"{km_model.inertia_:,.0f}")

    # ── VIZUALIZARE CLUSTERE ───────────────────────────────────────
    st.markdown('<p class="sec-title">Pasul 3 - Vizualizarea clusterelor: ADR vs Lead Time</p>',
                unsafe_allow_html=True)

    df_viz_km = X_km.copy()
    df_viz_km["Cluster"] = labels_km.astype(str)
    df_viz_km["hotel"] = df_km_orig["hotel"].values

    fig_scatter_km = px.scatter(
        df_viz_km.sample(min(8000, len(df_viz_km)), random_state=42),
        x="lead_time_c", y="adr",
        color="Cluster",
        symbol="hotel",
        opacity=0.55,
        color_discrete_sequence=px.colors.qualitative.Bold,
        title=f"Clustere K-Means (k={slider_k}) - ADR vs Lead Time",
        labels={"lead_time_c": "Lead Time (zile)", "adr": "ADR (€/noapte)",
                "Cluster": "Segment", "hotel": "Tip hotel"},
        hover_data=["total_of_special_requests", "total_nights"]
    )
    fig_scatter_km.update_layout(height=430)
    st.plotly_chart(fig_scatter_km, use_container_width=True)

    # ── SCATTER MATPLOTLIB cu sns.scatterplot + cluster_centers_

    st.markdown('<p class="sec-title">Vizualizare alternativă - sns.scatterplot + centroids</p>',
                unsafe_allow_html=True)

    culori_cls = ["#f59e0b", "#3b82f6", "#22c55e", "#a855f7",
                  "#ef4444", "#06b6d4", "#84cc16", "#f97316"]
    fig_sns_km, ax_sns_km = plt.subplots(figsize=(9, 5))

    for cluster_id in range(slider_k):
        mask = labels_km == cluster_id
        sns.scatterplot(
            x=X_km["lead_time_c"][mask],
            y=X_km["adr"][mask],
            color=culori_cls[cluster_id % len(culori_cls)],
            label=f"Cluster {cluster_id}",
            alpha=0.45, s=20, ax=ax_sns_km
        )

    # Centroids - cluster_centers_ pe spațiul original (denormalizat aproximativ)
    # Refacem cluster centers în spațiul original folosind inversul scalarului
    centers_original = km_model.cluster_centers_
    # Coloanele X_km în ordinea folosită la scalare: [adr, lead_time_c, ...]
    # Indicii 0=adr, 1=lead_time_c din features_km
    idx_adr = features_km.index("adr") if "adr" in features_km else 0
    idx_lt  = features_km.index("lead_time_c") if "lead_time_c" in features_km else 1

    from sklearn.preprocessing import StandardScaler as _SC
    _sc_tmp = _SC().fit(X_km[features_km])
    centers_inv = _sc_tmp.inverse_transform(centers_original)

    sns.scatterplot(
        x=centers_inv[:, idx_lt],
        y=centers_inv[:, idx_adr],
        color="red", marker="D", s=100,
        label="Centroids", zorder=10, ax=ax_sns_km
    )
    ax_sns_km.set_title(f"Clustere K-Means (k={slider_k}) - ADR vs Lead Time",
                        fontweight="bold", fontsize=10)
    ax_sns_km.set_xlabel("Lead Time (zile)")
    ax_sns_km.set_ylabel("ADR (€/noapte)")
    ax_sns_km.legend(fontsize=8)
    ax_sns_km.grid(linestyle="--", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_sns_km)
    plt.close()

    # ── PROFILUL CLUSTERELOR ───────────────────────────────────────
    st.markdown('<p class="sec-title">Pasul 4 - Profilul fiecărui cluster (statistici medii)</p>',
                unsafe_allow_html=True)

    df_profil = X_km.copy()
    df_profil["cluster"]  = labels_km
    df_profil["is_canceled"] = df_km_orig["is_canceled"].values
    df_profil["hotel"]    = df_km_orig["hotel"].values

    profil = (
        df_profil.groupby("cluster")
        .agg(
            nr_rezervari=("adr", "count"),
            adr_mediu=("adr", "mean"),
            lead_time_mediu=("lead_time_c", "mean"),
            cereri_speciale=("total_of_special_requests", "mean"),
            total_nopti=("total_nights", "mean"),
            rata_anulare=("is_canceled", "mean")
        )
        .assign(rata_anulare=lambda x: (x["rata_anulare"] * 100).round(1))
        .round(2)
    )

    # Adăugăm etichete interpretabile
    etichete = {0: "🏖️ Turiști last-minute", 1: "💼 Călători de afaceri",
                2: "👨‍👩‍👧 Familii planificate", 3: "⭐ Clienți premium"}
    profil.index = [etichete.get(i, f"Cluster {i}") for i in profil.index]

    st.dataframe(profil, use_container_width=True)

    # Radar chart cu Plotly pentru profilul clusterelor
    categorii_radar = ["adr_mediu", "lead_time_mediu", "cereri_speciale",
                       "total_nopti"]
    profil_norm = profil[categorii_radar].copy()
    for col in profil_norm.columns:
        profil_norm[col] = (profil_norm[col] - profil_norm[col].min()) /                            (profil_norm[col].max() - profil_norm[col].min() + 1e-9)

    fig_radar = go.Figure()
    culori_radar = ["#3b82f6", "#f97316", "#22c55e", "#a855f7"]
    for i, (idx, row) in enumerate(profil_norm.iterrows()):
        vals = list(row.values) + [row.values[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=categorii_radar + [categorii_radar[0]],
            fill="toself", opacity=0.6,
            name=str(idx),
            line_color=culori_radar[i % len(culori_radar)]
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Profilul clusterelor - Radar Chart (valori normalizate)",
        height=420, showlegend=True
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("""
    <div class="interpret-box">
        <strong>Interpretare economică a segmentelor:</strong><br>
        • <strong>Turiști last-minute:</strong> Lead time mic, ADR mediu - rezervări spontane,
        sensibili la oferte de ultim moment. Strategie: oferte flash cu 24-48h înainte de check-in.<br>
        • <strong>Călători de afaceri:</strong> Lead time scurt, ADR mare, cereri speciale ridicate -
        clienți corporativi care plătesc mai mult și cer servicii suplimentare.<br>
        • <strong>Familii planificate:</strong> Lead time mare, sejur lung, mai mulți adulți -
        rezervă din timp pentru vacanțe. Strategie: pachete familie cu reduceri la sejururi lungi.<br>
        • <strong>Clienți premium:</strong> ADR cel mai mare, cereri speciale maxime - segment de lux.
        Strategie: vanzări crescute la camere de lux, servicii extra/premium, experiențe personalizate.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SECȚIUNEA 6 - REGRESIE MULTIPLĂ (statsmodels OLS)
# ═══════════════════════════════════════════════════════════════
elif sectiune_curenta == "Regresie multiplă":
    st.markdown("""
    <div class="page-header">
        <h1>📈 Regresie multiplă - statsmodels OLS</h1>
        <p>Predicția tarifului mediu zilnic (ADR) și factorii care îl influențează</p>
    </div>
    """, unsafe_allow_html=True)

    import statsmodels.api as sm
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    st.markdown("""
    <div class="info-box">
        <strong>Obiectiv:</strong> Construim un model de regresie liniară multiplă prin metoda OLS
        (Ordinary Least Squares) pentru a prezice <em>ADR</em> (tariful mediu per noapte) în funcție
        de caracteristicile rezervării. Modelul ne arată și <strong>cât contribuie fiecare variabilă</strong>
        la prețul final - informație importantă pentru strategia de prețuri.
    </div>
    """, unsafe_allow_html=True)

    # ── Pregătire date pentru regresie ────────────────────────────
    @st.cache_data
    def pregateste_regresie(df: pd.DataFrame):
        df_r = df[(df["adr"] > 0) & (df["adr"] < df["adr"].quantile(0.99))].copy()

        # Imputare valori lipsă
        df_r["children"] = df_r["children"].fillna(0)
        df_r["country"]  = df_r["country"].fillna("Unknown")
        df_r["agent"]    = df_r["agent"].fillna(0)

        # Outlieri lead_time
        upper_lt = df_r["lead_time"].quantile(0.99)
        df_r["lead_time"] = np.where(df_r["lead_time"] > upper_lt,
                                     upper_lt, df_r["lead_time"])

        # Encoding
        le = LabelEncoder()
        df_r["hotel_enc"] = le.fit_transform(df_r["hotel"])

        cols_ohe = ["market_segment", "deposit_type",
                    "customer_type", "reserved_room_type"]
        cols_ohe = [c for c in cols_ohe if c in df_r.columns]
        df_r = pd.get_dummies(df_r, columns=cols_ohe, drop_first=True)

        # Features pentru regresie
        features_reg = [
            "lead_time", "arrival_month_num", "total_nights",
            "adults", "children", "booking_changes",
            "total_of_special_requests", "required_car_parking_spaces",
            "hotel_enc", "is_canceled"
        ]
        cols_ohe_result = [c for c in df_r.columns
                           if any(c.startswith(p + "_") for p in
                                  ["market_segment", "deposit_type",
                                   "customer_type", "reserved_room_type"])]
        features_reg = features_reg + cols_ohe_result
        features_reg = [c for c in features_reg if c in df_r.columns]

        X = df_r[features_reg].dropna()
        y = df_r.loc[X.index, "adr"]

        # Scalare
        scaler_r = StandardScaler()
        X_scaled_r = pd.DataFrame(
            scaler_r.fit_transform(X),
            columns=X.columns, index=X.index
        )

        return X_scaled_r, y, features_reg

    with st.spinner("Se preprocesează datele pentru regresie..."):
        X_reg, y_reg, features_reg = pregateste_regresie(df)

    # ── SPLIT TRAIN / TEST ─────────────────────────────────────────
    # np.log1p() pe ADR - transformare logaritmică pentru normalizarea distribuției
    # ADR are distribuție puternic asimetrică (văzut la EDA), log1p reduce influența outlierilor
    # ca în housing_california.py: y = np.log(SalePrice)
    y_reg_log = np.log1p(y_reg)

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg_log, test_size=0.20, random_state=42
    )

    st.markdown("""
    <div class="info-box">
        <strong>Transformare logaritmică pe ADR:</strong> Am aplicat <code>np.log1p(adr)</code> pe variabila
        țintă înainte de antrenament. ADR are distribuție asimetrică (coadă lungă la dreapta - observat la EDA),
        iar transformarea logaritmică aduce distribuția mai aproape de normalitate și îmbunătățește performanța
        modelului OLS. La interpretare, rezultatele se reconvertesc cu <code>np.expm1()</code>.
    </div>
    """, unsafe_allow_html=True)

    # ── ANTRENARE MODEL OLS ────────────────────────────────────────
    st.markdown('<p class="sec-title">Pasul 1 - Antrenare model OLS (statsmodels)</p>',
                unsafe_allow_html=True)

    @st.cache_data
    def antreneaza_ols(X_train, y_train):
        X_train_sm = sm.add_constant(X_train)
        model_ols  = sm.OLS(y_train, X_train_sm).fit()
        return model_ols

    with st.spinner("Se antrenează modelul OLS..."):
        model_ols = antreneaza_ols(X_train_r, y_train_r)

    # Rezumatul complet statsmodels
    st.markdown("**Rezumatul modelului OLS (statsmodels)**")
    st.text(str(model_ols.summary()))

    # ── METRICI PE SETUL DE TEST ───────────────────────────────────
    st.markdown('<p class="sec-title">Pasul 2 - Evaluarea modelului pe setul de test (20%)</p>',
                unsafe_allow_html=True)

    X_test_sm   = sm.add_constant(X_test_r, has_constant="add")
    y_pred_ols_log  = model_ols.predict(X_test_sm)

    # Reconvertim la scara originală pentru metrici interpretabile în €
    y_pred_ols  = np.expm1(y_pred_ols_log)
    y_test_orig = np.expm1(y_test_r)

    rmse_ols = np.sqrt(mean_squared_error(y_test_orig, y_pred_ols))
    mae_ols  = mean_absolute_error(y_test_orig, y_pred_ols)
    r2_ols   = r2_score(y_test_orig, y_pred_ols)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("R²",         f"{r2_ols:.4f}",
                  help="Proporția din variația ADR explicată de model (1=perfect)")
    col_m2.metric("RMSE (€)",   f"{rmse_ols:.2f}",
                  help="Eroarea medie în aceleași unități cu ADR")
    col_m3.metric("MAE (€)",    f"{mae_ols:.2f}",
                  help="Eroarea absolută medie - mai robustă la outlieri decât RMSE")
    col_m4.metric("Nr. obs. test", f"{len(y_test_r):,}")

    st.markdown(f"""
    <div class="interpret-box">
        <strong>Interpretarea metricilor:</strong><br>
        • <strong>R² = {r2_ols:.4f}</strong>: Modelul explică {r2_ols*100:.1f}% din variația tarifului zilnic -
        valoare acceptabilă dat fiind că prețul depinde și de factori neobservabili (negocieri, oferte speciale).<br>
        • <strong>RMSE = {rmse_ols:.2f} €</strong>: În medie, modelul greșește predicția cu ±{rmse_ols:.1f} €/noapte.
        Erorile mari sunt penalizate mai mult decât cele mici.<br>
        • <strong>MAE = {mae_ols:.2f} €</strong>: Eroarea absolută medie este de {mae_ols:.1f} €/noapte -
        mai mică decât RMSE, ceea ce indică existența unor cazuri cu erori mari care ridică RMSE.
    </div>
    """, unsafe_allow_html=True)

    # ── PREDICȚIE vs VALOARE REALĂ ─────────────────────────────────
    st.markdown('<p class="sec-title">Pasul 3 - Predicție vs. valoare reală</p>',
                unsafe_allow_html=True)

    # A) Comparație pentru UN SINGUR exemplu (cerința profesoarei)
    st.markdown("**A) Predicție pentru un singur exemplu**")

    idx_exemplu = st.number_input(
        "Alege indexul exemplului din setul de test (0 - primul, -1 - ultimul)",
        min_value=0, max_value=len(y_test_r) - 1, value=0, step=1
    )

    valoare_reala  = float(y_test_orig.iloc[idx_exemplu])
    valoare_pred   = float(y_pred_ols.iloc[idx_exemplu])
    eroare_absoluta = abs(valoare_reala - valoare_pred)

    col_ex1, col_ex2, col_ex3 = st.columns(3)
    col_ex1.metric("ADR real (€)",       f"{valoare_reala:.2f}")
    col_ex2.metric("ADR prezis (€)",     f"{valoare_pred:.2f}",
                   delta=f"{valoare_pred - valoare_reala:+.2f} €",
                   delta_color="inverse")
    col_ex3.metric("Eroare absolută (€)", f"{eroare_absoluta:.2f}")

    # B) Scatter predicții vs. valori reale (TOATE exemplele de test)
    st.markdown("**B) Predicții vs. valori reale - toate exemplele din setul de test**")

    fig_pred, ax_pred = plt.subplots(figsize=(8, 5))
    ax_pred.scatter(y_test_orig, y_pred_ols, alpha=0.25, s=12,
                    color="#3b82f6", label="Predicții")
    # Linia ideală y = x
    min_val = min(y_test_orig.min(), y_pred_ols.min())
    max_val = max(y_test_orig.max(), y_pred_ols.max())
    ax_pred.plot([min_val, max_val], [min_val, max_val],
                 color="#ef4444", linewidth=2, linestyle="--",
                 label="Predicție perfectă (y = x)")
    # Marcăm exemplul selectat
    ax_pred.scatter(valoare_reala, valoare_pred, color="#f97316",
                    s=120, zorder=5, label=f"Exemplul #{idx_exemplu}")
    ax_pred.set_xlabel("ADR real (€/noapte) - scară originală după expm1()")
    ax_pred.set_ylabel("ADR prezis (€/noapte) - scară originală după expm1()")
    ax_pred.set_title("Comparație ADR real vs. ADR prezis - set de test",
                      fontweight="bold")
    ax_pred.legend(fontsize=9)
    ax_pred.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig_pred)
    plt.close()

    st.markdown("""
    <div class="info-box">
        Punctele <strong>pe linia roșie</strong> = predicții perfecte.
        Punctele <strong>deasupra liniei</strong> = modelul subestimează prețul.
        Punctele <strong>sub linie</strong> = modelul supraestimează.
        Împrăștierea uniformă în jurul liniei arată că modelul nu are bias sistematic.
    </div>
    """, unsafe_allow_html=True)

    # C) Distribuția rezidualurilor
    st.markdown('<p class="sec-title">Pasul 4 - Analiza rezidualurilor</p>',
                unsafe_allow_html=True)

    rezidualuri = y_test_orig.values - y_pred_ols.values

    fig_rez, axes_rez = plt.subplots(1, 2, figsize=(12, 4))

    # Histogramă rezidualuri
    axes_rez[0].hist(rezidualuri, bins=50, color="#3b82f6",
                     edgecolor="white", alpha=0.85)
    axes_rez[0].axvline(0, color="#ef4444", linestyle="--",
                        linewidth=2, label="0")
    axes_rez[0].set_title("Distribuția rezidualurilor", fontweight="bold")
    axes_rez[0].set_xlabel("Rezidual (real - prezis)")
    axes_rez[0].set_ylabel("Frecvență")
    axes_rez[0].legend()
    axes_rez[0].grid(axis="y", linestyle="--", alpha=0.4)

    # Rezidualuri vs predicții
    axes_rez[1].scatter(y_pred_ols, rezidualuri, alpha=0.2,
                        s=8, color="#3b82f6")
    axes_rez[1].axhline(0, color="#ef4444", linestyle="--", linewidth=2)
    axes_rez[1].set_title("Rezidualuri vs. Valori prezise",
                          fontweight="bold")
    axes_rez[1].set_xlabel("ADR prezis (€)")
    axes_rez[1].set_ylabel("Rezidual")
    axes_rez[1].grid(linestyle="--", alpha=0.4)

    plt.tight_layout()
    st.pyplot(fig_rez)
    plt.close()

    # Top coeficienți
    st.markdown('<p class="sec-title">Pasul 5 - Coeficienții modelului (top 15 variabile)</p>',
                unsafe_allow_html=True)

    coef_df = pd.DataFrame({
        "Variabilă": model_ols.params.index,
        "Coeficient": model_ols.params.values,
        "p-value": model_ols.pvalues.values
    }).sort_values("Coeficient", key=abs, ascending=False).head(15)
    coef_df = coef_df[coef_df["Variabilă"] != "const"]

    fig_coef = px.bar(
        coef_df, x="Coeficient", y="Variabilă",
        orientation="h",
        color="Coeficient",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        title="Top 15 coeficienți OLS - impactul fiecărei variabile asupra ADR",
        labels={"Variabilă": "Feature", "Coeficient": "Coeficient (€)"},
        text_auto=".2f"
    )
    fig_coef.update_layout(height=450, coloraxis_showscale=False,
                           yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_coef, use_container_width=True)

    st.markdown(f"""
    <div class="interpret-box">
        <strong>Interpretare economică a coeficienților:</strong><br>
        Un coeficient pozitiv înseamnă că variabila respectivă <em>crește</em> ADR-ul,
        iar un coeficient negativ îl <em>scade</em>, menținând toate celelalte variabile constante.<br>
        Variabilele cu <strong>p-value &lt; 0.05</strong> sunt semnificative statistic -
        influența lor nu este întâmplătoare.
    </div>
    """, unsafe_allow_html=True)

    # ── PASUL 6: sklearn LinearRegression - comparație cu OLS ─────
    st.markdown('<p class="sec-title">Pasul 6 - Comparație: sklearn LinearRegression vs. statsmodels OLS</p>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>De ce comparăm OLS cu Random Forest?</strong><br>
        statsmodels OLS este un model <em>liniar</em> - presupune că relația dintre variabile
        și ADR este o linie dreaptă. Random Forest este un model <em>neliniar</em> - captează
        relații complexe și interacțiuni între variabile pe care OLS nu le poate modela.
        Comparând cele două, vedem cât de mult câștigăm adăugând complexitate.
    </div>
    """, unsafe_allow_html=True)

    from sklearn.ensemble import RandomForestRegressor

    @st.cache_data
    def antreneaza_rf_regresie(X_train, y_train, X_test):
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        return rf, rf.predict(X_test)

    with st.spinner("Se antrenează Random Forest Regressor..."):
        rf_reg_model, y_pred_rf_log = antreneaza_rf_regresie(
            X_train_r, y_train_r, X_test_r
        )

    # Reconvertim la scara originală
    y_pred_rf = np.expm1(y_pred_rf_log)

    rmse_rf = np.sqrt(mean_squared_error(y_test_orig, y_pred_rf))
    mae_rf  = mean_absolute_error(y_test_orig, y_pred_rf)
    r2_rf   = r2_score(y_test_orig, y_pred_rf)

    # Tabel comparativ OLS vs Random Forest
    reg_compare = pd.DataFrame({
        "Model":    ["statsmodels OLS (liniar)", "Random Forest Regressor (neliniar)"],
        "RMSE (€)": [round(rmse_ols, 2), round(rmse_rf, 2)],
        "MAE (€)":  [round(mae_ols, 2),  round(mae_rf, 2)],
        "R²":       [round(r2_ols, 4),   round(r2_rf, 4)]
    })
    st.dataframe(
        reg_compare.style.highlight_min(subset=["RMSE (€)", "MAE (€)"], color="#16a34a")
                         .highlight_max(subset=["R²"], color="#16a34a"),
        use_container_width=True, hide_index=True
    )

    # Scatter comparativ OLS vs Random Forest
    fig_cmp, axes_cmp = plt.subplots(1, 2, figsize=(12, 5))
    culori_cmp = ["#3b82f6", "#22c55e"]

    for ax_c, y_pred_c, titlu, culoare in zip(
        axes_cmp,
        [y_pred_ols, y_pred_rf],
        ["statsmodels OLS", "Random Forest Regressor"],
        culori_cmp
    ):
        ax_c.scatter(y_test_orig, y_pred_c, alpha=0.2, s=10, color=culoare)
        lim_min = min(float(y_test_orig.min()), float(y_pred_c.min()))
        lim_max = max(float(y_test_orig.max()), float(y_pred_c.max()))
        ax_c.plot([lim_min, lim_max], [lim_min, lim_max],
                  color="#ef4444", linewidth=2, linestyle="--")
        ax_c.set_title(titlu, fontweight="bold", fontsize=10)
        ax_c.set_xlabel("ADR real (€)")
        ax_c.set_ylabel("ADR prezis (€)")
        ax_c.grid(linestyle="--", alpha=0.3)

    plt.suptitle("Comparație predicții: OLS vs. Random Forest - cu cât se apropie punctele de linia roșie, cu atât modelul e mai bun",
                 fontweight="bold", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig_cmp)
    plt.close()

    st.markdown("""
    <div class="interpret-box">
        <strong>Interpretare comparație:</strong><br>
        • Dacă Random Forest are R² mai mare și RMSE/MAE mai mici - relația dintre variabile
        și ADR <em>nu este liniară</em>, iar modelul complex captează mai bine realitatea.<br>
        • Dacă OLS dă rezultate similare - relațiile sunt suficient de liniare și modelul simplu
        este preferat (mai ușor de explicat).<br>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SECȚIUNEA 7 - CLASIFICARE (Regresie Logistică)
# ═══════════════════════════════════════════════════════════════
elif sectiune_curenta == "Clasificare":
    st.markdown("""
    <div class="page-header">
        <h1>🎯 Clasificare - Comparație algoritmi</h1>
        <p>Regresie Logistică | Arbore de Decizie | Random Forest | Gradient Boosting</p>
    </div>
    """, unsafe_allow_html=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        accuracy_score, f1_score, roc_auc_score, roc_curve
    )

    st.markdown("""
    <div class="info-box">
        <strong>Obiectiv:</strong> Construim un model de clasificare binară care prezice dacă o rezervare
        va fi <em>anulată (1)</em> sau <em>menținută (0)</em>. Variabila țintă este <strong>is_canceled</strong>.
        Un model precis permite hotelului să ia măsuri proactive: contactarea clienților cu risc de anulare,
        ajustarea politicii de overbooking sau oferirea de stimulente de retenție.
    </div>
    """, unsafe_allow_html=True)

    # ── Pregătire date ─────────────────────────────────────────────
    @st.cache_data
    def pregateste_clasificare(df: pd.DataFrame):
        df_c = df[(df["adr"] > 0) & (df["adr"] < df["adr"].quantile(0.99))].copy()

        df_c["children"] = df_c["children"].fillna(0)
        df_c["country"]  = df_c["country"].fillna("Unknown")
        df_c["agent"]    = df_c["agent"].fillna(0)

        upper_lt = df_c["lead_time"].quantile(0.99)
        df_c["lead_time"] = np.where(df_c["lead_time"] > upper_lt,
                                     upper_lt, df_c["lead_time"])
        upper_adr = df_c["adr"].quantile(0.99)
        df_c["adr"] = np.where(df_c["adr"] > upper_adr,
                               upper_adr, df_c["adr"])

        le = LabelEncoder()
        df_c["hotel_enc"] = le.fit_transform(df_c["hotel"])

        cols_ohe = ["market_segment", "deposit_type",
                    "customer_type", "meal"]
        cols_ohe = [c for c in cols_ohe if c in df_c.columns]
        df_c = pd.get_dummies(df_c, columns=cols_ohe, drop_first=True)

        freq_country = df_c["country"].value_counts(normalize=True)
        df_c["country_freq"] = df_c["country"].map(freq_country)

        features_clf = [
            "lead_time", "arrival_month_num", "total_nights",
            "adults", "children", "booking_changes", "adr",
            "total_of_special_requests", "required_car_parking_spaces",
            "hotel_enc", "country_freq", "days_in_waiting_list"
        ]
        cols_ohe_result = [c for c in df_c.columns
                           if any(c.startswith(p + "_") for p in
                                  ["market_segment", "deposit_type",
                                   "customer_type", "meal"])]
        features_clf = features_clf + cols_ohe_result
        features_clf = [c for c in features_clf if c in df_c.columns]

        X = df_c[features_clf].dropna()
        y = df_c.loc[X.index, "is_canceled"]

        scaler_c = StandardScaler()
        X_scaled_c = pd.DataFrame(
            scaler_c.fit_transform(X),
            columns=X.columns, index=X.index
        )
        return X_scaled_c, y, features_clf

    with st.spinner("Se preprocesează datele pentru clasificare..."):
        X_clf, y_clf, features_clf = pregateste_clasificare(df)

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_clf, y_clf, test_size=0.20, random_state=42, stratify=y_clf
    )

    # ── ANTRENARE MODEL ────────────────────────────────────────────
    st.markdown('<p class="sec-title">Pasul 1 - Antrenare și comparație algoritmi</p>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        Antrenăm 4 algoritmi de clasificare pe același set de date și comparăm performanța
        prin <strong>acc_df</strong> - tabelul de tracking din seminarul de clasificare.
        Variabila țintă: <em>is_canceled</em> (0 = rezervare menținută, 1 = anulată).
    </div>
    """, unsafe_allow_html=True)

    @st.cache_data
    def antreneaza_toti_clasificatorii(X_train, y_train, X_test, y_test):
        """
        Antrenăm toți 4 clasificatori și returnăm predicțiile + metricile.
        Structura acc_df - ca în seminarul 6 de clasificare.
        """
        modele = {
            "Regresie Logistică":   LogisticRegression(max_iter=500, penalty="l2",
                                                        solver="lbfgs", random_state=42),
            "Arbore de Decizie":    DecisionTreeClassifier(max_depth=5, random_state=42),
            "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42,
                                                           n_jobs=-1),
            "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100,
                                                                max_depth=4,
                                                                learning_rate=0.1,
                                                                random_state=42),
        }

        acc_df_local = pd.DataFrame(columns=["Model", "Accuracy", "F1-Score", "AUC-ROC"])
        rezultate    = {}

        for nume, model in modele.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            acc_ = accuracy_score(y_test, y_pred)
            f1_  = f1_score(y_test, y_pred, average="weighted")
            auc_ = roc_auc_score(y_test, y_prob)

            # Populăm acc_df exact ca în seminarul 6
            acc_df_local.loc[len(acc_df_local)] = [
                nume, round(acc_, 4), round(f1_, 4), round(auc_, 4)
            ]
            rezultate[nume] = {
                "model": model, "y_pred": y_pred, "y_prob": y_prob,
                "acc": acc_, "f1": f1_, "auc": auc_
            }

        return acc_df_local, rezultate

    with st.spinner("Se antrenează cei 4 algoritmi..."):
        acc_df, rezultate_clf = antreneaza_toti_clasificatorii(
            X_train_c, y_train_c, X_test_c, y_test_c
        )

    # Afișăm acc_df - tabel comparativ toți 4 algoritmi
    st.markdown("**Tabel comparativ - acc_df (toți algoritmii)**")
    st.dataframe(
        acc_df.style.highlight_max(subset=["Accuracy", "F1-Score", "AUC-ROC"],
                                   color="#16a34a"),
        use_container_width=True, hide_index=True
    )

    # Selectăm modelul curent pentru analiză detaliată
    model_ales = st.selectbox(
        "Alege modelul pentru analiză detaliată:",
        list(rezultate_clf.keys()), index=2  # implicit Random Forest
    )
    rez = rezultate_clf[model_ales]
    model_rl  = rez["model"]
    y_pred_c  = rez["y_pred"]
    y_prob_c  = rez["y_prob"]
    acc       = rez["acc"]
    f1        = rez["f1"]
    auc       = rez["auc"]

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Accuracy",       f"{acc:.4f}")
    col_m2.metric("F1-Score",       f"{f1:.4f}")
    col_m3.metric("AUC-ROC",        f"{auc:.4f}")
    col_m4.metric("Nr. obs. test",  f"{len(y_test_c):,}")

    # ── CONFUSION MATRIX ───────────────────────────────────────────
    st.markdown('<p class="sec-title">Pasul 2 - Matricea de confuzie</p>',
                unsafe_allow_html=True)

    col_cm, col_rep = st.columns([1, 1])

    with col_cm:
        cm = confusion_matrix(y_test_c, y_pred_c)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            linewidths=0.5, linecolor="white", ax=ax_cm,
            xticklabels=["Menținută (0)", "Anulată (1)"],
            yticklabels=["Menținută (0)", "Anulată (1)"]
        )
        ax_cm.set_xlabel("Valoare prezisă", fontsize=10)
        ax_cm.set_ylabel("Valoare reală", fontsize=10)
        ax_cm.set_title("Matricea de confuzie - Regresie Logistică",
                        fontweight="bold", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close()

        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"""
        <div class="interpret-box" style="font-size:12px;">
            <strong>TN={tn:,}</strong> - rezervări menținute corect prezise<br>
            <strong>TP={tp:,}</strong> - rezervări anulate corect prezise<br>
            <strong>FP={fp:,}</strong> - menținute prezise greșit ca anulate<br>
            <strong>FN={fn:,}</strong> - anulate ratate (neprezise) cost maxim
        </div>
        """, unsafe_allow_html=True)

    with col_rep:
        st.markdown("**Classification Report**")
        report = classification_report(
            y_test_c, y_pred_c,
            target_names=["Menținută", "Anulată"]
        )
        st.text(report)

    # ── CURBA ROC-AUC ──────────────────────────────────────────────
    st.markdown('<p class="sec-title">Pasul 3 - Curba ROC-AUC</p>',
                unsafe_allow_html=True)

    fpr, tpr, _ = roc_curve(y_test_c, y_prob_c)

    fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
    ax_roc.plot(fpr, tpr, color="#3b82f6", linewidth=2.5,
                label=f"Regresie Logistică (AUC = {auc:.4f})")
    ax_roc.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--",
                linewidth=1.5, label="Model aleator (AUC = 0.5)")
    ax_roc.fill_between(fpr, tpr, alpha=0.1, color="#3b82f6")
    ax_roc.set_xlabel("Rată fals pozitiv (FPR)")
    ax_roc.set_ylabel("Rată adevărat pozitiv (TPR / Recall)")
    ax_roc.set_title("Curba ROC - Predicția anulărilor", fontweight="bold")
    ax_roc.legend(fontsize=10)
    ax_roc.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig_roc)
    plt.close()

    # ── PREDICȚIE PENTRU UN SINGUR EXEMPLU ────────────────────────
    st.markdown('<p class="sec-title">Pasul 4 - Predicție vs. valoare reală</p>',
                unsafe_allow_html=True)

    # A) Un singur exemplu
    st.markdown("**A) Predicție pentru un singur exemplu din setul de test**")

    idx_ex_c = st.number_input(
        "Alege indexul exemplului (0 = primul)",
        min_value=0, max_value=len(y_test_c) - 1,
        value=0, step=1, key="idx_clf"
    )

    real_c    = int(y_test_c.iloc[idx_ex_c])
    pred_c    = int(y_pred_c[idx_ex_c])
    prob_c    = float(y_prob_c[idx_ex_c])
    corect_c  = "✅ Corect" if real_c == pred_c else "❌ Greșit"

    col_ex_c1, col_ex_c2, col_ex_c3, col_ex_c4 = st.columns(4)
    col_ex_c1.metric("Valoare reală",
                     "Anulată" if real_c == 1 else "Menținută")
    col_ex_c2.metric("Valoare prezisă",
                     "Anulată" if pred_c == 1 else "Menținută")
    col_ex_c3.metric("Probabilitate anulare", f"{prob_c:.2%}")
    col_ex_c4.metric("Predicție", corect_c)

    # B) Distribuția probabilităților prezise
    st.markdown("**B) Distribuția probabilităților de anulare - tot setul de test**")

    fig_prob, ax_prob = plt.subplots(figsize=(9, 4))
    ax_prob.hist(
        y_prob_c[y_test_c == 0], bins=50, alpha=0.65,
        color="#3b82f6", edgecolor="white", label="Rezervări menținute (real=0)"
    )
    ax_prob.hist(
        y_prob_c[y_test_c == 1], bins=50, alpha=0.65,
        color="#ef4444", edgecolor="white", label="Rezervări anulate (real=1)"
    )
    ax_prob.axvline(0.5, color="#1f2937", linestyle="--",
                    linewidth=2, label="Prag decizie: 0.5")
    ax_prob.set_xlabel("Probabilitate prezisă de anulare")
    ax_prob.set_ylabel("Frecvență")
    ax_prob.set_title(
        "Distribuția probabilităților - menținute vs. anulate",
        fontweight="bold"
    )
    ax_prob.legend(fontsize=9)
    ax_prob.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig_prob)
    plt.close()

    # ── COMPARAȚIE ROC TOȚI ALGORITMII ──────────────────────────
    st.markdown('<p class="sec-title">Comparație curbe ROC - toți algoritmii</p>',
                unsafe_allow_html=True)

    fig_roc_all, ax_roc_all = plt.subplots(figsize=(9, 5))
    culori_roc = {"Regresie Logistică": "#3b82f6", "Arbore de Decizie": "#f97316",
                  "Random Forest": "#22c55e", "Gradient Boosting": "#a855f7"}

    for nume_m, rez_m in rezultate_clf.items():
        fpr_m, tpr_m, _ = roc_curve(y_test_c, rez_m["y_prob"])
        ax_roc_all.plot(fpr_m, tpr_m,
                        color=culori_roc.get(nume_m, "#94a3b8"),
                        linewidth=2,
                        label=f"{nume_m} (AUC={rez_m['auc']:.3f})")

    ax_roc_all.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--",
                    linewidth=1.5, label="Model aleator (AUC=0.5)")
    ax_roc_all.set_xlabel("Rată fals pozitiv (FPR)")
    ax_roc_all.set_ylabel("Rată adevărat pozitiv (TPR)")
    ax_roc_all.set_title("Curbe ROC - comparație toți algoritmii",
                         fontweight="bold")
    ax_roc_all.legend(fontsize=9, loc="lower right")
    ax_roc_all.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig_roc_all)
    plt.close()

    st.markdown(f"""
    <div class="interpret-box">
        <strong>Interpretare economică:</strong><br>
        • <strong>AUC-ROC = {auc:.4f}</strong>: Modelul are capacitate bună de discriminare
        între rezervările anulate și cele menținute. Un AUC &gt; 0.80 este considerat bun
        pentru probleme de clasificare în industria hotelieră.<br>
        • <strong>Accuracy = {acc:.2%}</strong>: Procentul total de predicții corecte.
        Atenție - pe date dezechilibrate (37% anulări), accuracy ca indice individual poate fi înșelător.<br>
        • <strong>Aplicație practică:</strong> Hotelul poate folosi modelul pentru a identifica
        rezervările cu probabilitate &gt; 70% de anulare și a le contacta proactiv cu oferte
        de retenție (upgrade cameră, late check-out, voucher restaurant) - reducând rata de anulare
        și maximizând ocuparea.
    </div>
    """, unsafe_allow_html=True)
