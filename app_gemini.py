"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          AGRI-SCM INTELLIGENCE COMMAND CENTER  v6.0                        ║
║          Production-Grade Agricultural Supply Chain Management Platform     ║
║          ML-Powered | Real-Time Analytics | Full ESG Tracking               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, silhouette_score
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import google.generativeai as genai
import requests
import json
import time

# ══════════════════════════════════════════════════════════════════════════════
# 1. PAGE CONFIG & THEME
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Agri-SCM Command Center v8",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

THEME = {
    'bg_primary':   '#0d1b0f',
    'bg_card':      '#13241a',
    'bg_panel':     '#1a2e20',
    'accent_green': '#4caf50',
    'accent_lime':  '#a5d63d',
    'accent_amber': '#ffb300',
    'accent_red':   '#ef5350',
    'accent_blue':  '#42a5f5',
    'text_primary': '#e8f5e9',
    'text_muted':   '#81c784',
    'border':       '#2e4d35',
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'IBM Plex Mono', monospace;
    background-color: {THEME['bg_primary']};
    color: {THEME['text_primary']};
}}

.main {{
    background: {THEME['bg_primary']};
    padding: 0 !important;
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: {THEME['bg_card']} !important;
    border-right: 1px solid {THEME['border']};
}}
section[data-testid="stSidebar"] * {{
    color: {THEME['text_primary']} !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {THEME['bg_card']};
    border-radius: 12px;
    padding: 6px;
    gap: 4px;
    border: 1px solid {THEME['border']};
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {THEME['text_muted']} !important;
    border-radius: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    padding: 8px 14px;
}}
.stTabs [aria-selected="true"] {{
    background: {THEME['accent_green']} !important;
    color: #000 !important;
    font-weight: 600;
}}

/* ── Metric cards ── */
[data-testid="stMetric"] {{
    background: {THEME['bg_card']};
    border: 1px solid {THEME['border']};
    border-radius: 12px;
    padding: 16px !important;
}}
[data-testid="stMetricValue"] {{
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem !important;
    color: {THEME['accent_lime']} !important;
}}
[data-testid="stMetricLabel"] {{
    color: {THEME['text_muted']} !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
[data-testid="stMetricDelta"] {{
    font-size: 12px !important;
}}

/* ── Selectbox / sliders ── */
.stSelectbox > div > div,
.stSlider > div {{
    background: {THEME['bg_panel']} !important;
    border-color: {THEME['border']} !important;
    color: {THEME['text_primary']} !important;
}}

/* ── Alert boxes ── */
.alert-critical {{
    padding: 14px 18px; border-radius: 10px; margin: 8px 0;
    border-left: 5px solid {THEME['accent_red']};
    background: rgba(239,83,80,0.12);
    color: {THEME['text_primary']};
    font-size: 13px;
}}
.alert-warning {{
    padding: 14px 18px; border-radius: 10px; margin: 8px 0;
    border-left: 5px solid {THEME['accent_amber']};
    background: rgba(255,179,0,0.10);
    color: {THEME['text_primary']};
    font-size: 13px;
}}
.alert-success {{
    padding: 14px 18px; border-radius: 10px; margin: 8px 0;
    border-left: 5px solid {THEME['accent_green']};
    background: rgba(76,175,80,0.12);
    color: {THEME['text_primary']};
    font-size: 13px;
}}
.alert-info {{
    padding: 14px 18px; border-radius: 10px; margin: 8px 0;
    border-left: 5px solid {THEME['accent_blue']};
    background: rgba(66,165,245,0.10);
    color: {THEME['text_primary']};
    font-size: 13px;
}}

/* ── Section headers ── */
.section-header {{
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: {THEME['accent_lime']};
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border-bottom: 1px solid {THEME['border']};
    padding-bottom: 6px;
    margin: 18px 0 12px 0;
}}

/* ── Stat chip ── */
.chip {{
    display: inline-block;
    background: {THEME['bg_panel']};
    border: 1px solid {THEME['border']};
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11px;
    color: {THEME['accent_lime']};
    margin: 2px;
}}

/* ── Hero banner ── */
.hero {{
    background: linear-gradient(135deg, {THEME['bg_card']} 0%, {THEME['bg_panel']} 100%);
    border: 1px solid {THEME['border']};
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 20px;
}}
.hero h1 {{
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: {THEME['accent_lime']};
    margin: 0;
    letter-spacing: -0.02em;
}}
.hero p {{
    color: {THEME['text_muted']};
    font-size: 13px;
    margin: 6px 0 0 0;
}}

/* ── Dataframe ── */
.stDataFrame {{
    background: {THEME['bg_card']} !important;
    border: 1px solid {THEME['border']};
    border-radius: 10px;
}}

/* ── Weather card ── */
.wx-card {{
    background: {THEME['bg_panel']};
    border: 1px solid {THEME['border']};
    border-radius: 10px;
    padding: 10px 8px;
    text-align: center;
    font-size: 11px;
    color: {THEME['text_primary']};
    line-height: 1.7;
}}
.wx-card b {{ color: {THEME['accent_lime']}; font-family: 'Syne', sans-serif; }}

/* ── Recommendation badge ── */
.badge-green {{ color: #69f0ae; font-weight: 600; }}
.badge-amber {{ color: #ffb300; font-weight: 600; }}
.badge-red   {{ color: #ef5350; font-weight: 600; }}

/* ── Divider ── */
hr {{ border-color: {THEME['border']} !important; }}

/* ── Plotly chart backgrounds ── */
.js-plotly-plot {{ border-radius: 12px; }}

/* ── Number input ── */
.stNumberInput input {{ background: {THEME['bg_panel']} !important; color: {THEME['text_primary']} !important; border-color: {THEME['border']} !important; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {THEME['bg_primary']}; }}
::-webkit-scrollbar-thumb {{ background: {THEME['border']}; border-radius: 3px; }}
</style>
""", unsafe_allow_html=True)

# Plot layout defaults
PLOT_BG    = THEME['bg_card']
PLOT_PAPER = THEME['bg_card']
PLOT_FONT  = dict(family="IBM Plex Mono", color=THEME['text_primary'], size=11)
PLOT_GRID  = dict(color=THEME['border'])

def dark_layout(fig, title="", height=320):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Syne", size=14, color=THEME['accent_lime'])),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_PAPER,
        font=PLOT_FONT, height=height,
        xaxis=dict(gridcolor=THEME['border'], zerolinecolor=THEME['border']),
        yaxis=dict(gridcolor=THEME['border'], zerolinecolor=THEME['border']),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA ENGINE — 10 COMMODITIES, 12 MANDIS, 3 YEARS
# ══════════════════════════════════════════════════════════════════════════════
class AgriDataEngine:

    COMMODITIES = ['Onion', 'Tomato', 'Potato', 'Wheat', 'Rice',
                   'Garlic', 'Carrot', 'Cabbage', 'Chilli', 'Maize']

    BASE_PRICES = {
        'Onion': 2200, 'Tomato': 1700, 'Potato': 1400,
        'Wheat': 2400, 'Rice': 3200,  'Garlic': 8000,
        'Carrot': 1600, 'Cabbage': 900, 'Chilli': 12000, 'Maize': 1800
    }

    PERISHABILITY = {
        'Tomato': 'High', 'Cabbage': 'High', 'Carrot': 'Medium',
        'Onion': 'Medium', 'Potato': 'Low', 'Garlic': 'Low',
        'Chilli': 'Low', 'Wheat': 'None', 'Rice': 'None', 'Maize': 'None'
    }

    SEASONAL_CURVES = {
        'Onion':  [1.3,1.4,1.2,0.9,0.7,0.8,1.0,1.1,1.0,0.9,1.0,1.2],
        'Tomato': [0.8,0.7,0.9,1.3,1.5,1.4,1.1,0.9,0.8,0.9,1.0,0.9],
        'Potato': [1.1,1.2,1.0,0.8,0.7,0.9,1.1,1.2,1.1,0.9,0.8,1.0],
        'Wheat':  [1.0,1.0,1.1,0.8,0.7,0.8,1.0,1.1,1.1,1.0,1.0,1.0],
        'Rice':   [1.0,1.0,1.0,1.0,1.1,1.2,1.1,0.9,0.8,0.9,1.0,1.0],
        'Garlic': [1.2,1.3,1.4,1.1,0.8,0.7,0.9,1.0,1.0,1.1,1.2,1.2],
        'Carrot': [1.0,1.1,1.2,1.0,0.7,0.6,0.8,1.0,1.1,1.1,1.0,1.0],
        'Cabbage':[1.1,1.0,0.9,0.8,0.7,0.8,1.0,1.1,1.2,1.1,1.0,1.0],
        'Chilli': [1.0,1.0,1.1,1.2,1.3,1.1,0.9,0.8,0.9,1.0,1.1,1.1],
        'Maize':  [1.0,1.0,1.0,1.1,1.2,1.1,0.9,0.8,0.9,1.0,1.0,1.0],
    }

    WEATHER_SENSITIVITY = {
        'Tomato': 3.0, 'Onion': 2.0, 'Carrot': 1.8, 'Cabbage': 1.8,
        'Potato': 1.2, 'Garlic': 1.0, 'Chilli': 1.0,
        'Wheat': 0.5, 'Rice': 0.5, 'Maize': 0.4
    }

    MANDI_PROFILES = {
        'Azadpur':    {'rel': (7,9),  'cap': 1.4, 'premium': 1.04, 'lat': 28.72, 'lon': 77.17, 'state': 'Delhi'},
        'Vashi':      {'rel': (7,10), 'cap': 1.5, 'premium': 1.06, 'lat': 19.07, 'lon': 73.00, 'state': 'Maharashtra'},
        'Bengaluru':  {'rel': (6,9),  'cap': 1.2, 'premium': 1.03, 'lat': 12.97, 'lon': 77.59, 'state': 'Karnataka'},
        'Pune':       {'rel': (5,8),  'cap': 1.0, 'premium': 1.00, 'lat': 18.52, 'lon': 73.85, 'state': 'Maharashtra'},
        'Nashik':     {'rel': (5,8),  'cap': 1.0, 'premium': 0.98, 'lat': 19.99, 'lon': 73.79, 'state': 'Maharashtra'},
        'Jaipur':     {'rel': (4,7),  'cap': 0.8, 'premium': 0.96, 'lat': 26.91, 'lon': 75.79, 'state': 'Rajasthan'},
        'Nagpur':     {'rel': (4,7),  'cap': 0.9, 'premium': 0.97, 'lat': 21.15, 'lon': 79.09, 'state': 'Maharashtra'},
        'Lucknow':    {'rel': (5,8),  'cap': 1.0, 'premium': 1.01, 'lat': 26.85, 'lon': 80.95, 'state': 'UP'},
        'Ahmedabad':  {'rel': (6,9),  'cap': 1.1, 'premium': 1.02, 'lat': 23.03, 'lon': 72.58, 'state': 'Gujarat'},
        'Hyderabad':  {'rel': (6,9),  'cap': 1.1, 'premium': 1.02, 'lat': 17.38, 'lon': 78.49, 'state': 'Telangana'},
        'Chennai':    {'rel': (6,8),  'cap': 1.0, 'premium': 1.01, 'lat': 13.08, 'lon': 80.27, 'state': 'Tamil Nadu'},
        'Kolkata':    {'rel': (5,8),  'cap': 1.0, 'premium': 1.00, 'lat': 22.57, 'lon': 88.36, 'state': 'West Bengal'},
    }

    @staticmethod
    @st.cache_data(show_spinner="⚙️ Generating market data...")
    def generate(_seed=42):
        np.random.seed(_seed)
        dates = pd.date_range(start="2023-01-01", end="2025-12-31", freq='D')
        rows  = []

        for date in dates:
            doy        = date.dayofyear
            mon        = date.month - 1
            year_drift = (date.year - 2023) * 0.055
            monsoon    = 5 <= date.month <= 9

            diesel  = 88 + np.sin(doy/365*np.pi)*4 + (date.year-2023)*2.2
            temp    = 22 + np.sin(doy/365*2*np.pi)*13 + np.random.normal(0, 1.5)
            hum     = 55 + np.cos(doy/365*2*np.pi)*26 + np.random.normal(0, 4)
            rain    = np.random.exponential(3.0 if monsoon else 0.3)
            wind    = np.random.exponential(8) + 3
            frost   = 1 if (date.month in [12,1,2] and np.random.random() < 0.05) else 0

            for loc, prof in AgriDataEngine.MANDI_PROFILES.items():
                rel = np.random.randint(*prof['rel'])

                for item in AgriDataEngine.COMMODITIES:
                    base_p   = AgriDataEngine.BASE_PRICES[item]
                    seasonal = AgriDataEngine.SEASONAL_CURVES[item][mon]
                    shock    = np.random.choice([1.0,1.3,0.78], p=[0.91,0.05,0.04])
                    wsens    = AgriDataEngine.WEATHER_SENSITIVITY[item]
                    w_fx     = 1 + (max(rain-5,0)*0.01*wsens)
                    frost_fx = (1.15 if frost and item in ['Tomato','Carrot','Cabbage'] else 1.0)

                    price   = max(100, (base_p * seasonal * shock * w_fx * frost_fx
                                        * prof['premium'] * (1+year_drift)
                                        + np.random.normal(0, base_p*0.018)))
                    arrival = max(20, 800 * prof['cap'] * (1/seasonal)
                                  + np.random.normal(0, 90))
                    traded  = arrival * np.random.uniform(0.7, 0.98)

                    rows.append([date, item, loc, round(price,2), round(arrival,1),
                                 round(traded,1), round(rain,2), round(temp,1),
                                 round(hum,1), round(diesel,2), rel,
                                 round(wind,1), frost,
                                 prof['lat'], prof['lon'], prof['state']])

        cols = ['Date','Commodity','Location','Price','Arrival','Traded',
                'Rainfall','Temp','Humidity','Diesel','Reliability',
                'Wind','Frost','Lat','Lon','State']
        return pd.DataFrame(rows, columns=cols)


# ══════════════════════════════════════════════════════════════════════════════
# 3. ML ENGINE — MULTIPLE MODELS
# ══════════════════════════════════════════════════════════════════════════════
class MLEngine:
    def __init__(self, df):
        self.df      = df.sort_values(['Commodity','Location','Date']).reset_index(drop=True)
        self.gb      = GradientBoostingRegressor(n_estimators=150, learning_rate=0.07,
                                                  max_depth=5, subsample=0.82, random_state=42)
        self.ridge   = Ridge(alpha=1.0)
        self.iso     = IsolationForest(contamination=0.03, random_state=42)
        self.kmeans  = KMeans(n_clusters=4, random_state=42, n_init=10)
        self.scaler       = StandardScaler()
        self.ridge_scaler = StandardScaler()
        self.le_loc  = LabelEncoder()
        self.le_com  = LabelEncoder()
        self.mae     = None
        self.feat_cols = []

    def _engineer(self, df):
        g = df.groupby(['Commodity','Location'])

        # Lag features
        for lag in [1, 3, 7, 14]:
            df[f'Price_L{lag}'] = g['Price'].shift(lag)

        # Rolling stats
        for w in [7, 14, 30]:
            df[f'Roll{w}_Mean'] = g['Price'].transform(lambda x: x.shift(1).rolling(w).mean())
            df[f'Roll{w}_Std']  = g['Price'].transform(lambda x: x.shift(1).rolling(w).std())

        df['Roll7_Min'] = g['Price'].transform(lambda x: x.shift(1).rolling(7).min())
        df['Roll7_Max'] = g['Price'].transform(lambda x: x.shift(1).rolling(7).max())
        df['Roll7_Range'] = df['Roll7_Max'] - df['Roll7_Min']

        # Arrival rolling
        df['Arr_Roll3'] = g['Arrival'].transform(lambda x: x.shift(1).rolling(3).mean())
        df['Arr_Roll7'] = g['Arrival'].transform(lambda x: x.shift(1).rolling(7).mean())

        # Seasonal
        df['Sin_DOY'] = np.sin(2*np.pi*df['Date'].dt.dayofyear/365)
        df['Cos_DOY'] = np.cos(2*np.pi*df['Date'].dt.dayofyear/365)
        df['Sin_M']   = np.sin(2*np.pi*df['Date'].dt.month/12)
        df['Cos_M']   = np.cos(2*np.pi*df['Date'].dt.month/12)
        df['Month']   = df['Date'].dt.month
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Quarter']   = df['Date'].dt.quarter

        # Interaction features
        df['Temp_Humidity'] = df['Temp'] * df['Humidity'] / 100
        df['Rain_Wind']     = df['Rainfall'] * df['Wind']
        df['Diesel_Dist']   = df['Diesel'] / 88  # normalised diesel

        # Encodings
        df['Loc_Enc'] = self.le_loc.fit_transform(df['Location'])
        df['Com_Enc'] = self.le_com.fit_transform(df['Commodity'])

        # Target
        df['Target'] = g['Price'].shift(-1)
        return df

    def train(self):
        df = self._engineer(self.df.copy())
        self.feat_cols = [
            'Price_L1','Price_L3','Price_L7','Price_L14',
            'Roll7_Mean','Roll7_Std','Roll14_Mean','Roll14_Std','Roll30_Mean',
            'Roll7_Min','Roll7_Max','Roll7_Range',
            'Arr_Roll3','Arr_Roll7',
            'Temp','Humidity','Diesel','Rainfall','Wind','Frost',
            'Temp_Humidity','Rain_Wind','Diesel_Dist',
            'Sin_DOY','Cos_DOY','Sin_M','Cos_M',
            'Month','DayOfWeek','Quarter',
            'Loc_Enc','Com_Enc'
        ]
        clean = df.dropna(subset=self.feat_cols+['Target'])
        mask  = clean['Date'].dt.year < 2025
        Xtr, ytr = clean.loc[mask,  self.feat_cols], clean.loc[mask,  'Target']
        Xvl, yvl = clean.loc[~mask, self.feat_cols], clean.loc[~mask, 'Target']

        self.gb.fit(Xtr, ytr)
        self.ridge.fit(self.ridge_scaler.fit_transform(Xtr), ytr)

        preds    = self.gb.predict(Xvl)
        self.mae = mean_absolute_error(yvl, preds)

        # Anomaly detection on full clean set
        self.iso.fit(clean[self.feat_cols])

        # Mandi clustering
        mandi_feats = df.groupby('Location').agg(
            Avg_Price=('Price','mean'), Avg_Arrival=('Arrival','mean'),
            Reliability=('Reliability','mean'), Avg_Temp=('Temp','mean')
        ).reset_index()
        self.mandi_cluster_df = mandi_feats.copy()
        X_cl = self.scaler.fit_transform(mandi_feats[['Avg_Price','Avg_Arrival','Reliability','Avg_Temp']])
        self.mandi_cluster_df['Cluster'] = self.kmeans.fit_predict(X_cl)

        return df

    def predict_price(self, row):
        return float(self.gb.predict(row[self.feat_cols].values.reshape(1,-1))[0])

    def detect_anomaly(self, row):
        score = self.iso.decision_function(row[self.feat_cols].values.reshape(1,-1))[0]
        flag  = self.iso.predict(row[self.feat_cols].values.reshape(1,-1))[0]
        return flag == -1, score

    def feature_importance(self):
        return pd.Series(self.gb.feature_importances_, index=self.feat_cols).sort_values(ascending=False)

    def price_forecast_series(self, last_row, steps=30):
        """Iterative multi-step forecast."""
        row   = last_row.copy()
        preds = []
        for _ in range(steps):
            p = self.predict_price(row)
            preds.append(p)
            row['Price_L14'] = row['Price_L7']
            row['Price_L7']  = row['Price_L3']
            row['Price_L3']  = row['Price_L1']
            row['Price_L1']  = p
        return preds


# ══════════════════════════════════════════════════════════════════════════════
# 4. LOGISTICS ENGINE
# ══════════════════════════════════════════════════════════════════════════════
LOGISTICS_MODES = {
    "Kisan Rail (Train)":  {"eff":40, "cost_km":8,  "co2":0.8, "cap_qt":5000, "speed_kmh":60},
    "Heavy Truck (16T)":   {"eff":4,  "cost_km":45, "co2":2.6, "cap_qt":160,  "speed_kmh":55},
    "LCV (3T)":            {"eff":12, "cost_km":20, "co2":1.5, "cap_qt":30,   "speed_kmh":50},
    "Reefer (Cold Chain)": {"eff":3,  "cost_km":65, "co2":3.2, "cap_qt":80,   "speed_kmh":50},
    "Drone Delivery":      {"eff":0,  "cost_km":120,"co2":0.2, "cap_qt":1,    "speed_kmh":80},
    "Electric Van (2T)":   {"eff":0,  "cost_km":18, "co2":0.3, "cap_qt":20,   "speed_kmh":45},
}

PERISH_FACTOR = {'High':1.8,'Medium':0.9,'Low':0.4,'None':0.08}
BASE_YIELD_Q  = {'Onion':120,'Tomato':200,'Potato':180,'Wheat':35,'Rice':45,
                  'Garlic':50,'Carrot':140,'Cabbage':220,'Chilli':30,'Maize':60}

def logistics_check(mode_name, commodity, qty_qt, dist_km):
    mode     = LOGISTICS_MODES[mode_name]
    warnings = []
    score    = 10
    perish   = AgriDataEngine.PERISHABILITY[commodity]

    if perish == 'High' and mode_name in ["Kisan Rail (Train)"]:
        warnings.append("⚠️ Train unsuitable for highly perishable cargo — use Reefer.")
        score -= 3
    if perish == 'High' and mode_name == "Heavy Truck (16T)" and dist_km > 600:
        warnings.append("⚠️ Long haul with perishables — consider Reefer instead.")
        score -= 2
    if qty_qt > mode['cap_qt']:
        n = int(np.ceil(qty_qt / mode['cap_qt']))
        warnings.append(f"ℹ️ {n} vehicles needed for this load size.")
        score -= 1
    if dist_km > 900 and mode_name == "LCV (3T)":
        warnings.append("⚠️ LCV inefficient beyond 900 km.")
        score -= 2
    if mode_name == "Drone Delivery" and qty_qt > 1:
        warnings.append("⚠️ Drone capacity is 1 quintal max — only for samples/express.")
        score -= 4

    transit_h    = dist_km / max(mode['speed_kmh'], 1)
    total_cost   = dist_km * mode['cost_km']
    co2_kg       = (dist_km / mode['eff']) * mode['co2'] if mode['eff'] > 0 else dist_km * 0.01
    cost_per_qt  = total_cost / max(qty_qt, 1)

    return score, total_cost, co2_kg, cost_per_qt, transit_h, warnings


# ══════════════════════════════════════════════════════════════════════════════
# 5. ALERT ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def generate_alerts(price_delta, rain, temp, frost, anomaly, commodity, mode, qty, dist):
    alerts = []
    perish = AgriDataEngine.PERISHABILITY[commodity]

    if anomaly:
        alerts.append(("critical","🚨 ANOMALY DETECTED: Price pattern is statistically unusual. Verify data or check for market manipulation."))
    if price_delta > 15:
        alerts.append(("critical",f"🔴 Price Surge +{price_delta:.1f}%: Accelerate procurement, consider alternate mandis or forward contracts."))
    elif price_delta > 8:
        alerts.append(("warning", f"🟠 Price Rise +{price_delta:.1f}%: Monitor closely. Review inventory before next purchase."))
    elif price_delta < -12:
        alerts.append(("success", f"🟢 Price Drop {price_delta:.1f}%: Ideal forward-buying window. Lock in bulk order now."))

    if rain > 35 and perish in ('High','Medium'):
        alerts.append(("critical",f"🌧️ Flood/Rain Risk: {rain:.1f}mm. {commodity} supply disruption likely within 3–5 days."))
    elif rain > 15 and perish == 'High':
        alerts.append(("warning", f"🌦️ Moderate Rain: {rain:.1f}mm. Expedite perishable dispatch."))

    if temp > 38 and perish == 'High':
        alerts.append(("critical",f"🌡️ Extreme Heat {temp:.1f}°C: Switch to Reefer immediately for {commodity}."))
    elif temp > 34 and perish == 'Medium':
        alerts.append(("warning", f"☀️ High Temp {temp:.1f}°C: Consider refrigerated storage."))

    if frost:
        alerts.append(("warning","❄️ Frost Event Detected: Crop damage risk for tomatoes, carrots, cabbage."))

    _, _, _, _, transit_h, mode_warns = logistics_check(mode, commodity, qty, dist)
    for w in mode_warns:
        alerts.append(("warning", w))

    if transit_h > 24 and perish == 'High':
        alerts.append(("critical",f"⏱️ Transit time {transit_h:.0f}h exceeds safe window for {commodity}. Choose faster route."))

    return alerts


# ══════════════════════════════════════════════════════════════════════════════
# 6. INITIALISE
# ══════════════════════════════════════════════════════════════════════════════
raw_df = AgriDataEngine.generate()
ml     = MLEngine(raw_df)
df_full = ml.train()

today_date = raw_df['Date'].max()

# ══════════════════════════════════════════════════════════════════════════════
# 7. SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0 20px 0;">
        <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;color:#a5d63d;">
            🌾 AGRI-SCM
        </div>
        <div style="font-size:10px;color:#81c784;letter-spacing:0.12em;">COMMAND CENTER v6</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**📦 Commodity & Market**")
    sel_item = st.selectbox("Commodity", sorted(AgriDataEngine.COMMODITIES))
    sel_loc  = st.selectbox("Primary Market", sorted(AgriDataEngine.MANDI_PROFILES.keys()))

    st.markdown("**🚛 Logistics**")
    logistics_mode = st.selectbox("Mode", list(LOGISTICS_MODES.keys()))
    dist_sim = st.slider("Route Distance (km)", 10, 2000, 300)
    f_qty    = st.number_input("Order Quantity (Quintals)", 1, 10000, 100)

    st.divider()

    mode_data = LOGISTICS_MODES[logistics_mode]
    _, t_cost, co2_val, cpq, transit_h, _ = logistics_check(logistics_mode, sel_item, f_qty, dist_sim)

    st.markdown("**🌱 ESG Snapshot**")
    c1, c2 = st.columns(2)
    c1.metric("CO₂ (kg)", f"{co2_val:.1f}")
    c2.metric("₹/Quintal", f"₹{cpq:.0f}")
    c1.metric("Transit", f"{transit_h:.1f}h")
    c2.metric("Total ₹", f"₹{t_cost:,.0f}")

    perish_label = AgriDataEngine.PERISHABILITY[sel_item]
    colour_map = {"High":"🔴","Medium":"🟡","Low":"🟢","None":"⚪"}
    st.info(f"Perishability: {colour_map[perish_label]} {perish_label}")

    st.divider()
    if ml.mae:
        st.markdown(f"<div class='chip'>Model MAE: ₹{ml.mae:.0f}/Q</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chip'>Data: {len(raw_df):,} records</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chip'>Last updated: {today_date.strftime('%d %b %Y')}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 8. CORE COMPUTATIONS
# ══════════════════════════════════════════════════════════════════════════════
mandi_df   = df_full[(df_full['Commodity']==sel_item) & (df_full['Location']==sel_loc)].dropna(subset=ml.feat_cols)
latest_row = mandi_df.sort_values('Date').iloc[-1]
pred_price = ml.predict_price(latest_row)
price_delta = ((pred_price - latest_row['Price']) / latest_row['Price']) * 100
anomaly_flag, anomaly_score = ml.detect_anomaly(latest_row)

# 30-day forecast
forecast_prices = ml.price_forecast_series(latest_row, steps=30)
forecast_dates  = [today_date + timedelta(days=i+1) for i in range(30)]

# ══════════════════════════════════════════════════════════════════════════════
# 9. HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
    <h1>🌾 AGRI-SCM COMMAND CENTER</h1>
    <p>ML-Powered Agricultural Supply Chain Intelligence &nbsp;|&nbsp;
       {sel_item} @ {sel_loc} &nbsp;|&nbsp;
       {today_date.strftime('%d %b %Y')} &nbsp;|&nbsp;
       Model MAE: ₹{ml.mae:.0f}/Q
    </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 10. ALERTS BAR
# ══════════════════════════════════════════════════════════════════════════════
alerts = generate_alerts(price_delta, latest_row['Rainfall'], latest_row['Temp'],
                          latest_row['Frost'], anomaly_flag, sel_item,
                          logistics_mode, f_qty, dist_sim)
if alerts:
    for a_type, a_msg in alerts:
        st.markdown(f'<div class="alert-{a_type}">{a_msg}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 11. TOP KPI ROW
# ══════════════════════════════════════════════════════════════════════════════
k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Market Price (₹/Q)",  f"₹{latest_row['Price']:,.0f}")
k2.metric("Predicted Tomorrow",  f"₹{pred_price:,.0f}", f"{price_delta:+.1f}%")
k3.metric("Arrivals (Q)",        f"{latest_row['Arrival']:,.0f}")
k4.metric("Reliability Score",   f"{latest_row['Reliability']}/10")
k5.metric("Anomaly Flag",        "⚠️ YES" if anomaly_flag else "✅ Normal",
          delta_color="inverse" if anomaly_flag else "normal")
k6.metric("Weather",             f"{latest_row['Temp']:.1f}°C / {latest_row['Rainfall']:.1f}mm")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# 12. TABS
# ══════════════════════════════════════════════════════════════════════════════
(t_exec, t_proc, t_farm, t_risk,
 t_insight, t_agro, t_report, t_scopt, t_agent) = st.tabs([
    "📊 Executive", "🏗️ Procurement", "🌾 Farmer Advisory",
    "📉 Risk & Spoilage", "🔍 Market Insights",
    "🌦️ Agro Intelligence", "📋 Sales Report", "🔗 SC Optimizer",
    "🤖 AI Agent"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with t_exec:
    st.markdown('<div class="section-header">Executive Overview</div>', unsafe_allow_html=True)

    # Price trend — all commodities last 60 days
    cut60 = raw_df[raw_df['Date'] >= today_date - timedelta(days=60)]
    trend_all = cut60.groupby(['Date','Commodity'])['Price'].mean().reset_index()
    fig_trend = px.line(trend_all, x='Date', y='Price', color='Commodity',
                        color_discrete_sequence=px.colors.qualitative.Safe)
    dark_layout(fig_trend, "All-Commodity Price Trends — Last 60 Days", 380)
    st.plotly_chart(fig_trend, use_container_width=True)

    col_l, col_r = st.columns(2)

    with col_l:
        # Revenue proxy waterfall
        cut15   = raw_df[raw_df['Date'] >= today_date - timedelta(days=15)]
        rev_df  = cut15.groupby('Commodity').apply(
            lambda x: (x['Price']*x['Traded']).sum()
        ).reset_index(name='Revenue')
        rev_df  = rev_df.sort_values('Revenue', ascending=False)
        fig_rev = go.Figure(go.Bar(
            x=rev_df['Commodity'], y=rev_df['Revenue'],
            marker=dict(color=rev_df['Revenue'],
                        colorscale='YlGn',
                        showscale=False),
            text=[f"₹{v/1e6:.1f}M" for v in rev_df['Revenue']],
            textposition='outside'
        ))
        dark_layout(fig_rev, "Revenue Proxy by Commodity — Last 15 Days", 300)
        st.plotly_chart(fig_rev, use_container_width=True)

    with col_r:
        # Mandi cluster scatter
        mc_df = ml.mandi_cluster_df.copy()
        fig_cl = px.scatter(mc_df, x='Avg_Price', y='Avg_Arrival',
                            color='Cluster', size='Reliability',
                            hover_name='Location',
                            color_continuous_scale='Viridis',
                            title="Mandi Clusters (KMeans)")
        dark_layout(fig_cl, "Mandi Segmentation — KMeans Clustering", 300)
        st.plotly_chart(fig_cl, use_container_width=True)

    # 30-day price forecast chart for selected commodity
    st.markdown('<div class="section-header">30-Day Price Forecast</div>', unsafe_allow_html=True)
    hist_df  = mandi_df.tail(60)
    hist_col = THEME['accent_lime']
    fore_col = THEME['accent_amber']

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['Price'],
                                mode='lines', name='Historical',
                                line=dict(color=hist_col, width=2)))
    fig_fc.add_trace(go.Scatter(x=forecast_dates, y=forecast_prices,
                                mode='lines', name='Forecast',
                                line=dict(color=fore_col, width=2, dash='dot')))
    # Confidence band (±10%)
    upper = [p*1.10 for p in forecast_prices]
    lower = [p*0.90 for p in forecast_prices]
    fig_fc.add_trace(go.Scatter(
        x=forecast_dates+forecast_dates[::-1],
        y=upper+lower[::-1],
        fill='toself', fillcolor='rgba(255,179,0,0.08)',
        line=dict(color='rgba(0,0,0,0)'), name='±10% Band'
    ))
    dark_layout(fig_fc, f"30-Day Price Forecast — {sel_item} @ {sel_loc}", 360)
    st.plotly_chart(fig_fc, use_container_width=True)

    # Heatmap: commodity × location (last 30d avg)
    cut30  = raw_df[raw_df['Date'] >= today_date - timedelta(days=30)]
    heat_p = cut30.groupby(['Commodity','Location'])['Price'].mean().unstack()
    fig_hm = px.imshow(heat_p, color_continuous_scale='RdYlGn_r', aspect='auto')
    dark_layout(fig_hm, "Price Heatmap — Commodity × Mandi (Last 30 Days)", 360)
    st.plotly_chart(fig_hm, use_container_width=True)


    # ── Real-Time Momentum Scoreboard ──
    st.markdown('<div class="section-header">📡 Real-Time Momentum Scoreboard</div>', unsafe_allow_html=True)
    mom_rows = []
    for crop in AgriDataEngine.COMMODITIES:
        c_df = raw_df[raw_df['Commodity']==crop].sort_values('Date')
        if len(c_df) < 30: continue
        p_now  = c_df['Price'].iloc[-1]
        p_7d   = c_df['Price'].iloc[-7]
        p_30d  = c_df['Price'].iloc[-30]
        v_now  = c_df['Arrival'].iloc[-7:].mean()
        v_30d  = c_df['Arrival'].iloc[-30:].mean()
        mom7   = (p_now-p_7d)/p_7d*100
        mom30  = (p_now-p_30d)/p_30d*100
        vol_ch = (v_now-v_30d)/v_30d*100
        trend  = "📈 Bullish" if mom7>3 else ("📉 Bearish" if mom7<-3 else "➡️ Sideways")
        mom_rows.append({"Crop":crop,"Price":round(p_now,0),"7d Chg%":round(mom7,1),
                          "30d Chg%":round(mom30,1),"Vol Chg%":round(vol_ch,1),"Trend":trend})
    mom_df = pd.DataFrame(mom_rows)
    st.dataframe(mom_df.style
                 .format({"Price":"₹{:.0f}","7d Chg%":"{:+.1f}%","30d Chg%":"{:+.1f}%","Vol Chg%":"{:+.1f}%"})
                 .background_gradient(subset=["7d Chg%"], cmap="RdYlGn"),
                 use_container_width=True)

    # ── Mandi Geo Map ──
    st.markdown('<div class="section-header">🗺️ Mandi Geo Performance Map</div>', unsafe_allow_html=True)
    geo_df = raw_df[raw_df["Date"] >= today_date-timedelta(days=30)].groupby("Location").agg(
        Avg_Price=("Price","mean"), Avg_Rel=("Reliability","mean"),
        Total_Arr=("Arrival","sum"), Lat=("Lat","first"), Lon=("Lon","first")
    ).reset_index()
    fig_geo = px.scatter_geo(geo_df, lat="Lat", lon="Lon",
                              text="Location", size="Total_Arr",
                              color="Avg_Price", color_continuous_scale="RdYlGn_r",
                              scope="asia", center=dict(lat=22, lon=78))
    fig_geo.update_geos(bgcolor=THEME["bg_card"], framecolor=THEME["border"],
                        landcolor="#1a2e20", oceancolor=THEME["bg_primary"],
                        showocean=True, showland=True)
    dark_layout(fig_geo, "Mandi Locations — Avg Price & Volume (Last 30 Days)", 420)
    st.plotly_chart(fig_geo, use_container_width=True)

    # ── Mandi Sell-Through Efficiency ──
    st.markdown('<div class="section-header">📦 Mandi Sell-Through Efficiency</div>', unsafe_allow_html=True)
    eff_df = raw_df[raw_df["Date"] >= today_date-timedelta(days=30)].groupby("Location").agg(
        Total_Arr=("Arrival","sum"), Total_Trd=("Traded","sum")
    ).reset_index()
    eff_df["Sell_Rate"] = (eff_df["Total_Trd"]/eff_df["Total_Arr"]*100).round(1)
    fig_eff = go.Figure()
    fig_eff.add_trace(go.Bar(name="Arrived", x=eff_df["Location"], y=eff_df["Total_Arr"],
                              marker_color=THEME["accent_blue"], opacity=0.7))
    fig_eff.add_trace(go.Bar(name="Traded",  x=eff_df["Location"], y=eff_df["Total_Trd"],
                              marker_color=THEME["accent_lime"]))
    fig_eff.update_layout(barmode="overlay")
    dark_layout(fig_eff, "Mandi Arrivals vs Traded Volume — Last 30 Days", 320)
    st.plotly_chart(fig_eff, use_container_width=True)

    # ── Supply Chain Health Index ──
    st.markdown('<div class="section-header">💚 Supply Chain Health Index</div>', unsafe_allow_html=True)
    avg_rel      = raw_df["Reliability"].mean()
    avg_sell_rt  = (raw_df["Traded"].sum() / raw_df["Arrival"].sum() * 100)
    price_vol    = raw_df.groupby("Commodity")["Price"].std().mean() / raw_df["Price"].mean() * 100
    alert_count  = len(alerts)
    health_score = (avg_rel/10*30 + avg_sell_rt/100*30 + max(0,(20-price_vol)/20)*25 + max(0,(5-alert_count)/5)*15)

    h1,h2,h3,h4,h5 = st.columns(5)
    h1.metric("Avg Reliability",  f"{avg_rel:.1f}/10")
    h2.metric("Sell-Through Rate",f"{avg_sell_rt:.1f}%")
    h3.metric("Price Volatility", f"{price_vol:.1f}%")
    h4.metric("Active Alerts",    f"{alert_count}")
    h5.metric("SC Health Score",  f"{health_score:.0f}/100",
              delta="Excellent" if health_score>=75 else ("Good" if health_score>=55 else "At Risk"))

    fig_health = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=health_score,
        delta={"reference":70, "valueformat":".0f"},
        title={"text":"Supply Chain Health","font":{"family":"Syne","color":THEME["accent_lime"],"size":13}},
        gauge={
            "axis":{"range":[0,100],"tickcolor":THEME["text_muted"]},
            "bar":{"color":THEME["accent_lime"]},
            "bgcolor":THEME["bg_panel"],
            "bordercolor":THEME["border"],
            "steps":[
                {"range":[0,40],"color":"rgba(239,83,80,0.2)"},
                {"range":[40,70],"color":"rgba(255,179,0,0.2)"},
                {"range":[70,100],"color":"rgba(165,214,61,0.2)"},
            ],
            "threshold":{"line":{"color":THEME["accent_lime"],"width":3},"value":70}
        },
        number={"font":{"family":"Syne","color":THEME["text_primary"],"size":36},"suffix":"/100"}
    ))
    fig_health.update_layout(plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_PAPER,
                              height=280, margin=dict(l=30,r=30,t=40,b=10))
    st.plotly_chart(fig_health, use_container_width=True)

    # ── Price Distribution Boxplot — All Commodities ──
    st.markdown('<div class="section-header">📊 Price Distribution — All Commodities</div>', unsafe_allow_html=True)
    cut90 = raw_df[raw_df["Date"] >= today_date-timedelta(days=90)]
    fig_box = px.box(cut90, x="Commodity", y="Price", color="Commodity",
                      color_discrete_sequence=px.colors.qualitative.Pastel,
                      notched=True)
    dark_layout(fig_box, "Price Distribution Boxplot — Last 90 Days (All Commodities)", 380)
    st.plotly_chart(fig_box, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PROCUREMENT
# ══════════════════════════════════════════════════════════════════════════════
with t_proc:
    st.markdown('<div class="section-header">Sourcing Intelligence</div>', unsafe_allow_html=True)

    p1,p2,p3,p4,p5 = st.columns(5)
    p1.metric("Current Price",     f"₹{latest_row['Price']:,.0f}")
    p2.metric("Next-Day Forecast", f"₹{pred_price:,.0f}", f"{price_delta:+.1f}%")
    p3.metric("Arrivals",          f"{latest_row['Arrival']:,.0f} Q")
    p4.metric("Reliability",       f"{latest_row['Reliability']}/10")
    p5.metric("Anomaly",           "⚠️ YES" if anomaly_flag else "✅ Normal")

    st.markdown('<div class="section-header">Cross-Mandi Arbitrage</div>', unsafe_allow_html=True)

    arb_df = df_full[(df_full['Date']==today_date) & (df_full['Commodity']==sel_item)].copy()
    _, lc, _, cpq_arb, th_arb, _ = logistics_check(logistics_mode, sel_item, f_qty, dist_sim)
    arb_df['Landed_Cost']  = arb_df['Price'] + cpq_arb
    arb_df['Value_Score']  = (arb_df['Reliability'] / arb_df['Landed_Cost'] * 1000).round(2)
    arb_df['Transit_h']    = dist_sim / LOGISTICS_MODES[logistics_mode]['speed_kmh']

    # Bubble: landed cost vs reliability, size=arrivals
    fig_arb = px.scatter(arb_df, x='Landed_Cost', y='Reliability',
                         size='Arrival', color='Value_Score',
                         color_continuous_scale='RdYlGn', hover_name='Location',
                         hover_data=['Price','Arrival','Landed_Cost','Value_Score'])
    dark_layout(fig_arb, "Mandi Selection: Landed Cost vs Reliability", 380)
    st.plotly_chart(fig_arb, use_container_width=True)

    # Best mandi recommendation
    best = arb_df.sort_values('Value_Score', ascending=False).iloc[0]
    st.markdown(f'<div class="alert-success">✅ Best Mandi: <b>{best["Location"]}</b> — Value Score {best["Value_Score"]:.2f} | Landed Cost ₹{best["Landed_Cost"]:.0f}/Q | Reliability {best["Reliability"]}/10</div>', unsafe_allow_html=True)

    arb_show = arb_df[['Location','Price','Reliability','Landed_Cost','Value_Score','Transit_h']].sort_values('Value_Score', ascending=False)
    st.dataframe(arb_show.style
                 .format({'Price':'₹{:.0f}','Landed_Cost':'₹{:.0f}','Value_Score':'{:.2f}','Transit_h':'{:.1f}h'})
                 .background_gradient(subset=['Value_Score'], cmap='RdYlGn'),
                 use_container_width=True)

    # Candlestick price chart
    st.markdown('<div class="section-header">Price OHLC — Weekly Candles</div>', unsafe_allow_html=True)
    ohlc = mandi_df.set_index('Date').resample('W')['Price'].agg(
        Open='first', High='max', Low='min', Close='last').reset_index()
    fig_cd = go.Figure(go.Candlestick(
        x=ohlc['Date'], open=ohlc['Open'], high=ohlc['High'],
        low=ohlc['Low'], close=ohlc['Close'],
        increasing_line_color=THEME['accent_lime'],
        decreasing_line_color=THEME['accent_red']
    ))
    dark_layout(fig_cd, f"Weekly OHLC — {sel_item} @ {sel_loc}", 360)
    fig_cd.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_cd, use_container_width=True)

    # Logistics mode comparison table
    st.markdown('<div class="section-header">Logistics Mode Comparison</div>', unsafe_allow_html=True)
    mode_comp = []
    for mn, mv in LOGISTICS_MODES.items():
        sc, tc, co2, cpq_m, th_m, warns = logistics_check(mn, sel_item, f_qty, dist_sim)
        mode_comp.append({'Mode': mn, 'Total Cost ₹': tc, '₹/Q': cpq_m,
                          'CO₂ kg': round(co2,1), 'Transit h': round(th_m,1),
                          'Suitability': sc, 'Issues': len(warns)})
    mc_df2 = pd.DataFrame(mode_comp).sort_values('Suitability', ascending=False)
    st.dataframe(mc_df2.style
                 .format({'Total Cost ₹':'₹{:,.0f}','₹/Q':'₹{:.0f}','CO₂ kg':'{:.1f}','Transit h':'{:.1f}','Suitability':'{}/10'})
                 .background_gradient(subset=['Suitability'], cmap='RdYlGn'),
                 use_container_width=True)


    # ── Forward Contract Simulator ──
    st.markdown('<div class="section-header">📜 Forward Contract Simulator</div>', unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        contract_qty   = st.number_input("Contract Quantity (Q)", 10, 10000, 500, key="fcs_q")
        contract_days  = st.slider("Contract Duration (Days)", 7, 180, 30, key="fcs_d")
    with fc2:
        contract_price = st.number_input("Agreed Price (₹/Q)", 100, 20000,
                                          int(latest_row["Price"]*1.02), key="fcs_p")
        premium_pct    = st.slider("Risk Premium (%)", 0, 15, 3, key="fcs_prem")
    with fc3:
        penalty_pct    = st.slider("Non-Delivery Penalty (%)", 0, 20, 5, key="fcs_pen")
        counterparty   = st.selectbox("Counterparty Type", ["Processor","Retailer","Exporter","Govt. Agency"])

    # Forecast price at contract expiry
    monthly_drift_fc  = (pred_price - latest_row["Price"]) / latest_row["Price"]
    forecast_at_expiry = latest_row["Price"] * ((1+monthly_drift_fc)**(contract_days/30))
    contract_value     = contract_price * contract_qty
    market_value       = forecast_at_expiry * contract_qty
    premium_cost       = contract_value * premium_pct/100
    if forecast_at_expiry > contract_price:
        buyer_gain  = (forecast_at_expiry - contract_price) * contract_qty
        seller_loss = -buyer_gain
    else:
        seller_gain = (contract_price - forecast_at_expiry) * contract_qty
        buyer_loss  = -seller_gain
        buyer_gain  = -seller_gain if forecast_at_expiry < contract_price else 0

    fg1,fg2,fg3,fg4 = st.columns(4)
    fg1.metric("Contract Value",       f"₹{contract_value:,.0f}")
    fg2.metric("Expected Mkt at Exp.", f"₹{forecast_at_expiry:,.0f}/Q")
    fg3.metric("Buyer P&L",            f"₹{buyer_gain:,.0f}", delta_color="normal")
    fg4.metric("Risk Premium Cost",    f"₹{premium_cost:,.0f}")

    # Price path vs contract level
    cp_dates = [today_date + timedelta(days=i) for i in range(contract_days+1)]
    cp_prices = [latest_row["Price"] * ((1+monthly_drift_fc)**(d/30)) for d in range(contract_days+1)]
    fig_cp = go.Figure()
    fig_cp.add_trace(go.Scatter(x=cp_dates, y=cp_prices, name="Forecast Price Path",
                                 line=dict(color=THEME["accent_lime"],width=2)))
    fig_cp.add_hline(y=contract_price, line_dash="dash", line_color=THEME["accent_amber"],
                      annotation_text=f"Contract ₹{contract_price}")
    fig_cp.add_hline(y=latest_row["Price"], line_dash="dot", line_color=THEME["text_muted"],
                      annotation_text="Spot Today")
    fig_cp.update_layout(xaxis_title="Date", yaxis_title="Price (₹/Q)")
    dark_layout(fig_cp, f"Forward Contract — Price Path vs Agreed Price", 320)
    st.plotly_chart(fig_cp, use_container_width=True)

    # ── Multi-Mandi Basket Procurement ──
    st.markdown('<div class="section-header">🧺 Multi-Mandi Basket Procurement</div>', unsafe_allow_html=True)
    st.caption("Diversify procurement across mandis to hedge against local supply shocks.")
    basket_mandis = st.multiselect("Select Mandis for Basket",
                                    sorted(AgriDataEngine.MANDI_PROFILES.keys()),
                                    default=list(sorted(AgriDataEngine.MANDI_PROFILES.keys()))[:4],
                                    key="basket_m")
    basket_qty    = st.slider("Total Quantity to Source (Q)", 50, 5000, 500, key="basket_q")

    if basket_mandis:
        basket_data = arb_df[arb_df["Location"].isin(basket_mandis)].copy()
        basket_data["Allocation_Q"] = basket_qty / len(basket_mandis)
        basket_data["Alloc_Cost"]   = basket_data["Landed_Cost"] * basket_data["Allocation_Q"]
        total_basket_cost = basket_data["Alloc_Cost"].sum()
        avg_basket_price  = basket_data["Landed_Cost"].mean()
        best_single_cost  = arb_df["Landed_Cost"].min() * basket_qty
        diversification_premium = total_basket_cost - best_single_cost

        bk1,bk2,bk3 = st.columns(3)
        bk1.metric("Total Basket Cost",       f"₹{total_basket_cost:,.0f}")
        bk2.metric("Avg Landed Price",         f"₹{avg_basket_price:,.0f}/Q")
        bk3.metric("Diversification Premium", f"₹{diversification_premium:,.0f}",
                   delta="vs single-source" if diversification_premium!=0 else None)

        fig_bk = px.pie(basket_data, names="Location", values="Allocation_Q",
                         color_discrete_sequence=px.colors.qualitative.Pastel,
                         title="Basket Allocation by Mandi")
        dark_layout(fig_bk, "Procurement Basket Allocation", 300)
        st.plotly_chart(fig_bk, use_container_width=True)

        st.dataframe(basket_data[["Location","Price","Landed_Cost","Allocation_Q","Alloc_Cost","Reliability"]]
                     .style.format({"Price":"₹{:.0f}","Landed_Cost":"₹{:.0f}",
                                    "Allocation_Q":"{:.0f}","Alloc_Cost":"₹{:,.0f}","Reliability":"{:.0f}"}),
                     use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FARMER ADVISORY
# ══════════════════════════════════════════════════════════════════════════════
with t_farm:
    st.markdown('<div class="section-header">Farmer Profit Optimizer</div>', unsafe_allow_html=True)

    _, transport_cost, _, cpq_farm, _, _ = logistics_check(logistics_mode, sel_item, f_qty, dist_sim)
    net_today = (latest_row['Price'] * f_qty) - transport_cost

    fa1, fa2, fa3 = st.columns(3)
    fa1.metric("Gross Revenue Today",  f"₹{latest_row['Price']*f_qty:,.0f}")
    fa2.metric("Transport Cost",       f"₹{transport_cost:,.0f}")
    fa3.metric("Net Profit (Today)",   f"₹{net_today:,.0f}")

    col_sell, col_hold = st.columns(2)
    with col_sell:
        st.markdown('<div class="section-header">📍 Immediate Sale Analysis</div>', unsafe_allow_html=True)
        # P&L waterfall
        fig_wf = go.Figure(go.Waterfall(
            name="P&L", orientation="v",
            measure=["absolute","relative","relative","total"],
            x=["Gross Revenue","Transport","Wastage Est.","Net Profit"],
            y=[latest_row['Price']*f_qty,
               -transport_cost,
               -(latest_row['Price']*f_qty*0.02),
               0],
            connector={"line":{"color":THEME['border']}},
            increasing={"marker":{"color":THEME['accent_lime']}},
            decreasing={"marker":{"color":THEME['accent_red']}},
            totals={"marker":{"color":THEME['accent_blue']}}
        ))
        dark_layout(fig_wf, "P&L Waterfall — Sell Today", 300)
        st.plotly_chart(fig_wf, use_container_width=True)

    with col_hold:
        st.markdown('<div class="section-header">❄️ Cold Storage ROI Optimizer</div>', unsafe_allow_html=True)
        storage_days = st.slider("Storage Duration (Days)", 7, 120, 30, key='sd_farm')
        rent_per_q   = 65

        monthly_drift   = (pred_price - latest_row['Price']) / latest_row['Price']
        future_price    = latest_row['Price'] * ((1+monthly_drift)**(storage_days/30))
        storage_cost    = (storage_days/30) * rent_per_q * f_qty
        perish_loss_pct = PERISH_FACTOR[AgriDataEngine.PERISHABILITY[sel_item]] * storage_days * 0.5 / 100
        perish_loss_pct = min(perish_loss_pct, 0.95)
        saleable_qty    = f_qty * (1 - perish_loss_pct)
        net_future      = (future_price * saleable_qty) - storage_cost - transport_cost
        roi             = net_future - net_today

        fh1,fh2 = st.columns(2)
        fh1.metric("Future Price Est.", f"₹{future_price:,.0f}", f"{((future_price/latest_row['Price'])-1)*100:+.1f}%")
        fh2.metric("Storage Cost",      f"₹{storage_cost:,.0f}")
        fh1.metric("Spoilage Loss",     f"{perish_loss_pct*100:.1f}%")
        fh2.metric("Net ROI vs Now",    f"₹{roi:,.0f}", delta_color="normal")

        if roi > 0:
            st.markdown(f'<div class="alert-success">✅ HOLD: Storing {storage_days}d yields ₹{roi:,.0f} extra.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-critical">❌ SELL NOW: Net loss of ₹{abs(roi):,.0f} from holding.</div>', unsafe_allow_html=True)

    # ROI sweep chart
    st.markdown('<div class="section-header">Storage ROI Sweep</div>', unsafe_allow_html=True)
    dur_range = range(7, 121, 7)
    roi_sweep = []
    for d in dur_range:
        fp  = latest_row['Price'] * ((1+monthly_drift)**(d/30))
        sc  = (d/30)*rent_per_q*f_qty
        pl  = min(PERISH_FACTOR[AgriDataEngine.PERISHABILITY[sel_item]]*d*0.5/100, 0.95)
        sq  = f_qty*(1-pl)
        roi_sweep.append((fp*sq - sc - transport_cost) - net_today)

    fig_roi = go.Figure(go.Bar(
        x=list(dur_range), y=roi_sweep,
        marker_color=[THEME['accent_lime'] if v>0 else THEME['accent_red'] for v in roi_sweep]
    ))
    fig_roi.add_hline(y=0, line_dash='dash', line_color=THEME['text_muted'])
    dark_layout(fig_roi, "Net ROI vs Sell Today (₹) — by Storage Duration", 280)
    fig_roi.update_layout(xaxis_title="Storage Days", yaxis_title="ROI (₹)")
    st.plotly_chart(fig_roi, use_container_width=True)

    # Market price distribution violin
    st.markdown('<div class="section-header">Price Distribution — All Mandis</div>', unsafe_allow_html=True)
    dist_df = raw_df[(raw_df['Commodity']==sel_item) & (raw_df['Date'] >= today_date - timedelta(days=90))]
    fig_vio = px.violin(dist_df, x='Location', y='Price', color='Location',
                        box=True, points=False,
                        color_discrete_sequence=px.colors.qualitative.Pastel)
    dark_layout(fig_vio, f"{sel_item} Price Distribution by Mandi — Last 90 Days", 360)
    st.plotly_chart(fig_vio, use_container_width=True)


    # ── Commodity Comparison Calculator ──
    st.markdown('<div class="section-header">⚖️ Crop Profitability Comparison</div>', unsafe_allow_html=True)
    comp_qty = f_qty
    comp_dist = dist_sim
    profit_compare = []
    for crop in AgriDataEngine.COMMODITIES:
        c_latest = raw_df[raw_df["Commodity"]==crop].sort_values("Date").iloc[-1]
        _, tc_c, _, _, _, _ = logistics_check(logistics_mode, crop, comp_qty, comp_dist)
        gross = c_latest["Price"] * comp_qty
        net   = gross - tc_c
        profit_compare.append({"Crop":crop, "Price ₹/Q":round(c_latest["Price"],0),
                                 "Gross ₹":round(gross,0), "Net After Transport":round(net,0),
                                 "Transport ₹":round(tc_c,0)})
    pc_df = pd.DataFrame(profit_compare).sort_values("Net After Transport", ascending=False)
    fig_pc = px.bar(pc_df, x="Crop", y="Net After Transport",
                    color="Net After Transport", color_continuous_scale="YlGn",
                    text=[f"₹{v/1e3:.0f}K" for v in pc_df["Net After Transport"]])
    fig_pc.update_traces(textposition="outside")
    dark_layout(fig_pc, f"Net Profit Comparison — {comp_qty}Q via {logistics_mode}", 320)
    st.plotly_chart(fig_pc, use_container_width=True)
    st.dataframe(pc_df.style.format({"Price ₹/Q":"₹{:.0f}","Gross ₹":"₹{:,.0f}",
                                      "Net After Transport":"₹{:,.0f}","Transport ₹":"₹{:,.0f}"})
                 .background_gradient(subset=["Net After Transport"], cmap="YlGn"),
                 use_container_width=True)

    # ── Break-even Analysis ──
    st.markdown('<div class="section-header">📐 Break-Even Price Analysis</div>', unsafe_allow_html=True)
    be1, be2 = st.columns(2)
    with be1:
        cost_of_production = st.number_input("Cost of Production (₹/Q)", 100, 5000, 800, key="cop")
        packaging_cost     = st.number_input("Packaging Cost (₹/Q)", 0, 500, 50, key="pack")
    with be2:
        labour_cost        = st.number_input("Labour Cost (₹/Q)", 0, 1000, 100, key="lab")
        misc_cost          = st.number_input("Misc. Expenses (₹/Q)", 0, 500, 30, key="misc")

    total_cost_pq   = cost_of_production + packaging_cost + labour_cost + misc_cost + cpq
    breakeven_price = total_cost_pq
    current_margin  = latest_row["Price"] - breakeven_price
    margin_pct      = (current_margin / breakeven_price) * 100 if breakeven_price > 0 else 0

    bb1,bb2,bb3,bb4 = st.columns(4)
    bb1.metric("Total Cost/Q",     f"₹{breakeven_price:,.0f}")
    bb2.metric("Break-Even Price", f"₹{breakeven_price:,.0f}")
    bb3.metric("Current Margin",   f"₹{current_margin:,.0f}", f"{margin_pct:+.1f}%")
    bb4.metric("Margin Status",    "✅ Profitable" if current_margin>0 else "❌ Loss")

    # Break-even waterfall
    fig_be = go.Figure(go.Waterfall(
        name="Cost Build-Up", orientation="v",
        measure=["absolute","relative","relative","relative","relative","relative","total"],
        x=["Production","Packaging","Labour","Misc","Transport","—","Break-Even"],
        y=[cost_of_production, packaging_cost, labour_cost, misc_cost, cpq, 0, 0],
        connector={"line":{"color":THEME["border"]}},
        increasing={"marker":{"color":THEME["accent_red"]}},
        totals={"marker":{"color":THEME["accent_amber"]}}
    ))
    dark_layout(fig_be, "Break-Even Cost Waterfall", 300)
    st.plotly_chart(fig_be, use_container_width=True)

    # ── Commodity Profitability Comparison ──
    st.markdown('<div class="section-header">⚖️ Crop Profitability Comparison</div>', unsafe_allow_html=True)
    profit_compare = []
    for crop in AgriDataEngine.COMMODITIES:
        c_latest = raw_df[raw_df["Commodity"]==crop].sort_values("Date").iloc[-1]
        _, tc_c, _, cpq_c, _, _ = logistics_check(logistics_mode, crop, f_qty, dist_sim)
        gross = c_latest["Price"] * f_qty
        net   = gross - tc_c
        profit_compare.append({"Crop":crop,"Price ₹/Q":round(c_latest["Price"],0),
                                 "Gross ₹":round(gross,0),"Transport ₹":round(tc_c,0),"Net ₹":round(net,0)})
    pc_df = pd.DataFrame(profit_compare).sort_values("Net ₹", ascending=False)
    fig_pc = px.bar(pc_df, x="Crop", y="Net ₹", color="Net ₹", color_continuous_scale="YlGn",
                    text=[f"₹{v/1e3:.0f}K" for v in pc_df["Net ₹"]])
    fig_pc.update_traces(textposition="outside")
    dark_layout(fig_pc, f"Net Profit Comparison — {f_qty}Q via {logistics_mode}", 320)
    st.plotly_chart(fig_pc, use_container_width=True)
    st.dataframe(pc_df.style.format({"Price ₹/Q":"₹{:.0f}","Gross ₹":"₹{:,.0f}",
                                      "Transport ₹":"₹{:,.0f}","Net ₹":"₹{:,.0f}"})
                 .background_gradient(subset=["Net ₹"], cmap="YlGn"), use_container_width=True)

    # ── Break-Even Analysis ──
    st.markdown('<div class="section-header">📐 Break-Even Price Analysis</div>', unsafe_allow_html=True)
    be1,be2 = st.columns(2)
    with be1:
        cop  = st.number_input("Cost of Production (₹/Q)",  100, 8000, 800, key="cop2")
        pack = st.number_input("Packaging + Sorting (₹/Q)", 0,   500,  50,  key="pack2")
    with be2:
        lab  = st.number_input("Labour Cost (₹/Q)",         0,   1000, 100, key="lab2")
        misc = st.number_input("Misc. Expenses (₹/Q)",      0,   500,  30,  key="misc2")
    _, _, _, cpq_be, _, _ = logistics_check(logistics_mode, sel_item, f_qty, dist_sim)
    breakeven = cop + pack + lab + misc + cpq_be
    margin    = latest_row["Price"] - breakeven
    margin_p  = (margin/breakeven*100) if breakeven>0 else 0
    bb1,bb2,bb3,bb4 = st.columns(4)
    bb1.metric("Total Cost/Q",     f"₹{breakeven:,.0f}")
    bb2.metric("Current Margin",   f"₹{margin:,.0f}", f"{margin_p:+.1f}%")
    bb3.metric("Margin Status",    "✅ Profit" if margin>0 else "❌ Loss")
    bb4.metric("Margin vs Pred",   f"₹{pred_price-breakeven:,.0f}", delta_color="normal")
    fig_bev = go.Figure(go.Waterfall(
        measure=["absolute","relative","relative","relative","relative","total"],
        x=["Production","Packaging","Labour","Misc","Transport","Break-Even"],
        y=[cop,pack,lab,misc,cpq_be,0],
        connector={"line":{"color":THEME["border"]}},
        increasing={"marker":{"color":THEME["accent_red"]}},
        totals={"marker":{"color":THEME["accent_amber"]}}
    ))
    dark_layout(fig_bev, "Break-Even Cost Waterfall (₹/Q)", 300)
    st.plotly_chart(fig_bev, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RISK & SPOILAGE
# ══════════════════════════════════════════════════════════════════════════════
with t_risk:
    st.markdown('<div class="section-header">Risk Dashboard</div>', unsafe_allow_html=True)

    transit_days = st.slider("Transit Time (Days)", 1, 15, 3, key='risk_transit')
    mode_factor  = 0.18 if logistics_mode == "Reefer (Cold Chain)" else (
                   0.35 if logistics_mode == "Electric Van (2T)" else 1.0)
    pf_val       = PERISH_FACTOR[AgriDataEngine.PERISHABILITY[sel_item]]
    spoilage_pct = min(((latest_row['Temp']*0.42)*(transit_days**1.35)*pf_val*mode_factor)/10, 100)
    fin_loss     = (f_qty * spoilage_pct / 100) * latest_row['Price']

    r1,r2,r3,r4 = st.columns(4)
    r1.metric("Spoilage Loss",        f"{spoilage_pct:.1f}%")
    r2.metric("Revenue at Risk",      f"₹{fin_loss:,.0f}")
    r3.metric("Safe Sell Window",     f"{max(0,int(5-transit_days))} days")
    r4.metric("Weather Risk Score",   f"{min(int(latest_row['Temp']/5 + latest_row['Rainfall']/10),10)}/10")

    col_a, col_b = st.columns(2)
    with col_a:
        # Mode comparison spoilage
        mc_rows = []
        for mn in LOGISTICS_MODES:
            mf = 0.18 if mn=="Reefer (Cold Chain)" else (0.35 if mn=="Electric Van (2T)" else 1.0)
            sp = min(((latest_row['Temp']*0.42)*(transit_days**1.35)*pf_val*mf)/10, 100)
            mc_rows.append({'Mode':mn,'Spoilage %':round(sp,1),'Cost ₹':dist_sim*LOGISTICS_MODES[mn]['cost_km']})
        mc_comp = pd.DataFrame(mc_rows)
        fig_sp = px.bar(mc_comp, x='Mode', y='Spoilage %',
                        color='Spoilage %', color_continuous_scale='RdYlGn_r',
                        text='Spoilage %')
        fig_sp.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        dark_layout(fig_sp, "Spoilage by Logistics Mode", 320)
        st.plotly_chart(fig_sp, use_container_width=True)

    with col_b:
        # Spoilage vs transit days — all commodities
        sp_lines = {}
        for c in AgriDataEngine.COMMODITIES:
            pf_c = PERISH_FACTOR[AgriDataEngine.PERISHABILITY[c]]
            sp_lines[c] = [min(((latest_row['Temp']*0.42)*(d**1.35)*pf_c*mode_factor)/10,100)
                           for d in range(1,16)]
        sp_df = pd.DataFrame(sp_lines, index=range(1,16)).reset_index()
        sp_df = sp_df.melt('index', var_name='Commodity', value_name='Spoilage %')
        fig_spl = px.line(sp_df, x='index', y='Spoilage %', color='Commodity',
                          color_discrete_sequence=px.colors.qualitative.Pastel)
        dark_layout(fig_spl, "Spoilage % vs Transit Days — All Crops", 320)
        fig_spl.update_layout(xaxis_title="Transit Days")
        st.plotly_chart(fig_spl, use_container_width=True)

    # Anomaly detection scatter
    st.markdown('<div class="section-header">Anomaly Detection — Price Outliers</div>', unsafe_allow_html=True)
    anom_data = df_full[(df_full['Commodity']==sel_item) & (df_full['Location']==sel_loc)].dropna(subset=ml.feat_cols).tail(120)
    if len(anom_data) > 10:
        anom_preds = ml.iso.predict(anom_data[ml.feat_cols])
        anom_data  = anom_data.copy()
        anom_data['Anomaly'] = ['Anomaly' if p==-1 else 'Normal' for p in anom_preds]
        fig_an = px.scatter(anom_data, x='Date', y='Price', color='Anomaly',
                            color_discrete_map={'Normal':THEME['accent_lime'],'Anomaly':THEME['accent_red']},
                            size_max=8)
        dark_layout(fig_an, f"Price Anomaly Detection — {sel_item} @ {sel_loc}", 320)
        st.plotly_chart(fig_an, use_container_width=True)

    # Weather risk trend
    st.markdown('<div class="section-header">Weather Risk — Last 60 Days</div>', unsafe_allow_html=True)
    wx_hist = mandi_df.tail(60)[['Date','Temp','Humidity','Rainfall']].copy()
    fig_wx  = make_subplots(rows=3, cols=1, shared_xaxes=True,
                             subplot_titles=["Temperature (°C)","Humidity (%)","Rainfall (mm)"])
    fig_wx.add_trace(go.Scatter(x=wx_hist['Date'], y=wx_hist['Temp'],
                                mode='lines', line=dict(color='#ef5350',width=1.5), name='Temp'), row=1,col=1)
    fig_wx.add_trace(go.Scatter(x=wx_hist['Date'], y=wx_hist['Humidity'],
                                mode='lines', line=dict(color='#42a5f5',width=1.5), name='Humidity'), row=2,col=1)
    fig_wx.add_trace(go.Bar(x=wx_hist['Date'], y=wx_hist['Rainfall'],
                            marker_color='#4fc3f7', name='Rainfall'), row=3,col=1)
    fig_wx.update_layout(plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_PAPER,
                         font=PLOT_FONT, height=420, showlegend=False,
                         margin=dict(l=10,r=10,t=40,b=10))
    for ax in fig_wx.layout:
        if ax.startswith('xaxis') or ax.startswith('yaxis'):
            fig_wx.layout[ax].update(gridcolor=THEME['border'], zerolinecolor=THEME['border'])
    st.plotly_chart(fig_wx, use_container_width=True)


    # ── VaR (Value at Risk) ──
    st.markdown('<div class="section-header">💰 Value at Risk (VaR) Analysis</div>', unsafe_allow_html=True)
    hist_returns = mandi_df["Price"].pct_change().dropna()
    var_95 = np.percentile(hist_returns, 5)
    var_99 = np.percentile(hist_returns, 1)
    var_95_val = latest_row["Price"] * abs(var_95) * f_qty
    var_99_val = latest_row["Price"] * abs(var_99) * f_qty

    vr1,vr2,vr3,vr4 = st.columns(4)
    vr1.metric("Daily VaR (95%)",   f"₹{var_95_val:,.0f}")
    vr2.metric("Daily VaR (99%)",   f"₹{var_99_val:,.0f}")
    vr3.metric("Worst Day Return",  f"{hist_returns.min()*100:.1f}%")
    vr4.metric("Best Day Return",   f"{hist_returns.max()*100:.1f}%")

    fig_ret = go.Figure()
    fig_ret.add_trace(go.Histogram(x=hist_returns*100, nbinsx=60,
                                    marker_color=THEME["accent_lime"], opacity=0.7))
    fig_ret.add_vline(x=var_95*100, line_dash="dash", line_color=THEME["accent_amber"],
                       annotation_text="VaR 95%")
    fig_ret.add_vline(x=var_99*100, line_dash="dash", line_color=THEME["accent_red"],
                       annotation_text="VaR 99%")
    dark_layout(fig_ret, f"Daily Return Distribution — {sel_item} @ {sel_loc}", 320)
    fig_ret.update_layout(xaxis_title="Daily Return %", yaxis_title="Frequency")
    st.plotly_chart(fig_ret, use_container_width=True)

    # ── Stress Test ──
    st.markdown('<div class="section-header">🧨 Scenario Stress Test</div>', unsafe_allow_html=True)
    scenarios = {
        "Base Case":        {"price_shock": 0,    "demand_drop": 0,    "transit_extra": 0},
        "Moderate Rain":    {"price_shock": 0.10, "demand_drop": 0.05, "transit_extra": 1},
        "Flood/Disruption": {"price_shock": 0.25, "demand_drop": 0.20, "transit_extra": 3},
        "Drought":          {"price_shock": 0.30, "demand_drop": 0.10, "transit_extra": 0},
        "Market Crash":     {"price_shock":-0.20, "demand_drop": 0.30, "transit_extra": 0},
        "Fuel Price Surge": {"price_shock": 0.05, "demand_drop": 0.02, "transit_extra": 2},
    }
    stress_rows = []
    base_net = (latest_row["Price"] * f_qty) - transport_cost
    for scen, params in scenarios.items():
        s_price   = latest_row["Price"] * (1 + params["price_shock"])
        s_qty     = f_qty * (1 - params["demand_drop"])
        s_transit = transit_days + params["transit_extra"]
        mf_s      = 0.18 if logistics_mode=="Reefer (Cold Chain)" else 1.0
        pf_s      = PERISH_FACTOR[AgriDataEngine.PERISHABILITY[sel_item]]
        s_spoil   = min(((latest_row["Temp"]*0.42)*(s_transit**1.35)*pf_s*mf_s)/10,100)
        s_net     = (s_price * s_qty * (1-s_spoil/100)) - transport_cost
        s_impact  = s_net - base_net
        stress_rows.append({"Scenario":scen,"Price ₹":round(s_price,0),
                             "Saleable Q":round(s_qty*(1-s_spoil/100),0),
                             "Net Revenue ₹":round(s_net,0),"P&L Impact ₹":round(s_impact,0),
                             "Spoilage %":round(s_spoil,1)})
    stress_df = pd.DataFrame(stress_rows)
    fig_st = go.Figure(go.Bar(
        x=stress_df["Scenario"], y=stress_df["P&L Impact ₹"],
        marker_color=[THEME["accent_lime"] if v>=0 else THEME["accent_red"] for v in stress_df["P&L Impact ₹"]],
        text=[f"₹{v:,.0f}" for v in stress_df["P&L Impact ₹"]],
        textposition="outside"
    ))
    fig_st.add_hline(y=0, line_dash="dash", line_color=THEME["text_muted"])
    dark_layout(fig_st, "Scenario Stress Test — P&L Impact vs Base Case", 320)
    st.plotly_chart(fig_st, use_container_width=True)
    st.dataframe(stress_df.style
                 .format({"Price ₹":"₹{:.0f}","Saleable Q":"{:.0f}",
                          "Net Revenue ₹":"₹{:,.0f}","P&L Impact ₹":"₹{:,.0f}","Spoilage %":"{:.1f}%"})
                 .background_gradient(subset=["P&L Impact ₹"], cmap="RdYlGn"),
                 use_container_width=True)

    # ── Insurance Premium Estimator ──
    st.markdown('<div class="section-header">🛡️ Crop Insurance Premium Estimator</div>', unsafe_allow_html=True)
    ins1, ins2 = st.columns(2)
    with ins1:
        insured_val    = st.number_input("Insured Crop Value (Rs)", 10000, 5000000, int(latest_row["Price"]*f_qty), key="ins_val")
        coverage_pct   = st.slider("Coverage Level (%)", 50, 100, 80, key="ins_cov")
    with ins2:
        scheme         = st.selectbox("Scheme Type", ["PMFBY","RWBCIS","MNAIS","Private"], key="ins_scheme")
        years_claim    = st.slider("Years Since Last Claim", 0, 10, 2, key="ins_yr")

    base_premium_rates = {"PMFBY":0.02,"RWBCIS":0.05,"MNAIS":0.08,"Private":0.12}
    base_rate  = base_premium_rates[scheme]
    risk_load  = (0.01 * max(0, latest_row["Temp"]-35) +
                  0.005 * max(0, latest_row["Rainfall"]-20) +
                  0.002 * max(0, 5 - latest_row["Reliability"]))
    ncb        = max(0, 0.05 * years_claim)  # no-claim bonus
    final_rate = max(0.005, base_rate + risk_load - ncb)
    premium    = insured_val * final_rate * (coverage_pct/100)
    subsidy    = premium * 0.50 if scheme == "PMFBY" else 0
    net_prem   = premium - subsidy

    ip1,ip2,ip3,ip4 = st.columns(4)
    ip1.metric("Gross Premium", f"Rs{premium:,.0f}")
    ip2.metric("Govt Subsidy",  f"Rs{subsidy:,.0f}")
    ip3.metric("Net Premium",   f"Rs{net_prem:,.0f}")
    ip4.metric("Premium Rate",  f"{final_rate*100:.2f}%")

    # Premium sensitivity to coverage
    cov_range = range(50, 101, 5)
    prem_range = [insured_val * final_rate * (c/100) - (insured_val * final_rate * (c/100) * 0.50 if scheme=="PMFBY" else 0)
                  for c in cov_range]
    fig_ins = go.Figure(go.Scatter(x=list(cov_range), y=prem_range,
                                    mode="lines+markers", fill="tozeroy",
                                    line=dict(color=THEME["accent_amber"],width=2),
                                    fillcolor="rgba(255,179,0,0.1)"))
    dark_layout(fig_ins, "Premium vs Coverage Level", 280)
    fig_ins.update_layout(xaxis_title="Coverage %", yaxis_title="Net Premium (Rs)")
    st.plotly_chart(fig_ins, use_container_width=True)

    # ── Commodity Risk Scorecard ──
    st.markdown('<div class="section-header">📊 Commodity Risk Scorecard</div>', unsafe_allow_html=True)
    risk_cards = []
    for crop in AgriDataEngine.COMMODITIES:
        c_data = raw_df[raw_df["Commodity"]==crop]["Price"]
        vol    = c_data.std() / c_data.mean() * 10
        perish_score = {"High":9,"Medium":6,"Low":3,"None":1}[AgriDataEngine.PERISHABILITY[crop]]
        weather_s    = 5 + (latest_row["Temp"]-22)*0.2
        total_risk   = (vol*0.4 + perish_score*0.35 + min(weather_s,10)*0.25)
        risk_cards.append({"Crop":crop,"Volatility":round(vol,1),"Perishability Risk":perish_score,
                           "Weather Risk":round(min(weather_s,10),1),"Total Risk":round(total_risk,1)})
    risk_sc_df = pd.DataFrame(risk_cards).sort_values("Total Risk",ascending=False)
    fig_rsc = px.scatter(risk_sc_df, x="Volatility", y="Perishability Risk",
                          size="Total Risk", color="Total Risk",
                          color_continuous_scale="RdYlGn_r",
                          hover_name="Crop", text="Crop")
    fig_rsc.update_traces(textposition="top center")
    dark_layout(fig_rsc, "Risk Scatter: Volatility vs Perishability (size=total risk)", 360)
    st.plotly_chart(fig_rsc, use_container_width=True)
    st.dataframe(risk_sc_df.style.background_gradient(subset=["Total Risk"],cmap="RdYlGn_r"),
                 use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — MARKET INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with t_insight:
    st.markdown('<div class="section-header">Market Intelligence</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Feature importance
        fi_df = ml.feature_importance().head(15).reset_index()
        fi_df.columns = ['Feature','Importance']
        fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                        color='Importance', color_continuous_scale='YlGn')
        dark_layout(fig_fi, "Top Price Drivers — GB Feature Importance", 420)
        fig_fi.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_fi, use_container_width=True)

    with col2:
        # Correlation matrix
        corr_cols = ['Price','Arrival','Temp','Humidity','Rainfall','Diesel','Reliability']
        corr_data = raw_df[raw_df['Commodity']==sel_item][corr_cols].corr()
        fig_corr  = px.imshow(corr_data, color_continuous_scale='RdBu_r',
                               zmin=-1, zmax=1, aspect='auto',
                               text_auto='.2f')
        dark_layout(fig_corr, f"Correlation Matrix — {sel_item}", 420)
        st.plotly_chart(fig_corr, use_container_width=True)

    # Volatility comparison
    st.markdown('<div class="section-header">Price Volatility Analysis</div>', unsafe_allow_html=True)
    vol_df = raw_df.groupby('Commodity')['Price'].agg(
        CV=lambda x: x.std()/x.mean()*100,
        Avg=np.mean, Std=np.std
    ).reset_index().sort_values('CV', ascending=False)

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(x=vol_df['Commodity'], y=vol_df['CV'],
                              name='CV%', marker_color=THEME['accent_amber'],
                              text=[f"{v:.1f}%" for v in vol_df['CV']],
                              textposition='outside'))
    dark_layout(fig_vol, "Price Volatility (CV%) by Commodity", 300)
    st.plotly_chart(fig_vol, use_container_width=True)

    # Price seasonality heatmap per commodity
    st.markdown('<div class="section-header">Seasonality Heatmap — Monthly Avg Price</div>', unsafe_allow_html=True)
    raw_df['Month_Num'] = raw_df['Date'].dt.month
    season_heat = raw_df[raw_df['Commodity']==sel_item].groupby(
        ['Month_Num','Location'])['Price'].mean().unstack()
    season_heat.index = ['Jan','Feb','Mar','Apr','May','Jun',
                          'Jul','Aug','Sep','Oct','Nov','Dec'][:len(season_heat)]
    fig_sh = px.imshow(season_heat, color_continuous_scale='RdYlGn_r', aspect='auto')
    dark_layout(fig_sh, f"Monthly Price Seasonality — {sel_item} by Mandi", 360)
    st.plotly_chart(fig_sh, use_container_width=True)

    # Supply vs demand scatter (arrivals vs price)
    st.markdown('<div class="section-header">Supply vs Price Dynamics</div>', unsafe_allow_html=True)
    sup_df = raw_df[(raw_df['Commodity']==sel_item) & (raw_df['Date'] >= today_date-timedelta(days=90))]
    fig_sd = px.scatter(sup_df, x='Arrival', y='Price', color='Location',
                        trendline='ols',
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                        opacity=0.6)
    dark_layout(fig_sd, f"Arrivals vs Price — {sel_item} (Last 90 Days, with LOWESS trend)", 360)
    st.plotly_chart(fig_sd, use_container_width=True)

    # PCA on mandi features
    st.markdown('<div class="section-header">Mandi Feature PCA</div>', unsafe_allow_html=True)
    pca_data = raw_df.groupby('Location').agg(
        Price=('Price','mean'), Arrival=('Arrival','mean'),
        Temp=('Temp','mean'), Humidity=('Humidity','mean'),
        Reliability=('Reliability','mean'), Rainfall=('Rainfall','mean')
    ).reset_index()
    pca_feats = ['Price','Arrival','Temp','Humidity','Reliability','Rainfall']
    scaler_pca = StandardScaler()
    pca_scaled = scaler_pca.fit_transform(pca_data[pca_feats])
    pca_model  = PCA(n_components=2)
    pca_result = pca_model.fit_transform(pca_scaled)
    pca_df     = pd.DataFrame(pca_result, columns=['PC1','PC2'])
    pca_df['Location'] = pca_data['Location'].values
    pca_df['Reliability'] = pca_data['Reliability'].values
    fig_pca = px.scatter(pca_df, x='PC1', y='PC2', text='Location',
                         color='Reliability', size='Reliability',
                         color_continuous_scale='RdYlGn')
    dark_layout(fig_pca, f"Mandi PCA — PC1: {pca_model.explained_variance_ratio_[0]*100:.0f}% | PC2: {pca_model.explained_variance_ratio_[1]*100:.0f}%", 360)
    fig_pca.update_traces(textposition='top center')
    st.plotly_chart(fig_pca, use_container_width=True)


    # ── Price Elasticity ──
    st.markdown('<div class="section-header">📐 Price Elasticity of Supply</div>', unsafe_allow_html=True)
    elas_rows = []
    for crop in AgriDataEngine.COMMODITIES:
        c_df = raw_df[raw_df["Commodity"]==crop].dropna()
        if len(c_df) < 60: continue
        from sklearn.linear_model import LinearRegression as LR
        X_e = c_df["Arrival"].values.reshape(-1,1)
        y_e = c_df["Price"].values
        lr_e = LR().fit(X_e, y_e)
        elas_rows.append({"Crop":crop,"Price~Arrival Slope":round(lr_e.coef_[0],4),
                           "R²":round(lr_e.score(X_e,y_e),3)})
    elas_df = pd.DataFrame(elas_rows)
    col_el1, col_el2 = st.columns(2)
    with col_el1:
        fig_el = px.bar(elas_df, x="Crop", y="Price~Arrival Slope",
                        color="Price~Arrival Slope", color_continuous_scale="RdBu",
                        title="Price Sensitivity to Arrivals (slope)")
        dark_layout(fig_el, "Price Elasticity by Commodity", 300)
        st.plotly_chart(fig_el, use_container_width=True)
    with col_el2:
        fig_r2 = px.bar(elas_df, x="Crop", y="R²",
                        color="R²", color_continuous_scale="Blues")
        dark_layout(fig_r2, "R² Score — Arrivals explaining Price", 300)
        st.plotly_chart(fig_r2, use_container_width=True)

    # ── Rolling Correlation ──
    st.markdown('<div class="section-header">📊 Rolling 30-Day Correlation: Price vs Arrivals</div>', unsafe_allow_html=True)
    roll_corr = (mandi_df.set_index("Date")[["Price","Arrival"]]
                 .rolling(30).corr().unstack()["Price"]["Arrival"].reset_index())
    roll_corr.columns = ["Date","Rolling_Corr"]
    roll_corr = roll_corr.dropna()
    fig_rc = go.Figure(go.Scatter(x=roll_corr["Date"], y=roll_corr["Rolling_Corr"],
                                   mode="lines", fill="tozeroy",
                                   line=dict(color=THEME["accent_lime"],width=1.5),
                                   fillcolor="rgba(165,214,61,0.1)"))
    fig_rc.add_hline(y=0, line_dash="dash", line_color=THEME["text_muted"])
    dark_layout(fig_rc, f"Rolling 30-Day Correlation: Price vs Arrivals — {sel_item} @ {sel_loc}", 280)
    st.plotly_chart(fig_rc, use_container_width=True)

    # ── ML Model Comparison ──
    st.markdown('<div class="section-header">🤖 ML Model Comparison</div>', unsafe_allow_html=True)
    comp_data = mandi_df.dropna(subset=ml.feat_cols+["Target"]).tail(200)
    X_c = comp_data[ml.feat_cols]
    y_c = comp_data["Target"]
    from sklearn.ensemble import RandomForestRegressor as RFR
    from sklearn.linear_model import LinearRegression as LR2
    models = {
        "Gradient Boosting": ml.gb,
        "Ridge Regression":  ml.ridge,
        "Random Forest":     RFR(n_estimators=50,random_state=42).fit(X_c,y_c),
        "Linear Regression": LR2().fit(X_c,y_c)
    }
    model_perf = []
    for name, mdl in models.items():
        if name == "Ridge Regression":
            preds = mdl.predict(ml.ridge_scaler.transform(X_c))
        else:
            preds = mdl.predict(X_c)
        mae_v = mean_absolute_error(y_c, preds)
        model_perf.append({"Model":name,"MAE ₹":round(mae_v,1)})
    perf_df = pd.DataFrame(model_perf).sort_values("MAE ₹")
    fig_mcomp = px.bar(perf_df, x="Model", y="MAE ₹",
                        color="MAE ₹", color_continuous_scale="RdYlGn_r",
                        text="MAE ₹")
    fig_mcomp.update_traces(texttemplate="₹%{text:.1f}", textposition="outside")
    dark_layout(fig_mcomp, "Model Comparison — MAE (lower is better)", 300)
    st.plotly_chart(fig_mcomp, use_container_width=True)

    # ── Inter-Commodity Price Correlation ──
    st.markdown('<div class="section-header">🔗 Inter-Commodity Price Correlation Network</div>', unsafe_allow_html=True)
    pivot_corr = raw_df.groupby(["Date","Commodity"])["Price"].mean().unstack()
    corr_matrix = pivot_corr.corr()
    fig_icorr = px.imshow(corr_matrix, color_continuous_scale="RdBu_r",
                           zmin=-1, zmax=1, text_auto=".2f", aspect="auto")
    dark_layout(fig_icorr, "Inter-Commodity Price Correlation Matrix", 400)
    st.plotly_chart(fig_icorr, use_container_width=True)

    # ── Mandi Efficiency Ranking ──
    st.markdown('<div class="section-header">🏅 Mandi Efficiency Ranking</div>', unsafe_allow_html=True)
    eff_rank = raw_df.groupby("Location").agg(
        Avg_Price=("Price","mean"), Avg_Arr=("Arrival","mean"),
        Avg_Trd=("Traded","mean"), Avg_Rel=("Reliability","mean"),
        Price_Std=("Price","std")
    ).reset_index()
    eff_rank["Sell_Rate%"]  = (eff_rank["Avg_Trd"]/eff_rank["Avg_Arr"]*100).round(1)
    eff_rank["Price_CV%"]   = (eff_rank["Price_Std"]/eff_rank["Avg_Price"]*100).round(1)
    eff_rank["Eff_Score"]   = (
        eff_rank["Avg_Rel"]*0.35 +
        eff_rank["Sell_Rate%"]/10*0.35 +
        (10 - eff_rank["Price_CV%"]/5).clip(0,10)*0.30
    ).round(2)
    eff_rank = eff_rank.sort_values("Eff_Score", ascending=False)

    fig_eff_rank = px.bar(eff_rank, x="Location", y="Eff_Score",
                           color="Eff_Score", color_continuous_scale="YlGn",
                           text=[f"{v:.1f}" for v in eff_rank["Eff_Score"]])
    fig_eff_rank.update_traces(textposition="outside")
    dark_layout(fig_eff_rank, "Mandi Efficiency Score (Reliability + Sell-Rate + Price Stability)", 320)
    st.plotly_chart(fig_eff_rank, use_container_width=True)

    st.dataframe(eff_rank[["Location","Avg_Price","Sell_Rate%","Price_CV%","Avg_Rel","Eff_Score"]]
                 .style.format({"Avg_Price":"₹{:.0f}","Sell_Rate%":"{:.1f}%",
                                "Price_CV%":"{:.1f}%","Avg_Rel":"{:.1f}","Eff_Score":"{:.2f}"})
                 .background_gradient(subset=["Eff_Score"], cmap="YlGn"),
                 use_container_width=True)

    # ── Year-over-Year Price Comparison ──
    st.markdown('<div class="section-header">📅 Year-over-Year Price Comparison</div>', unsafe_allow_html=True)
    raw_df["Year"] = raw_df["Date"].dt.year
    raw_df["Month_Label"] = raw_df["Date"].dt.strftime("%b")
    raw_df["Month_Num"]   = raw_df["Date"].dt.month
    yoy_df = raw_df[raw_df["Commodity"]==sel_item].groupby(["Year","Month_Num","Month_Label"])["Price"].mean().reset_index()
    yoy_df = yoy_df.sort_values("Month_Num")
    fig_yoy = px.line(yoy_df, x="Month_Label", y="Price", color="Year",
                       color_discrete_sequence=[THEME["accent_lime"],THEME["accent_amber"],THEME["accent_red"]],
                       markers=True,
                       category_orders={"Month_Label":["Jan","Feb","Mar","Apr","May","Jun",
                                                        "Jul","Aug","Sep","Oct","Nov","Dec"]})
    dark_layout(fig_yoy, f"Year-over-Year Monthly Avg Price — {sel_item}", 320)
    fig_yoy.update_layout(xaxis_title="Month", yaxis_title="Avg Price (₹/Q)")
    st.plotly_chart(fig_yoy, use_container_width=True)

    # ── Moving Average Crossover Signal ──
    st.markdown('<div class="section-header">📉 Moving Average Crossover Signal</div>', unsafe_allow_html=True)
    ma_df = mandi_df[["Date","Price"]].copy().set_index("Date").sort_index()
    ma_df["MA7"]  = ma_df["Price"].rolling(7).mean()
    ma_df["MA21"] = ma_df["Price"].rolling(21).mean()
    ma_df["Signal"] = np.where(ma_df["MA7"] > ma_df["MA21"], 1, -1)
    ma_df["Cross"]  = ma_df["Signal"].diff().fillna(0)
    ma_df = ma_df.reset_index().tail(120)

    buy_signals  = ma_df[ma_df["Cross"]==2]
    sell_signals = ma_df[ma_df["Cross"]==-2]

    fig_mac = go.Figure()
    fig_mac.add_trace(go.Scatter(x=ma_df["Date"],y=ma_df["Price"],mode="lines",
                                  name="Price",line=dict(color=THEME["text_muted"],width=1)))
    fig_mac.add_trace(go.Scatter(x=ma_df["Date"],y=ma_df["MA7"],mode="lines",
                                  name="MA7",line=dict(color=THEME["accent_lime"],width=2)))
    fig_mac.add_trace(go.Scatter(x=ma_df["Date"],y=ma_df["MA21"],mode="lines",
                                  name="MA21",line=dict(color=THEME["accent_amber"],width=2)))
    fig_mac.add_trace(go.Scatter(x=buy_signals["Date"],y=buy_signals["Price"],
                                  mode="markers",name="Buy Signal",
                                  marker=dict(color=THEME["accent_lime"],size=10,symbol="triangle-up")))
    fig_mac.add_trace(go.Scatter(x=sell_signals["Date"],y=sell_signals["Price"],
                                  mode="markers",name="Sell Signal",
                                  marker=dict(color=THEME["accent_red"],size=10,symbol="triangle-down")))
    dark_layout(fig_mac, f"MA7/MA21 Crossover Signals — {sel_item} @ {sel_loc}", 360)
    st.plotly_chart(fig_mac, use_container_width=True)
    st.caption("▲ Buy signal when 7-day MA crosses above 21-day MA | ▼ Sell signal on downward cross")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — AGRO INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
with t_agro:
    st.markdown('<div class="section-header">🌤️ 14-Day Weather Forecast</div>', unsafe_allow_html=True)

    forecast_wx = []
    for i in range(1, 15):
        fd   = today_date + timedelta(days=i)
        doy  = fd.dayofyear
        t_fc = 22 + np.sin(doy/365*2*np.pi)*13 + np.random.normal(0, 1.2)
        h_fc = 55 + np.cos(doy/365*2*np.pi)*26 + np.random.normal(0, 3)
        r_fc = np.random.exponential(3.0 if 5<=fd.month<=9 else 0.3)
        w_fc = np.random.exponential(7)+3
        cond = ("🌧️ Rain" if r_fc>3 else ("⛅ Cloudy" if h_fc>72 else "☀️ Sunny"))
        frost_fc = (fd.month in [12,1,2] and t_fc < 5)
        forecast_wx.append({'Day':fd.strftime('%d %b'),'Temp':round(t_fc,1),
                             'Hum':round(h_fc,1),'Rain':round(r_fc,2),
                             'Wind':round(w_fc,1),'Cond':cond,'Frost':frost_fc})
    fc_df = pd.DataFrame(forecast_wx)

    # 7-col weather cards × 2 rows
    for row_start in [0, 7]:
        cols_wx = st.columns(7)
        for i, col in enumerate(cols_wx):
            d  = fc_df.iloc[row_start+i]
            bg = ("#1a2e20" if "Sunny" in d['Cond'] else
                  ("#1a2035" if "Cloudy" in d['Cond'] else "#2e1a20"))
            bd = (THEME['accent_lime'] if "Sunny" in d['Cond'] else
                  (THEME['accent_blue'] if "Cloudy" in d['Cond'] else THEME['accent_red']))
            frost_str = " ❄️" if d['Frost'] else ""
            col.markdown(f"""
            <div class="wx-card" style="background:{bg};border-color:{bd};">
                <b>{d['Day']}</b><br>{d['Cond']}{frost_str}<br>
                🌡️{d['Temp']}°<br>💧{d['Rain']}mm<br>💨{d['Wind']}kph
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Weather forecast chart
    fig_wf2 = make_subplots(specs=[[{"secondary_y":True}]])
    fig_wf2.add_trace(go.Scatter(x=fc_df['Day'], y=fc_df['Temp'], name='Temp °C',
                                  mode='lines+markers',
                                  line=dict(color=THEME['accent_red'],width=2)), secondary_y=False)
    fig_wf2.add_trace(go.Bar(x=fc_df['Day'], y=fc_df['Rain'], name='Rain mm',
                              marker_color=THEME['accent_blue'], opacity=0.6), secondary_y=True)
    fig_wf2.update_layout(plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_PAPER,
                           font=PLOT_FONT, height=280,
                           legend=dict(bgcolor='rgba(0,0,0,0)'),
                           title=dict(text="14-Day Forecast: Temperature & Rainfall",
                                      font=dict(family="Syne",color=THEME['accent_lime'],size=13)),
                           margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_wf2, use_container_width=True)

    st.divider()

    # ── YIELD PREDICTION ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">🌱 Harvest Yield Prediction</div>', unsafe_allow_html=True)

    yc1, yc2, yc3 = st.columns(3)
    with yc1:
        yield_crop  = st.selectbox("Crop", AgriDataEngine.COMMODITIES, key='yc2')
        acreage     = st.number_input("Field Area (Acres)", 1.0, 1000.0, 10.0, step=1.0, key='acre')
    with yc2:
        soil_qual   = st.selectbox("Soil Quality", ["Poor","Average","Good","Excellent"])
        irrigation  = st.selectbox("Irrigation Type", ["Rain-fed","Drip","Sprinkler","Canal Flood","Micro-irrigation"])
    with yc3:
        fertilizer  = st.selectbox("Fertilizer Level", ["None","Low","Medium","High","Precision"])
        seed_type   = st.selectbox("Seed Variety", ["Local","Hybrid","HYV (High Yielding)","GM/Biotech"])

    soil_mult  = {"Poor":0.58,"Average":0.78,"Good":1.00,"Excellent":1.18}
    irrig_mult = {"Rain-fed":0.72,"Drip":1.22,"Sprinkler":1.06,"Canal Flood":0.88,"Micro-irrigation":1.30}
    fert_mult  = {"None":0.62,"Low":0.80,"Medium":1.00,"High":1.20,"Precision":1.35}
    seed_mult  = {"Local":0.78,"Hybrid":1.12,"HYV (High Yielding)":1.28,"GM/Biotech":1.45}

    avg_t_fc = fc_df['Temp'].mean()
    avg_r_fc = fc_df['Rain'].mean()
    wx_yld   = 1.0
    if avg_t_fc > 38: wx_yld -= 0.18
    elif avg_t_fc < 10: wx_yld -= 0.12
    if avg_r_fc > 6 and yield_crop in ('Rice','Maize'): wx_yld += 0.08
    elif avg_r_fc > 6 and yield_crop in ('Tomato','Carrot'): wx_yld -= 0.06
    if fc_df['Frost'].any() and yield_crop in ('Tomato','Carrot','Cabbage'): wx_yld -= 0.20

    base_y   = BASE_YIELD_Q[yield_crop]
    est_q    = (base_y * acreage * soil_mult[soil_qual] * irrig_mult[irrigation]
                * fert_mult[fertilizer] * seed_mult[seed_type] * wx_yld)
    est_rev  = est_q * AgriDataEngine.BASE_PRICES[yield_crop]

    ya,yb,yc_m,yd = st.columns(4)
    ya.metric("Estimated Yield",    f"{est_q:,.0f} Q")
    yb.metric("Per Acre",           f"{est_q/acreage:.1f} Q/acre")
    yc_m.metric("Est. Revenue",     f"₹{est_rev:,.0f}")
    yd.metric("Weather Modifier",   f"{(wx_yld-1)*100:+.1f}%", delta_color="normal")

    # Yield sensitivity heatmap
    soil_o = list(soil_mult.keys())
    irr_o  = list(irrig_mult.keys())
    sens   = []
    for s in soil_o:
        for ir in irr_o:
            y = base_y*acreage*soil_mult[s]*irrig_mult[ir]*fert_mult[fertilizer]*seed_mult[seed_type]*wx_yld
            sens.append({'Soil':s,'Irrigation':ir,'Yield (Q)':round(y,0)})
    sens_df = pd.DataFrame(sens).pivot(index='Soil', columns='Irrigation', values='Yield (Q)')
    fig_sh2 = px.imshow(sens_df, color_continuous_scale='YlGn', text_auto=True, aspect='auto')
    dark_layout(fig_sh2, f"Yield Sensitivity — {yield_crop} ({acreage:.0f} acres)", 320)
    st.plotly_chart(fig_sh2, use_container_width=True)

    st.divider()

    # ── CROP RECOMMENDATIONS ──────────────────────────────────────────────
    st.markdown('<div class="section-header">🏆 High-Yield Crop Recommendations</div>', unsafe_allow_html=True)

    curr_mon = today_date.month - 1
    rec_data = []
    for crop in AgriDataEngine.COMMODITIES:
        seas_now  = AgriDataEngine.SEASONAL_CURVES[crop][curr_mon]
        seas_next = AgriDataEngine.SEASONAL_CURVES[crop][(curr_mon+1)%12]
        mom       = (seas_next - seas_now)/seas_now * 100

        crop_rc   = raw_df[raw_df['Commodity']==crop]
        price_trend = (crop_rc['Price'].iloc[-1] - crop_rc['Price'].iloc[-30])/crop_rc['Price'].iloc[-30]*100 if len(crop_rc)>30 else 0

        yld_s   = BASE_YIELD_Q[crop] / max(BASE_YIELD_Q.values()) * 10
        wx_s    = max(0, 10 - (max(avg_t_fc-35,0)*0.5) - (1 if fc_df['Frost'].any() and crop in ('Tomato','Carrot','Cabbage') else 0))
        mkt_s   = min(seas_now*10, 10)
        price_s = min(max((price_trend/10)+5, 0), 10)
        comp    = yld_s*0.28 + wx_s*0.28 + mkt_s*0.22 + price_s*0.12 + max(mom*0.05,0)*0.10

        rec_data.append({
            'Crop': crop,
            'Yield Score': round(yld_s,1),
            'Weather Fit': round(wx_s,1),
            'Market Score': round(mkt_s,1),
            'Price Momentum': round(price_s,1),
            'Overall': round(comp,2),
            'Signal': ('🟢 Plant Now' if comp>=6.5 else ('🟡 Consider' if comp>=4.5 else '🔴 Skip'))
        })

    rec_df = pd.DataFrame(rec_data).sort_values('Overall', ascending=False)

    # Radar for top 3
    top3  = rec_df.head(3)
    cats  = ['Yield Score','Weather Fit','Market Score','Price Momentum']
    fig_r = go.Figure()
    for idx, (_, row) in enumerate(top3.iterrows()):
        vals = [row[c] for c in cats] + [row[cats[0]]]
        fig_r.add_trace(go.Scatterpolar(
            r=vals, theta=cats+[cats[0]],
            fill='toself', name=row['Crop'],
            line_color=[THEME['accent_lime'],THEME['accent_amber'],THEME['accent_blue']][idx],
            opacity=0.72
        ))
    fig_r.update_layout(
        polar=dict(
            bgcolor=THEME['bg_panel'],
            radialaxis=dict(range=[0,12], gridcolor=THEME['border'], color=THEME['text_muted']),
            angularaxis=dict(gridcolor=THEME['border'], color=THEME['text_muted'])
        ),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_PAPER, font=PLOT_FONT, height=380,
        title=dict(text="Top 3 Crop Radar",font=dict(family="Syne",color=THEME['accent_lime'],size=13)),
        margin=dict(l=10,r=10,t=40,b=10)
    )
    st.plotly_chart(fig_r, use_container_width=True)

    st.dataframe(rec_df.style
                 .format({'Yield Score':'{:.1f}','Weather Fit':'{:.1f}','Market Score':'{:.1f}',
                          'Price Momentum':'{:.1f}','Overall':'{:.2f}'})
                 .background_gradient(subset=['Overall'], cmap='RdYlGn'),
                 use_container_width=True)


    # ── Soil Health Advisory ──
    st.markdown('<div class="section-header">🪱 Soil Health & Fertilizer Advisory</div>', unsafe_allow_html=True)
    sh1, sh2, sh3 = st.columns(3)
    with sh1:
        soil_ph   = st.slider("Soil pH", 4.0, 9.0, 6.5, 0.1, key="soil_ph")
        nitrogen  = st.selectbox("Nitrogen Level", ["Very Low","Low","Medium","High","Excess"], key="nitro")
    with sh2:
        phosphorus = st.selectbox("Phosphorus Level", ["Low","Medium","High"], key="phos")
        potassium  = st.selectbox("Potassium Level",  ["Low","Medium","High"], key="potas")
    with sh3:
        organic_m  = st.slider("Organic Matter (%)", 0.1, 8.0, 1.5, 0.1, key="orgm")
        texture    = st.selectbox("Soil Texture", ["Sandy","Loamy","Clay","Silt","Sandy Loam"], key="tex")

    # pH suitability
    ph_suit = {
        "Onion":(6.0,7.0),"Tomato":(5.5,7.5),"Potato":(4.8,5.8),
        "Wheat":(6.0,7.5),"Rice":(5.0,6.5),"Garlic":(6.0,7.0),
        "Carrot":(5.5,7.0),"Cabbage":(6.0,7.5),"Chilli":(6.0,7.5),"Maize":(5.8,7.0)
    }
    low_ph, high_ph = ph_suit.get(yield_crop, (5.5,7.5))
    ph_ok = low_ph <= soil_ph <= high_ph

    n_recs = {"Very Low":"Apply 150kg Urea/acre","Low":"Apply 80kg Urea/acre",
               "Medium":"Apply 40kg Urea/acre","High":"No additional N needed","Excess":"Flush with water & skip N"}

    sa1,sa2,sa3 = st.columns(3)
    sa1.metric("pH Status", f"{soil_ph:.1f} — {'✅ Optimal' if ph_ok else '⚠️ Adjust'}")
    sa2.metric("Optimal pH Range", f"{low_ph}–{high_ph}")
    sa3.metric("Correction", "None" if ph_ok else ("Add lime" if soil_ph<low_ph else "Add sulfur"))

    st.markdown(f'<div class="alert-info">🌿 <b>Fertilizer Advice:</b> {n_recs[nitrogen]}<br>'
                f'Phosphorus: {"Apply DAP 50kg/acre" if phosphorus=="Low" else "Adequate"} &nbsp;|&nbsp; '
                f'Potassium: {"Apply MOP 40kg/acre" if potassium=="Low" else "Adequate"}</div>',
                unsafe_allow_html=True)

    # pH suitability radar for all crops
    ph_scores = []
    for crop in AgriDataEngine.COMMODITIES:
        lo, hi = ph_suit.get(crop,(5.5,7.5))
        dist   = 0 if lo<=soil_ph<=hi else min(abs(soil_ph-lo), abs(soil_ph-hi))
        score  = max(0, 10 - dist*5)
        ph_scores.append({"Crop":crop,"pH Suitability":round(score,1)})
    ph_df = pd.DataFrame(ph_scores).sort_values("pH Suitability",ascending=False)
    fig_ph = px.bar(ph_df, x="Crop", y="pH Suitability",
                    color="pH Suitability", color_continuous_scale="RdYlGn",
                    text="pH Suitability")
    fig_ph.update_traces(textposition="outside")
    dark_layout(fig_ph, f"Crop Suitability at Current pH {soil_ph:.1f}", 300)
    st.plotly_chart(fig_ph, use_container_width=True)

    # ── Irrigation Schedule Builder ──
    st.markdown('<div class="section-header">💧 Smart Irrigation Schedule</div>', unsafe_allow_html=True)
    is1, is2 = st.columns(2)
    with is1:
        crop_stage   = st.selectbox("Crop Growth Stage",
                                    ["Germination","Seedling","Vegetative","Flowering","Fruiting","Harvest"],
                                    key="crop_stage")
        field_acres  = st.number_input("Field Size (Acres)", 0.5, 500.0, 5.0, key="field_ac")
    with is2:
        water_source = st.selectbox("Water Source", ["Borewell","Canal","Rainwater","Tank","River"])
        soil_type_w  = st.selectbox("Soil Water Retention", ["Low (Sandy)","Medium (Loamy)","High (Clay)"])

    # Water requirement by stage
    stage_water_mm = {
        "Germination":4,"Seedling":6,"Vegetative":8,"Flowering":10,"Fruiting":9,"Harvest":5
    }
    retention_factor = {"Low (Sandy)":1.4,"Medium (Loamy)":1.0,"High (Clay)":0.7}
    daily_req_mm = stage_water_mm[crop_stage] * retention_factor[soil_type_w]

    # Adjust for forecast rain
    rain_cover = min(fc_df["Rain"].mean() * 0.7, daily_req_mm)
    net_irrig  = max(0, daily_req_mm - rain_cover)
    total_water_liter = net_irrig * field_acres * 4047  # mm * acres * m2_per_acre / 1000 * 1000

    iw1,iw2,iw3,iw4 = st.columns(4)
    iw1.metric("Daily Water Need", f"{daily_req_mm:.1f} mm/day")
    iw2.metric("Rain Coverage",    f"{rain_cover:.1f} mm/day")
    iw3.metric("Net Irrigation",   f"{net_irrig:.1f} mm/day")
    iw4.metric("Total Volume",     f"{total_water_liter/1000:.1f} KL/day")

    # 14-day irrigation schedule
    irrig_sched = []
    for i, row_wx in fc_df.iterrows():
        rain_d = row_wx["Rain"]
        rain_cov_d = min(rain_d*0.7, daily_req_mm)
        net_d = max(0, daily_req_mm - rain_cov_d)
        irrig_sched.append({"Day":row_wx["Day"],"Rain mm":row_wx["Rain"],
                             "Rain Cover mm":round(rain_cov_d,1),"Irrigate mm":round(net_d,1),
                             "Action":"💧 Irrigate" if net_d>0 else "🌧️ Skip — Rain sufficient"})
    irrig_df = pd.DataFrame(irrig_sched)

    fig_irrig = go.Figure()
    fig_irrig.add_trace(go.Bar(x=irrig_df["Day"],y=irrig_df["Irrigate mm"],name="Irrigation Needed",
                                marker_color=THEME["accent_blue"]))
    fig_irrig.add_trace(go.Bar(x=irrig_df["Day"],y=irrig_df["Rain Cover mm"],name="Rain Coverage",
                                marker_color=THEME["accent_lime"]))
    fig_irrig.update_layout(barmode="stack")
    dark_layout(fig_irrig, "14-Day Irrigation Schedule (mm/day)", 300)
    st.plotly_chart(fig_irrig, use_container_width=True)

    st.dataframe(irrig_df.style.map(
        lambda v: f"color:{THEME['accent_lime']}" if "Skip" in str(v) else f"color:{THEME['accent_blue']}",
        subset=["Action"]), use_container_width=True)

    # ── Pest & Disease Risk Index ──
    st.markdown('<div class="section-header">🦟 Pest & Disease Risk Assessment</div>', unsafe_allow_html=True)
    avg_hum   = fc_df["Hum"].mean()
    avg_temp_pest = fc_df["Temp"].mean()
    avg_rain_pest = fc_df["Rain"].mean()

    PEST_RISKS = {
        "Aphids":      {"temp_range":(15,30), "hum_thresh":60, "crops":["Tomato","Cabbage","Carrot","Chilli"]},
        "Whitefly":    {"temp_range":(25,38), "hum_thresh":55, "crops":["Tomato","Chilli","Cabbage"]},
        "Late Blight": {"temp_range":(15,22), "hum_thresh":80, "crops":["Potato","Tomato"]},
        "Powdery Mildew":{"temp_range":(20,28),"hum_thresh":65,"crops":["Onion","Garlic","Wheat"]},
        "Root Rot":    {"temp_range":(22,35), "hum_thresh":85, "crops":["Carrot","Potato","Onion"]},
        "Stem Borer":  {"temp_range":(25,38), "hum_thresh":70, "crops":["Rice","Maize","Wheat"]},
    }

    pest_rows = []
    for pest, info in PEST_RISKS.items():
        t_low, t_high = info["temp_range"]
        temp_risk = 1 if t_low <= avg_temp_pest <= t_high else 0
        hum_risk  = 1 if avg_hum >= info["hum_thresh"] else 0
        crop_risk = 1 if yield_crop in info["crops"] else 0
        risk_score = (temp_risk*4 + hum_risk*3 + crop_risk*3)
        pest_rows.append({"Pest/Disease":pest,"Temp Risk":temp_risk,"Humidity Risk":hum_risk,
                           "Crop Susceptible":crop_risk,"Risk Score":risk_score,
                           "Risk Level":"🔴 High" if risk_score>=7 else ("🟡 Medium" if risk_score>=4 else "🟢 Low")})
    pest_df = pd.DataFrame(pest_rows).sort_values("Risk Score",ascending=False)

    fig_pest = px.bar(pest_df, x="Pest/Disease", y="Risk Score",
                       color="Risk Score", color_continuous_scale="RdYlGn_r",
                       text="Risk Level")
    fig_pest.update_traces(textposition="outside")
    dark_layout(fig_pest, f"Pest & Disease Risk Index — {yield_crop} (next 14 days)", 320)
    st.plotly_chart(fig_pest, use_container_width=True)
    st.dataframe(pest_df.style.background_gradient(subset=["Risk Score"],cmap="RdYlGn_r"),
                 use_container_width=True)

    # ── Heatwave & Cold Snap Forecast ──
    st.markdown('<div class="section-header">🌡️ Extreme Weather Event Forecast</div>', unsafe_allow_html=True)
    hw_days = [d for d in forecast_wx if d["Temp"] > 38]
    cs_days = [d for d in forecast_wx if d["Temp"] < 10]
    frost_days_list = [d for d in forecast_wx if d["Frost"]]
    ew1,ew2,ew3 = st.columns(3)
    ew1.metric("Heatwave Days (>38°C)",    f"{len(hw_days)} days",
               delta="High Risk" if len(hw_days)>3 else "Manageable")
    ew2.metric("Cold Snap Days (<10°C)",   f"{len(cs_days)} days")
    ew3.metric("Frost Events Forecast",    f"{len(frost_days_list)} days",
               delta="Critical" if len(frost_days_list)>0 else "None")

    if hw_days:
        st.markdown(f'<div class="alert-critical">🌡️ Heatwave Alert: {len(hw_days)} days above 38°C forecast. '
                    f'For {yield_crop}: increase irrigation frequency, deploy shade nets, harvest earlier.</div>',
                    unsafe_allow_html=True)
    if frost_days_list:
        st.markdown('<div class="alert-warning">❄️ Frost Risk: Use row covers, smoke screens, or micro-sprinklers '
                    'to protect crop canopy during frost nights.</div>', unsafe_allow_html=True)

    # 14-day temp chart with risk bands
    fig_ex = go.Figure()
    fig_ex.add_hrect(y0=38, y1=50, fillcolor="rgba(239,83,80,0.12)", line_width=0, annotation_text="Heatwave Zone")
    fig_ex.add_hrect(y0=-5, y1=10, fillcolor="rgba(66,165,245,0.12)", line_width=0, annotation_text="Cold Snap Zone")
    fig_ex.add_trace(go.Scatter(x=fc_df["Day"], y=fc_df["Temp"], mode="lines+markers",
                                 line=dict(color=THEME["accent_amber"],width=2.5),
                                 marker=dict(size=8,
                                             color=[THEME["accent_red"] if t>38 else
                                                    (THEME["accent_blue"] if t<10 else THEME["accent_lime"])
                                                    for t in fc_df["Temp"]]),
                                 name="Forecast Temp"))
    dark_layout(fig_ex, "14-Day Temperature Forecast with Risk Zones", 300)
    st.plotly_chart(fig_ex, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — SALES REPORT
# ══════════════════════════════════════════════════════════════════════════════
with t_report:
    st.markdown('<div class="section-header">Sales Intelligence Report</div>', unsafe_allow_html=True)

    rep_days = st.slider("Lookback Window (days)", 1, 15, 7, key='rep2')
    cutoff   = today_date - timedelta(days=rep_days)
    recent   = raw_df[raw_df['Date'] >= cutoff]

    rk1,rk2,rk3,rk4,rk5 = st.columns(5)
    rk1.metric("Period",               f"Last {rep_days}d")
    rk2.metric("Records",              f"{len(recent):,}")
    rk3.metric("Avg Price (₹/Q)",      f"₹{recent['Price'].mean():,.0f}")
    rk4.metric("Total Arrivals (Q)",   f"{recent['Arrival'].sum():,.0f}")
    rk5.metric("Total Traded (Q)",     f"{recent['Traded'].sum():,.0f}")

    # Best-selling — revenue proxy bar
    st.markdown('<div class="section-header">🏆 Best-Selling Crops</div>', unsafe_allow_html=True)
    past = recent.groupby('Commodity').agg(
        Avg_Price=('Price','mean'), Total_Arrivals=('Arrival','sum'),
        Total_Traded=('Traded','sum'), Avg_Rel=('Reliability','mean')
    ).reset_index()
    past['Revenue'] = (past['Avg_Price']*past['Total_Traded']).round(0)
    past = past.sort_values('Revenue', ascending=False)
    medals = ['🥇','🥈','🥉','4️⃣','5️⃣','6️⃣','7️⃣','8️⃣','9️⃣','🔟']
    past['Rank'] = medals[:len(past)]

    fig_bp = go.Figure(go.Bar(
        x=past['Commodity'], y=past['Revenue'],
        text=[f"₹{v/1e6:.2f}M" for v in past['Revenue']],
        textposition='outside',
        marker=dict(color=past['Revenue'], colorscale='YlGn', showscale=False)
    ))
    dark_layout(fig_bp, f"Revenue Proxy — Last {rep_days} Days", 320)
    st.plotly_chart(fig_bp, use_container_width=True)

    st.dataframe(past[['Rank','Commodity','Avg_Price','Total_Arrivals','Total_Traded','Revenue','Avg_Rel']]
                 .style.format({'Avg_Price':'₹{:.0f}','Total_Arrivals':'{:.0f}',
                                'Total_Traded':'{:.0f}','Revenue':'₹{:.0f}','Avg_Rel':'{:.1f}'})
                 .background_gradient(subset=['Revenue'], cmap='YlGn'),
                 use_container_width=True)

    # Forward forecast
    st.markdown('<div class="section-header">🔮 Forward Sales Forecast — Next 15 Days</div>', unsafe_allow_html=True)
    fwd = []
    for crop in AgriDataEngine.COMMODITIES:
        crop_d = df_full[(df_full['Commodity']==crop)].dropna(subset=ml.feat_cols)
        if crop_d.empty: continue
        lat_r   = crop_d.sort_values('Date').iloc[-1]
        pred_p  = ml.predict_price(lat_r)
        c_seas  = AgriDataEngine.SEASONAL_CURVES[crop][today_date.month-1]
        f_seas  = AgriDataEngine.SEASONAL_CURVES[crop][(today_date.month)%12]
        seas_ch = (f_seas-c_seas)/c_seas*100
        arr_avg = recent[recent['Commodity']==crop]['Arrival'].mean()
        dem_s   = min(arr_avg/800*10, 10)
        p_ch    = (pred_p - lat_r['Price'])/lat_r['Price']*100
        signal  = ('🟢 Strong Buy' if seas_ch>5 and p_ch>0 else
                   ('🟡 Neutral' if abs(seas_ch)<=5 else '🔴 Cautious'))
        fwd.append({'Crop':crop,'Cur Price':round(lat_r['Price'],0),
                    'Pred Price':round(pred_p,0),'Δ%':round(p_ch,1),
                    'Seas Mom%':round(seas_ch,1),'Demand':round(dem_s,1),'Signal':signal})

    fwd_df = pd.DataFrame(fwd).sort_values('Δ%', ascending=False)

    # Grouped bar
    fig_fwd = go.Figure()
    fig_fwd.add_trace(go.Bar(x=fwd_df['Crop'], y=fwd_df['Cur Price'],
                              name='Current', marker_color='#546e7a'))
    fig_fwd.add_trace(go.Bar(x=fwd_df['Crop'], y=fwd_df['Pred Price'],
                              name='Predicted', marker_color=THEME['accent_lime']))
    fig_fwd.update_layout(barmode='group', yaxis_title="₹/Q")
    dark_layout(fig_fwd, "Current vs Predicted Price", 320)
    st.plotly_chart(fig_fwd, use_container_width=True)

    # Seasonal momentum
    fig_mom = px.bar(fwd_df, x='Crop', y='Seas Mom%',
                     color='Seas Mom%', color_continuous_scale='RdYlGn',
                     text=[f"{v:+.1f}%" for v in fwd_df['Seas Mom%']])
    fig_mom.add_hline(y=0, line_dash='dash', line_color=THEME['text_muted'])
    fig_mom.update_traces(textposition='outside')
    dark_layout(fig_mom, "Seasonal Momentum — Next 15 Days", 300)
    st.plotly_chart(fig_mom, use_container_width=True)

    st.dataframe(fwd_df.style
                 .format({'Cur Price':'₹{:.0f}','Pred Price':'₹{:.0f}',
                          'Δ%':'{:+.1f}%','Seas Mom%':'{:+.1f}%','Demand':'{:.1f}'})
                 .background_gradient(subset=['Δ%'], cmap='RdYlGn'),
                 use_container_width=True)

    # Mandi-wise performance
    st.markdown('<div class="section-header">📍 Mandi-wise Performance</div>', unsafe_allow_html=True)
    mp = recent.groupby('Location').agg(
        Avg_Price=('Price','mean'), Total_Arr=('Arrival','sum'),
        Total_Traded=('Traded','sum'), Avg_Rel=('Reliability','mean')
    ).reset_index()
    mp['Sell_Rate%'] = (mp['Total_Traded']/mp['Total_Arr']*100).round(1)
    fig_mp = px.scatter(mp, x='Total_Arr', y='Avg_Price',
                        size='Avg_Rel', color='Sell_Rate%',
                        color_continuous_scale='RdYlGn',
                        hover_name='Location', text='Location')
    fig_mp.update_traces(textposition='top center')
    dark_layout(fig_mp, "Mandi Volume vs Price (size=reliability, colour=sell rate%)", 380)
    st.plotly_chart(fig_mp, use_container_width=True)


    # ── Price Band Analysis ──
    st.markdown('<div class="section-header">📊 Price Band & Support/Resistance</div>', unsafe_allow_html=True)
    pb_hist = raw_df[(raw_df["Commodity"]==sel_item)&(raw_df["Date"]>=today_date-timedelta(days=90))]["Price"]
    p10 = pb_hist.quantile(0.10); p25 = pb_hist.quantile(0.25)
    p50 = pb_hist.quantile(0.50); p75 = pb_hist.quantile(0.75); p90 = pb_hist.quantile(0.90)
    pb1,pb2,pb3,pb4,pb5 = st.columns(5)
    pb1.metric("Support (P10)",    f"₹{p10:.0f}")
    pb2.metric("Lower Band (P25)", f"₹{p25:.0f}")
    pb3.metric("Median (P50)",     f"₹{p50:.0f}")
    pb4.metric("Upper Band (P75)", f"₹{p75:.0f}")
    pb5.metric("Resistance (P90)", f"₹{p90:.0f}")
    band_hist = raw_df[(raw_df["Commodity"]==sel_item)&(raw_df["Location"]==sel_loc)].tail(90)
    fig_band = go.Figure()
    fig_band.add_trace(go.Scatter(x=band_hist["Date"],y=band_hist["Price"],mode="lines",name="Price",
                                   line=dict(color=THEME["accent_lime"],width=1.5)))
    for lvl,nm,clr in [(p10,"Support","#ef5350"),(p25,"P25","#ff8a65"),
                        (p50,"Median","#ffb300"),(p75,"P75","#aed581"),(p90,"Resistance","#4caf50")]:
        fig_band.add_hline(y=lvl, line_dash="dot", line_color=clr,
                           annotation_text=nm, annotation_position="right")
    dark_layout(fig_band, f"Price Bands — {sel_item} @ {sel_loc} (Last 90 Days)", 340)
    st.plotly_chart(fig_band, use_container_width=True)

    # ── Market Absorption Rate ──
    st.markdown('<div class="section-header">📦 Market Absorption Rate</div>', unsafe_allow_html=True)
    abs_df = raw_df[(raw_df["Commodity"]==sel_item)&(raw_df["Location"]==sel_loc)].tail(60).copy()
    abs_df["Absorption"] = abs_df["Traded"]/abs_df["Arrival"]*100
    fig_abs = go.Figure()
    fig_abs.add_trace(go.Scatter(x=abs_df["Date"],y=abs_df["Absorption"],mode="lines",fill="tozeroy",
                                  line=dict(color=THEME["accent_blue"],width=1.5),
                                  fillcolor="rgba(66,165,245,0.1)"))
    fig_abs.add_hline(y=85, line_dash="dash", line_color=THEME["accent_lime"], annotation_text="Target 85%")
    dark_layout(fig_abs, f"Market Absorption Rate — {sel_item} @ {sel_loc}", 280)
    st.plotly_chart(fig_abs, use_container_width=True)

    # ── Procurement Action Summary ──
    st.markdown('<div class="section-header">✅ Procurement Action Plan</div>', unsafe_allow_html=True)
    action_items = []
    if price_delta > 8:
        action_items.append({"Priority":"🔴 HIGH","Action":"Lock forward contract NOW","Reason":f"Price rising +{price_delta:.1f}%","Deadline":"Today"})
    else:
        action_items.append({"Priority":"🟢 LOW","Action":"Monitor daily, no urgent action","Reason":"Stable pricing","Deadline":"Ongoing"})
    if latest_row["Reliability"] < 6:
        action_items.append({"Priority":"🟡 MED","Action":"Diversify to higher reliability mandi","Reason":f"Reliability only {latest_row['Reliability']}/10","Deadline":"3 days"})
    if AgriDataEngine.PERISHABILITY[sel_item] == "High" and logistics_mode != "Reefer (Cold Chain)":
        action_items.append({"Priority":"🟡 MED","Action":"Switch to Reefer cold chain","Reason":"High perishability risk","Deadline":"Before next dispatch"})
    if anomaly_flag:
        action_items.append({"Priority":"🔴 HIGH","Action":"Investigate price anomaly","Reason":"Statistical outlier detected","Deadline":"Immediate"})
    if not action_items:
        action_items.append({"Priority":"🟢 LOW","Action":"Routine monitoring","Reason":"All systems normal","Deadline":"Weekly review"})

    act_df = pd.DataFrame(action_items)
    st.dataframe(act_df, use_container_width=True, hide_index=True)

    # ── Summary Report Card ──
    st.markdown('<div class="section-header">📄 Session Summary Report Card</div>', unsafe_allow_html=True)
    summary_data = {
        "Report Generated":       today_date.strftime("%d %b %Y %H:%M"),
        "Commodity Analysed":     sel_item,
        "Primary Market":         sel_loc,
        "Current Market Price":   f"Rs{latest_row['Price']:,.0f}/Q",
        "Next-Day Prediction":    f"Rs{pred_price:,.0f}/Q ({price_delta:+.1f}%)",
        "Anomaly Detected":       "Yes" if anomaly_flag else "No",
        "Logistics Mode":         logistics_mode,
        "Route Distance":         f"{dist_sim} km",
        "Order Quantity":         f"{f_qty} Q",
        "Logistics Cost":         f"Rs{t_cost:,.0f}",
        "Carbon Footprint":       f"{co2_val:.1f} kg CO2",
        "Mandi Reliability":      f"{latest_row['Reliability']}/10",
        "Weather (Temp/Rain)":    f"{latest_row['Temp']:.1f}C / {latest_row['Rainfall']:.1f}mm",
        "Perishability":          AgriDataEngine.PERISHABILITY[sel_item],
        "Model MAE":              f"Rs{ml.mae:.0f}/Q",
        "Active Alerts":          str(len(alerts)),
    }
    sr_df = pd.DataFrame(list(summary_data.items()), columns=["Parameter","Value"])
    st.dataframe(sr_df, use_container_width=True, hide_index=True)
    st.caption("Export this report using your browser's Print or Save as PDF function.")
    # Daily price line per commodity (sparklines)
    st.markdown('<div class="section-header">📈 Daily Price Trend — All Commodities</div>', unsafe_allow_html=True)
    daily_all = raw_df[raw_df['Date'] >= today_date-timedelta(days=rep_days)].groupby(['Date','Commodity'])['Price'].mean().reset_index()
    fig_daily = px.line(daily_all, x='Date', y='Price', color='Commodity',
                        color_discrete_sequence=px.colors.qualitative.Pastel)
    dark_layout(fig_daily, f"Daily Avg Price — Last {rep_days} Days", 320)
    st.plotly_chart(fig_daily, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — SUPPLY CHAIN OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════
with t_scopt:
    st.markdown('<div class="section-header">🔗 Supply Chain Optimizer</div>', unsafe_allow_html=True)
    st.caption("End-to-end chain: Farm → Mandi → Warehouse → Retailer | Optimise cost, time, CO₂")

    sc1, sc2 = st.columns(2)
    with sc1:
        farm_loc   = st.selectbox("Farm Location (Origin)", sorted(AgriDataEngine.MANDI_PROFILES.keys()), key='farm_loc')
        retail_loc = st.selectbox("Retail Destination", sorted(AgriDataEngine.MANDI_PROFILES.keys()), key='retail_loc')
        sc_crop    = st.selectbox("Crop", AgriDataEngine.COMMODITIES, key='sc_crop')
        sc_qty     = st.number_input("Batch Size (Q)", 1, 5000, 200, key='sc_qty')
    with sc2:
        farm_mode  = st.selectbox("Farm→Mandi Mode", list(LOGISTICS_MODES.keys()), key='fm')
        mandi_mode = st.selectbox("Mandi→Warehouse Mode", list(LOGISTICS_MODES.keys()), key='mm')
        wh_mode    = st.selectbox("Warehouse→Retail Mode", list(LOGISTICS_MODES.keys()), key='wm')
        cold_store = st.checkbox("Include Cold Storage at Warehouse?", value=False)
        wh_days    = st.slider("Warehouse Storage (Days)", 0, 60, 7, key='wh_d') if cold_store else 0

    # Leg distances (simulated fixed)
    leg_dists = {'Farm→Mandi': 80, 'Mandi→Warehouse': 200, 'Warehouse→Retail': 120}
    leg_modes = {'Farm→Mandi': farm_mode, 'Mandi→Warehouse': mandi_mode, 'Warehouse→Retail': wh_mode}

    chain_rows = []
    cumul_cost = 0
    cumul_co2  = 0
    cumul_time = 0
    cumul_spoil = 0

    for leg, dist in leg_dists.items():
        mode_n = leg_modes[leg]
        _, tc, co2, cpq_l, th_l, w = logistics_check(mode_n, sc_crop, sc_qty, dist)
        mf = 0.18 if mode_n=="Reefer (Cold Chain)" else (0.35 if mode_n=="Electric Van (2T)" else 1.0)
        transit_d = th_l / 24
        sp = min(((latest_row['Temp']*0.42)*(transit_d**1.35)*PERISH_FACTOR[AgriDataEngine.PERISHABILITY[sc_crop]]*mf)/10, 30)
        cumul_cost  += tc
        cumul_co2   += co2
        cumul_time  += th_l
        cumul_spoil += sp
        chain_rows.append({'Leg':leg,'Mode':mode_n,'Dist km':dist,
                           'Cost ₹':round(tc,0),'CO₂ kg':round(co2,1),
                           'Time h':round(th_l,1),'Spoilage %':round(sp,1),
                           'Warnings':len(w)})

    # Warehouse storage cost
    wh_cost = wh_days/30 * 65 * sc_qty if cold_store else 0
    wh_spoil = min(PERISH_FACTOR[AgriDataEngine.PERISHABILITY[sc_crop]]*wh_days*0.5/100*100, 50) if cold_store else 0
    if cold_store:
        cumul_cost  += wh_cost
        cumul_spoil += wh_spoil
        chain_rows.append({'Leg':'Warehouse Storage','Mode':'Cold Room','Dist km':0,
                           'Cost ₹':round(wh_cost,0),'CO₂ kg':0,
                           'Time h':wh_days*24,'Spoilage %':round(wh_spoil,1),'Warnings':0})

    chain_df = pd.DataFrame(chain_rows)

    # Summary KPIs
    s1,s2,s3,s4,s5 = st.columns(5)
    s1.metric("Total Chain Cost", f"₹{cumul_cost:,.0f}")
    s2.metric("Total CO₂",        f"{cumul_co2:.1f} kg")
    s3.metric("Total Time",       f"{cumul_time:.0f} h")
    s4.metric("Total Spoilage",   f"{min(cumul_spoil,100):.1f}%")
    s5.metric("Net Saleable Qty", f"{sc_qty*(1-min(cumul_spoil,100)/100):,.0f} Q")

    # Chain breakdown table
    st.dataframe(chain_df.style
                 .format({'Cost ₹':'₹{:,.0f}','CO₂ kg':'{:.1f}','Time h':'{:.1f}','Spoilage %':'{:.1f}%'})
                 .background_gradient(subset=['Spoilage %'], cmap='RdYlGn_r'),
                 use_container_width=True)

    # Sankey diagram
    st.markdown('<div class="section-header">Supply Chain Flow (Sankey)</div>', unsafe_allow_html=True)
    nodes  = ['Farm', 'Mandi', 'Warehouse', 'Retail']
    values = [chain_df.loc[chain_df['Leg']=='Farm→Mandi','Cost ₹'].values[0] if len(chain_df)>0 else 100,
              chain_df.loc[chain_df['Leg']=='Mandi→Warehouse','Cost ₹'].values[0] if len(chain_df)>1 else 100,
              chain_df.loc[chain_df['Leg']=='Warehouse→Retail','Cost ₹'].values[0] if len(chain_df)>2 else 100]
    fig_sk = go.Figure(go.Sankey(
        node=dict(
            label=nodes,
            color=[THEME['accent_lime'],THEME['accent_amber'],THEME['accent_blue'],THEME['accent_red']],
            pad=20, thickness=20
        ),
        link=dict(
            source=[0,1,2],
            target=[1,2,3],
            value=values,
            color=['rgba(165,214,61,0.3)','rgba(255,179,0,0.3)','rgba(66,165,245,0.3)']
        )
    ))
    dark_layout(fig_sk, "Supply Chain Cost Flow (Sankey)", 360)
    st.plotly_chart(fig_sk, use_container_width=True)

    # Cost breakdown donut
    st.markdown('<div class="section-header">Cost Breakdown</div>', unsafe_allow_html=True)
    col_d, col_co2 = st.columns(2)

    with col_d:
        fig_donut = go.Figure(go.Pie(
            labels=chain_df['Leg'], values=chain_df['Cost ₹'],
            hole=0.55,
            marker=dict(colors=[THEME['accent_lime'],THEME['accent_amber'],
                                 THEME['accent_blue'],THEME['accent_red']])
        ))
        dark_layout(fig_donut, "Cost Split by Leg", 300)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_co2:
        fig_co2 = go.Figure(go.Bar(
            x=chain_df['Leg'], y=chain_df['CO₂ kg'],
            marker_color=THEME['accent_amber'],
            text=[f"{v:.1f}" for v in chain_df['CO₂ kg']],
            textposition='outside'
        ))
        dark_layout(fig_co2, "CO₂ Footprint by Leg", 300)
        st.plotly_chart(fig_co2, use_container_width=True)

    # Multi-mode optimiser: find cheapest, fastest, cleanest
    st.markdown('<div class="section-header">⚡ Mode Optimisation Matrix</div>', unsafe_allow_html=True)
    opt_rows = []
    for mn in LOGISTICS_MODES:
        t_c_all = 0; t_co2 = 0; t_time = 0; t_sp = 0
        for leg, dist in leg_dists.items():
            _, tc_l, co2_l, _, th_l, _ = logistics_check(mn, sc_crop, sc_qty, dist)
            mf_l = 0.18 if mn=="Reefer (Cold Chain)" else (0.35 if mn=="Electric Van (2T)" else 1.0)
            td   = th_l/24
            sp_l = min(((latest_row['Temp']*0.42)*(td**1.35)*PERISH_FACTOR[AgriDataEngine.PERISHABILITY[sc_crop]]*mf_l)/10, 30)
            t_c_all += tc_l; t_co2 += co2_l; t_time += th_l; t_sp += sp_l
        opt_rows.append({'Mode':mn,'Total Cost ₹':round(t_c_all,0),
                         'Total CO₂ kg':round(t_co2,1),'Total Time h':round(t_time,1),
                         'Total Spoilage %':round(min(t_sp,100),1)})
    opt_df = pd.DataFrame(opt_rows)

    fig_opt = px.scatter(opt_df, x='Total Cost ₹', y='Total Spoilage %',
                         size='Total CO₂ kg', color='Total Time h',
                         color_continuous_scale='RdYlGn_r',
                         hover_name='Mode', text='Mode')
    fig_opt.update_traces(textposition='top center')
    dark_layout(fig_opt, "Mode Optimisation: Cost vs Spoilage (size=CO₂, colour=time)", 380)
    st.plotly_chart(fig_opt, use_container_width=True)

    st.dataframe(opt_df.style
                 .format({'Total Cost ₹':'₹{:,.0f}','Total CO₂ kg':'{:.1f}',
                          'Total Time h':'{:.1f}','Total Spoilage %':'{:.1f}%'})
                 .background_gradient(subset=['Total Cost ₹'], cmap='RdYlGn')
                 .background_gradient(subset=['Total Spoilage %'], cmap='RdYlGn_r'),
                 use_container_width=True)


    # ── Route Planner ──
    st.markdown('<div class="section-header">🗺️ Multi-Stop Route Planner</div>', unsafe_allow_html=True)
    st.caption("Plan optimal collection routes across mandis before consolidating for delivery.")
    route_stops = st.multiselect("Select Route Stops (Mandis)",
                                  sorted(AgriDataEngine.MANDI_PROFILES.keys()),
                                  default=list(sorted(AgriDataEngine.MANDI_PROFILES.keys()))[:3],
                                  key="route_stops")
    route_mode  = st.selectbox("Route Vehicle Mode", list(LOGISTICS_MODES.keys()), key="route_mode")

    if len(route_stops) >= 2:
        route_rows = []
        total_route_cost = 0; total_route_co2 = 0; total_route_time = 0
        for i in range(len(route_stops)-1):
            leg_name = f"{route_stops[i]} -> {route_stops[i+1]}"
            leg_d    = np.random.randint(80, 350)
            _, r_cost, r_co2, _, r_time, _ = logistics_check(route_mode, sc_crop, max(1,sc_qty//len(route_stops)), leg_d)
            total_route_cost += r_cost; total_route_co2 += r_co2; total_route_time += r_time
            route_rows.append({"Leg":leg_name,"Dist km":leg_d,"Cost":round(r_cost,0),"CO2 kg":round(r_co2,1),"Time h":round(r_time,1)})
        route_df = pd.DataFrame(route_rows)
        rt1,rt2,rt3 = st.columns(3)
        rt1.metric("Route Cost", f"Rs{total_route_cost:,.0f}")
        rt2.metric("Total CO2",  f"{total_route_co2:.1f} kg")
        rt3.metric("Total Time", f"{total_route_time:.1f} h")
        st.dataframe(route_df, use_container_width=True)

        route_locs = [AgriDataEngine.MANDI_PROFILES[m] for m in route_stops]
        route_geo  = pd.DataFrame([{"Location":m,"Lat":p["lat"],"Lon":p["lon"],"Stop":i+1}
                                    for i,(m,p) in enumerate(zip(route_stops,route_locs))])
        fig_route = px.scatter_geo(route_geo, lat="Lat", lon="Lon", text="Location",
                                    size="Stop", scope="asia", center=dict(lat=22, lon=78))
        fig_route.add_trace(go.Scattergeo(
            lat=[p["lat"] for p in route_locs], lon=[p["lon"] for p in route_locs],
            mode="lines", line=dict(color=THEME["accent_lime"],width=2), name="Route"))
        fig_route.update_geos(bgcolor=THEME["bg_card"],landcolor="#1a2e20",
                               oceancolor=THEME["bg_primary"],showocean=True)
        dark_layout(fig_route, "Collection Route Map", 380)
        st.plotly_chart(fig_route, use_container_width=True)

    # ── Carbon Offset Calculator ──
    st.markdown('<div class="section-header">🌿 Carbon Offset Calculator</div>', unsafe_allow_html=True)
    co2_per_tree = 21.0
    trees_needed = cumul_co2 / co2_per_tree
    offset_cost  = trees_needed * 250
    ox1,ox2,ox3 = st.columns(3)
    ox1.metric("Total CO2 Emitted", f"{cumul_co2:.1f} kg")
    ox2.metric("Trees to Offset",   f"{trees_needed:.1f}")
    ox3.metric("Offset Cost (Rs)",  f"Rs{offset_cost:,.0f}")
    bench_df = pd.DataFrame([{"Mode":m,"CO2_per_km":LOGISTICS_MODES[m]["co2"]} for m in LOGISTICS_MODES])
    fig_co2b = px.bar(bench_df, x="Mode", y="CO2_per_km",
                       color="CO2_per_km", color_continuous_scale="RdYlGn_r",
                       text=[f"{v:.1f}" for v in bench_df["CO2_per_km"]])
    fig_co2b.update_traces(textposition="outside")
    dark_layout(fig_co2b, "CO2 Emission per km by Mode", 300)
    st.plotly_chart(fig_co2b, use_container_width=True)

    # ESG Score
    st.markdown('<div class="section-header">🌱 ESG Score Card</div>', unsafe_allow_html=True)
    co2_score  = max(0, 10 - (cumul_co2/50))
    cost_score = max(0, 10 - (cumul_cost/50000))
    sp_score   = max(0, 10 - (min(cumul_spoil,100)/10))
    esg_total  = (co2_score*0.35 + sp_score*0.35 + cost_score*0.30)

    e1,e2,e3,e4 = st.columns(4)
    e1.metric("CO₂ Score",        f"{co2_score:.1f}/10")
    e2.metric("Waste Score",       f"{sp_score:.1f}/10")
    e3.metric("Efficiency Score",  f"{cost_score:.1f}/10")
    e4.metric("ESG Total",         f"{esg_total:.1f}/10",
              delta="Excellent" if esg_total>=7 else ("Fair" if esg_total>=4 else "Poor"))

    fig_esg = go.Figure(go.Indicator(
        mode="gauge+number",
        value=esg_total,
        title={"text":"ESG Score","font":{"family":"Syne","color":THEME['accent_lime'],"size":14}},
        gauge={
            "axis":{"range":[0,10],"tickcolor":THEME['text_muted']},
            "bar":{"color":THEME['accent_lime']},
            "bgcolor":THEME['bg_panel'],
            "bordercolor":THEME['border'],
            "steps":[
                {"range":[0,4],"color":"rgba(239,83,80,0.25)"},
                {"range":[4,7],"color":"rgba(255,179,0,0.25)"},
                {"range":[7,10],"color":"rgba(165,214,61,0.25)"},
            ],
            "threshold":{"line":{"color":THEME['accent_lime'],"width":3},"value":7}
        },
        number={"font":{"family":"Syne","color":THEME['text_primary'],"size":36}}
    ))
    fig_esg.update_layout(plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_PAPER,
                           height=280, margin=dict(l=30,r=30,t=40,b=10))
    st.plotly_chart(fig_esg, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — AI AGENT (Claude-powered, full SCM expert)
# ══════════════════════════════════════════════════════════════════════════════
with t_agent:
    st.markdown('''
    <div style="background:linear-gradient(135deg,#13241a,#1a3a22);border:1px solid #2e4d35;
                border-radius:14px;padding:20px 24px;margin-bottom:20px;">
        <div style="font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;color:#a5d63d;margin-bottom:6px;">
            🤖 Agri-SCM AI Agent — Full Supply Chain Intelligence
        </div>
        <div style="color:#81c784;font-size:12px;line-height:1.7;">
            Powered by Google Gemini with native Google Search. Ask anything about prices, weather, logistics,
            crop recommendations, procurement strategy, spoilage risk, or market trends.
            The agent fetches live commodity prices &amp; weather from the internet and reasons over
            your ML dashboard data to give precise, actionable answers.
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # ── Quick-fire example buttons ────────────────────────────────────────
    st.markdown("**⚡ Quick Questions:**")
    qcols = st.columns(3)
    quick_qs = [
        "What are live Onion prices across major Indian mandis today?",
        "Should I sell my Tomato crop now or store for 30 days?",
        "What is the weather forecast impact on my supply chain?",
        "Which crop gives maximum profit this season?",
        "Compare Heavy Truck vs Reefer for 300km Tomato delivery",
        "Give me a full procurement strategy for 500Q of Rice this week",
        "What spoilage risk do I face shipping Tomato via LCV in this weather?",
        "Which mandi has best price-to-reliability ratio for Onion right now?",
        "Explain the current supply chain risks I should be aware of",
    ]
    for i, q in enumerate(quick_qs):
        with qcols[i % 3]:
            if st.button(q, key=f"qq_{i}", use_container_width=True):
                st.session_state["agent_prefill"] = q

    st.divider()

    # ── API config ────────────────────────────────────────────────────────
    with st.expander("⚙️ API Configuration", expanded=not bool(st.session_state.get("agent_api_key",""))):
        agent_api_key  = st.text_input("Google Gemini API Key", type="password",
                                        placeholder="AIzaSy...",
                                        value=st.session_state.get("agent_api_key",""),
                                        key="api_key_input",
                                        help="Get yours at aistudio.google.com")
        if agent_api_key:
            st.session_state["agent_api_key"] = agent_api_key
        agent_model    = st.selectbox("Model", ["gemini-2.0-flash","gemini-2.5-flash","gemini-1.5-flash"],
                                       key="agent_model_sel")
        agent_max_tok  = st.slider("Max Response Tokens", 500, 4000, 2000, key="agent_max_tok")
        web_search_ena = st.checkbox("🌐 Live Web Search (prices + weather)", value=True, key="ws_ena")
        st.caption("Web search fetches live Indian commodity prices (data.gov.in / agmarknet) and weather from wttr.in")

    # ── Context builder ───────────────────────────────────────────────────
    def fetch_live_weather(location="Mumbai"):
        """Fetch live weather from wttr.in (free, no key needed)."""
        try:
            resp = requests.get(
                f"https://wttr.in/{location}?format=j1",
                timeout=8,
                headers={"User-Agent": "AgriSCM/1.0"}
            )
            if resp.status_code == 200:
                w = resp.json()
                cc = w["current_condition"][0]
                return {
                    "temp_c":        cc.get("temp_C","N/A"),
                    "feels_like_c":  cc.get("FeelsLikeC","N/A"),
                    "humidity":      cc.get("humidity","N/A"),
                    "desc":          cc["weatherDesc"][0]["value"],
                    "wind_kmph":     cc.get("windspeedKmph","N/A"),
                    "precip_mm":     cc.get("precipMM","N/A"),
                    "visibility":    cc.get("visibility","N/A"),
                    "uv_index":      cc.get("uvIndex","N/A"),
                }
        except Exception:
            pass
        return None

    def fetch_live_prices_context():
        """Try to fetch a brief live price snippet from data.gov.in commodity API."""
        try:
            url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
            params = {
                "api-key": "579b464db66ec23bdd000001cdd3946e44ce4adf589fb76b6e571f2",
                "format": "json",
                "limit": "20",
                "filters[commodity]": sel_item
            }
            resp = requests.get(url, params=params, timeout=8)
            if resp.status_code == 200:
                records = resp.json().get("records", [])
                if records:
                    lines = []
                    for r in records[:8]:
                        lines.append(
                            f"  {r.get('market','?')} ({r.get('state','?')}): "
                            f"Min Rs{r.get('min_price','?')} / "
                            f"Max Rs{r.get('max_price','?')} / "
                            f"Modal Rs{r.get('modal_price','?')} — "
                            f"{r.get('arrival_date','?')}"
                        )
                    return "LIVE AGMARKNET PRICES:\n" + "\n".join(lines)
        except Exception:
            pass
        return "Live price API unavailable — using ML model data."

    def build_full_context():
        """Build comprehensive context snapshot for the agent."""
        # All commodity prices
        all_px = raw_df[raw_df["Date"]==today_date].groupby("Commodity")["Price"].mean().to_dict()
        # Cross-mandi for selected item
        mandi_px_30d = raw_df[
            (raw_df["Commodity"]==sel_item) &
            (raw_df["Date"] >= today_date-timedelta(days=30))
        ].groupby("Location")["Price"].mean().to_dict()
        # Arrivals
        mandi_arr = raw_df[raw_df["Date"]==today_date].groupby("Location")["Arrival"].mean().to_dict()
        # Spoilage
        pf_c  = PERISH_FACTOR[AgriDataEngine.PERISHABILITY[sel_item]]
        mf_c  = 0.18 if logistics_mode=="Reefer (Cold Chain)" else 1.0
        sp_3d = min(((latest_row["Temp"]*0.42)*(3**1.35)*pf_c*mf_c)/10, 100)
        sp_7d = min(((latest_row["Temp"]*0.42)*(7**1.35)*pf_c*mf_c)/10, 100)
        # Storage ROI
        md_c  = (pred_price - latest_row["Price"]) / latest_row["Price"]
        fp_30 = latest_row["Price"] * ((1+md_c)**1)
        _, tc_c, co2_c, cpq_c, th_c, _ = logistics_check(logistics_mode, sel_item, f_qty, dist_sim)
        roi_30 = (fp_30*f_qty*0.97 - 65*f_qty/30 - tc_c) - (latest_row["Price"]*f_qty - tc_c)
        # Top mandis
        arb_c = df_full[(df_full["Date"]==today_date)&(df_full["Commodity"]==sel_item)].copy()
        if not arb_c.empty:
            arb_c["VS"] = arb_c["Reliability"] / (arb_c["Price"]+cpq_c) * 1000
            top3m = arb_c.nlargest(3,"VS")[["Location","Price","Reliability"]].to_dict("records")
        else:
            top3m = []
        # Crop recs
        cm = today_date.month-1
        crops_ranked = sorted(
            [(c, AgriDataEngine.SEASONAL_CURVES[c][cm]*10) for c in AgriDataEngine.COMMODITIES],
            key=lambda x: x[1], reverse=True
        )

        ctx = f"""
╔══════════════════════════════════════════════════════════╗
║         LIVE AGRI-SCM INTELLIGENCE CONTEXT               ║
║         {today_date.strftime("%d %b %Y")} | ML Model MAE: Rs{ml.mae:.0f}/Q          ║
╚══════════════════════════════════════════════════════════╝

▌ CURRENT SELECTION
  Commodity      : {sel_item}
  Market         : {sel_loc}
  Logistics      : {logistics_mode}
  Quantity       : {f_qty} Q
  Route Distance : {dist_sim} km
  Perishability  : {AgriDataEngine.PERISHABILITY[sel_item]}

▌ PRICE INTELLIGENCE
  Current Price  : Rs{latest_row["Price"]:,.0f}/Q
  Tomorrow (ML)  : Rs{pred_price:,.0f}/Q  ({price_delta:+.1f}%)
  30-Day Forecast: Rs{forecast_prices[0]:,.0f} → Rs{forecast_prices[-1]:,.0f}/Q
  Anomaly        : {"⚠️ YES — unusual price detected" if anomaly_flag else "✅ Normal"}
  Reliability    : {latest_row["Reliability"]}/10
  Arrivals Today : {latest_row["Arrival"]:,.0f} Q

▌ ALL COMMODITY PRICES TODAY (ML Avg across mandis)
{chr(10).join([f"  {k:<12}: Rs{v:>8,.0f}/Q" for k,v in sorted(all_px.items())])}

▌ {sel_item} PRICE — 30-DAY MANDI COMPARISON
{chr(10).join([f"  {k:<14}: Rs{v:>8,.0f}/Q" for k,v in sorted(mandi_px_30d.items(), key=lambda x:x[1])])}

▌ TOP 3 MANDIS FOR {sel_item} (value score = reliability/price)
{chr(10).join([f"  {i+1}. {m['Location']:<14} Price Rs{m['Price']:,.0f}  Reliability {m['Reliability']}/10" for i,m in enumerate(top3m)])}

▌ LOGISTICS ANALYSIS
  Total Cost     : Rs{tc_c:,.0f}
  Cost per Q     : Rs{cpq_c:,.0f}
  Transit Time   : {th_c:.1f} hours
  CO2 Footprint  : {co2_c:.1f} kg

▌ SPOILAGE RISK
  After 3 days   : {sp_3d:.1f}%
  After 7 days   : {sp_7d:.1f}%
  Reefer reduces spoilage by ~80%

▌ STORAGE DECISION
  30-Day Hold ROI: Rs{roi_30 :,.0f}  → {"✅ HOLD" if roi_30>0 else "❌ SELL NOW"}
  Net Profit Now : Rs{latest_row["Price"]*f_qty-tc_c:,.0f}

▌ CURRENT WEATHER (at {sel_loc})
  Temperature    : {latest_row["Temp"]:.1f}°C
  Humidity       : {latest_row["Humidity"]:.1f}%
  Rainfall       : {latest_row["Rainfall"]:.1f} mm
  Wind           : {latest_row["Wind"]:.1f} kph
  Frost Event    : {"YES ❄️" if latest_row["Frost"] else "No"}

▌ ACTIVE ALERTS ({len(alerts)} total)
{chr(10).join(["  ["+t.upper()+"] "+m for t,m in alerts]) if alerts else "  None — all clear"}

▌ CROP SEASONAL RANKINGS (current month)
{chr(10).join([f"  {i+1}. {c:<12} Score: {s:.1f}/10" for i,(c,s) in enumerate(crops_ranked[:5])])}

▌ LOGISTICS MODE COMPARISON (for {dist_sim}km, {f_qty}Q)
{chr(10).join([f"  {mn:<26}: Rs{dist_sim*mv['cost_km']:>7,.0f}  CO2:{dist_sim/max(mv['eff'],1)*mv['co2']:>5.1f}kg  {dist_sim/mv['speed_kmh']:>4.1f}h" for mn,mv in LOGISTICS_MODES.items()])}
"""
        return ctx

    # ── Agent API call ────────────────────────────────────────────────────
    def call_gemini_agent(question, api_key, model_name, web_search):
        from google import genai as new_genai
        from google.genai import types as genai_types

        client = new_genai.Client(api_key=api_key)
        ctx    = build_full_context()

        system_instruction = """You are an elite Agricultural Supply Chain Management AI Agent for Indian markets.
You have access to a live ML dashboard with real-time price predictions, spoilage models, and logistics analysis.
Use Google Search to find current Indian commodity prices, weather, news, and market conditions.
Rules:
1. Give SPECIFIC, ACTIONABLE answers with exact numbers from the data provided.
2. When comparing options, pick a clear winner with reasoning.
3. Structure answers: verdict first, then detailed reasoning.
4. Quote prices, percentages, and costs from the live data snapshot.
5. Be direct — traders and farmers need decisions, not essays.
6. Use Rs for rupees, Q for quintals."""

        user_msg = f"LIVE ML DATA CONTEXT:\n{ctx}\n\nUSER QUESTION:\n{question}\n\nUse Google Search for live Indian market prices, weather, and news relevant to this question."

        tools = [genai_types.Tool(google_search=genai_types.GoogleSearch())] if web_search else []

        # Build chat history
        history_genai = []
        for msg in st.session_state.get("agent_msgs_v2", []):
            role = "user" if msg["role"] == "user" else "model"
            history_genai.append(genai_types.Content(
                role=role,
                parts=[genai_types.Part(text=msg["content"])]
            ))

        config = genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=st.session_state.get("agent_max_tok", 2000),
            temperature=0.7,
            tools=tools if tools else None,
        )

        try:
            chat = client.chats.create(model=model_name, history=history_genai, config=config)
            response = chat.send_message(user_msg)
            return response.text
        except Exception as e:
            return f"❌ Gemini API Error: {str(e)}"

    # ── Chat UI ───────────────────────────────────────────────────────────
    if "agent_msgs_v2" not in st.session_state:
        st.session_state["agent_msgs_v2"] = []
    if "agent_prefill" not in st.session_state:
        st.session_state["agent_prefill"] = ""

    # Chat history display
    for msg in st.session_state["agent_msgs_v2"]:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🌾"):
                st.markdown(msg["content"])

    # Input — pre-filled from quick buttons
    prefill_val = st.session_state.pop("agent_prefill", "")
    user_q = st.chat_input("Ask the Agri-SCM AI Agent anything...",
                            key="agent_chat_input")

    # Handle prefill from buttons via session state trick
    if prefill_val and not user_q:
        user_q = prefill_val

    if user_q:
        key_val = st.session_state.get("agent_api_key","").strip()
        if not key_val:
            st.warning("⚙️ Please enter your Google Gemini API key in the API Configuration section above.")
        else:
            st.session_state["agent_msgs_v2"].append({"role":"user","content":user_q})
            with st.chat_message("user"):
                st.write(user_q)

            with st.chat_message("assistant", avatar="🌾"):
                with st.spinner("🤖 Fetching live data + reasoning..."):
                    try:
                        answer = call_gemini_agent(
                            user_q,
                            key_val,
                            st.session_state.get("agent_model_sel","gemini-2.0-flash"),
                            st.session_state.get("ws_ena", True)
                        )
                    except requests.exceptions.Timeout:
                        answer = "⏱️ Request timed out (90s). Try a shorter question or disable web search."
                    except Exception as e:
                        answer = f"❌ Error: {str(e)}"
                st.markdown(answer)
            st.session_state["agent_msgs_v2"].append({"role":"assistant","content":answer})

    col_cl, col_ctx2 = st.columns([1,4])
    with col_cl:
        if st.button("🗑️ Clear Chat", key="clear_chat_v2"):
            st.session_state["agent_msgs_v2"] = []
            st.rerun()
    with col_ctx2:
        if st.button("📋 View Live Data Context", key="ctx_btn_v2"):
            with st.expander("What the agent sees", expanded=True):
                st.code(build_full_context(), language="text")


    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # LOCAL INTELLIGENCE ENGINE — No API Key Required
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0d1b2e,#1a2e3a);border:1px solid #2e4d35;
                border-radius:14px;padding:18px 22px;margin:16px 0;">
        <div style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;color:#42a5f5;margin-bottom:6px;">
            🧠 Local Intelligence Engine — No API Key Needed
        </div>
        <div style="color:#81c784;font-size:12px;line-height:1.7;">
            Fully offline. Analyses your live ML dashboard data directly — prices, spoilage,
            logistics, forecasts, risk scores — and gives structured answers using built-in
            reasoning rules. Works without any external API or internet connection.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Local question engine ────────────────────────────────────────────
    def local_agent_answer(question: str) -> str:
        """
        Deep analytical engine — mines raw_df, df_full, ML predictions,
        and all computed metrics to answer any SCM question.
        """
        import re as _re
        q = question.lower().strip()

        # ══════════════════════════════════════════════════════════════════
        # CORE DATA MINING — compute everything upfront
        # ══════════════════════════════════════════════════════════════════
        today   = today_date
        all_px  = raw_df[raw_df["Date"]==today].groupby("Commodity")["Price"].mean().to_dict()
        all_arr = raw_df[raw_df["Date"]==today].groupby("Commodity")["Arrival"].mean().to_dict()
        all_trd = raw_df[raw_df["Date"]==today].groupby("Commodity")["Traded"].mean().to_dict()

        # Logistics
        _, tc, co2, cpq, th, mode_warns = logistics_check(logistics_mode, sel_item, f_qty, dist_sim)
        pf  = PERISH_FACTOR[AgriDataEngine.PERISHABILITY[sel_item]]
        mf  = 0.18 if logistics_mode == "Reefer (Cold Chain)" else 1.0
        sp3 = min(((latest_row["Temp"]*0.42)*(3**1.35)*pf*mf)/10, 100)
        sp7 = min(((latest_row["Temp"]*0.42)*(7**1.35)*pf*mf)/10, 100)

        # Price momentum
        md    = (pred_price - latest_row["Price"]) / latest_row["Price"]
        fp30  = latest_row["Price"] * ((1+md)**1)
        fp60  = latest_row["Price"] * ((1+md)**2)
        fp90  = latest_row["Price"] * ((1+md)**3)
        net_now = latest_row["Price"]*f_qty - tc
        roi30   = (fp30*f_qty*0.97 - 65*f_qty/30 - tc) - net_now
        roi60   = (fp60*f_qty*0.94 - 65*f_qty/15 - tc) - net_now
        roi90   = (fp90*f_qty*0.91 - 65*f_qty/10 - tc) - net_now

        # Cross mandi arbitrage
        arb = df_full[(df_full["Date"]==today)&(df_full["Commodity"]==sel_item)].copy()
        if not arb.empty:
            arb["Landed"] = arb["Price"] + cpq
            arb["VS"]     = (arb["Reliability"] / arb["Landed"] * 1000).round(2)
            arb_sorted    = arb.sort_values("VS", ascending=False)
            best_m        = arb_sorted.iloc[0]
            worst_m       = arb_sorted.iloc[-1]
        else:
            arb_sorted = best_m = worst_m = None

        # Historical sales (all time)
        hist = raw_df[raw_df["Commodity"]==sel_item].copy()
        hist_loc = hist[hist["Location"]==sel_loc].copy()

        # Volatility
        vol_map = {}
        for c in AgriDataEngine.COMMODITIES:
            cd = raw_df[raw_df["Commodity"]==c]["Price"]
            vol_map[c] = cd.std()/cd.mean()*100 if len(cd)>1 else 0

        # Seasonal crop rankings
        cm = today.month - 1
        crops_ranked = sorted(
            [(c, AgriDataEngine.SEASONAL_CURVES[c][cm]*10) for c in AgriDataEngine.COMMODITIES],
            key=lambda x: x[1], reverse=True
        )

        # Revenue analysis
        def revenue_stats(df_in, period_label):
            if df_in.empty:
                return {}
            rev = (df_in["Price"] * df_in["Traded"]).sum()
            avg_p = df_in["Price"].mean()
            tot_arr = df_in["Arrival"].sum()
            tot_trd = df_in["Traded"].sum()
            sell_rate = tot_trd/tot_arr*100 if tot_arr>0 else 0
            return {"revenue": rev, "avg_price": avg_p, "arrivals": tot_arr,
                    "traded": tot_trd, "sell_rate": sell_rate, "label": period_label}

        r7   = revenue_stats(hist_loc[hist_loc["Date"] >= today-timedelta(days=7)],  "Last 7 Days")
        r30  = revenue_stats(hist_loc[hist_loc["Date"] >= today-timedelta(days=30)], "Last 30 Days")
        r90  = revenue_stats(hist_loc[hist_loc["Date"] >= today-timedelta(days=90)], "Last 90 Days")
        r365 = revenue_stats(hist_loc[hist_loc["Date"] >= today-timedelta(days=365)],"Last 12 Months")

        # Best selling periods
        hist_copy = hist.copy()
        hist_copy["Year"]  = hist_copy["Date"].dt.year
        hist_copy["Month"] = hist_copy["Date"].dt.month
        monthly = hist_copy.groupby(["Year","Month"]).agg(
            AvgPrice=("Price","mean"), Arrivals=("Arrival","sum"), Traded=("Traded","sum")
        ).reset_index()
        monthly["RevenueProxy"] = monthly["AvgPrice"] * monthly["Traded"]
        best_month  = monthly.nlargest(1,"RevenueProxy").iloc[0] if len(monthly)>0 else None
        worst_month = monthly.nsmallest(1,"RevenueProxy").iloc[0] if len(monthly)>0 else None

        # Commodity revenue comparison (last 30d)
        cut30 = raw_df[raw_df["Date"] >= today-timedelta(days=30)]
        rev_by_crop = cut30.groupby("Commodity").apply(
            lambda x: (x["Price"]*x["Traded"]).sum()
        ).sort_values(ascending=False)

        # Sell rate by mandi (last 30d)
        sell_rate_mandi = cut30.groupby("Location").apply(
            lambda x: x["Traded"].sum()/x["Arrival"].sum()*100 if x["Arrival"].sum()>0 else 0
        ).sort_values(ascending=False)

        # Price forecast series (30 steps)
        fc_prices = ml.price_forecast_series(latest_row, steps=90)
        fc_dates  = [today + timedelta(days=i+1) for i in range(90)]

        # Future revenue estimate
        def future_rev(days):
            fp = fc_prices[min(days-1, len(fc_prices)-1)]
            sp = min(PERISH_FACTOR[AgriDataEngine.PERISHABILITY[sel_item]]*days*0.5/100, 0.95)
            sq = f_qty*(1-sp)
            sc = (days/30)*65*f_qty if days>7 else 0
            return fp*sq - sc - tc

        # Price support/resistance
        p10 = hist_loc["Price"].quantile(0.10) if len(hist_loc)>0 else 0
        p25 = hist_loc["Price"].quantile(0.25) if len(hist_loc)>0 else 0
        p50 = hist_loc["Price"].quantile(0.50) if len(hist_loc)>0 else 0
        p75 = hist_loc["Price"].quantile(0.75) if len(hist_loc)>0 else 0
        p90 = hist_loc["Price"].quantile(0.90) if len(hist_loc)>0 else 0

        # ══════════════════════════════════════════════════════════════════
        # INTENT DETECTION — broad keyword matching
        # ══════════════════════════════════════════════════════════════════
        lines = []

        # ── PAST SALES / REVENUE HISTORY ─────────────────────────────────
        if any(x in q for x in ["past sales","past revenue","historical","history","previous","last week",
                                  "last month","last year","how much did","how many","sold before",
                                  "sales data","sales report","revenue report","sold"]):
            lines += [
                f"## 📈 Historical Sales & Revenue — {sel_item} @ {sel_loc}",
                "",
                "### Revenue Summary by Period",
                "| Period | Avg Price | Total Arrivals | Total Traded | Sell Rate | Revenue Proxy |",
                "|--------|-----------|---------------|--------------|-----------|---------------|",
            ]
            for r in [r7, r30, r90, r365]:
                if r:
                    lines.append(
                        f"| {r['label']} | Rs{r['avg_price']:,.0f} | {r['arrivals']:,.0f}Q |"
                        f" {r['traded']:,.0f}Q | {r['sell_rate']:.1f}% | Rs{r['revenue']/1e6:.2f}M |"
                    )

            lines += ["", "### Best & Worst Sales Periods"]
            if best_month is not None:
                import calendar
                bm_name = calendar.month_abbr[int(best_month["Month"])]
                wm_name = calendar.month_abbr[int(worst_month["Month"])]
                lines += [
                    f"- 🏆 **Best Month:** {bm_name} {int(best_month['Year'])} — "
                    f"Avg Rs{best_month['AvgPrice']:,.0f}/Q, Revenue Rs{best_month['RevenueProxy']/1e6:.2f}M",
                    f"- 📉 **Worst Month:** {wm_name} {int(worst_month['Year'])} — "
                    f"Avg Rs{worst_month['AvgPrice']:,.0f}/Q, Revenue Rs{worst_month['RevenueProxy']/1e6:.2f}M",
                ]

            lines += [
                "",
                "### All-Commodity Revenue Ranking (Last 30 Days)",
                "| Rank | Commodity | Revenue Proxy | % of Total |",
                "|------|-----------|--------------|------------|",
            ]
            total_rev = rev_by_crop.sum()
            medals = ["🥇","🥈","🥉","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣","🔟"]
            for i, (crop, rev) in enumerate(rev_by_crop.items()):
                pct = rev/total_rev*100 if total_rev>0 else 0
                lines.append(f"| {medals[i]} | {crop} | Rs{rev/1e6:.2f}M | {pct:.1f}% |")

        # ── FUTURE REVENUE / FORECAST ─────────────────────────────────────
        elif any(x in q for x in ["future","forecast","expected","predict","next","upcoming","projection",
                                    "will sell","how much will","expected revenue","future revenue",
                                    "what will","price in","30 day","60 day","90 day"]):
            lines += [
                f"## 🔮 Future Revenue Forecast — {sel_item} @ {sel_loc}",
                f"*Based on ML Gradient Boosting model (MAE: Rs{ml.mae:.0f}/Q)*",
                "",
                "### Price Forecast (ML Model)",
                "| Horizon | Predicted Price | Change | Est. Revenue ({f_qty}Q) | ROI vs Today |",
                "|---------|----------------|--------|------------------------|--------------|",
            ]
            for days, label in [(1,"Tomorrow"),(7,"1 Week"),(14,"2 Weeks"),
                                  (30,"1 Month"),(60,"2 Months"),(90,"3 Months")]:
                fp = fc_prices[min(days-1, len(fc_prices)-1)]
                chg = (fp - latest_row["Price"])/latest_row["Price"]*100
                est_rev = future_rev(days)
                roi     = est_rev - net_now
                lines.append(
                    f"| {label} | Rs{fp:,.0f} | {chg:+.1f}% |"
                    f" Rs{est_rev:,.0f} | Rs{roi:+,.0f} |"
                )

            lines += [
                "",
                "### Seasonal Price Outlook (Next 12 Months)",
                "| Month | Seasonal Index | Expected Price | Signal |",
                "|-------|---------------|----------------|--------|",
            ]
            import calendar
            for i in range(12):
                mon_idx  = (today.month - 1 + i) % 12
                seas     = AgriDataEngine.SEASONAL_CURVES[sel_item][mon_idx]
                exp_p    = latest_row["Price"] * seas
                mon_name = calendar.month_abbr[mon_idx+1]
                signal   = "🟢 High" if seas >= 1.2 else ("🟡 Medium" if seas >= 0.9 else "🔴 Low")
                lines.append(f"| {mon_name} | {seas:.2f}x | Rs{exp_p:,.0f} | {signal} |")

            lines += [
                "",
                f"**Best Month to Sell:** {calendar.month_abbr[AgriDataEngine.SEASONAL_CURVES[sel_item].index(max(AgriDataEngine.SEASONAL_CURVES[sel_item]))+1]}",
                f"**Worst Month to Sell:** {calendar.month_abbr[AgriDataEngine.SEASONAL_CURVES[sel_item].index(min(AgriDataEngine.SEASONAL_CURVES[sel_item]))+1]}",
            ]

        # ── SELL RATE / MARKET ABSORPTION ────────────────────────────────
        elif any(x in q for x in ["sell rate","selling rate","absorption","how fast","traded","turnover",
                                    "how quickly","market rate","offtake","sold rate","clearance"]):
            lines += [
                f"## 📊 Sell Rate & Market Absorption — {sel_item}",
                "",
                "### Sell Rate by Mandi (Last 30 Days)",
                "| Mandi | Sell Rate % | Total Arrived | Total Traded | Avg Price |",
                "|-------|-------------|---------------|--------------|-----------|",
            ]
            for loc in sell_rate_mandi.index:
                loc_df = cut30[cut30["Location"]==loc]
                arr  = loc_df["Arrival"].sum()
                trd  = loc_df["Traded"].sum()
                rate = trd/arr*100 if arr>0 else 0
                avgp = loc_df["Price"].mean()
                tag  = " ✅" if rate == sell_rate_mandi.max() else ""
                lines.append(f"| {loc}{tag} | {rate:.1f}% | {arr:,.0f}Q | {trd:,.0f}Q | Rs{avgp:,.0f} |")

            lines += [
                "",
                "### Sell Rate by Commodity (Last 30 Days)",
                "| Commodity | Sell Rate % | Revenue Proxy |",
                "|-----------|-------------|---------------|",
            ]
            for c in AgriDataEngine.COMMODITIES:
                c_df = cut30[cut30["Commodity"]==c]
                arr  = c_df["Arrival"].sum()
                trd  = c_df["Traded"].sum()
                rate = trd/arr*100 if arr>0 else 0
                rev  = (c_df["Price"]*c_df["Traded"]).sum()
                lines.append(f"| {c} | {rate:.1f}% | Rs{rev/1e6:.2f}M |")

            lines += [
                "",
                f"**{sel_item} Current Sell Rate @ {sel_loc}:**",
                f"- Today: {latest_row['Traded']:,.0f}Q traded / {latest_row['Arrival']:,.0f}Q arrived = "
                f"{latest_row['Traded']/latest_row['Arrival']*100:.1f}%",
                f"- Market signal: {'🟢 Strong demand' if latest_row['Traded']/latest_row['Arrival']>0.85 else ('🟡 Moderate' if latest_row['Traded']/latest_row['Arrival']>0.70 else '🔴 Weak demand')}",
            ]

        # ── WHERE TO SELL / BEST MARKET ───────────────────────────────────
        elif any(x in q for x in ["where to sell","where should","best market","which market","which mandi",
                                    "best place","sell where","which location","best location","where can i sell",
                                    "best mandi","at what rate","which rate","what rate","best rate"]):
            lines += [
                f"## 🏪 Where & At What Rate to Sell — {sel_item}",
                "",
                "### Full Mandi Comparison",
                "| Rank | Mandi | State | Price | Reliability | Sell Rate | Landed Cost | Value Score | Verdict |",
                "|------|-------|-------|-------|-------------|-----------|-------------|-------------|---------|",
            ]
            if arb_sorted is not None:
                for i, (_, row) in enumerate(arb_sorted.iterrows()):
                    loc_df   = cut30[cut30["Location"]==row["Location"]]
                    arr      = loc_df["Arrival"].sum()
                    trd      = loc_df["Traded"].sum()
                    s_rate   = trd/arr*100 if arr>0 else 0
                    state    = AgriDataEngine.MANDI_PROFILES.get(row["Location"],{}).get("state","?")
                    verdict  = "🥇 BEST" if i==0 else ("🥈" if i==1 else ("🥉" if i==2 else "—"))
                    lines.append(
                        f"| {i+1} | {row['Location']} | {state} |"
                        f" Rs{row['Price']:,.0f} | {row['Reliability']}/10 |"
                        f" {s_rate:.1f}% | Rs{row['Landed']:,.0f} | {row['VS']:.2f} | {verdict} |"
                    )

                lines += [
                    "",
                    f"### 🎯 Recommendation for {f_qty}Q of {sel_item}",
                    f"",
                    f"**Sell at: {best_m['Location']}**",
                    f"- **Price:** Rs{best_m['Price']:,.0f}/Q",
                    f"- **Reliability:** {best_m['Reliability']}/10",
                    f"- **Landed Cost (your end):** Rs{best_m['Landed']:,.0f}/Q",
                    f"- **Net Revenue:** Rs{(best_m['Price']-cpq)*f_qty:,.0f}",
                    f"- **Why:** Highest value score = best reliability per rupee spent on transport",
                    "",
                    f"**Avoid: {worst_m['Location']}** — lowest value score ({worst_m['VS']:.2f})",
                    "",
                    f"### Optimal Selling Rate",
                    f"| Timing | Price | Action |",
                    f"|--------|-------|--------|",
                    f"| Today (Spot) | Rs{latest_row['Price']:,.0f}/Q | {'✅ Sell' if roi30<=0 else '⏳ Wait'} |",
                    f"| Tomorrow (ML) | Rs{pred_price:,.0f}/Q | {'+' if price_delta>0 else ''}{price_delta:.1f}% |",
                    f"| 30-Day Forward | Rs{fp30:,.0f}/Q | {'✅ Better' if roi30>0 else '❌ Worse than today'} |",
                    f"| Price Support | Rs{p25:,.0f}/Q | Don't sell below this |",
                    f"| Price Resistance | Rs{p75:,.0f}/Q | Target this for forward contracts |",
                ]

        # ── SELL vs STORE ─────────────────────────────────────────────────
        elif any(x in q for x in ["sell","store","hold","storage","wait","keep","when to sell","should i sell"]):
            verdict = "✅ HOLD" if roi30 > 0 else "❌ SELL NOW"
            lines += [
                f"## 📦 Sell vs Store Decision — {sel_item} ({f_qty}Q)",
                f"**Verdict: {verdict}**",
                "",
                "### Revenue Comparison",
                "| Option | Gross Revenue | Costs | Net Profit | vs Today |",
                "|--------|--------------|-------|------------|---------|",
                f"| Sell Today | Rs{latest_row['Price']*f_qty:,.0f} | Rs{tc:,.0f} | Rs{net_now:,.0f} | Baseline |",
                f"| Hold 30 Days | Rs{fp30*f_qty*0.97:,.0f} | Rs{tc+65*f_qty/30:,.0f} | Rs{net_now+roi30:,.0f} | Rs{roi30:+,.0f} |",
                f"| Hold 60 Days | Rs{fp60*f_qty*0.94:,.0f} | Rs{tc+65*f_qty/15:,.0f} | Rs{net_now+roi60:,.0f} | Rs{roi60:+,.0f} |",
                f"| Hold 90 Days | Rs{fp90*f_qty*0.91:,.0f} | Rs{tc+65*f_qty/10:,.0f} | Rs{net_now+roi90:,.0f} | Rs{roi90:+,.0f} |",
                "",
                "### Spoilage Risk",
                "| Days Stored | Spoilage % | Quantity Lost | Revenue Lost |",
                "|-------------|------------|---------------|--------------|",
            ]
            for d in [7,14,30,60,90]:
                sp = min(PERISH_FACTOR[AgriDataEngine.PERISHABILITY[sel_item]]*d*0.5/100, 0.95)
                ql = f_qty*sp
                rl = ql*latest_row["Price"]
                lines.append(f"| {d} days | {sp*100:.1f}% | {ql:.0f}Q | Rs{rl:,.0f} |")

            lines += [
                "",
                f"**Best Action:** {'Hold for maximum ' + str(30 if roi30>=roi60 else (60 if roi60>=roi90 else 90)) + ' days' if max(roi30,roi60,roi90)>0 else 'Sell today — all storage options lose money'}",
            ]

        # ── PRICE ANALYSIS ────────────────────────────────────────────────
        elif any(x in q for x in ["price","rate","cost","worth","value","expensive","cheap","how much"]):
            lines += [
                f"## 💰 Complete Price Intelligence — {sel_item}",
                "",
                "### Current Snapshot",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Spot Price ({sel_loc}) | Rs{latest_row['Price']:,.0f}/Q |",
                f"| Tomorrow (ML) | Rs{pred_price:,.0f}/Q ({price_delta:+.1f}%) |",
                f"| 30-Day Forecast | Rs{fp30:,.0f}/Q |",
                f"| Support (P25) | Rs{p25:,.0f}/Q |",
                f"| Fair Value (P50) | Rs{p50:,.0f}/Q |",
                f"| Resistance (P75) | Rs{p75:,.0f}/Q |",
                f"| Historical Low (P10) | Rs{p10:,.0f}/Q |",
                f"| Historical High (P90) | Rs{p90:,.0f}/Q |",
                f"| Current vs Fair Value | {((latest_row['Price']-p50)/p50*100):+.1f}% |",
                f"| Anomaly | {'⚠️ Unusual' if anomaly_flag else '✅ Normal'} |",
                "",
                "### All Commodity Prices Today",
                "| Commodity | Price | 30d Trend | Volatility | Signal |",
                "|-----------|-------|-----------|------------|--------|",
            ]
            for c in AgriDataEngine.COMMODITIES:
                cp = all_px.get(c, 0)
                c_hist = raw_df[raw_df["Commodity"]==c]
                trend  = (c_hist["Price"].iloc[-1]-c_hist["Price"].iloc[-30])/c_hist["Price"].iloc[-30]*100 if len(c_hist)>30 else 0
                vol    = vol_map.get(c, 0)
                sig    = "🟢" if trend>5 else ("🔴" if trend<-5 else "🟡")
                lines.append(f"| {c} | Rs{cp:,.0f} | {trend:+.1f}% | {vol:.1f}% | {sig} |")

            lines += [
                "",
                "### Price Across All Mandis (30-Day Avg)",
                "| Mandi | Avg Price | vs Median | Sell Rate |",
                "|-------|-----------|-----------|-----------|",
            ]
            for loc, px in sorted(
                raw_df[(raw_df["Commodity"]==sel_item)&(raw_df["Date"]>=today-timedelta(days=30))]
                .groupby("Location")["Price"].mean().items(), key=lambda x: x[1], reverse=True
            ):
                vs_med = (px-p50)/p50*100
                sr_df  = cut30[cut30["Location"]==loc]
                sr     = sr_df["Traded"].sum()/sr_df["Arrival"].sum()*100 if sr_df["Arrival"].sum()>0 else 0
                lines.append(f"| {loc} | Rs{px:,.0f} | {vs_med:+.1f}% | {sr:.1f}% |")

        # ── LOGISTICS ─────────────────────────────────────────────────────
        elif any(x in q for x in ["logistic","transport","truck","reefer","rail","mode","delivery",
                                    "ship","transit","cold","lcv","electric","vehicle","cost to transport"]):
            lines += [
                f"## 🚛 Complete Logistics Analysis — {sel_item}, {f_qty}Q over {dist_sim}km",
                "",
                "### All Modes Comparison",
                "| Mode | Total Cost | Per Quintal | Transit | CO₂ kg | Spoilage% | Suitability | Best For |",
                "|------|-----------|-------------|---------|--------|-----------|-------------|---------|",
            ]
            for mn, mv in LOGISTICS_MODES.items():
                sc, tc_m, co2_m, cpq_m, th_m, warns_m = logistics_check(mn, sel_item, f_qty, dist_sim)
                mf_m  = 0.18 if mn=="Reefer (Cold Chain)" else (0.35 if mn=="Electric Van (2T)" else 1.0)
                sp_m  = min(((latest_row["Temp"]*0.42)*((th_m/24)**1.35)*pf*mf_m)/10, 100)
                best_for = {"Kisan Rail (Train)":"Bulk grains, long haul",
                            "Heavy Truck (16T)":"Large volumes 200-600km",
                            "LCV (3T)":"Small loads, urban last-mile",
                            "Reefer (Cold Chain)":"Perishables, premium produce",
                            "Drone Delivery":"Samples, emergency micro-delivery",
                            "Electric Van (2T)":"Urban, eco-friendly short haul"}.get(mn,"General")
                active = " ✅" if mn==logistics_mode else ""
                lines.append(
                    f"| {mn}{active} | Rs{tc_m:,.0f} | Rs{cpq_m:.0f} |"
                    f" {th_m:.1f}h | {co2_m:.1f} | {sp_m:.1f}% | {sc}/10 | {best_for} |"
                )

            lines += [
                "",
                f"### Current Selection: {logistics_mode}",
                f"- **Total Cost:** Rs{tc:,.0f}",
                f"- **Cost per Quintal:** Rs{cpq:,.0f}",
                f"- **Transit Time:** {th:.1f} hours",
                f"- **CO₂ Footprint:** {co2:.1f} kg",
                f"- **Spoilage Risk (transit):** {sp3:.1f}%",
            ]
            if mode_warns:
                lines += ["", "**⚠️ Issues with current mode:**"]
                for w in mode_warns:
                    lines.append(f"- {w}")

            # Recommendation
            perish = AgriDataEngine.PERISHABILITY[sel_item]
            if perish == "High":
                rec = "Reefer (Cold Chain)"
                reason = "Reduces spoilage by 80% for highly perishable cargo"
            elif dist_sim > 800:
                rec = "Kisan Rail (Train)"
                reason = "Most cost-efficient for long-distance bulk transport"
            elif f_qty > 160:
                rec = "Heavy Truck (16T)"
                reason = "High capacity, good for large orders"
            else:
                rec = "LCV (3T)"
                reason = "Cost-effective for smaller loads"

            _, tc_rec, _, cpq_rec, th_rec, _ = logistics_check(rec, sel_item, f_qty, dist_sim)
            lines += [
                "",
                f"### 🎯 Recommendation: **{rec}**",
                f"- Reason: {reason}",
                f"- Cost: Rs{tc_rec:,.0f} (Rs{cpq_rec:.0f}/Q)",
                f"- Transit: {th_rec:.1f}h",
                f"- Savings vs current: Rs{tc-tc_rec:+,.0f}",
            ]

        # ── SPOILAGE ──────────────────────────────────────────────────────
        elif any(x in q for x in ["spoilage","spoil","rot","perish","waste","damage","fresh","shelf life"]):
            lines += [
                f"## 🦠 Complete Spoilage Analysis — {sel_item}",
                "",
                f"**Perishability Class:** {AgriDataEngine.PERISHABILITY[sel_item]}",
                f"**Current Conditions:** {latest_row['Temp']:.1f}°C | {latest_row['Humidity']:.1f}% RH | {latest_row['Rainfall']:.1f}mm rain",
                "",
                "### Spoilage by Transit Duration (All Modes)",
                "| Days | Ambient | Reefer | Electric Van | Loss (Ambient) | Loss (Reefer) |",
                "|------|---------|--------|-------------|---------------|--------------|",
            ]
            for d in [1,2,3,5,7,10,14]:
                sp_amb = min(((latest_row["Temp"]*0.42)*(d**1.35)*pf*1.0)/10, 100)
                sp_ref = min(((latest_row["Temp"]*0.42)*(d**1.35)*pf*0.18)/10, 100)
                sp_ev  = min(((latest_row["Temp"]*0.42)*(d**1.35)*pf*0.35)/10, 100)
                loss_a = sp_amb/100*f_qty*latest_row["Price"]
                loss_r = sp_ref/100*f_qty*latest_row["Price"]
                lines.append(
                    f"| {d}d | {sp_amb:.1f}% | {sp_ref:.1f}% | {sp_ev:.1f}% |"
                    f" Rs{loss_a:,.0f} | Rs{loss_r:,.0f} |"
                )

            lines += [
                "",
                "### Spoilage by Commodity (3-Day Transit, Ambient)",
                "| Commodity | Perishability | 3d Spoilage | Revenue Risk |",
                "|-----------|--------------|-------------|--------------|",
            ]
            for c in AgriDataEngine.COMMODITIES:
                pf_c  = PERISH_FACTOR[AgriDataEngine.PERISHABILITY[c]]
                sp_c  = min(((latest_row["Temp"]*0.42)*(3**1.35)*pf_c*1.0)/10, 100)
                risk  = sp_c/100*f_qty*all_px.get(c,0)
                lines.append(
                    f"| {c} | {AgriDataEngine.PERISHABILITY[c]} |"
                    f" {sp_c:.1f}% | Rs{risk:,.0f} |"
                )

        # ── CROP RECOMMENDATION ───────────────────────────────────────────
        elif any(x in q for x in ["crop","plant","grow","recommend","which crop","best crop","cultivat",
                                    "what to grow","what should i plant","most profitable crop"]):
            lines += [
                f"## 🌱 Comprehensive Crop Recommendations — {today.strftime('%B %Y')}",
                "",
                "### Full Scoring Matrix",
                "| Rank | Crop | Yield Score | Seasonal | Price Trend | Volatility | Revenue/Q | Overall | Signal |",
                "|------|------|------------|----------|-------------|------------|-----------|---------|--------|",
            ]
            medals = ["🥇","🥈","🥉","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣","🔟"]
            for i, (c, s) in enumerate(crops_ranked):
                c_df   = raw_df[raw_df["Commodity"]==c]
                trend  = (c_df["Price"].iloc[-1]-c_df["Price"].iloc[-30])/c_df["Price"].iloc[-30]*100 if len(c_df)>30 else 0
                vol    = vol_map.get(c, 0)
                rev_q  = all_px.get(c,0) * (1-PERISH_FACTOR[AgriDataEngine.PERISHABILITY[c]]*0.01)
                yld_s  = BASE_YIELD_Q[c]/max(BASE_YIELD_Q.values())*10
                signal = "🟢 Plant" if s>=7 else ("🟡 Consider" if s>=5 else "🔴 Skip")
                lines.append(
                    f"| {medals[i]} | {c} | {yld_s:.1f} | {s:.1f}/10 |"
                    f" {trend:+.1f}% | {vol:.1f}% | Rs{rev_q:,.0f} | {(s+yld_s)/2:.1f} | {signal} |"
                )

            top_crop = crops_ranked[0][0]
            lines += [
                "",
                f"### 🏆 Top Pick: {top_crop}",
                f"- Seasonal Score: {crops_ranked[0][1]:.1f}/10",
                f"- Current Price: Rs{all_px.get(top_crop,0):,.0f}/Q",
                f"- Base Yield: {BASE_YIELD_Q[top_crop]} Q/acre",
                f"- Perishability: {AgriDataEngine.PERISHABILITY[top_crop]}",
                f"- Revenue potential (10 acres): Rs{all_px.get(top_crop,0)*BASE_YIELD_Q[top_crop]*10:,.0f}",
                "",
                "### Month-by-Month Seasonal Outlook",
                "| Month | " + " | ".join(c for c,_ in crops_ranked[:5]) + " |",
                "|-------|" + "|".join(["------"]*5) + "|",
            ]
            import calendar
            for mon_i in range(12):
                mon_name = calendar.month_abbr[mon_i+1]
                row_vals = []
                for c, _ in crops_ranked[:5]:
                    s = AgriDataEngine.SEASONAL_CURVES[c][mon_i]
                    emoji = "🟢" if s>=1.2 else ("🟡" if s>=0.9 else "🔴")
                    row_vals.append(f"{emoji}{s:.1f}x")
                lines.append(f"| {mon_name} | " + " | ".join(row_vals) + " |")

        # ── SUPPLY CHAIN RISKS ────────────────────────────────────────────
        elif any(x in q for x in ["risk","aware","concern","threat","challenge","disruption","supply chain risk"]):
            risk_score = min(int(latest_row["Temp"]/5 + latest_row["Rainfall"]/10), 10)
            lines += [
                f"## ⚠️ Complete Supply Chain Risk Assessment",
                f"**Overall Risk: {'🔴 HIGH' if risk_score>=7 else ('🟡 MEDIUM' if risk_score>=4 else '🟢 LOW')} ({risk_score}/10)**",
                "",
                "### Risk Matrix",
                "| Risk Category | Level | Score | Impact | Mitigation |",
                "|---------------|-------|-------|--------|------------|",
                f"| Price Volatility | {'🔴' if vol_map.get(sel_item,0)>25 else ('🟡' if vol_map.get(sel_item,0)>15 else '🟢')} | {vol_map.get(sel_item,0):.1f}% CV | Revenue uncertainty | Forward contracts |",
                f"| Anomaly | {'🔴' if anomaly_flag else '🟢'} | {'High' if anomaly_flag else 'Low'} | Price manipulation risk | Investigate immediately |",
                f"| Temperature | {'🔴' if latest_row['Temp']>38 else ('🟡' if latest_row['Temp']>34 else '🟢')} | {latest_row['Temp']:.1f}°C | Spoilage acceleration | Cold chain |",
                f"| Rainfall | {'🔴' if latest_row['Rainfall']>30 else ('🟡' if latest_row['Rainfall']>15 else '🟢')} | {latest_row['Rainfall']:.1f}mm | Supply disruption | Expedite dispatch |",
                f"| Frost | {'🔴' if latest_row['Frost'] else '🟢'} | {'Present' if latest_row['Frost'] else 'None'} | Crop damage | Row covers |",
                f"| Perishability | {'🔴' if AgriDataEngine.PERISHABILITY[sel_item]=='High' else ('🟡' if AgriDataEngine.PERISHABILITY[sel_item]=='Medium' else '🟢')} | {AgriDataEngine.PERISHABILITY[sel_item]} | Transit loss | Reefer logistics |",
                f"| Mandi Reliability | {'🔴' if latest_row['Reliability']<5 else ('🟡' if latest_row['Reliability']<7 else '🟢')} | {latest_row['Reliability']}/10 | Supply disruption | Multi-mandi sourcing |",
                f"| Price Forecast | {'🔴' if price_delta>15 else ('🟡' if price_delta>8 else ('🟢' if price_delta>-5 else '🔴'))} | {price_delta:+.1f}% | {('Surge — buy now' if price_delta>8 else ('Drop — sell now' if price_delta<-8 else 'Stable'))} | {'Forward contract' if price_delta>8 else 'Monitor daily'} |",
                "",
                "### Active Alerts",
            ]
            if alerts:
                for a_type, a_msg in alerts:
                    emoji = {"critical":"🔴","warning":"🟡","success":"🟢","info":"🔵"}.get(a_type,"⚪")
                    lines.append(f"- {emoji} {a_msg}")
            else:
                lines.append("- ✅ No active alerts")

            lines += ["", "### Prioritised Action Plan"]
            actions = []
            if anomaly_flag:
                actions.append("1. 🔍 **URGENT** — Investigate price anomaly immediately")
            if price_delta > 10:
                actions.append(f"2. 📜 Lock forward contract — price rising {price_delta:.1f}%")
            if latest_row["Temp"]>35 and AgriDataEngine.PERISHABILITY[sel_item]=="High":
                actions.append("3. ❄️ Switch to Reefer cold chain immediately")
            if latest_row["Rainfall"]>20:
                actions.append("4. 🚛 Expedite all dispatches — flood risk within 48h")
            if latest_row["Reliability"]<6:
                actions.append("5. 🏪 Diversify to higher-reliability mandis")
            if not actions:
                actions.append("✅ All risk indicators normal — continue routine operations")
            lines.extend(actions)

        # ── PROFIT / MARGIN ANALYSIS ──────────────────────────────────────
        elif any(x in q for x in ["profit","margin","income","earn","revenue","money","return","roi",
                                    "how much can","net","gross","p&l","breakeven","break even"]):
            lines += [
                f"## 💰 Complete Profit & Margin Analysis — {sel_item} ({f_qty}Q)",
                "",
                "### P&L Breakdown (Sell Today)",
                f"| Item | Amount |",
                f"|------|--------|",
                f"| Gross Revenue | Rs{latest_row['Price']*f_qty:,.0f} |",
                f"| Transport Cost | Rs{-tc:,.0f} |",
                f"| Spoilage Loss (est 2%) | Rs{-latest_row['Price']*f_qty*0.02:,.0f} |",
                f"| **Net Profit** | **Rs{net_now:,.0f}** |",
                f"| Net per Quintal | Rs{net_now/f_qty:,.0f} |",
                "",
                "### Scenario Comparison",
                "| Scenario | Net Revenue | Margin % | vs Today |",
                "|----------|------------|---------|---------|",
                f"| Sell Today | Rs{net_now:,.0f} | {net_now/(latest_row['Price']*f_qty)*100:.1f}% | — |",
                f"| Sell Tomorrow | Rs{pred_price*f_qty-tc:,.0f} | {(pred_price*f_qty-tc)/(pred_price*f_qty)*100:.1f}% | Rs{pred_price*f_qty-tc-net_now:+,.0f} |",
                f"| Hold 30 Days | Rs{net_now+roi30:,.0f} | {(net_now+roi30)/(fp30*f_qty)*100:.1f}% | Rs{roi30:+,.0f} |",
                f"| Hold 60 Days | Rs{net_now+roi60:,.0f} | {(net_now+roi60)/(fp60*f_qty)*100:.1f}% | Rs{roi60:+,.0f} |",
                "",
                "### Profit by Commodity (Sell Today, Same Qty)",
                "| Commodity | Price | Gross | Transport | Net | Margin% |",
                "|-----------|-------|-------|-----------|-----|---------|",
            ]
            for c in sorted(AgriDataEngine.COMMODITIES, key=lambda x: all_px.get(x,0)*f_qty-tc, reverse=True):
                cp    = all_px.get(c,0)
                gross = cp*f_qty
                _, tc_c,_,_,_,_ = logistics_check(logistics_mode, c, f_qty, dist_sim)
                net_c = gross - tc_c
                mg    = net_c/gross*100 if gross>0 else 0
                lines.append(f"| {c} | Rs{cp:,.0f} | Rs{gross:,.0f} | Rs{tc_c:,.0f} | Rs{net_c:,.0f} | {mg:.1f}% |")

        # ── FULL DASHBOARD / GENERAL ───────────────────────────────────────
        else:
            import calendar
            lines += [
                f"## 📊 Complete SCM Intelligence Report",
                f"**{sel_item} @ {sel_loc} | {today.strftime('%d %b %Y')} | {f_qty}Q via {logistics_mode}**",
                "",
                "### 🔑 Key Metrics",
                f"| Metric | Value | Status |",
                f"|--------|-------|--------|",
                f"| Spot Price | Rs{latest_row['Price']:,.0f}/Q | {'⚠️ Anomaly' if anomaly_flag else '✅ Normal'} |",
                f"| Tomorrow ML Forecast | Rs{pred_price:,.0f}/Q | {price_delta:+.1f}% |",
                f"| Net Profit Today | Rs{net_now:,.0f} | {'✅' if net_now>0 else '❌'} |",
                f"| 30-Day Storage ROI | Rs{roi30:,.0f} | {'✅ Hold' if roi30>0 else '❌ Sell now'} |",
                f"| Spoilage (3 days) | {sp3:.1f}% | {'🔴' if sp3>20 else ('🟡' if sp3>10 else '🟢')} |",
                f"| Logistics Cost | Rs{tc:,.0f} | Rs{cpq:.0f}/Q |",
                f"| Best Mandi | {best_m['Location'] if best_m is not None else sel_loc} | Score {best_m['VS']:.2f} if best_m is not None else '' |",
                f"| Top Crop This Season | {crops_ranked[0][0]} | Score {crops_ranked[0][1]:.1f}/10 |",
                "",
                "### 📈 Revenue History (This Location)",
                "| Period | Avg Price | Traded | Revenue |",
                "|--------|-----------|--------|---------|",
            ]
            for r in [r7, r30, r90, r365]:
                if r:
                    lines.append(f"| {r['label']} | Rs{r['avg_price']:,.0f} | {r['traded']:,.0f}Q | Rs{r['revenue']/1e6:.2f}M |")

            lines += [
                "",
                "### 🔮 Price Forecast",
                "| Horizon | Price | Change |",
                "|---------|-------|--------|",
            ]
            for days, label in [(1,"Tomorrow"),(7,"1 Week"),(30,"1 Month"),(90,"3 Months")]:
                fp = fc_prices[min(days-1, len(fc_prices)-1)]
                chg = (fp-latest_row["Price"])/latest_row["Price"]*100
                lines.append(f"| {label} | Rs{fp:,.0f} | {chg:+.1f}% |")

            lines += [
                "",
                "### ⚠️ Active Alerts",
            ]
            if alerts:
                for a_type, a_msg in alerts:
                    emoji = {"critical":"🔴","warning":"🟡","success":"🟢","info":"🔵"}.get(a_type,"⚪")
                    lines.append(f"- {emoji} {a_msg}")
            else:
                lines.append("- ✅ All clear")

            lines += [
                "",
                "### 🌱 Top 3 Crops This Season",
            ]
            for c, s in crops_ranked[:3]:
                lines.append(f"- **{c}** — Score {s:.1f}/10 | Price Rs{all_px.get(c,0):,.0f}/Q")

        return "\n".join(lines)


    # ── Local agent chat UI ───────────────────────────────────────────────
    st.markdown("**💡 Quick Questions (No API Key):**")
    local_qcols = st.columns(3)
    local_qs = [
        "What are the supply chain risks right now?",
        "Should I sell or store my crop?",
        "Which crop is most profitable this season?",
        "What is the spoilage risk for my shipment?",
        "Which mandi should I buy from?",
        "Give me a full profit analysis",
        "What is the weather impact on my supply chain?",
        "Compare all logistics modes for my route",
        "Give me the full dashboard report",
    ]
    for i, q in enumerate(local_qs):
        with local_qcols[i % 3]:
            if st.button(q, key=f"lq_{i}", use_container_width=True):
                st.session_state["local_prefill"] = q

    if "local_msgs" not in st.session_state:
        st.session_state["local_msgs"] = []
    if "local_prefill" not in st.session_state:
        st.session_state["local_prefill"] = ""

    # Display local chat history
    for msg in st.session_state["local_msgs"]:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🧠"):
                st.markdown(msg["content"])

    local_prefill = st.session_state.pop("local_prefill", "")
    local_q = st.chat_input("Ask the Local Intelligence Engine (no API key needed)...",
                             key="local_chat_input")
    if local_prefill and not local_q:
        local_q = local_prefill

    if local_q:
        st.session_state["local_msgs"].append({"role":"user","content":local_q})
        with st.chat_message("user"):
            st.write(local_q)
        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("🧠 Analysing your ML data..."):
                local_answer = local_agent_answer(local_q)
            st.markdown(local_answer)
        st.session_state["local_msgs"].append({"role":"assistant","content":local_answer})

    if st.button("🗑️ Clear Local Chat", key="clear_local"):
        st.session_state["local_msgs"] = []
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# FLOATING CHAT WIDGET — Gemini AI (appears on every tab)
# ══════════════════════════════════════════════════════════════════════════════
import streamlit.components.v1 as components
_WIDGET_HTML = '<!DOCTYPE html>\n<html>\n<head>\n<style>\n  body{margin:0;background:transparent;font-family:\'IBM Plex Mono\',monospace}\n  #fb{position:fixed;bottom:20px;right:20px;z-index:9999;background:linear-gradient(135deg,#4caf50,#a5d63d);color:#0d1b0f;border:none;border-radius:50px;padding:13px 20px;font-weight:800;font-size:13px;cursor:pointer;box-shadow:0 4px 20px rgba(165,214,61,0.45);transition:all 0.2s}\n  #fb:hover{transform:translateY(-3px);box-shadow:0 8px 28px rgba(165,214,61,0.6)}\n  #fp{position:fixed;bottom:72px;right:20px;z-index:9998;width:370px;max-height:520px;display:none;flex-direction:column;background:#13241a;border:1px solid #2e4d35;border-radius:16px;box-shadow:0 12px 40px rgba(0,0,0,0.7);overflow:hidden}\n  #fp.open{display:flex}\n  .hd{background:linear-gradient(135deg,#1a3a22,#2e4d35);padding:12px 16px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #2e4d35}\n  .ht{font-family:Syne,sans-serif;font-weight:800;color:#a5d63d;font-size:13px}\n  .hc{cursor:pointer;color:#81c784;font-size:17px;padding:2px 6px;border-radius:5px}\n  .hc:hover{background:#2e4d35;color:#e8f5e9}\n  .ak{padding:8px 12px;border-bottom:1px solid #2e4d35;display:flex;gap:6px;align-items:center}\n  .akl{font-size:10px;color:#81c784;white-space:nowrap}\n  #ki{flex:1;background:#1a2e20;border:1px solid #2e4d35;border-radius:6px;padding:6px 9px;color:#e8f5e9;font-size:11px;outline:none}\n  #ki:focus{border-color:#4caf50}\n  #ks{background:#2e4d35;border:none;border-radius:6px;padding:5px 9px;color:#a5d63d;font-size:11px;cursor:pointer;font-weight:700;white-space:nowrap}\n  #ks:hover{background:#4caf50;color:#0d1b0f}\n  #ms{flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:8px}\n  .mu{align-self:flex-end;background:#1a3a22;border:1px solid #2e4d35;border-radius:12px 12px 2px 12px;padding:9px 13px;max-width:88%;font-size:11px;color:#e8f5e9}\n  .ma{align-self:flex-start;background:#0d1b0f;border:1px solid #2e4d35;border-radius:12px 12px 12px 2px;padding:9px 13px;max-width:92%;font-size:11px;color:#e8f5e9;white-space:pre-wrap}\n  .lb{font-size:10px;font-weight:700;margin-bottom:3px}\n  .lu{color:#a5d63d} .la{color:#42a5f5}\n  .ty{color:#81c784;font-style:italic;font-size:11px}\n  .ia{padding:10px;border-top:1px solid #2e4d35;display:flex;gap:7px;align-items:flex-end}\n  #ti{flex:1;background:#1a2e20;border:1px solid #2e4d35;border-radius:8px;padding:8px 11px;color:#e8f5e9;font-family:\'IBM Plex Mono\',monospace;font-size:11px;outline:none;resize:none}\n  #ti:focus{border-color:#4caf50}\n  #gb{background:#4caf50;color:#0d1b0f;border:none;border-radius:8px;padding:8px 13px;cursor:pointer;font-weight:800;font-size:13px}\n  #gb:hover{background:#a5d63d}\n  #gb:disabled{background:#2e4d35;color:#81c784;cursor:default}\n</style>\n</head>\n<body>\n<button id="fb" onclick="tog()">✨ Gemini AI</button>\n<div id="fp">\n  <div class="hd"><div class="ht">🤖 Agri-SCM Gemini Agent</div><span class="hc" onclick="tog()">✕</span></div>\n  <div class="ak"><span class="akl">API Key:</span><input id="ki" type="password" placeholder="AIzaSy..."/><button id="ks" onclick="svk()">Save</button></div>\n  <div id="ms"><div class="ma"><div class="lb la">✨ Gemini</div>Hi! Enter your Google Gemini API key above, then ask me anything about prices, logistics, spoilage, crop strategy or market trends!</div></div>\n  <div class="ia"><textarea id="ti" rows="2" placeholder="Ask about prices, crops, logistics..."></textarea><button id="gb" onclick="snd()">⚡</button></div>\n</div>\n<script>\nvar op=false,hs=[],ak=localStorage.getItem("agri_gemini_key")||"";\nif(ak)document.getElementById("ki").value=ak;\ndocument.getElementById("ti").addEventListener("keydown",function(e){if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();snd();}});\nfunction tog(){op=!op;document.getElementById("fp").className=op?"open":"";}\nfunction svk(){ak=document.getElementById("ki").value.trim();localStorage.setItem("agri_gemini_key",ak);var b=document.getElementById("ks");b.textContent="Saved!";setTimeout(function(){b.textContent="Save";},1500);}\nfunction am(r,t){var box=document.getElementById("ms");var d=document.createElement("div");d.className=r==="user"?"mu":"ma";var l=document.createElement("div");l.className="lb "+(r==="user"?"lu":"la");l.textContent=r==="user"?"You":"✨ Gemini";var b=document.createElement("div");b.textContent=t;d.appendChild(l);d.appendChild(b);box.appendChild(d);box.scrollTop=box.scrollHeight;}\nfunction sty(){var box=document.getElementById("ms");var d=document.createElement("div");d.id="ty";d.className="ma";d.innerHTML=\'<div class="lb la">✨ Gemini</div><div class="ty">Searching Google & reasoning...</div>\';box.appendChild(d);box.scrollTop=box.scrollHeight;}\nfunction hty(){var e=document.getElementById("ty");if(e)e.remove();}\nasync function snd(){\n  var inp=document.getElementById("ti"),q=inp.value.trim();\n  if(!q)return;\n  if(!ak){am("agent","⚠️ Please enter and save your Google Gemini API key above.");return;}\n  inp.value="";am("user",q);hs.push({role:"user",content:q});\n  document.getElementById("gb").disabled=true;sty();\n  try{\n    var contents=hs.map(function(h){return{role:h.role==="user"?"user":"model",parts:[{text:h.content}]};});\n    var r=await fetch("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key="+ak,{\n      method:"POST",\n      headers:{"Content-Type":"application/json"},\n      body:JSON.stringify({\n        systemInstruction:{parts:[{text:"You are an expert Agricultural Supply Chain AI Agent for Indian markets. Answer questions about prices, logistics, crop recommendations, spoilage, and market trends. Use Google Search for live Indian commodity prices and weather. Be concise and actionable."}]},\n        contents:contents,\n        tools:[{googleSearch:{}}]\n      })\n    });\n    var data=await r.json();\n    if(data.error){hty();am("agent","API Error: "+data.error.message);document.getElementById("gb").disabled=false;return;}\n    var text="No response received.";\n    if(data.candidates&&data.candidates[0]){text=data.candidates[0].content.parts.filter(function(p){return p.text;}).map(function(p){return p.text;}).join("\n");}\n    hty();am("agent",text);hs.push({role:"model",content:text});\n  }catch(e){hty();am("agent","Error: "+e.message);}\n  document.getElementById("gb").disabled=false;\n}\n</script>\n</body>\n</html>'

components.html(_WIDGET_HTML, height=0, scrolling=False)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown(f"""
<div style="text-align:center;color:{THEME['text_muted']};font-size:11px;padding:8px 0;">
    🌾 Agri-SCM Intelligence Command Center v8.0 &nbsp;|&nbsp;
    Gradient Boosting · KMeans · IsolationForest · PCA · Ridge · Gemini AI Agent · Floating Chat Widget &nbsp;|&nbsp;
    10 Crops · 12 Mandis · 3 Years · {len(raw_df):,} Records &nbsp;|&nbsp;
    Model MAE ₹{ml.mae:.0f}/Q
</div>
""", unsafe_allow_html=True)

# NOTE: appended sections marker — do not remove
