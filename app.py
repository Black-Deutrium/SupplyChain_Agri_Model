import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. SYSTEM CONFIGURATION & UI STYLING
# ==========================================
st.set_page_config(
    page_title="Agri-SCM Intelligence Hub v4",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main { background: #f8fafb; }
.stMetric { border: 1px solid #e0e7ef; padding: 15px; border-radius: 12px; background: #ffffff; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
.stTabs [aria-selected="true"] { background-color: #2e7d32 !important; color: white !important; border-radius: 8px; }
.alert-critical { padding: 14px 18px; border-radius: 10px; margin-bottom: 16px; border-left: 6px solid #d32f2f; background: #fff5f5; }
.alert-warning  { padding: 14px 18px; border-radius: 10px; margin-bottom: 16px; border-left: 6px solid #f57c00; background: #fff8e1; }
.alert-info     { padding: 14px 18px; border-radius: 10px; margin-bottom: 16px; border-left: 6px solid #1565c0; background: #e8f0fe; }
.insight-card   { background:#ffffff; border-radius:12px; padding:14px 18px; margin:6px 0; border:1px solid #e0e7ef; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. REALISTIC DATA ENGINE
# ==========================================
class AgriDataEngine:

    # Seasonal price multipliers per commodity (monthly index, Jan=0)
    SEASONAL_CURVES = {
        'Onion':  [1.3, 1.4, 1.2, 0.9, 0.7, 0.8, 1.0, 1.1, 1.0, 0.9, 1.0, 1.2],
        'Tomato': [0.8, 0.7, 0.9, 1.3, 1.5, 1.4, 1.1, 0.9, 0.8, 0.9, 1.0, 0.9],
        'Potato': [1.1, 1.2, 1.0, 0.8, 0.7, 0.9, 1.1, 1.2, 1.1, 0.9, 0.8, 1.0],
        'Wheat':  [1.0, 1.0, 1.1, 0.8, 0.7, 0.8, 1.0, 1.1, 1.1, 1.0, 1.0, 1.0],
        'Rice':   [1.0, 1.0, 1.0, 1.0, 1.1, 1.2, 1.1, 0.9, 0.8, 0.9, 1.0, 1.0],
    }

    # Mandi-specific reliability, capacity, and typical price premium
    MANDI_PROFILES = {
        'Azadpur':   {'reliability': (7, 9),  'cap_factor': 1.3, 'price_premium': 1.04},
        'Vashi':     {'reliability': (7, 10), 'cap_factor': 1.4, 'price_premium': 1.06},
        'Bengaluru': {'reliability': (6, 9),  'cap_factor': 1.1, 'price_premium': 1.03},
        'Pune':      {'reliability': (5, 8),  'cap_factor': 1.0, 'price_premium': 1.00},
        'Nashik':    {'reliability': (5, 8),  'cap_factor': 1.0, 'price_premium': 0.98},
        'Jaipur':    {'reliability': (4, 7),  'cap_factor': 0.8, 'price_premium': 0.96},
        'Nagpur':    {'reliability': (4, 7),  'cap_factor': 0.9, 'price_premium': 0.97},
        'Lucknow':   {'reliability': (5, 8),  'cap_factor': 1.0, 'price_premium': 1.01},
        'Ahmedabad': {'reliability': (6, 9),  'cap_factor': 1.1, 'price_premium': 1.02},
    }

    BASE_PRICES = {'Onion': 2200, 'Tomato': 1700, 'Potato': 1400, 'Wheat': 2400, 'Rice': 3200}

    @staticmethod
    @st.cache_data
    def generate_industry_data():
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", end="2025-12-31", freq='D')
        commodities = list(AgriDataEngine.BASE_PRICES.keys())
        locations   = list(AgriDataEngine.MANDI_PROFILES.keys())
        rows = []

        for date in dates:
            doy = date.dayofyear
            mon = date.month - 1  # 0-indexed
            year_drift = (date.year - 2023) * 0.05  # 5% annual inflation

            diesel = 88 + np.sin(doy / 365 * np.pi) * 4 + (date.year - 2023) * 2
            temp   = 22 + np.sin(doy / 365 * 2 * np.pi) * 12 + np.random.normal(0, 1.5)
            hum    = 55 + np.cos(doy / 365 * 2 * np.pi) * 25 + np.random.normal(0, 4)
            # Monsoon rainfall pulse
            rain   = (np.random.exponential(2) if 5 <= date.month <= 9 else np.random.exponential(0.3))

            for loc in locations:
                profile = AgriDataEngine.MANDI_PROFILES[loc]
                rel = np.random.randint(*profile['reliability'])

                for item in commodities:
                    base_p = AgriDataEngine.BASE_PRICES[item]
                    seasonal = AgriDataEngine.SEASONAL_CURVES[item][mon]

                    # Price shock: rare supply disruptions
                    shock = np.random.choice([1.0, 1.25, 0.80], p=[0.93, 0.04, 0.03])

                    # Weather effect on price (perishables respond more)
                    weather_effect = 1 + (max(rain - 5, 0) * 0.01 * {'Tomato': 3, 'Onion': 2, 'Potato': 1, 'Wheat': 0.3, 'Rice': 0.3}[item])

                    price = (base_p * seasonal * shock * weather_effect * profile['price_premium']
                             * (1 + year_drift) + np.random.normal(0, base_p * 0.02))

                    arrival = (800 * profile['cap_factor'] * (1 / seasonal)
                               + np.random.normal(0, 100))
                    arrival = max(arrival, 50)

                    rows.append([date, item, loc, round(price, 2), round(arrival, 1),
                                 round(rain, 2), round(temp, 1), round(hum, 1),
                                 round(diesel, 2), rel])

        df = pd.DataFrame(rows, columns=['Date', 'Commodity', 'Location', 'Price', 'Arrival',
                                         'Rainfall', 'Temp', 'Humidity', 'Diesel', 'Reliability'])
        return df


# ==========================================
# 3. SMARTER ML PIPELINE
# ==========================================
class SCM_Predictor:
    def __init__(self, df):
        self.df    = df.sort_values(['Commodity', 'Location', 'Date']).reset_index(drop=True)
        self.model = GradientBoostingRegressor(n_estimators=120, learning_rate=0.08,
                                               max_depth=4, subsample=0.85, random_state=42)
        self.le_loc  = LabelEncoder()
        self.le_comp = LabelEncoder()
        self.trained = False

    def _add_features(self, df):
        g = df.groupby(['Commodity', 'Location'])

        # Lag features
        df['Price_L1'] = g['Price'].shift(1)
        df['Price_L7'] = g['Price'].shift(7)

        # Rolling statistics
        df['Price_Roll7_Mean'] = g['Price'].transform(lambda x: x.shift(1).rolling(7).mean())
        df['Price_Roll7_Std']  = g['Price'].transform(lambda x: x.shift(1).rolling(7).std())
        df['Arrival_Roll3']    = g['Arrival'].transform(lambda x: x.shift(1).rolling(3).mean())

        # Seasonal encoding
        df['Sin_DOY'] = np.sin(2 * np.pi * df['Date'].dt.dayofyear / 365)
        df['Cos_DOY'] = np.cos(2 * np.pi * df['Date'].dt.dayofyear / 365)
        df['Month']   = df['Date'].dt.month

        # Encoded categoricals
        df['Loc_Enc']  = self.le_loc.fit_transform(df['Location'])
        df['Item_Enc'] = self.le_comp.fit_transform(df['Commodity'])

        # Target: next-day price
        df['Target'] = g['Price'].shift(-1)
        return df

    def prepare_and_train(self):
        df = self._add_features(self.df.copy())
        self.feat_cols = [
            'Price_L1', 'Price_L7', 'Price_Roll7_Mean', 'Price_Roll7_Std',
            'Arrival_Roll3', 'Temp', 'Humidity', 'Diesel', 'Rainfall',
            'Sin_DOY', 'Cos_DOY', 'Month', 'Loc_Enc', 'Item_Enc'
        ]
        train_df = df.dropna(subset=self.feat_cols + ['Target'])
        # Train on 2023-2024 only; leave 2025 for "live" data
        mask = train_df['Date'].dt.year < 2025
        X_train, y_train = train_df.loc[mask, self.feat_cols], train_df.loc[mask, 'Target']
        X_val,   y_val   = train_df.loc[~mask, self.feat_cols], train_df.loc[~mask, 'Target']

        self.model.fit(X_train, y_train)
        val_preds = self.model.predict(X_val)
        self.mae  = mean_absolute_error(y_val, val_preds)
        self.trained = True
        return df

    def predict_next(self, row):
        feat = row[self.feat_cols].values.reshape(1, -1)
        return self.model.predict(feat)[0]

    def feature_importance(self):
        return pd.Series(self.model.feature_importances_, index=self.feat_cols).sort_values(ascending=False)


# ==========================================
# 4. LOGISTICS ENGINE
# ==========================================
LOGISTICS_MODES = {
    "Kisan Rail (Train)":   {"eff": 40,  "cost_km": 8,  "co2": 0.8, "capacity_qt": 5000, "perishable_ok": False},
    "Heavy Truck (16T)":    {"eff": 4,   "cost_km": 45, "co2": 2.6, "capacity_qt": 160,  "perishable_ok": True},
    "LCV (3T)":             {"eff": 12,  "cost_km": 20, "co2": 1.5, "capacity_qt": 30,   "perishable_ok": True},
    "Reefer (Cold Chain)":  {"eff": 3,   "cost_km": 65, "co2": 3.2, "capacity_qt": 80,   "perishable_ok": True},
}

PERISHABILITY = {'Tomato': 'High', 'Onion': 'Medium', 'Potato': 'Low', 'Wheat': 'None', 'Rice': 'None'}
PERISHABILITY_FACTOR = {'High': 1.6, 'Medium': 0.9, 'Low': 0.4, 'None': 0.1}

def logistics_suitability(mode_name, commodity, qty_qt, dist_km):
    mode = LOGISTICS_MODES[mode_name]
    warnings_list = []
    score = 10

    perish = PERISHABILITY[commodity]
    if perish == 'High' and mode_name == "Kisan Rail (Train)":
        warnings_list.append("⚠️ Train not recommended for highly perishable cargo — consider Reefer.")
        score -= 3
    if qty_qt > mode['capacity_qt']:
        n_vehicles = int(np.ceil(qty_qt / mode['capacity_qt']))
        warnings_list.append(f"ℹ️ Load exceeds single vehicle capacity. {n_vehicles} vehicles needed.")
        score -= 1
    if dist_km > 800 and mode_name == "LCV (3T)":
        warnings_list.append("⚠️ LCV inefficient beyond 800 km — switch to Heavy Truck or Rail.")
        score -= 2

    total_cost = dist_km * mode['cost_km']
    co2 = (dist_km / mode['eff']) * mode['co2']
    return score, total_cost, co2, warnings_list


# ==========================================
# 5. SMART ALERT ENGINE
# ==========================================
def generate_alerts(price_delta, rainfall, temp, commodity, mode_name, qty_qt, dist_km):
    alerts = []
    perish = PERISHABILITY[commodity]

    if price_delta > 12:
        alerts.append(("critical", f"🔴 Price Surge Alert: +{price_delta:.1f}% forecast for {commodity}. "
                        "Consider accelerating procurement or switching to alternate mandi."))
    elif price_delta < -10:
        alerts.append(("info", f"🟢 Price Drop Opportunity: {price_delta:.1f}% decrease forecast. "
                        "Ideal window to buy forward stock."))

    if rainfall > 30 and perish in ('High', 'Medium'):
        alerts.append(("warning", f"🌧️ Heavy Rain Risk: {rainfall:.1f}mm expected. "
                        f"{commodity} losses may rise. Switch to Reefer or expedite dispatch."))

    if temp > 35 and perish == 'High':
        alerts.append(("critical", f"🌡️ Heat Stress: {temp:.1f}°C. "
                        "Tomato/Leafy spoilage accelerates above 32°C. Use cold chain immediately."))

    _, total_cost, co2, mode_warns = logistics_suitability(mode_name, commodity, qty_qt, dist_km)
    for w in mode_warns:
        alerts.append(("warning", w))

    return alerts


# ==========================================
# 6. INIT
# ==========================================
engine  = AgriDataEngine()
raw_df  = engine.generate_industry_data()
predictor = SCM_Predictor(raw_df)
df_full   = predictor.prepare_and_train()

# ==========================================
# 7. SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/wheat.png", width=60)
    st.title("Control Panel")

    logistics_mode = st.selectbox("Logistics Mode", list(LOGISTICS_MODES.keys()))
    sel_item = st.selectbox("Commodity", sorted(raw_df['Commodity'].unique()))
    sel_loc  = st.selectbox("Primary Market", sorted(raw_df['Location'].unique()))
    dist_sim = st.slider("Route Distance (km)", 10, 1500, 300)
    f_qty    = st.number_input("Your Yield / Order (Quintals)", 1, 5000, 50)

    st.divider()
    mode      = LOGISTICS_MODES[logistics_mode]
    _, t_cost, co2_val, _ = logistics_suitability(logistics_mode, sel_item, f_qty, dist_sim)

    st.subheader("🌱 ESG Snapshot")
    col_a, col_b = st.columns(2)
    col_a.metric("CO₂ Footprint", f"{co2_val:.1f} kg")
    col_b.metric("Transport Cost", f"₹{t_cost:,.0f}")
    perish_label = PERISHABILITY[sel_item]
    colour = {"High": "🔴", "Medium": "🟡", "Low": "🟢", "None": "⚪"}[perish_label]
    st.info(f"Perishability: {colour} {perish_label}")

    st.divider()
    if predictor.trained:
        st.caption(f"Model MAE (validation): ₹{predictor.mae:.1f}/quintal")

# ==========================================
# 8. PREDICTION & ALERTS
# ==========================================
mandi_df   = df_full[(df_full['Commodity'] == sel_item) & (df_full['Location'] == sel_loc)].dropna(subset=predictor.feat_cols)
latest_day = mandi_df.iloc[-1]
pred_price = predictor.predict_next(latest_day)
price_delta = ((pred_price - latest_day['Price']) / latest_day['Price']) * 100

alerts = generate_alerts(price_delta, latest_day['Rainfall'], latest_day['Temp'],
                         sel_item, logistics_mode, f_qty, dist_sim)

st.title("🌾 Agri-SCM Intelligence Hub")
st.caption(f"Live context: {sel_item} @ {sel_loc} | Mode: {logistics_mode} | {datetime.now().strftime('%d %b %Y')}")

if alerts:
    for a_type, a_msg in alerts:
        css_cls = {"critical": "alert-critical", "warning": "alert-warning", "info": "alert-info"}[a_type]
        st.markdown(f'<div class="{css_cls}">{a_msg}</div>', unsafe_allow_html=True)

# ==========================================
# 9. MAIN TABS
# ==========================================
t_proc, t_farm, t_risk, t_insight, t_agro, t_report = st.tabs([
    "🏗️ Procurement", "🌾 Farmer Advisory", "📉 Risk & Spoilage",
    "📊 Market Insights", "🌦️ Agro Intelligence", "📋 Sales Report"
])

# ------------------------------------------
# TAB 1: PROCUREMENT
# ------------------------------------------
with t_proc:
    st.subheader("Sourcing Intelligence")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price (₹/Q)", f"₹{latest_day['Price']:,.0f}")
    c2.metric("Predicted Next Day", f"₹{pred_price:,.0f}", f"{price_delta:+.1f}%")
    c3.metric("Mandi Reliability", f"{latest_day['Reliability']}/10")
    c4.metric("Arrivals (Q)", f"{latest_day['Arrival']:,.0f}")

    st.markdown("---")
    st.write("### 🌎 Cross-Mandi Arbitrage — Price vs Reliability")
    latest_date = mandi_df['Date'].max()
    arb_df = df_full[(df_full['Date'] == latest_date) & (df_full['Commodity'] == sel_item)].copy()
    _, logi_cost, _, _ = logistics_suitability(logistics_mode, sel_item, f_qty, dist_sim)
    arb_df['Landed_Cost'] = arb_df['Price'] + logi_cost / max(f_qty, 1)
    arb_df['Value_Score'] = (arb_df['Reliability'] / arb_df['Landed_Cost'] * 1000).round(2)

    fig_arb = px.scatter(arb_df, x="Landed_Cost", y="Reliability", size="Arrival",
                         color="Value_Score", color_continuous_scale="RdYlGn",
                         hover_name="Location", hover_data=["Price", "Arrival", "Landed_Cost"],
                         title="Mandi Selection: Landed Cost vs Reliability (bubble = arrivals, colour = value score)")
    fig_arb.update_layout(height=400)
    st.plotly_chart(fig_arb, use_container_width=True)

    arb_disp = arb_df[['Location', 'Price', 'Reliability', 'Landed_Cost', 'Value_Score']].sort_values('Value_Score', ascending=False)
    st.dataframe(arb_disp.style.background_gradient(subset=['Value_Score'], cmap='RdYlGn'), use_container_width=True)

    # 30-day price trend
    st.write("### 📈 30-Day Price Trend")
    trend_df = mandi_df.tail(30)
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(x=trend_df['Date'], y=trend_df['Price'], mode='lines+markers',
                               name='Actual', line=dict(color='#2e7d32', width=2)))
    fig_t.add_trace(go.Scatter(x=trend_df['Date'], y=trend_df['Price_Roll7_Mean'], mode='lines',
                               name='7-Day MA', line=dict(color='#f57c00', dash='dot')))
    fig_t.update_layout(height=300, title=f"{sel_item} Price — {sel_loc}")
    st.plotly_chart(fig_t, use_container_width=True)

# ------------------------------------------
# TAB 2: FARMER ADVISORY
# ------------------------------------------
with t_farm:
    st.subheader("👨‍🌾 Farmer Profit Optimizer")

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### 📍 Sell Today")
        transport_total = dist_sim * mode['cost_km']
        net_today = (latest_day['Price'] * f_qty) - transport_total
        st.metric("Gross Revenue", f"₹{latest_day['Price'] * f_qty:,.0f}")
        st.metric("Transport Cost", f"₹{transport_total:,.0f}")
        st.metric("Net Profit Today", f"₹{net_today:,.0f}")

    with col2:
        st.write("#### ❄️ Cold Storage ROI")
        storage_days = st.slider("Storage Duration (Days)", 7, 120, 30)
        rent_per_q   = 60  # ₹ per quintal per month
        # Smarter: use model prediction trend compounded
        monthly_drift = (pred_price - latest_day['Price']) / latest_day['Price']
        future_price  = latest_day['Price'] * ((1 + monthly_drift) ** (storage_days / 30))
        storage_cost  = (storage_days / 30) * rent_per_q * f_qty
        # Perishability loss during storage
        perish_loss_pct = PERISHABILITY_FACTOR[PERISHABILITY[sel_item]] * storage_days * 0.5 / 100
        saleable_qty = f_qty * (1 - perish_loss_pct)
        net_future   = (future_price * saleable_qty) - storage_cost - transport_total
        roi          = net_future - net_today

        st.metric("Estimated Future Price", f"₹{future_price:,.0f}", f"{((future_price/latest_day['Price'])-1)*100:+.1f}%")
        st.metric("Storage Cost", f"₹{storage_cost:,.0f}")
        st.metric("Spoilage Loss (est.)", f"{perish_loss_pct*100:.1f}%")
        st.metric("Net ROI vs Sell Now", f"₹{roi:,.0f}", delta_color="normal")

    if roi > 0:
        st.success(f"✅ HOLD: Storing for {storage_days} days is estimated to yield ₹{roi:,.0f} extra after all costs.")
    else:
        st.error(f"❌ SELL NOW: Holding costs + spoilage exceed the predicted price gain by ₹{abs(roi):,.0f}.")

    # Multi-day ROI sweep
    st.write("#### 📊 Storage ROI by Duration")
    durations = list(range(7, 121, 7))
    roi_vals  = []
    for d in durations:
        fp    = latest_day['Price'] * ((1 + monthly_drift) ** (d / 30))
        sc    = (d / 30) * rent_per_q * f_qty
        pl    = PERISHABILITY_FACTOR[PERISHABILITY[sel_item]] * d * 0.5 / 100
        sq    = f_qty * (1 - pl)
        roi_vals.append((fp * sq - sc - transport_total) - net_today)

    fig_roi = go.Figure(go.Bar(x=durations, y=roi_vals,
                                marker_color=['#2e7d32' if v > 0 else '#d32f2f' for v in roi_vals]))
    fig_roi.update_layout(title="Net ROI vs Immediate Sale (₹)", xaxis_title="Storage Days",
                          yaxis_title="ROI (₹)", height=280)
    st.plotly_chart(fig_roi, use_container_width=True)

# ------------------------------------------
# TAB 3: RISK & SPOILAGE
# ------------------------------------------
with t_risk:
    st.subheader("📉 Risk Dashboard")

    transit_days = st.slider("Transit Time (Days)", 1, 15, 3)
    mode_factor  = 0.2 if logistics_mode == "Reefer (Cold Chain)" else 1.0
    pf = PERISHABILITY_FACTOR[PERISHABILITY[sel_item]]
    spoilage_pct = ((latest_day['Temp'] * 0.4) * (transit_days ** 1.3) * pf * mode_factor) / 10
    spoilage_pct = min(spoilage_pct, 100)

    fin_loss = (f_qty * spoilage_pct / 100) * latest_day['Price']

    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Spoilage Loss", f"{spoilage_pct:.1f}%",
               delta="Reefer active" if logistics_mode == "Reefer (Cold Chain)" else None)
    rc2.metric("Revenue at Risk", f"₹{fin_loss:,.0f}")
    rc3.metric("Safe Sell Window", f"{max(0, 5 - transit_days)} days")

    # Spoilage by mode comparison
    st.write("### Mode Comparison — Spoilage Risk")
    modes_comp = []
    for mname, mval in LOGISTICS_MODES.items():
        mf = 0.2 if mname == "Reefer (Cold Chain)" else 1.0
        sp = min(((latest_day['Temp'] * 0.4) * (transit_days ** 1.3) * pf * mf) / 10, 100)
        cost_mode = dist_sim * mval['cost_km']
        modes_comp.append({'Mode': mname, 'Spoilage %': round(sp, 1), 'Transport Cost ₹': cost_mode})

    comp_df = pd.DataFrame(modes_comp)
    fig_comp = make_subplots(specs=[[{"secondary_y": True}]])
    fig_comp.add_trace(go.Bar(x=comp_df['Mode'], y=comp_df['Spoilage %'],
                               name='Spoilage %', marker_color='#ef5350'), secondary_y=False)
    fig_comp.add_trace(go.Scatter(x=comp_df['Mode'], y=comp_df['Transport Cost ₹'],
                                  name='Transport Cost ₹', mode='lines+markers',
                                  line=dict(color='#1565c0', width=2)), secondary_y=True)
    fig_comp.update_layout(height=340, title="Spoilage vs Transport Cost by Logistics Mode")
    st.plotly_chart(fig_comp, use_container_width=True)

    # Weather risk map
    st.write("### 🌦️ Weather Risk Factors")
    wc1, wc2, wc3 = st.columns(3)
    wc1.metric("Temperature", f"{latest_day['Temp']:.1f}°C",
               delta="High Risk" if latest_day['Temp'] > 35 else "Normal")
    wc2.metric("Humidity", f"{latest_day['Humidity']:.1f}%",
               delta="Mold Risk" if latest_day['Humidity'] > 80 else "Normal")
    wc3.metric("Rainfall", f"{latest_day['Rainfall']:.1f} mm",
               delta="Flood Risk" if latest_day['Rainfall'] > 30 else "Normal")

# ------------------------------------------
# TAB 4: MARKET INSIGHTS
# ------------------------------------------
with t_insight:
    st.subheader("📊 Market Intelligence")

    # Feature importance
    st.write("### 🤖 What Drives Price Predictions?")
    fi_df = predictor.feature_importance().reset_index()
    fi_df.columns = ['Feature', 'Importance']
    fig_fi = px.bar(fi_df.head(10), x='Importance', y='Feature', orientation='h',
                    color='Importance', color_continuous_scale='Greens',
                    title="Top Predictors (Gradient Boosting Feature Importance)")
    fig_fi.update_layout(height=320, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_fi, use_container_width=True)

    # All-commodity price heatmap (last 90 days)
    st.write("### 🌡️ Commodity × Mandi Price Heatmap (Last 90 Days Avg)")
    cut = raw_df[raw_df['Date'] >= raw_df['Date'].max() - pd.Timedelta(days=90)]
    heat = cut.groupby(['Commodity', 'Location'])['Price'].mean().unstack()
    fig_heat = px.imshow(heat, color_continuous_scale='RdYlGn_r', aspect='auto',
                          title="Average Price Heatmap (₹/Q) — Last 90 Days")
    fig_heat.update_layout(height=340)
    st.plotly_chart(fig_heat, use_container_width=True)

    # Commodity price volatility
    st.write("### 📉 Price Volatility (CV%) by Commodity")
    vol_df = raw_df.groupby('Commodity')['Price'].agg(lambda x: x.std() / x.mean() * 100).reset_index()
    vol_df.columns = ['Commodity', 'CV%']
    fig_vol = px.bar(vol_df.sort_values('CV%'), x='Commodity', y='CV%',
                     color='CV%', color_continuous_scale='Reds',
                     title="Price Volatility — Coefficient of Variation (%)")
    fig_vol.update_layout(height=300)
    st.plotly_chart(fig_vol, use_container_width=True)


# ------------------------------------------
# TAB 5: AGRO INTELLIGENCE (Weather + Yield + Crop Recommendations)
# ------------------------------------------
with t_agro:
    st.subheader("🌦️ Agro Intelligence — Weather, Yield & Crop Recommendations")

    # ── SECTION A: Weather Forecast (14-day simulated) ──────────────────
    st.write("### 🌤️ 14-Day Weather Forecast")
    today       = raw_df['Date'].max()
    doy_base    = today.dayofyear
    mon_base    = today.month - 1

    forecast_rows = []
    for i in range(1, 15):
        fd   = today + pd.Timedelta(days=i)
        doy  = fd.dayofyear
        mon  = fd.month - 1
        t_fc = 22 + np.sin(doy / 365 * 2 * np.pi) * 12 + np.random.normal(0, 1.2)
        h_fc = 55 + np.cos(doy / 365 * 2 * np.pi) * 25 + np.random.normal(0, 3)
        r_fc = (np.random.exponential(2.5) if 5 <= fd.month <= 9 else np.random.exponential(0.4))
        cond = ("🌧️ Rain" if r_fc > 3 else ("⛅ Cloudy" if h_fc > 70 else "☀️ Sunny"))
        forecast_rows.append({'Day': fd.strftime('%d %b'), 'Temp (°C)': round(t_fc, 1),
                               'Humidity (%)': round(h_fc, 1), 'Rain (mm)': round(r_fc, 2),
                               'Condition': cond})

    fc_df = pd.DataFrame(forecast_rows)

    # Colour-coded forecast cards in 7-column rows
    for row_start in [0, 7]:
        cols = st.columns(7)
        for i, col in enumerate(cols):
            d = fc_df.iloc[row_start + i]
            bg = "#e8f5e9" if "Sunny" in d['Condition'] else ("#e3f2fd" if "Cloudy" in d['Condition'] else "#fce4ec")
            col.markdown(f"""
            <div style="background:{bg};border-radius:10px;padding:10px 6px;text-align:center;border:1px solid #ccc;">
                <b>{d['Day']}</b><br>{d['Condition']}<br>
                🌡️ {d['Temp (°C)']}°C<br>💧 {d['Rain (mm)']} mm
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Temp & Rain trend chart
    fig_wf = make_subplots(specs=[[{"secondary_y": True}]])
    fig_wf.add_trace(go.Scatter(x=fc_df['Day'], y=fc_df['Temp (°C)'], name='Temp (°C)',
                                 mode='lines+markers', line=dict(color='#e53935', width=2)), secondary_y=False)
    fig_wf.add_trace(go.Bar(x=fc_df['Day'], y=fc_df['Rain (mm)'], name='Rain (mm)',
                             marker_color='#42a5f5', opacity=0.6), secondary_y=True)
    fig_wf.update_layout(height=280, title="14-Day Temperature & Rainfall Forecast", legend=dict(orientation='h'))
    fig_wf.update_yaxes(title_text="Temp (°C)", secondary_y=False)
    fig_wf.update_yaxes(title_text="Rain (mm)", secondary_y=True)
    st.plotly_chart(fig_wf, use_container_width=True)

    st.markdown("---")

    # ── SECTION B: Harvest Yield Prediction ─────────────────────────────
    st.write("### 🌱 Harvest Yield Prediction")

    # Yield model: base yield modified by weather, season, and soil params
    CROP_BASE_YIELD = {'Onion': 120, 'Tomato': 200, 'Potato': 180, 'Wheat': 35, 'Rice': 45}  # Q/acre

    yc1, yc2, yc3 = st.columns(3)
    with yc1:
        yield_crop   = st.selectbox("Crop for Yield Forecast", list(CROP_BASE_YIELD.keys()), key='yc')
        acreage      = st.number_input("Field Area (Acres)", 1.0, 500.0, 10.0, step=0.5)
    with yc2:
        soil_quality = st.selectbox("Soil Quality", ["Poor", "Average", "Good", "Excellent"])
        irrigation   = st.selectbox("Irrigation Type", ["Rain-fed", "Drip", "Sprinkler", "Canal Flood"])
    with yc3:
        fertilizer   = st.selectbox("Fertilizer Level", ["None", "Low", "Medium", "High"])
        seed_variety = st.selectbox("Seed Variety", ["Local", "Hybrid", "HYV (High Yielding)"])

    # Multiplier lookup tables
    soil_mult  = {"Poor": 0.60, "Average": 0.80, "Good": 1.00, "Excellent": 1.15}
    irrig_mult = {"Rain-fed": 0.75, "Drip": 1.20, "Sprinkler": 1.05, "Canal Flood": 0.90}
    fert_mult  = {"None": 0.65, "Low": 0.80, "Medium": 1.00, "High": 1.18}
    seed_mult  = {"Local": 0.80, "Hybrid": 1.10, "HYV (High Yielding)": 1.25}

    # Average forecast weather effect
    avg_temp_fc = fc_df['Temp (°C)'].mean()
    avg_rain_fc = fc_df['Rain (mm)'].mean()
    weather_yld_mult = 1.0
    if avg_temp_fc > 38:
        weather_yld_mult -= 0.15
    if avg_rain_fc > 5:
        weather_yld_mult += 0.05 if yield_crop in ('Rice', 'Wheat') else -0.05

    base_y      = CROP_BASE_YIELD[yield_crop]
    est_yield_q = (base_y * acreage
                   * soil_mult[soil_quality]
                   * irrig_mult[irrigation]
                   * fert_mult[fertilizer]
                   * seed_mult[seed_variety]
                   * weather_yld_mult)
    est_revenue = est_yield_q * latest_day['Price'] if yield_crop == sel_item else est_yield_q * AgriDataEngine.BASE_PRICES[yield_crop]

    ya, yb, yc_col, yd = st.columns(4)
    ya.metric("Estimated Yield", f"{est_yield_q:,.0f} Q")
    yb.metric("Per Acre", f"{est_yield_q/acreage:.1f} Q/acre")
    yc_col.metric("Estimated Revenue", f"₹{est_revenue:,.0f}")
    yd.metric("Weather Impact", f"{(weather_yld_mult-1)*100:+.1f}%",
              delta_color="normal")

    # Yield sensitivity chart
    soil_opts = list(soil_mult.keys())
    irrig_opts = list(irrig_mult.keys())
    sens_rows = []
    for s in soil_opts:
        for ir in irrig_opts:
            y = base_y * acreage * soil_mult[s] * irrig_mult[ir] * fert_mult[fertilizer] * seed_mult[seed_variety] * weather_yld_mult
            sens_rows.append({'Soil': s, 'Irrigation': ir, 'Yield (Q)': round(y, 1)})
    sens_df = pd.DataFrame(sens_rows)
    fig_sens = px.density_heatmap(sens_df, x='Irrigation', y='Soil', z='Yield (Q)',
                                   color_continuous_scale='YlGn',
                                   title=f"Yield Sensitivity — {yield_crop} ({acreage} acres)")
    fig_sens.update_layout(height=300)
    st.plotly_chart(fig_sens, use_container_width=True)

    st.markdown("---")

    # ── SECTION C: Crop Recommendations ─────────────────────────────────
    st.write("### 🏆 High-Yield Crop Recommendations")
    st.caption("Based on current season, forecast weather, and market price momentum")

    current_month = today.month - 1  # 0-indexed

    rec_rows = []
    for crop, base_p in AgriDataEngine.BASE_PRICES.items():
        seasonal_idx  = AgriDataEngine.SEASONAL_CURVES[crop][current_month]
        # Price momentum: last 30 days trend
        crop_recent   = raw_df[(raw_df['Commodity'] == crop)].tail(30 * 9)  # all locations
        price_mom     = (crop_recent['Price'].iloc[-1] - crop_recent['Price'].iloc[0]) / crop_recent['Price'].iloc[0] * 100
        # Yield score
        base_yld      = CROP_BASE_YIELD[crop]
        yld_score     = base_yld / max(CROP_BASE_YIELD.values()) * 10
        # Weather suitability
        weather_suit  = 8 if avg_temp_fc < 35 else 5
        if crop in ('Rice',) and avg_rain_fc > 2:
            weather_suit = min(10, weather_suit + 2)
        if crop == 'Tomato' and avg_temp_fc > 35:
            weather_suit -= 2
        # Market score: seasonal high = good
        mkt_score     = seasonal_idx * 10
        # Composite
        composite     = (yld_score * 0.3 + weather_suit * 0.3 + mkt_score * 0.25 + max(price_mom, 0) * 0.01 * 0.15)
        rec_rows.append({
            'Crop': crop,
            'Yield Score': round(yld_score, 1),
            'Weather Fit': round(weather_suit, 1),
            'Market Score': round(mkt_score, 1),
            'Price Momentum %': round(price_mom, 1),
            'Overall Score': round(composite, 2),
            'Recommendation': '🟢 Plant Now' if composite >= 7 else ('🟡 Consider' if composite >= 5 else '🔴 Avoid')
        })

    rec_df = pd.DataFrame(rec_rows).sort_values('Overall Score', ascending=False)

    # Radar chart for top 3 crops
    top3 = rec_df.head(3)
    categories = ['Yield Score', 'Weather Fit', 'Market Score']
    fig_radar = go.Figure()
    colors_r = ['#2e7d32', '#f57c00', '#1565c0']
    for idx, (_, row) in enumerate(top3.iterrows()):
        vals = [row[c] for c in categories]
        vals += [vals[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            fill='toself',
            name=row['Crop'],
            line_color=colors_r[idx],
            opacity=0.7
        ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(range=[0, 12])),
                             title="Top 3 Crops — Multi-Factor Radar", height=380)
    st.plotly_chart(fig_radar, use_container_width=True)

    st.dataframe(rec_df.style.background_gradient(subset=['Overall Score'], cmap='RdYlGn')
                              .map(lambda v: 'color: green' if '🟢' in str(v) else
                                        ('color: orange' if '🟡' in str(v) else 'color: red'),
                                        subset=['Recommendation']),
                 use_container_width=True)


# ------------------------------------------
# TAB 6: SALES REPORT
# ------------------------------------------
with t_report:
    st.subheader("📋 Sales Intelligence Report")

    rep_days = st.slider("Lookback Window (days)", 1, 15, 7, key='rep_days')
    cutoff   = raw_df['Date'].max() - pd.Timedelta(days=rep_days)
    recent   = raw_df[raw_df['Date'] >= cutoff]

    # ── KPI Row ────────────────────────────────────────────────────────
    rk1, rk2, rk3, rk4 = st.columns(4)
    rk1.metric("Period", f"Last {rep_days} days")
    rk2.metric("Records Analysed", f"{len(recent):,}")
    rk3.metric("Avg Price (₹/Q)", f"₹{recent['Price'].mean():,.0f}")
    rk4.metric("Total Arrivals (Q)", f"{recent['Arrival'].sum():,.0f}")

    st.markdown("---")

    # ── Best-Selling Crops (past window) ───────────────────────────────
    st.write(f"### 🏆 Best-Selling Crops — Last {rep_days} Days")

    past_perf = recent.groupby('Commodity').agg(
        Avg_Price=('Price', 'mean'),
        Total_Arrivals=('Arrival', 'sum'),
        Avg_Reliability=('Reliability', 'mean')
    ).reset_index()
    past_perf['Revenue_Proxy'] = (past_perf['Avg_Price'] * past_perf['Total_Arrivals']).round(0)
    past_perf['Rank'] = past_perf['Revenue_Proxy'].rank(ascending=False).astype(int)
    past_perf = past_perf.sort_values('Revenue_Proxy', ascending=False)

    # Medal badges
    medals = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣']
    past_perf['Rank'] = [medals[i] for i in range(len(past_perf))]

    fig_past = px.bar(past_perf, x='Commodity', y='Revenue_Proxy',
                      color='Avg_Price', color_continuous_scale='Greens',
                      text='Rank', title=f"Revenue Proxy (Price × Arrivals) — Last {rep_days} Days")
    fig_past.update_traces(textposition='outside')
    fig_past.update_layout(height=320)
    st.plotly_chart(fig_past, use_container_width=True)

    st.dataframe(past_perf[['Rank', 'Commodity', 'Avg_Price', 'Total_Arrivals', 'Revenue_Proxy', 'Avg_Reliability']]
                 .style.format({'Avg_Price': '₹{:.0f}', 'Total_Arrivals': '{:.0f}',
                                'Revenue_Proxy': '₹{:.0f}', 'Avg_Reliability': '{:.1f}'})
                 .background_gradient(subset=['Revenue_Proxy'], cmap='YlGn'),
                 use_container_width=True)

    st.markdown("---")

    # ── Forward Sales Forecast (next 15 days) ──────────────────────────
    st.write("### 🔮 Forward Sales Forecast — Next 15 Days")
    st.caption("Uses GB model predictions + seasonal curves to rank which crops will sell most")

    future_forecast = []
    for crop in AgriDataEngine.BASE_PRICES.keys():
        crop_locs = df_full[(df_full['Commodity'] == crop)].dropna(subset=predictor.feat_cols)
        if crop_locs.empty:
            continue
        latest_crop = crop_locs.sort_values('Date').iloc[-1]
        pred_p = predictor.predict_next(latest_crop)

        # Forward seasonal momentum
        future_mon  = (raw_df['Date'].max() + pd.Timedelta(days=8)).month - 1
        curr_seas   = AgriDataEngine.SEASONAL_CURVES[crop][(raw_df['Date'].max().month - 1)]
        fut_seas    = AgriDataEngine.SEASONAL_CURVES[crop][future_mon]
        seas_change = (fut_seas - curr_seas) / curr_seas * 100

        # Demand score based on arrival trend
        arr_trend = recent[recent['Commodity'] == crop]['Arrival'].mean()
        demand_score = min(arr_trend / 800 * 10, 10)

        future_forecast.append({
            'Crop': crop,
            'Current Avg Price': round(latest_crop['Price'], 0),
            'Predicted Price': round(pred_p, 0),
            'Price Change %': round(((pred_p - latest_crop['Price']) / latest_crop['Price']) * 100, 1),
            'Seasonal Momentum %': round(seas_change, 1),
            'Demand Score': round(demand_score, 1),
            'Sell Signal': ('🟢 Strong Buy' if seas_change > 5 and pred_p > latest_crop['Price']
                            else '🟡 Neutral' if abs(seas_change) <= 5
                            else '🔴 Cautious')
        })

    fwd_df = pd.DataFrame(future_forecast).sort_values('Price Change %', ascending=False)

    # Dual bar: current vs predicted
    fig_fwd = go.Figure()
    fig_fwd.add_trace(go.Bar(x=fwd_df['Crop'], y=fwd_df['Current Avg Price'],
                              name='Current Price', marker_color='#90a4ae'))
    fig_fwd.add_trace(go.Bar(x=fwd_df['Crop'], y=fwd_df['Predicted Price'],
                              name='Predicted Price', marker_color='#2e7d32'))
    fig_fwd.update_layout(barmode='group', title="Current vs Predicted Price — Next Day Forecast",
                          height=320, yaxis_title="₹ / Quintal")
    st.plotly_chart(fig_fwd, use_container_width=True)

    # Seasonal momentum sparkline
    fig_mom = px.bar(fwd_df, x='Crop', y='Seasonal Momentum %',
                     color='Seasonal Momentum %', color_continuous_scale='RdYlGn',
                     title="Seasonal Momentum (Next 15 Days) — Positive = Demand Rising")
    fig_mom.add_hline(y=0, line_dash='dash', line_color='grey')
    fig_mom.update_layout(height=280)
    st.plotly_chart(fig_mom, use_container_width=True)

    st.dataframe(fwd_df.style
                 .format({'Current Avg Price': '₹{:.0f}', 'Predicted Price': '₹{:.0f}',
                          'Price Change %': '{:+.1f}%', 'Seasonal Momentum %': '{:+.1f}%',
                          'Demand Score': '{:.1f}'})
                 .background_gradient(subset=['Price Change %'], cmap='RdYlGn')
                 .map(lambda v: 'color: green' if '🟢' in str(v) else
                           ('color: orange' if '🟡' in str(v) else 'color: red'),
                           subset=['Sell Signal']),
                 use_container_width=True)

    st.markdown("---")

    # ── Location-wise sales breakdown ──────────────────────────────────
    st.write(f"### 📍 Mandi-wise Performance — Last {rep_days} Days")
    mandi_perf = recent.groupby('Location').agg(
        Avg_Price=('Price', 'mean'),
        Total_Arrivals=('Arrival', 'sum'),
        Avg_Reliability=('Reliability', 'mean')
    ).reset_index().sort_values('Total_Arrivals', ascending=False)

    fig_mandi = px.scatter(mandi_perf, x='Total_Arrivals', y='Avg_Price',
                            size='Avg_Reliability', color='Avg_Reliability',
                            color_continuous_scale='RdYlGn', hover_name='Location',
                            text='Location',
                            title="Mandi Volume vs Price (bubble = reliability)")
    fig_mandi.update_traces(textposition='top center')
    fig_mandi.update_layout(height=380)
    st.plotly_chart(fig_mandi, use_container_width=True)


st.divider()
st.caption("Agri-SCM Intelligence Hub v5 | Gradient Boosting | Weather Forecast | Yield Predictor | Crop Recommendations | Sales Report")
