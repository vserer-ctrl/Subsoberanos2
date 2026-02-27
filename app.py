import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.optimize import fsolve
import plotly.express as px

st.set_page_config(page_title="Dashboard Bonos Subsoberanos", layout="wide")

# --- FUNCIONES ---
@st.cache_data(ttl=60)
def get_realtime_prices():
    try:
        response = requests.get('https://data912.com/live/arg_bonds', timeout=10)
        return {item['symbol']: item['c'] for item in response.json()}
    except:
        return {}

def calculate_financials(ticker_df, price, ref_date):
    future = ticker_df[ticker_df['payment_date'] > ref_date].copy()
    if future.empty or not price or price <= 0 or pd.isna(price): 
        return np.nan, np.nan
    
    future['t'] = (future['payment_date'] - ref_date).dt.days / 365.0
    times = [0] + future['t'].tolist()
    cfs = [-price] + future['cash_flow'].tolist()
    
    try:
        tir = fsolve(lambda y: sum(cf / (1 + y)**t for cf, t in zip(cfs, times)), 0.15)[0]
        pv_cfs = [cf / (1 + tir)**t for cf, t in zip(future['cash_flow'], future['t'])]
        dur = sum(t * pv / sum(pv_cfs) for t, pv in zip(future['t'], pv_cfs)) / (1 + tir)
        return tir, dur
    except: 
        return np.nan, np.nan

# --- APP ---
st.title("🏛️ Monitor de Bonos Subsoberanos Argentinos")

# Carga de datos
@st.cache_data
def load_data():
    df = pd.read_csv('docta-capital-data.xlsx - Cashflow de Sub Soberano.csv')
    df['payment_date'] = pd.to_datetime(df['payment_date'])
    return df

df = load_data()
prices = get_realtime_prices()
today = pd.Timestamp.now()

results = []
for ticker in df['ticker'].unique():
    t_df = df[df['ticker'] == ticker]
    px_val = prices.get(ticker, np.nan)
    
    tir, dur = calculate_financials(t_df, px_val, today)
    
    if not np.isnan(px_val):
        results.append({
            'Ticker': ticker, 
            'Precio': px_val, 
            'TIR (%)': tir * 100, 
            'Duration': dur
        })

df_res = pd.DataFrame(results)

if not df_res.empty:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.scatter(
            df_res, x="Duration", y="TIR (%)", text="Ticker", color="TIR (%)",
            title="Relación Riesgo/Retorno (TIR vs Duration)",
            labels={"Duration": "Duración Modificada (años)"}
        )
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.write("### Tabla de Precios y Métricas")
        st.dataframe(
            df_res.sort_values("TIR (%)", ascending=False).style.format({
                'Precio': '{:.2f}',
                'TIR (%)': '{:.2f}%',
                'Duration': '{:.2f}'
            }),
            use_container_width=True,
            hide_index=True
        )
else:
    st.warning("No se pudieron obtener precios en vivo de la API o no hay coincidencias de tickers.")
