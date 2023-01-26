import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import pandas as pd

N = norm.cdf

def bs(S, K, vol, T, call=True):  
    r= 0
    d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    if call: return S * N(d1) - K *  N(d2) * np.exp(-r * T)
    else: return K * N(-d2) * np.exp(-r * T) - S * N(-d1)

def binbs(S, K, vol, T, call=True):    
    d1 = (np.log(S/K) + (0.5*vol**2)*T) / (vol*np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    if call: return N(d2)
    else: return N(-d2)

st.sidebar.title('Ithaca Products')


strike1 = st.sidebar.number_input('Strike 1', key="strike1", step=100, min_value=1000, value=1200)
strike2 = st.sidebar.number_input('Strike 2', key="strike2", step=100, min_value=1000, value=1600)
strike3 = st.sidebar.number_input('Strike 3', key="strike3", step=100, min_value=1000, value=1700)
strike4 = st.sidebar.number_input('Strike 4', key="strike4", step=100, min_value=1000, value=1800)
strike5 = st.sidebar.number_input('Strike 5', key="strike5", step=100, min_value=1000, value=2000)

st.sidebar.write("Other Params")
v = st.sidebar.number_input('Vol (%)', key="vol", step=1, min_value=1, value=70)
spot = st.sidebar.number_input('Spot', key="spot", step=50, min_value=1, value=1600)
t = st.sidebar.number_input('Expiry (days)', key="t", step=1, min_value=1, value=90)


T = t / 365
vol = v / 100

products = {
    1: {
        "label": "Riskless lend + call",
        "legs": [
            ('BUY',  1,  "SPOT",  None),
            ('BUY',  1,  "PUT",   strike3),
            ('SELL', 1,  "CALL",  strike3),
            ('BUY',  1,  "CALL",  strike5),
        ]
    },
    2: {
        "label": "Short Put Spread + UnI Call",
        "legs":     [
            ('BUY',   1, "PUT",   strike1),
            ('SELL',  1, "PUT",   strike3),
            ('BUY',   1, "CALL",  strike4),
            ('BUY',  200, "BCALL", strike4),        
            ('SELL',  1, "CALL",  strike5),        
        ],
    },
    3: {
        "label": "Covered Call",
        "legs": [
            ('SELL',  1, "CALL",  strike4),            
        ]
    },
    4: {
        "label": "Cash protected put",
        "legs": [
            ('SELL',  1, "PUT",  strike1),            
        ]
    },
    5: {
        "label": "Premium adjusted ATM call",
        "legs": [
            ('BUY',   1, "CALL",  strike2),            
            ('SELL', 20, "BCALL", strike2),            
        ]
    },
    6: {
        "label": "Long Binary Spread",
        "legs": [
            ('BUY',  40, "BCALL", strike2),            
            ('SELL', 40, "BCALL", strike4),            
        ]
    },
    7: {
        "label": "Short Binary Spread",
        "legs": [
            ('SELL', 20, "BCALL", strike2),            
            ('BUY',  20, "BCALL", strike5),            
        ]
    },
    8: {
        "label": "Long Binary Spread 2",
        "legs": [
            ('BUY',  20, "BCALL", strike4),
            ('SELL', 20, "BCALL", strike5),            
        ]
    }    
}

class Strategy:
    plot_min = 1000
    plot_max = 3000
    
    def __init__(self, strat_id):
        self.legs = products.get(strat_id).get('legs')
        self.label = products.get(strat_id).get('label')
        self.range = np.linspace(self.plot_min, self.plot_max, 500)
        self.exp = np.array(self.exposure()).T
        
    def exposure(self, S=spot, vol=vol, T=T):
        self.payoffs = []
        for side, qty, instr, strike in self.legs:
            side = 1 if side == "BUY" else -1
            payoff = []
            for i in self.range:
                if instr == "SPOT" or instr == "FORWARD":
                    cf = side * (i - spot)
                if instr == "CALL":
                    cf = max(i - strike, 0) * qty * side
                if instr == "PUT":
                    cf = max(strike - i, 0) * qty * side
                if instr == "BCALL":
                    cf =  (1 if i > strike else 0 ) * qty * side
                if instr == "BPUT":
                    cf =  (1 if i < strike else 0 ) * qty * side
                payoff.append(cf)
            self.payoffs.append(payoff)
        return self.payoffs
    
    def plot(self, ax=None):
        if not ax:
            fig, ax = plt.subplots()
        ax.plot(self.range, self.exp, alpha=0.1)
        ax.plot(self.range, self.exp.sum(1), )
        ax.set_title(self.label)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    def get_price(self, S=spot, vol=vol, T=T):
        px = []
        for side, qty, instr, strike in self.legs:
            side = 1 if side == "BUY" else -1            
            if instr == "SPOT" or instr == "FORWARD":
                px.append(0)            
            if instr == "CALL" or instr == "PUT":
                isCall = instr == "CALL"
                px.append(bs(S, strike, vol, T, call=isCall) * qty * side)
            if instr == "BCALL" or instr == "BPUT":
                isCall = instr == "CALL"
                px.append(binbs(S, strike, vol, T, call=isCall) * qty * side)
        return sum(px)

st.subheader("Products")
col1, col2, col3, col4 = st.columns(4)

for col, product in zip(st.columns(4), range(1,5)):
    with col:
        st.write(products.get(product).get('label'))
        for side, qty, instr, strike in products.get(product).get('legs'):
            st.caption(f"{side} {qty} {instr} @ {strike}")


for col, product in zip(st.columns(4), range(5,9)):
    with col:
        st.write(products.get(product).get('label'))
        for side, qty, instr, strike in products.get(product).get('legs'):
            st.caption(f"{side} {qty} {instr} @ {strike}")


st.subheader("Exposures")
fig, ax = plt.subplots(2,4, figsize=(12,6), sharey=False)
ax = ax.flatten()
for i in range(8):
    s = Strategy(i+1)
    s.plot(ax=ax[i])
plt.tight_layout()
st.pyplot(fig)


st.subheader("Premiums")
st.write("Current premium for each strategy based on current spot, vol and time to expiry (flat vol BS)")
data = []
for t in [6/12, 3/12, 1/12, 7/365]:
    for strat in range(1, 9):
        s = Strategy(strat)
        px = s.get_price(T=t)
        data.append({"t": t, "strat": strat, "px": px})
df = pd.DataFrame(data).pivot(index='t', columns='strat', values='px')
st.dataframe(df)


fig, ax = plt.subplots(2,4, figsize=(12,6), sharey=True)
ax = ax.flatten()

tt = np.linspace(180, 1)
for strat in range(1, 9):
    s = Strategy(strat)
    px = [s.get_price(T=t/360) for t in tt]
    ax[strat-1].plot(px)
    ax[strat-1].set_title(s.label)
plt.tight_layout()
st.pyplot(fig)