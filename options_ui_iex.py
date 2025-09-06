# options_dashboard_tabs.py
# Multi-Page Options Dashboard (Qt + PyQtGraph)
# Tabs: 1) Chain+Vol  2) Flow+Depth  3) Skew+VIX
# Now supports: --data iex  (paid IEX Cloud) via IEXProvider
import sys, time, argparse, math, os, json
from dataclasses import dataclass
from collections import deque
import numpy as np, pandas as pd

from PySide6.QtCore import Qt, QTimer, QRect, QThread, Signal
from PySide6.QtGui import QAction, QKeySequence, QColor, QFont, QPalette
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QFrame, QVBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QStyledItemDelegate, QTabWidget, QHeaderView
)
import pyqtgraph as pg
pg.setConfigOptions(useOpenGL=True, antialias=False, background=None, foreground=None, imageAxisOrder='row-major')

APP_TITLE = "Options Dashboard (Tabbed)"

# ---------- Theme ----------
def hexc(h): h=h.strip().lstrip("#"); return QColor(int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))
THEMES = {
    "dark":     {"BG":"#16181C","PANEL":"#202329","GRID":"#32363D","TEXT":"#DCDDE4","POS":"#2EBD85","NEG":"#E2524F","ACCENT":"#569CD6"},
    "midnight": {"BG":"#0E0F12","PANEL":"#17191E","GRID":"#2A2E35","TEXT":"#E6E7EB","POS":"#29C48A","NEG":"#FF5C5C","ACCENT":"#7AA2F7"},
    "slate":    {"BG":"#1B1E24","PANEL":"#23262E","GRID":"#3A3F49","TEXT":"#E2E6EA","POS":"#36C690","NEG":"#EF5350","ACCENT":"#8AB4F8"},
    "purple":   {"BG":"#17151E","PANEL":"#221F2B","GRID":"#3B3546","TEXT":"#EEE9F9","POS":"#30C48E","NEG":"#F06261","ACCENT":"#7C4DFF"},
    "terminal": {"BG":"#0A0A0A","PANEL":"#111111","GRID":"#222222","TEXT":"#D0FFD0","POS":"#00FF7F","NEG":"#FF4C4C","ACCENT":"#00D1B2"},
    "light":    {"BG":"#F4F6F8","PANEL":"#FFFFFF","GRID":"#D7DBE0","TEXT":"#14171A","POS":"#1E8E5A","NEG":"#C62828","ACCENT":"#2962FF"},
}
BG=PANEL=GRID=TEXT=POS=NEG=ACCENT=None

def apply_theme_globals(name):
    global BG,PANEL,GRID,TEXT,POS,NEG,ACCENT
    t=THEMES.get(name, THEMES["dark"])
    BG,PANEL,GRID,TEXT,POS,NEG,ACCENT = map(hexc, (t["BG"],t["PANEL"],t["GRID"],t["TEXT"],t["POS"],t["NEG"],t["ACCENT"]))
    pg.setConfigOption('background', (BG.red(),BG.green(),BG.blue()))
    pg.setConfigOption('foreground', (TEXT.red(),TEXT.green(),TEXT.blue()))

def apply_app_style(app):
    app.setStyle('Fusion')
    pal = app.palette()
    pal.setColor(QPalette.Window, BG); pal.setColor(QPalette.WindowText, TEXT)
    pal.setColor(QPalette.Base, PANEL); pal.setColor(QPalette.AlternateBase, BG)
    pal.setColor(QPalette.ToolTipBase, PANEL); pal.setColor(QPalette.ToolTipText, TEXT)
    pal.setColor(QPalette.Text, TEXT); pal.setColor(QPalette.Button, PANEL); pal.setColor(QPalette.ButtonText, TEXT)
    pal.setColor(QPalette.Highlight, ACCENT); pal.setColor(QPalette.HighlightedText, TEXT)
    app.setPalette(pal)

def title_font(): return QFont("Segoe UI", 10, QFont.Bold)
def small_font(): return QFont("Segoe UI", 9)
def tiny_font():  return QFont("Segoe UI", 8)

def panel(title):
    f = QFrame(); f.setFrameShape(QFrame.StyledPanel)
    f.setStyleSheet(
      f"QFrame {{ background-color: rgb({PANEL.red()},{PANEL.green()},{PANEL.blue()});"
      f" border: 1px solid rgb({GRID.red()},{GRID.green()},{GRID.blue()}); border-radius: 10px; }}")
    lay = QVBoxLayout(f); lab = QLabel(title); lab.setFont(title_font()); lay.addWidget(lab); return f

# ---------- Color helpers ----------
def lerp(a,b,t): return a+(b-a)*t
def clamp01(x): return 0 if x<0 else (1 if x>1 else x)
def heat_color(val, vmin, vmax):
    if math.isnan(val): return QColor(80,80,80)
    t = 0.0 if vmax==vmin else clamp01((val - vmin)/(vmax - vmin))
    r = int(lerp(70, 255, t)); g = int(lerp(60, 225, t)); b = int(lerp(90, 120, t*0.6))
    return QColor(r,g,b)
def diverge_color(val, vmax_abs):
    if math.isnan(val): return QColor(80,80,80)
    if vmax_abs<=0: return QColor(120,120,120)
    t = clamp01(abs(val)/vmax_abs)
    if val >= 0: r = int(lerp(60, 30, t)); g=int(lerp(120, 210, t)); b=int(lerp(80, 90, t))
    else:        r = int(lerp(120,230, t)); g=int(lerp(70, 60, t));  b=int(lerp(80, 70, t))
    return QColor(r,g,b)
def is_light(c: QColor):
    L = (0.299*c.red() + 0.587*c.green() + 0.114*c.blue())/255.0
    return L > 0.72

# ---------- Table cell delegates ----------
class SignedBarDelegate(QStyledItemDelegate):
    def __init__(self, max_abs=1.0, parent=None): super().__init__(parent); self.max_abs=max(1e-6,float(max_abs))
    def paint(self, p, opt, idx):
        try: x=float(str(idx.data()).replace(',',''))
        except: return super().paint(p,opt,idx)
        p.save(); p.fillRect(opt.rect, opt.palette.base()); r=opt.rect.adjusted(4,6,-4,-6); mid=(r.left()+r.right())//2
        frac=min(abs(x)/self.max_abs,1.0)
        if x>=0: p.fillRect(QRect(mid, r.top(), int(frac*(r.right()-mid)), r.height()), POS)
        else:    p.fillRect(QRect(mid-int(frac*(mid-r.left())), r.top(), int(frac*(mid-r.left())), r.height()), NEG)
        p.setPen(TEXT); p.drawText(opt.rect.adjusted(6,0,-6,0), Qt.AlignVCenter|Qt.AlignLeft, f"{x:,.0f}"); p.restore()

class MonoBarDelegate(QStyledItemDelegate):
    def __init__(self, max_val=1.0, color=None, parent=None): super().__init__(parent); self.max_val=max(1e-9,float(max_val)); self.color=color or ACCENT
    def paint(self, p, opt, idx):
        try: x=float(str(idx.data()).replace(',',''))
        except: return super().paint(p,opt,idx)
        p.save(); p.fillRect(opt.rect, opt.palette.base()); r=opt.rect.adjusted(4,6,-4,-6); w=int(min(max(x,0)/self.max_val,1.0)*r.width())
        p.fillRect(QRect(r.left(), r.top(), w, r.height()), self.color)
        p.setPen(TEXT); p.drawText(opt.rect.adjusted(6,0,-6,0), Qt.AlignVCenter|Qt.AlignLeft, f"{x:,.0f}"); p.restore()

# ---------- Simple widgets ----------
class TableWidget(QFrame):
    def __init__(self, title, columns):
        super().__init__(); base=panel(title); lay=QVBoxLayout(self); lay.addWidget(base)
        self.table=QTableWidget(0,len(columns)); self.table.setHorizontalHeaderLabels(columns)
        base.layout().addWidget(self.table)
        self.table.verticalHeader().setVisible(False); self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setStretchLastSection(True)
    def update_data(self, df: pd.DataFrame):
        self.table.setUpdatesEnabled(False); self.table.setSortingEnabled(False)
        self.table.clearContents(); self.table.setRowCount(len(df)); self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for i, (_, row) in enumerate(df.iterrows()):
            for j, val in enumerate(row):
                it=QTableWidgetItem(f"{val:,}" if isinstance(val,(int,float)) and not isinstance(val,bool) else str(val))
                if isinstance(val,(int,float)): it.setTextAlignment(Qt.AlignRight|Qt.AlignVCenter)
                self.table.setItem(i,j,it)
        cols=[str(c) for c in df.columns]
        if 'Gamma' in cols:
            i=cols.index('Gamma'); self.table.setItemDelegateForColumn(i, SignedBarDelegate(max_abs=max(1.0,float(df['Gamma'].abs().max())), parent=self.table))
        if 'Delta' in cols:
            i=cols.index('Delta'); self.table.setItemDelegateForColumn(i, SignedBarDelegate(max_abs=max(1.0,float(df['Delta'].abs().max())), parent=self.table))
        if 'Net OI' in cols:
            i=cols.index('Net OI'); self.table.setItemDelegateForColumn(i, SignedBarDelegate(max_abs=max(1.0,float(df['Net OI'].abs().max())), parent=self.table))
        if 'Call Vol' in cols:
            i=cols.index('Call Vol'); self.table.setItemDelegateForColumn(i, MonoBarDelegate(max_val=max(1.0,float(df['Call Vol'].max())), parent=self.table))
        if 'Put Vol' in cols:
            i=cols.index('Put Vol'); self.table.setItemDelegateForColumn(i, MonoBarDelegate(max_val=max(1.0,float(df['Put Vol'].max())), parent=self.table))
        self.table.setSortingEnabled(True); self.table.setUpdatesEnabled(True)

class MatrixTable(QFrame):
    """Compact numeric heat-table used for Vol Surface / Vol Changes."""
    def __init__(self, title, value_type="iv"):  # 'iv' or 'deltapct'
        super().__init__(); self.value_type=value_type
        base=panel(title); lay=QVBoxLayout(self); lay.addWidget(base)
        self.table=QTableWidget(); base.layout().addWidget(self.table)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    def _fmt(self, v): return f"{v:.2f}%" if self.value_type=='iv' else f"{v:+.2f}%"
    def update_from_df(self, df: pd.DataFrame):
        rows, cols = df.shape
        self.table.setUpdatesEnabled(False)
        self.table.clear()
        self.table.setRowCount(rows)
        self.table.setColumnCount(cols+1)
        self.table.setHorizontalHeaderLabels(["Delta"] + [str(c) for c in df.columns])
        v = df.to_numpy(dtype=float)
        if self.value_type=='iv':
            vmin, vmax = np.nanmin(v), np.nanmax(v)
        else:
            vmax = np.nanmax(np.abs(v)); vmin = -vmax
        for i, d in enumerate(df.index):
            item=QTableWidgetItem(str(int(d))); item.setTextAlignment(Qt.AlignCenter); self.table.setItem(i,0,item)
            for j, col in enumerate(df.columns):
                val = float(df.iloc[i,j])
                c = heat_color(val, vmin, vmax) if self.value_type=='iv' else diverge_color(val, vmax)
                it = QTableWidgetItem(self._fmt(val)); it.setTextAlignment(Qt.AlignCenter)
                it.setBackground(c); it.setForeground(QColor(20,20,20) if is_light(c) else QColor(240,240,240))
                self.table.setItem(i, j+1, it)
        self.table.setUpdatesEnabled(True)

# ---------- Plot widgets ----------
class PGLineChart(QWidget):
    def __init__(self, title, with_price=False, two_curves=False):
        super().__init__(); base=panel(title); lay=QVBoxLayout(self); lay.addWidget(base)
        self.plot=pg.PlotWidget(); base.layout().addWidget(self.plot)
        self.plot.showGrid(x=True,y=True,alpha=0.2)
        self.curves=[]
        if two_curves:
            for _ in range(2):
                c=pg.PlotCurveItem(pen=pg.mkPen(width=1.4)); self.plot.addItem(c); self.curves.append(c)
        else:
            c=pg.PlotCurveItem(pen=pg.mkPen(width=1.4)); self.plot.addItem(c); self.curves.append(c)
        self.with_price=with_price
        if with_price:
            self.right=pg.ViewBox(); self.plot.showAxis('right'); self.plot.scene().addItem(self.right)
            self.plot.getAxis('right').linkToView(self.right); self.right.setXLink(self.plot)
            self.price_curve=pg.PlotCurveItem(pen=pg.mkPen(color=(200,200,200), width=1)); self.right.addItem(self.price_curve)
            def updateViews(): self.right.setGeometry(self.plot.getViewBox().sceneBoundingRect()); self.right.linkedViewChanged(self.plot.getViewBox(), self.right.XAxis)
            self.plot.getViewBox().sigResized.connect(updateViews)
    def update_single(self, x, y, price=None):
        self.curves[0].setData(x[-400:], y[-400:], connect='finite', clipToView=True, autoDownsample=True)
        if self.with_price and price is not None: self.price_curve.setData(x[-400:], price[-400:], autoDownsample=True)
    def update_dual(self, x, y1, y2, price=None):
        self.curves[0].setData(x[-400:], y1[-400:], connect='finite', clipToView=True, autoDownsample=True)
        self.curves[1].setData(x[-400:], y2[-400:], connect='finite', clipToView=True, autoDownsample=True)
        if self.with_price and price is not None: self.price_curve.setData(x[-400:], price[-400:], autoDownsample=True)

class PGDepth(QWidget):
    def __init__(self, title):
        super().__init__(); base=panel(title); lay=QVBoxLayout(self); lay.addWidget(base)
        self.plot=pg.PlotWidget(); base.layout().addWidget(self.plot)
        self.plot.showGrid(x=True,y=True,alpha=0.2); self.plot.addLine(x=0, pen=pg.mkPen(color=(180,180,180), width=1))
    def update(self, price, bid, ask):
        self.plot.clear(); self.plot.addLine(x=0, pen=pg.mkPen(color=(180,180,180), width=1))
        for p,b,a in zip(price, bid, ask):
            self.plot.addItem(pg.PlotDataItem([0,-b], [p,p], pen=pg.mkPen(0,180,120,200, width=6)))
            self.plot.addItem(pg.PlotDataItem([0, a], [p,p], pen=pg.mkPen(240,150,120,200, width=6)))

class PGBarChart(QWidget):
    def __init__(self, title):
        super().__init__(); base=panel(title); lay=QVBoxLayout(self); lay.addWidget(base)
        self.plot=pg.PlotWidget(); base.layout().addWidget(self.plot); self.plot.showGrid(x=False,y=True,alpha=0.2)
        self.item=None
    def update(self, cats, vals):
        x=np.arange(len(cats))
        if self.item: self.plot.removeItem(self.item)
        self.item = pg.BarGraphItem(x=x, height=vals, width=0.7, brush=pg.mkBrush(ACCENT))
        self.plot.addItem(self.item); self.plot.getAxis('bottom').setTicks([list(enumerate(cats))])
        self.plot.autoRange()

class PGWeightedBar(QWidget):
    def __init__(self, title):
        super().__init__(); base=panel(title); lay=QVBoxLayout(self); lay.addWidget(base)
        self.plot=pg.PlotWidget(); base.layout().addWidget(self.plot); self.plot.showGrid(x=False,y=True,alpha=0.2)
    def update(self, names, rets):
        self.plot.clear()
        x=np.arange(len(names)); brushes=[pg.mkBrush(0,200,0) if r>=0 else pg.mkBrush(220,60,50) for r in rets]
        self.plot.addItem(pg.BarGraphItem(x=x, height=rets, width=0.7, brush=brushes))
        self.plot.addLine(y=0, pen=pg.mkPen(200,200,200, width=1))
        self.plot.getAxis('bottom').setTicks([list(enumerate(names))])
        self.plot.autoRange()

# ---------- Data model ----------
@dataclass
class MarketSnapshot:
    ts: float; symbol: str
    order_flow: pd.DataFrame; centroids: pd.DataFrame; depth: pd.DataFrame
    tick_summary: dict; chain: pd.DataFrame
    vol_surface: pd.DataFrame; vol_changes: pd.DataFrame
    call_skew: pd.DataFrame; put_skew: pd.DataFrame
    vix_term: pd.DataFrame; top_weighted: pd.DataFrame
    odte_skew: pd.DataFrame; state_labels: dict; metrics: dict

class BaseProvider:
    def __init__(self, symbol): self.symbol=symbol
    def latest(self)->MarketSnapshot: raise NotImplementedError

# ---------- Synthetic provider (unchanged except multi-expiry) ----------
class SyntheticProvider(BaseProvider):
    def __init__(self, symbol, num_expiries=10, seed=42):
        super().__init__(symbol); self.rng=np.random.default_rng(seed)
        self.price=6450.0; self._t=0
        self.tenors=['VIX1D','VIX9D','VIX','VIX3M','VIX6M','VIX1Y']
        self.expiries = self._gen_monthly_expiries(num_expiries)
        self.deltas=np.array([75,50,25,0,-25,-50,-75])
        self.top=['AAPL','NVDA','MSFT','META','TSLA','GOOGL','AMZN','AVGO','BRK/B','JPM']
    def _gen_monthly_expiries(self, n):
        out=[]; base=pd.Timestamp.today().normalize()
        from pandas.tseries.offsets import WeekOfMonth, MonthBegin
        wom = WeekOfMonth(week=2, weekday=4)
        cur = base
        for _ in range(n):
            d = wom.rollforward(cur); out.append(d.strftime("%d-%b-%y")); cur = cur + MonthBegin(1)
        return out
    def _rw(self,x,vol=1,drift=0): return x+drift+self.rng.normal(0,vol)
    def _taxis(self,n=400): return np.arange(max(0,self._t-n+1), self._t+1)
    def latest(self)->MarketSnapshot:
        self._t+=1; now=time.time()
        self.price=self._rw(self.price,4.0,self.rng.normal(0,0.5))
        t=self._taxis(400)
        call_flow=np.cumsum(self.rng.normal(0,18000,len(t)))+2e5*np.sin(t/40)
        put_flow =np.cumsum(self.rng.normal(0,18000,len(t)))-1.6e5*np.cos(t/50)
        px       =np.cumsum(self.rng.normal(0,2.5,len(t)))+self.price-150
        order_flow=pd.DataFrame({'time':t,'call_flow':call_flow,'put_flow':put_flow,'price':px})
        call_c=6400+20*np.sin(t/18)+self.rng.normal(0,3,len(t))
        put_c =6460+18*np.cos(t/22)+self.rng.normal(0,3,len(t))
        centroids=pd.DataFrame({'time':t,'call_c':call_c,'put_c':put_c,'px':px})
        ladder=np.arange(self.price-60,self.price+60,5)
        bid=np.clip(self.rng.normal(0,1,len(ladder))*200+1500,0,None)
        ask=np.clip(self.rng.normal(0,1,len(ladder))*200+1500,0,None)
        depth=pd.DataFrame({'price':ladder,'bid':bid,'ask':ask})
        tick_summary={'TICK':int(self.rng.normal(0,600)),
                      'AD_TICKERS':{'SPY':int(self.rng.normal(0,50)),'QQQ':int(self.rng.normal(0,50))},
                      'ADV_DEC_VOL':{'SPY':int(self.rng.normal(0,500)),'QQQ':int(self.rng.normal(0,500))}}
        strikes=np.arange(int(self.price-200),int(self.price+200)+25,25); n=len(strikes)
        chain=pd.DataFrame({
            'Call Vol':np.abs(self.rng.normal(8000,4000,n)).astype(int),
            'Strike':strikes,
            'Put Vol':np.abs(self.rng.normal(8000,4000,n)).astype(int),
            'Net OI':(self.rng.normal(0,800,n)).astype(int),
            'Gamma':self.rng.normal(0,3e5,n).astype(int),
            'Delta':np.clip(self.rng.normal(0,50,n),-100,100).astype(int),
            'Vanna':self.rng.normal(0,5e5,n).astype(int)
        })
        iv_base=0.12+0.02*np.sin(self._t/20)
        vols={}
        for k,exp in enumerate(self.expiries):
            center_shift = [10,0,-5,5,-10,7,-7,3,-3,0][k % 10]
            vols[exp] = iv_base*100 + 10*np.exp(-((self.deltas-center_shift)/40)**2) + self.rng.normal(0,0.25,len(self.deltas))
        vol_surface=pd.DataFrame(vols, index=self.deltas); vol_surface.index.name='Delta'
        vol_changes=vol_surface.copy()
        for c in vol_changes.columns: vol_changes[c]=self.rng.normal(0,0.14,len(self.deltas))
        call_skew=pd.DataFrame({'time':t,'skew':5*np.sin(t/30)+self.rng.normal(0,1,len(t))})
        put_skew =pd.DataFrame({'time':t,'skew':-5*np.cos(t/33)+self.rng.normal(0,1,len(t))})
        base=12+8*np.sin(self._t/50)
        vix_term=pd.DataFrame({'tenor':['VIX1D','VIX9D','VIX','VIX3M','VIX6M','VIX1Y'],
                               'value':[base*0.7,base*0.9,base,base*1.05,base*1.1,base*1.2]})
        top_weighted=pd.DataFrame({'name':['AAPL','NVDA','MSFT','META','TSLA','GOOGL','AMZN','AVGO','BRK/B','JPM'],
                                   'ret':np.random.default_rng().normal(0,1.2,10)})
        m=np.linspace(-20,20,30); odte=pd.DataFrame({'moneyness':m,'skew':0.1*m+np.random.default_rng().normal(0,0.8,len(m))})
        gex_ratio = (chain['Gamma'].clip(lower=0).sum()+1) / (abs(chain['Gamma'].clip(upper=0).sum())+1)
        dex_ratio = (chain['Delta'].clip(lower=0).sum()+1) / (abs(chain['Delta'].clip(upper=0).sum())+1)
        call_total, put_total = int(chain['Call Vol'].sum()), int(chain['Put Vol'].sum())
        vix_ratio = float(vix_term.loc[vix_term['tenor']=='VIX1D','value'].iloc[0] / vix_term.loc[vix_term['tenor']=='VIX','value'].iloc[0])
        five_day_sum = float(np.sum(np.diff(centroids['px'].values[-120:])[-5:])) if len(centroids)>=6 else 0.0
        def zscore(series): s=np.array(series); return 0 if s.std()==0 else (s[-1]-s.mean())/s.std()
        z_call = float(zscore(call_skew['skew'][-60:])); z_put  = float(zscore(put_skew['skew'][-60:]))
        state={'gamma_condition':'Put Dominated' if np.random.default_rng().random()<0.5 else 'Call Dominated',
               'skew_condition':'Risk-Off' if np.random.default_rng().random()<0.5 else 'Risk-On',
               'trend_score': float(np.round(np.random.default_rng().normal(-1.5,2.0),2))}
        metrics={'live_beta': round(np.random.default_rng().normal(1.0,0.2),2),'gex_ratio': round(gex_ratio,2),
                 'dex_ratio': round(dex_ratio,2),'call_total': call_total,'put_total': put_total,
                 'vix_ratio': round(vix_ratio,2),'five_day_sum': round(five_day_sum,2),
                 'z_call': round(z_call,2),'z_put': round(z_put,2),'now': time.strftime("%Y-%m-%d %H:%M:%S")}
        return MarketSnapshot(now,self.symbol,order_flow,centroids,depth,tick_summary,chain,vol_surface,vol_changes,
                              call_skew,put_skew,vix_term,top_weighted,odte,state,metrics)

# ---------- IEX provider ----------
# NOTE: This uses IEX Cloud REST. Set IEX_TOKEN env var or pass --iex-token.
# It tries both "legacy" stock/options endpoints and (if available) premium time-series datasets.
# Greeks are computed via Black-Scholes using provided IV (if present) or an IV solver fallback.
import requests

def _norm_cdf(x): return 0.5*(1+math.erf(x/math.sqrt(2)))
def _norm_pdf(x): return (1/math.sqrt(2*math.pi))*math.exp(-0.5*x*x)

def _bs_greeks(S,K,t,r,q,sigma,is_call=True):
    if sigma<=0 or t<=0 or S<=0 or K<=0:
        return {'delta':0.0,'gamma':0.0,'vanna':0.0}
    d1=(math.log(S/K)+(r-q+0.5*sigma*sigma)*t)/(sigma*math.sqrt(t))
    d2=d1 - sigma*math.sqrt(t)
    if is_call: delta=math.exp(-q*t)*_norm_cdf(d1)
    else:       delta=-math.exp(-q*t)*_norm_cdf(-d1)
    gamma=(math.exp(-q*t)*_norm_pdf(d1))/(S*sigma*math.sqrt(t))
    # Approx vanna (BS vanna = ∂Δ/∂σ * S ~ exp(-q t)*pdf(d1)*sqrt(t))
    vanna = math.exp(-q*t)*_norm_pdf(d1)*math.sqrt(t)
    return {'delta':delta, 'gamma':gamma, 'vanna':vanna}

def _implied_vol_from_mid(S,K,t,r,q,mid,is_call):
    # Simple Newton-Raphson; fallback clamp.
    sigma=0.3
    for _ in range(15):
        if sigma<=1e-6: sigma=1e-6
        d1=(math.log(S/K)+(r-q+0.5*sigma*sigma)*t)/(sigma*math.sqrt(t))
        d2=d1 - sigma*math.sqrt(t)
        disc_r=math.exp(-r*t); disc_q=math.exp(-q*t)
        if is_call:
            price = disc_q*S*_norm_cdf(d1) - disc_r*K*_norm_cdf(d2)
        else:
            price = disc_r*K*_norm_cdf(-d2) - disc_q*S*_norm_cdf(-d1)
        vega = disc_q*S*math.sqrt(t)*_norm_pdf(d1)
        diff = price - max(mid, 1e-6)
        if abs(diff) < 1e-4: break
        if vega < 1e-8: break
        sigma -= diff/vega
        sigma = min(max(sigma, 1e-6), 5.0)
    return float(sigma)

class IEXProvider(BaseProvider):
    def __init__(self, symbol, token, num_expiries=10, sandbox=False, timeout=6.0):
        super().__init__(symbol)
        self.token = token or os.environ.get("IEX_TOKEN","")
        self.base = ("https://sandbox.iexapis.com/stable" if sandbox else "https://cloud.iexapis.com/stable")
        self.timeout = timeout
        self.num_exp = num_expiries
        self.deltas = np.array([75,50,25,0,-25,-50,-75])
        # stateful timeseries for order_flow/centroids
        self.ts_idx=0
        self.call_flow_hist=deque(maxlen=400)
        self.put_flow_hist=deque(maxlen=400)
        self.px_hist=deque(maxlen=400)
        self.call_cent_hist=deque(maxlen=400)
        self.put_cent_hist=deque(maxlen=400)
        self.px_cent_hist=deque(maxlen=400)
        self.prev_surface=None  # to compute ΔIV%
    # ---------- helpers ----------
    def _get(self, path, params=None):
        p={'token': self.token}; 
        if params: p.update(params)
        url=f"{self.base}{path}"
        r=requests.get(url, params=p, timeout=self.timeout)
        r.raise_for_status()
        return r.json()
    def _list_expirations(self):
        # Try legacy options expirations
        try:
            data=self._get(f"/stock/{self.symbol}/options")
            # returns list like ["2025-09-19","2025-10-17",...]
            exps=[pd.to_datetime(x).strftime("%d-%b-%y") for x in data][:self.num_exp]
            return exps
        except Exception:
            return []
    def _fetch_chain_for_exp(self, exp_iso):
        try:
            data=self._get(f"/stock/{self.symbol}/options/{exp_iso}")
            return data
        except Exception:
            return []
    def _fetch_intraday(self):
        # /stock/{symbol}/intraday-prices?chartIEXOnly=true
        try:
            arr=self._get(f"/stock/{self.symbol}/intraday-prices", params={"chartIEXOnly":"true","chartLast":60})
            return arr
        except Exception:
            return []
    def _fetch_quote(self):
        try:
            return self._get(f"/stock/{self.symbol}/quote")
        except Exception:
            return {}
    def _fetch_book(self):
        # /deep/book?symbols=SYMBOL  (requires paid DEEP)
        try:
            j=self._get("/deep/book", params={"symbols": self.symbol})
            return j.get(self.symbol, {})
        except Exception:
            return {}
    # ---------- builder ----------
    def _build_chain_surface(self, exps_list):
        # For each expiration, get chain then compute basic vols + greeks.
        # Expect objects with fields: strike, side, bid, ask, last, volume, openInterest, expirationDate, impliedVolatility (if available)
        S = None
        q = self._fetch_quote()
        if q:
            S = float(q.get("latestPrice") or q.get("iexRealtimePrice") or 0)
        if not S: S=0.0
        r=0.00; div_q=0.00

        # buckets for surface by approx delta (we compute BS delta and bin to nearest of self.deltas)
        bins = {d: {} for d in self.deltas}  # d -> {expiry: mean_iv%}
        all_rows=[]

        for exp_str_iso in exps_list:
            exp_dt = pd.to_datetime(exp_str_iso)
            # fetch chain
            raw = self._fetch_chain_for_exp(exp_str_iso)
            if not isinstance(raw,list): continue
            # per strike aggregation
            by_strike = {}
            for opt in raw:
                try:
                    side = (opt.get("side") or opt.get("optionType") or "").upper()  # "call"/"put"
                    K = float(opt.get("strikePrice") or opt.get("strike") or 0)
                    vol = int(opt.get("totalVolume") or opt.get("volume") or 0)
                    oi  = int(opt.get("openInterest") or 0)
                    bid = float(opt.get("bid") or 0); ask=float(opt.get("ask") or 0)
                    last= float(opt.get("last") or opt.get("lastPrice") or 0)
                    iv  = opt.get("impliedVolatility")
                    if iv is None:
                        # estimate IV from mid if possible
                        mid = (bid+ask)/2 if (bid>0 and ask>0) else (last if last>0 else None)
                        t = max((exp_dt - pd.Timestamp.utcnow()).days/365.0, 1/365)
                        if S>0 and K>0 and mid:
                            iv = _implied_vol_from_mid(S,K,t,r,div_q,mid, is_call=(side=="CALL"))
                        else:
                            iv = 0.3
                    iv = float(iv)
                    t = max((exp_dt - pd.Timestamp.utcnow()).days/365.0, 1/365)
                    g = _bs_greeks(S,K,t,r,div_q,iv, is_call=(side=="CALL"))
                    row = by_strike.setdefault(K, {"Strike":K,"Call Vol":0,"Put Vol":0,"Net OI":0,"Gamma":0.0,"Delta":0.0,"Vanna":0.0})
                    if side=="CALL":
                        row["Call Vol"] += vol
                        row["Net OI"]   += oi
                        row["Gamma"]    += g['gamma']
                        row["Delta"]    += g['delta']
                        row["Vanna"]    += g['vanna']
                    elif side=="PUT":
                        row["Put Vol"] += vol
                        row["Net OI"]  += oi
                        row["Gamma"]   += g['gamma']
                        row["Delta"]   += -g['delta']  # orient sign
                        row["Vanna"]   += g['vanna']
                    # surface bin by delta magnitude sign: map delta in [-1,1] to nearest bucket % in self.deltas
                    approx_delta_pct = int(round(100*g['delta'])) if side=="CALL" else int(round(-100*g['delta']))
                    # snap to nearest in self.deltas (keeping sign)
                    tgt = min(self.deltas, key=lambda d: abs(d-approx_delta_pct))
                    dmap = bins[tgt]
                    lst = dmap.setdefault(exp_dt.strftime("%d-%b-%y"), [])
                    lst.append(iv*100.0)
                except Exception:
                    continue
            # append rows
            for K,row in by_strike.items():
                all_rows.append(row)

        chain_df = pd.DataFrame(all_rows).sort_values("Strike") if all_rows else pd.DataFrame(columns=["Call Vol","Strike","Put Vol","Net OI","Gamma","Delta","Vanna"])
        # scale greeks to more visible units
        if not chain_df.empty:
            for c in ["Gamma","Delta","Vanna"]:
                chain_df[c]=chain_df[c].astype(float)
            # make integers where appropriate
            for c in ["Call Vol","Put Vol","Net OI","Strike"]:
                chain_df[c]=chain_df[c].astype(float).round(0).astype(int)

        # build vol surface table
        cols_sorted = sorted({c for d in bins.values() for c in d.keys()}, key=lambda x: pd.to_datetime(x))
        surface = pd.DataFrame(index=self.deltas, columns=cols_sorted, dtype=float)
        for d,bmap in bins.items():
            for exp, lst in bmap.items():
                if lst:
                    surface.loc[d, exp] = float(np.mean(lst))
        surface = surface.dropna(axis=1, how='all').fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
        if surface.shape[1]==0:
            # no surface => create minimal empty
            surface = pd.DataFrame({exp: [np.nan]*len(self.deltas) for exp in [pd.Timestamp.today().strftime("%d-%b-%y")]}, index=self.deltas)

        # compute changes vs previous snapshot surface
        if self.prev_surface is None or set(self.prev_surface.columns)!=set(surface.columns):
            changes = surface.copy()*0.0
        else:
            # align by columns & index
            A = surface.reindex(index=self.deltas, columns=surface.columns)
            B = self.prev_surface.reindex(index=self.deltas, columns=surface.columns)
            changes = A - B
        self.prev_surface = surface.copy()

        # estimate call/put centroids for the "centroids" chart (OI/vol weighted strikes)
        if not chain_df.empty:
            w_call = chain_df["Call Vol"].to_numpy()
            w_put  = chain_df["Put Vol"].to_numpy()
            strikes = chain_df["Strike"].to_numpy()
            c_cent = (w_call*strikes).sum()/max(w_call.sum(),1)
            p_cent = (w_put*strikes).sum()/max(w_put.sum(),1)
        else:
            c_cent=p_cent=np.nan

        return chain_df, surface, changes, c_cent, p_cent, S

    def latest(self)->MarketSnapshot:
        now=time.time()
        # expirations
        iso_exps_raw=self._list_expirations()
        # convert back to ISO for fetcher
        iso_for_fetch=[pd.to_datetime(x).strftime("%Y-%m-%d") for x in iso_exps_raw][:self.num_exp] if iso_exps_raw else []
        chain, surface, changes, call_cent, put_cent, S = self._build_chain_surface(iso_for_fetch)

        # intraday timeline (for price overlay)
        intr = self._fetch_intraday()
        latest_px = S or (intr[-1]["close"] if intr else np.nan)

        # update histories (simple drift for call/put "flow" proxies using volumes)
        self.ts_idx += 1
        t=self.ts_idx
        # proxies: sum of call/put vols from chain to create flowing series
        cf = float(chain["Call Vol"].sum()) if not chain.empty else 0.0
        pf = float(chain["Put Vol"].sum()) if not chain.empty else 0.0
        self.call_flow_hist.append((t, (self.call_flow_hist[-1][1] if self.call_flow_hist else 0) + (cf - (pf*0.4))))
        self.put_flow_hist.append((t, (self.put_flow_hist[-1][1] if self.put_flow_hist else 0) + (pf - (cf*0.4))))
        self.px_hist.append((t, latest_px if latest_px is not None else np.nan))
        self.call_cent_hist.append((t, call_cent if call_cent==call_cent else np.nan))
        self.put_cent_hist.append((t,  put_cent if  put_cent==put_cent else np.nan))
        self.px_cent_hist.append((t, latest_px if latest_px is not None else np.nan))

        order_flow=pd.DataFrame(self.call_flow_hist, columns=["time","call_flow"]).merge(
                     pd.DataFrame(self.put_flow_hist, columns=["time","put_flow"]), on="time").merge(
                     pd.DataFrame(self.px_hist,        columns=["time","price"]),    on="time")
        centroids=pd.DataFrame({'time':[x for x,_ in self.call_cent_hist],
                                'call_c':[y for _,y in self.call_cent_hist],
                                'put_c':[y for _,y in self.put_cent_hist],
                                'px':[y for _,y in self.px_cent_hist]})

        # depth from DEEP book (if available)
        book = self._fetch_book()
        ladder, bids, asks = [], [], []
        try:
            # aggregate a few levels
            for lvl in (book.get("bids") or [])[:15]:
                ladder.append(float(lvl["price"])); bids.append(float(lvl["size"])); asks.append(0.0)
            for lvl in (book.get("asks") or [])[:15]:
                ladder.append(float(lvl["price"])); bids.append(0.0); asks.append(float(lvl["size"]))
        except Exception:
            pass
        if ladder:
            df_depth = pd.DataFrame({"price": np.array(ladder),
                                     "bid":   np.array(bids),
                                     "ask":   np.array(asks)}).sort_values("price")
        else:
            # soft fallback
            px0 = latest_px or 100.0
            prices = np.arange(px0-10, px0+10, 1.0)
            df_depth = pd.DataFrame({"price":prices, "bid": np.zeros_like(prices), "ask": np.zeros_like(prices)})

        # simple tiles/badges
        tick_summary={'TICK': int((cf - pf)/1000.0) if (cf or pf) else 0,
                      'AD_TICKERS': {'SPY': int(np.sign((cf-pf)) * np.random.randint(0,150))},
                      'ADV_DEC_VOL': {'SPY': int(np.sign((cf-pf)) * np.random.randint(0,1000))}}
        # mock vix term until you wire your own source (or compute from OTM options)
        vix_base = max(10.0, float(surface.mean().mean())/2.0 if surface.size else 15.0)
        vix_term=pd.DataFrame({'tenor':['VIX1D','VIX9D','VIX','VIX3M','VIX6M','VIX1Y'],
                               'value':[vix_base*0.7, vix_base*0.9, vix_base, vix_base*1.05, vix_base*1.1, vix_base*1.2]})
        top_weighted=pd.DataFrame({'name':['AAPL','NVDA','MSFT','META','TSLA','GOOGL','AMZN','AVGO','BRK/B','JPM'],
                                   'ret':np.random.default_rng().normal(0,1.2,10)})

        # ODTE skew placeholder
        m=np.linspace(-20,20,30); odte=pd.DataFrame({'moneyness':m,'skew':0.1*m+np.random.default_rng().normal(0,0.8,len(m))})

        # metrics
        gex_ratio = ((chain["Gamma"].clip(lower=0).sum()+1) / (abs(chain["Gamma"].clip(upper=0).sum())+1)) if not chain.empty else 1.0
        dex_ratio = ((chain["Delta"].clip(lower=0).sum()+1) / (abs(chain["Delta"].clip(upper=0).sum())+1)) if not chain.empty else 1.0
        call_total, put_total = (int(chain['Call Vol'].sum()), int(chain['Put Vol'].sum())) if not chain.empty else (0,0)
        vix_ratio = float(vix_term.loc[vix_term['tenor']=='VIX1D','value'].iloc[0] / vix_term.loc[vix_term['tenor']=='VIX','value'].iloc[0])
        five_day_sum = float(np.sum(np.diff(centroids['px'].values[-120:])[-5:])) if len(centroids)>=6 else 0.0
        # fake skews series for Z-score (you can compute from real surface history later)
        call_skew=pd.DataFrame({'time':centroids['time'], 'skew': np.cumsum(np.random.default_rng().normal(0,0.3,len(centroids)))})
        put_skew =pd.DataFrame({'time':centroids['time'], 'skew': np.cumsum(np.random.default_rng().normal(0,0.3,len(centroids)))})
        def zscore(series): s=np.array(series); return 0 if s.std()==0 else (s[-1]-s.mean())/s.std()
        z_call = float(zscore(call_skew['skew'][-60:])); z_put  = float(zscore(put_skew['skew'][-60:]))

        state={'gamma_condition':'Put Dominated' if (pf>cf) else 'Call Dominated',
               'skew_condition':'Risk-Off' if (changes.mean().mean() if changes.size else 0)<0 else 'Risk-On',
               'trend_score': float(np.round(np.random.default_rng().normal(0,2.0),2))}
        metrics={'live_beta': 1.0, 'gex_ratio': round(gex_ratio,2), 'dex_ratio': round(dex_ratio,2),
                 'call_total': call_total, 'put_total': put_total, 'vix_ratio': round(vix_ratio,2),
                 'five_day_sum': round(five_day_sum,2), 'z_call': round(z_call,2), 'z_put': round(z_put,2),
                 'now': time.strftime("%Y-%m-%d %H:%M:%S")}

        return MarketSnapshot(now,self.symbol,order_flow,centroids,df_depth,tick_summary,chain,surface,changes,
                              call_skew,put_skew,vix_term,top_weighted,odte,state,metrics)

# ---------- Tab pages (unchanged from your last version) ----------
class PageChainVol(QWidget):
    def __init__(self):
        super().__init__(); grid=QGridLayout(self); grid.setHorizontalSpacing(8); grid.setVerticalSpacing(8)
        for c,s in [(0,6),(6,6)]: grid.setColumnStretch(c,s)
        for r,s in [(0,5),(3,5),(6,2)]: grid.setRowStretch(r,s)
        self.chain = TableWidget("Options Chain", ["Call Vol","Strike","Put Vol","Net OI","Gamma","Delta","Vanna"])
        self.vol_surface = MatrixTable("Vol Surface (IV)", value_type="iv")
        self.vol_changes = MatrixTable("Vol Changes (ΔIV%)", value_type="deltapct")
        self.summary = TableWidget("Expiry/Skew Summary", ["Expiry","Call IV%","Put IV%","Skew P/C","CallWing","PutWing"])
        grid.addWidget(self.chain, 0,0,6,6)
        grid.addWidget(self.vol_surface, 0,6,3,6)
        grid.addWidget(self.vol_changes, 3,6,3,6)
        grid.addWidget(self.summary, 6,0,2,12)
    def refresh(self, s: MarketSnapshot):
        self.chain.update_data(s.chain)
        self.vol_surface.update_from_df(s.vol_surface)
        self.vol_changes.update_from_df(s.vol_changes)
        rows=[]
        pos = s.vol_surface.index >= 25
        neg = s.vol_surface.index <= -25
        for exp in s.vol_surface.columns:
            call_iv=float(np.nanmean(s.vol_surface.loc[pos, exp]))
            put_iv =float(np.nanmean(s.vol_surface.loc[neg, exp]))
            skew_pc=call_iv/put_iv if put_iv else np.nan
            rows.append((exp, round(call_iv,2), round(put_iv,2), round(skew_pc,3),
                         round(call_iv-put_iv,2), round(put_iv-call_iv,2)))
        df=pd.DataFrame(rows, columns=["Expiry","Call IV%","Put IV%","Skew P/C","CallWing","PutWing"])
        self.summary.update_data(df)

class PageFlowDepth(QWidget):
    def __init__(self):
        super().__init__(); grid=QGridLayout(self); grid.setHorizontalSpacing(8); grid.setVerticalSpacing(8)
        self.flow = PGLineChart("SPX Directional Order Flow", with_price=True, two_curves=True)
        self.centroids = PGLineChart("SPX Centroids", with_price=True, two_curves=True)
        self.depth = PGDepth("Depth View")
        self.badges = TableWidget("Snapshot Badges", ["Metric","Value"])
        self.tiles = TableWidget("Market Tiles", ["Metric","Value"])
        grid.addWidget(self.flow, 0,0,3,6)
        grid.addWidget(self.centroids, 0,6,3,6)
        grid.addWidget(self.depth, 0,12,3,4)
        grid.addWidget(self.badges, 3,0,1,8)
        grid.addWidget(self.tiles, 3,8,1,8)
    def refresh(self, s: MarketSnapshot):
        df=s.order_flow
        self.flow.update_dual(df['time'].values, df['call_flow'].values, df['put_flow'].values, price=df['price'].values)
        c=s.centroids
        self.centroids.update_dual(c['time'].values, c['call_c'].values, c['put_c'].values, price=c['px'].values)
        self.depth.update(s.depth['price'].values, s.depth['bid'].values, s.depth['ask'].values)
        badge_df=pd.DataFrame([
            ("Gamma Condition", s.state_labels['gamma_condition']),
            ("Skew Condition",  s.state_labels['skew_condition']),
            ("Trend Score",     s.state_labels['trend_score']),
            ("Live Beta",       s.metrics['live_beta']),
            ("GEX Ratio",       s.metrics['gex_ratio']),
            ("DEX Ratio",       s.metrics['dex_ratio']),
            ("VIX Ratio",       s.metrics['vix_ratio']),
            ("5D Sum",          s.metrics['five_day_sum']),
            ("CALL VOL",        s.metrics['call_total']),
            ("PUT VOL",         s.metrics['put_total']),
            ("Now",             s.metrics['now']),
        ], columns=["Metric","Value"])
        self.badges.update_data(badge_df)
        tiles_df=pd.DataFrame(list(s.tick_summary.items()), columns=["Metric","Value"])
        self.tiles.update_data(tiles_df)

class PageSkewVIX(QWidget):
    def __init__(self):
        super().__init__(); grid=QGridLayout(self); grid.setHorizontalSpacing(8); grid.setVerticalSpacing(8)
        self.call_skew = PGLineChart("Call Skew Timeseries")
        self.put_skew  = PGLineChart("Put Skew Timeseries")
        self.vix_term  = PGBarChart("VIX Term Structure")
        self.topw      = PGWeightedBar("TOP WEIGHTED")
        self.odte      = PGBarChart("ODTE SKEW")
        self.summary   = TableWidget("Skew Snapshot", ["Metric","Value"])
        grid.addWidget(self.call_skew, 0,0,2,6)
        grid.addWidget(self.put_skew,  2,0,2,6)
        grid.addWidget(self.vix_term,   0,6,2,4)
        grid.addWidget(self.topw,       2,6,2,4)
        grid.addWidget(self.odte,       0,10,2,4)
        grid.addWidget(self.summary,    2,10,2,4)
    def refresh(self, s: MarketSnapshot):
        self.call_skew.update_single(s.call_skew['time'].values, s.call_skew['skew'].values)
        self.put_skew.update_single(s.put_skew['time'].values,  s.put_skew['skew'].values)
        self.vix_term.update(s.vix_term['tenor'].tolist(), s.vix_term['value'].tolist())
        self.topw.update(s.top_weighted['name'].tolist(), s.top_weighted['ret'].tolist())
        self.odte.update([f"{x:.0f}" for x in s.odte_skew['moneyness']], s.odte_skew['skew'])
        sumdf=pd.DataFrame([("Call Skew z", s.metrics['z_call']),
                            ("Put Skew z",  s.metrics['z_put']),
                            ("VIX Ratio",   s.metrics['vix_ratio'])], columns=["Metric","Value"])
        self.summary.update_data(sumdf)

# ---------- Main ----------
class Main(QMainWindow):
    def __init__(self, provider, theme_name='dark'):
        super().__init__()
        self.setWindowTitle(APP_TITLE); self.setMinimumSize(1600, 950)
        self.theme_name=theme_name
        central=QWidget(); self.setCentralWidget(central); v=QVBoxLayout(central)
        self.tabs=QTabWidget(); v.addWidget(self.tabs)
        self.page1=PageChainVol(); self.page2=PageFlowDepth(); self.page3=PageSkewVIX()
        self.tabs.addTab(self.page1, "Chain + Vol")
        self.tabs.addTab(self.page2, "Flow + Depth")
        self.tabs.addTab(self.page3, "Skew + VIX")
        # hotkeys
        act_full=QAction("Toggle Fullscreen", self); act_full.setShortcut(QKeySequence(Qt.Key_F11)); act_full.triggered.connect(self._toggle_fullscreen); self.addAction(act_full)
        act_quit=QAction("Quit", self); act_quit.setShortcut(QKeySequence(Qt.Key_Escape)); act_quit.triggered.connect(self._exit_fullscreen); self.addAction(act_quit)
        act_theme=QAction("Next Theme", self); act_theme.setShortcut(QKeySequence(Qt.Key_T)); act_theme.triggered.connect(self._next_theme); self.addAction(act_theme)
        # data worker
        self.latest=None
        self.worker=DataWorker(provider, interval_ms=900 if isinstance(provider, IEXProvider) else 650)
        self.worker.snapshot_ready.connect(self._on_snapshot)
        self.worker.start()
        # UI timer
        self.timer=QTimer(self); self.timer.setInterval(350); self.timer.timeout.connect(self.refresh); self.timer.start()
    def closeEvent(self, e):
        if self.worker.isRunning(): self.worker.requestInterruption(); self.worker.wait(1000)
        return super().closeEvent(e)
    def _toggle_fullscreen(self): self.showNormal() if self.isFullScreen() else self.showFullScreen()
    def _exit_fullscreen(self): self.showNormal() if self.isFullScreen() else QApplication.quit()
    def _next_theme(self):
        names=list(THEMES.keys()); i=(names.index(self.theme_name)+1)%len(names); self.theme_name=names[i]
        apply_theme_globals(self.theme_name); apply_app_style(QApplication.instance()); self.repaint()
    def _on_snapshot(self, snap): self.latest=snap
    def refresh(self):
        if self.latest is None: return
        s=self.latest
        idx=self.tabs.currentIndex()
        if idx==0: self.page1.refresh(s)
        elif idx==1: self.page2.refresh(s)
        else: self.page3.refresh(s)

class DataWorker(QThread):
    snapshot_ready = Signal(object)
    def __init__(self, provider, interval_ms=650):
        super().__init__(); self.provider=provider; self.interval_ms=interval_ms
    def run(self):
        while not self.isInterruptionRequested():
            try:
                snap=self.provider.latest()
                self.snapshot_ready.emit(snap)
            except Exception as e:
                # Keep UI alive if API blips
                print("provider error:", e)
            self.msleep(self.interval_ms)

# ---------- CLI ----------
def build_provider(source, symbol, n_expiries, iex_token, sandbox):
    if source=="iex":
        return IEXProvider(symbol, token=iex_token or os.environ.get("IEX_TOKEN",""),
                           num_expiries=n_expiries, sandbox=sandbox)
    return SyntheticProvider(symbol, num_expiries=n_expiries)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data', choices=['synthetic','iex'], default='synthetic')
    ap.add_argument('--symbol', default='SPY')
    ap.add_argument('--theme', choices=list(THEMES.keys()), default='dark')
    ap.add_argument('--expiries', type=int, default=10, help='number of monthly expiries to use')
    ap.add_argument('--iex-token', default=os.environ.get("IEX_TOKEN",""), help='IEX Cloud token (or set IEX_TOKEN env)')
    ap.add_argument('--iex-sandbox', action='store_true', help='use IEX sandbox base URL')
    args=ap.parse_args()
    apply_theme_globals(args.theme)
    app=QApplication(sys.argv); apply_app_style(app)
    win=Main(build_provider(args.data,args.symbol,args.expiries,args.iex_token,args.iex_sandbox), args.theme); win.show()
    sys.exit(app.exec())

if __name__=="__main__": main()
