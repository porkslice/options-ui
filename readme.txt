# install deps (once)
pip install PySide6 pyqtgraph numpy pandas requests

# IEX live
python options_dashboard_tabs.py --data iex --symbol SPY --theme purple --expiries 12 --iex-token pk_xxxxxxxxxxxxx

# IEX sandbox (if you want to test without burning credits)
python options_dashboard_tabs.py --data iex --symbol SPY --theme purple --expiries 12 --iex-token Tsk_xxxxxxxxx --iex-sandbox


and to use the options ui with synthetic data so you can mess around with it use options_ui and use  
python options_ui.py --data synthetic --symbol SPX --theme purple --expiries 20 

 
