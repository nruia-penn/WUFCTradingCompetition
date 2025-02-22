import csv
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from ParticipantManager import ParticipantManager
from OrderBookManager import OrderBookManager
from PriceGenerator import PriceGenerator
from orderForTree import Order
from liquidityBot import LiquidityBot
import threading
from competitor_template import CompetitorBoilerplate
import random


ROUND_ENDED_EVENT = threading.Event()

app = Flask(__name__)
app.secret_key = 'secret_key_for_session'


participant_manager = ParticipantManager()
price_generator = PriceGenerator(seed=42)
order_book_manager = OrderBookManager(participant_manager)
price_generator.start()


order_queue_manager = order_book_manager.orderQueue
has_started = False
returns = []
# Add some securities
csv_file = 'possibleScenarios.csv'

symbol_map = {}

with open(csv_file, 'r', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        sym = row["symbol"].strip()

        drift_val = float(row["drift"])
        vol_val   = float(row["volatility"])

        if sym not in symbol_map:
            symbol_map[sym] = []
        symbol_map[sym].append((drift_val, vol_val))


securities = [
    {"symbol": "NVR",  "initial_price": 150.0, "drift": 0.01,  "volatility": 0.05, "time_step": 1.0},
    {"symbol": "CPMD", "initial_price": 175.0, "drift": 0.0005,"volatility": 0.25, "time_step": 1.0},
    {"symbol": "MFH",  "initial_price": 200.0, "drift": 0.0007,"volatility": 0.12, "time_step": 1.0},
    {"symbol": "ANG",  "initial_price": 60.0, "drift": 0.003,"volatility": 0.012, "time_step": 1.0},
    {"symbol": "TVW",  "initial_price": 10.0, "drift": 0.009,"volatility": 0.32, "time_step": 1.0},
]

for sec in securities:
    sym = sec["symbol"]
    if sym in symbol_map and symbol_map[sym]:
        (random_drift, random_vol) = random.choice(symbol_map[sym])
        sec["drift"]      = random_drift
        sec["volatility"] = random_vol
    else:
        print(f"Warning: No CSV entries for symbol: {sym}. Using default drift/vol.")

for sec in securities:
    price_generator.add_security(
        symbol=sec["symbol"],
        initial_price=sec["initial_price"],
        drift=sec["drift"],
        volatility=sec["volatility"],
        time_step=sec["time_step"]
    )
    order_book_manager.add_order_book(sec["symbol"])


num_bots = 100
for i in range(1, num_bots + 1):
    participant_id = f"LiquidityBot_{i}"
    bot = LiquidityBot(
        participant_id=participant_id,
        order_queue_manager=order_queue_manager, 
        price_generator=price_generator,
        market_order_probability=0.2,
        order_book_manager=order_book_manager,
    )
    participant_manager.add_participant(bot)
    bot.start(ROUND_ENDED_EVENT)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        participant_id = request.form.get('participant_id', '').strip()
        if participant_id:
            p = participant_manager.get_participant(participant_id)
            if not p:
                p = CompetitorBoilerplate(participant_id=participant_id, order_book_manager=order_book_manager, order_queue_manager=order_queue_manager)
                participant_manager.add_participant(p)
                

                p.start(ROUND_ENDED_EVENT)
            session['participant_id'] = participant_id
            return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/')
def index():
    if 'participant_id' in session:
        return redirect(url_for('dashboard'))
    else:
        return redirect(url_for('login'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'participant_id' not in session:
        return redirect(url_for('login'))

    participant_id = session['participant_id']
    participant = participant_manager.get_participant(participant_id)

    if request.method == 'POST':
        side = request.form.get('side', 'buy')
        order_type = request.form.get('order_type', 'limit')
        symbol = request.form.get('symbol', securities[0]['symbol'])
        quantity_str = request.form.get('quantity', '1')
        price_str = request.form.get('price', '')

        if not quantity_str.isdigit():
            quantity = 1
        else:
            quantity = int(quantity_str)

        if order_type == 'limit':
            try:
                price = float(price_str)
            except ValueError:
                price = None
        else:
            price = None
        if order_type == 'limit' and price is not None:
            order = Order.create_limit_order(
                price=price, size=quantity, side=side, participant_id=participant_id, symbol=symbol
            )
        else:
            order = Order.create_market_order(
                size=quantity, side=side, participant_id=participant_id, symbol=symbol
            )
        if participant:
            participant._place_order_in_queue(order)

    selected_symbol = request.args.get('symbol', securities[0]['symbol'])
    snapshot = order_book_manager.get_order_book_snapshot(selected_symbol)

    bids = snapshot['bids']
    asks = snapshot['asks']

    if participant is None:
        return redirect(url_for('login'))

    holdings = participant.get_portfolio
    balance = participant.get_balance
    best_bid_prices = {sym: order_book_manager.get_order_book(sym).get_best_price(askForBid=True) for sym in holdings.keys()}
    portfolio_value = sum(best_bid_prices.get(sym, 0) * qty for sym, qty in holdings.items())
    pnl = portfolio_value + balance - 100000.0


    return render_template(
        'dashboard.html',
        participant_id=participant_id,
        securities=[sec['symbol'] for sec in securities],
        selected_symbol=selected_symbol,
        bids=bids,
        asks=asks,
        holdings=holdings,
        balance=balance,
        pnl=pnl
    )

@app.route('/orderbook_data')
def orderbook_data():
    symbol = request.args.get('symbol', securities[0]['symbol'])
    snapshot = order_book_manager.get_order_book_snapshot(symbol, 30)

    raw_bids = snapshot['bids']
    raw_asks = snapshot['asks']

    return jsonify({
        "bids": raw_bids,
        "asks": raw_asks
    })

@app.route('/participant_data')
def participant_data():
    if 'participant_id' not in session:
        return jsonify({"error": "Not logged in"}), 403

    participant_id = session['participant_id']
    participant = participant_manager.get_participant(participant_id)
    if not participant:
        return jsonify({"error": "Participant not found"}), 404

    holdings = participant.get_portfolio
    balance = participant.get_balance
    best_bid_prices = {sym: order_book_manager.get_order_book(sym).get_best_price(askForBid=True) for sym in holdings.keys()}
    portfolio_value = sum(best_bid_prices.get(sym, 0) * qty for sym, qty in holdings.items())
    pnl = portfolio_value + balance - 100000.0

    returns.append(pnl)
    return jsonify({
        "holdings": holdings,
        "balance": balance,
        "pnl": pnl
    })

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=num_bots*50)

@app.route('/call_all_strategies')
def call_all_strategies():
    if ROUND_ENDED_EVENT.is_set():
        return "Round has ended. No more trading."
    participants_copy = list(participant_manager.participants.values())

    for participant in participants_copy:
        if hasattr(participant, 'strategy') and callable(participant.strategy):
            executor.submit(
                participant.strategy
                #order_queue_manager,
                #order_book_manager
            )

    return "All strategies submitted."


@app.route('/end_round')
def end_round():

    participant_id = session.get('participant_id', None)
    if not participant_id:
        return redirect(url_for('login'))

    participant = participant_manager.get_participant(participant_id)
    if not participant:
        return redirect(url_for('login'))
    
    ROUND_ENDED_EVENT.set()

    
    holdings = participant.get_portfolio
    balance = participant.get_balance
    best_bid_prices = {sym: order_book_manager.get_order_book(sym).get_best_price(askForBid=True) for sym in holdings.keys()}
    portfolio_value = sum(best_bid_prices.get(sym, 0) * qty for sym, qty in holdings.items())
    final_profit = portfolio_value + balance - 100000.0
    sharpe = calculate_sharpe_ratio(returns)

    return render_template(
        'ending.html',
        profit=final_profit,
        sharpe_ratio=sharpe
    )

import numpy as np



def calculate_sharpe_ratio(pnl_values, risk_free_rate=4.0):

    pnl_values = np.array(pnl_values, dtype=np.float64)

    nonzero_indices = np.where(pnl_values > 1e-12)[0]
    if len(nonzero_indices) == 0:
        
        return 0.0
    start_idx = nonzero_indices[0]
    pnl_values = pnl_values[start_idx:]

    if len(pnl_values) < 2:
        return 0.0


    denominators = np.clip(pnl_values[:-1], 1e-12, None)
    daily_returns = np.diff(pnl_values) / denominators

    daily_rfr = (risk_free_rate / 100.0) / 252.0

    excess_returns = daily_returns - daily_rfr

    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns, ddof=1)

    if std_excess_return == 0.0:
        return 0.0

    sharpe_ratio = (mean_excess_return / std_excess_return) * np.sqrt(252)

    return sharpe_ratio


@app.teardown_appcontext
def shutdown_session(exception=None):
    pass



if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8081)