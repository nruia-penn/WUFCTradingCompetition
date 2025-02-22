"""
Boilerplate Competitor Class
----------------------------

Instructions for Participants:
1. Do not import external libraries beyond what's provided.
2. Focus on implementing the `strategy()` method with your trading logic.
3. Use the provided methods to interact with the exchange:
   - self.create_limit_order(price, size, side, symbol) -> order_id if succesfully placed in order book or None
   - self.create_market_order(size, side, symbol) -> order_id if succesfully placed in order book or None
   - self.remove_order(order_id, symbol) -> bool
   - self.get_order_book_snapshot(symbol) -> dict
   - self.get_balance() -> float
   - self.get_portfolio() -> dict

   
Happy Trading!
"""

from typing import Optional, List, Dict
from Participant import Participant
import numpy as np
import random

class CompetitorBoilerplate(Participant):
    def __init__(self, 
                 participant_id: str,
                 order_book_manager=None,
                 order_queue_manager=None,
                 balance: float = 100000.0):
        """
        Initializes the competitor with default strategy parameters.
        
        :param participant_id: Unique ID for the competitor.
        :param order_book_manager: Reference to the OrderBookManager.
        :param order_queue_manager: Reference to the OrderQueueManager.
        :param balance: Starting balance for the competitor.
        """
        super().__init__(
            participant_id=participant_id,
            balance=balance,
            order_book_manager=order_book_manager,
            order_queue_manager=order_queue_manager
        )

        # Strategy parameters (fixed defaults)
        self.symbols: List[str] = ["NVR", "CPMD", "MFH", "ANG", "TVW"]
        
    def strategy(self):
        if not hasattr(self, "price_history"):
            self.price_history = {s: [] for s in self.symbols}

        population_size = 8
        generations = 3
        lookback = 15
        sharpe_evaluate_window = 10
        sharpe_threshold = 0.0825
        
        if not hasattr(self, "gp_population"):
            self.gp_population = []
            for _ in range(population_size):
                individual = {
                    "weights": [random.uniform(-1, 1) for _ in range(3)],
                    "fitness": None
                }
                self.gp_population.append(individual)
        
        if not hasattr(self, "best_rule"):
            self.best_rule = {"weights": [0, 0, 0], "fitness": -999}
        
        if not hasattr(self, "gp_runs"):
            self.gp_runs = 0
        
        for symbol in self.symbols:
            snapshot = self.get_order_book_snapshot(symbol)
            if not snapshot['bids'] or not snapshot['asks']:
                continue

            raw_best_bid = snapshot['bids'][0][0]
            raw_best_ask = snapshot['asks'][0][0]
            if raw_best_bid is None or raw_best_ask is None:
                continue

            mid_price = (raw_best_bid + raw_best_ask) / 2
            self.price_history[symbol].append(mid_price)
            if len(self.price_history[symbol]) > lookback:
                self.price_history[symbol].pop(0)

        if self.gp_runs < 1:
            for _ in range(generations):
                for ind in self.gp_population:
                    all_returns = []
                    for symbol in self.symbols:
                        prices = self.price_history[symbol]
                        if len(prices) < sharpe_evaluate_window:
                            continue
                        
                        momentum = (prices[-1] / prices[0]) - 1
                        volatility = np.std(prices)
                        slope = (prices[-1] - prices[0]) / (sharpe_evaluate_window + 1e-6)

                        w = ind["weights"]
                        signal = w[0]*momentum + w[1]*volatility + w[2]*slope

                        daily_returns = []
                        for i in range(len(prices) - 1):
                            ret = (prices[i+1] - prices[i]) / (prices[i] + 1e-6)
                            daily_returns.append(ret if signal > 0 else -ret)

                        if len(daily_returns) > 1:
                            sr = np.mean(daily_returns) / (np.std(daily_returns) + 1e-9)
                            all_returns.append(sr)
                    
                    if len(all_returns) == 0:
                        ind["fitness"] = -999
                    else:
                        ind["fitness"] = np.mean(all_returns)
                
                self.gp_population.sort(key=lambda x: x["fitness"], reverse=True)
                half = population_size // 2
                parents = self.gp_population[:half]
                children = []
                
                while len(children) < half:
                    p1, p2 = random.sample(parents, 2)
                    child_w = [(a + b)/2.0 for a, b in zip(p1["weights"], p2["weights"])]
                    idx = random.randint(0, 2)
                    child_w[idx] += random.uniform(-0.1, 0.1)
                    children.append({"weights": child_w, "fitness": None})
                
                self.gp_population = parents + children
            
            if self.gp_population:
                for ind in self.gp_population:
                    if ind["fitness"] is None:
                        ind["fitness"] = -999
                self.best_rule = max(self.gp_population, key=lambda x: x["fitness"])
            else:
                self.best_rule = {"weights": [0, 0, 0], "fitness": -999}
            
            self.gp_runs += 1
        
        w0, w1, w2 = self.best_rule["weights"]
        
        for symbol in self.symbols:
            snapshot = self.get_order_book_snapshot(symbol)
            if not snapshot['bids'] or not snapshot['asks']:
                continue
            raw_best_bid = snapshot['bids'][0][0]
            raw_best_ask = snapshot['asks'][0][0]
            if raw_best_bid is None or raw_best_ask is None:
                continue

            if len(self.price_history[symbol]) < 2:
                continue

            best_bid = raw_best_bid
            best_ask = raw_best_ask
            prices = self.price_history[symbol]

            momentum = (prices[-1]/prices[0] - 1) if len(prices) > 1 else 0
            volatility = np.std(prices)
            slope = (prices[-1] - prices[0]) / (len(prices) + 1e-6)

            signal = w0*momentum + w1*volatility + w2*slope

            if signal > sharpe_threshold:
                self.create_limit_order(price=best_bid + 0.01, size=10, side='buy', symbol=symbol)
            elif signal < -sharpe_threshold:
                self.create_limit_order(price=best_ask - 0.01, size=10, side='sell', symbol=symbol)

"""
Boilerplate Competitor Class
----------------------------

Instructions for Participants:
1. Do not import external libraries beyond what's provided.
2. Focus on implementing the `strategy()` method with your trading logic.
3. Use the provided methods to interact with the exchange:
   - self.create_limit_order(price, size, side, symbol) -> order_id if succesfully placed in order book or None
   - self.create_market_order(size, side, symbol) -> order_id if succesfully placed in order book or None
   - self.remove_order(order_id, symbol) -> bool
   - self.get_order_book_snapshot(symbol) -> dict
   - self.get_balance() -> float
   - self.get_portfolio() -> dict

   
Happy Trading!
"""

from typing import Optional, List, Dict

from Participant import Participant
import numpy as np

class CompetitorBoilerplate(Participant):
    def __init__(self, 
                 participant_id: str,
                 order_book_manager=None,
                 order_queue_manager=None,
                 balance: float = 100000.0):
        """
        Initializes the competitor with default strategy parameters.
        
        :param participant_id: Unique ID for the competitor.
        :param order_book_manager: Reference to the OrderBookManager.
        :param order_queue_manager: Reference to the OrderQueueManager.
        :param balance: Starting balance for the competitor.
        """
        super().__init__(
            participant_id=participant_id,
            balance=balance,
            order_book_manager=order_book_manager,
            order_queue_manager=order_queue_manager
        )

        # Strategy parameters (fixed defaults)
        self.symbols: List[str] = ["NVR", "CPMD", "MFH", "ANG", "TVW"]
        self.portfolio = {symbol: 0 for symbol in self.symbols}  # Initialize portfolio
        
## ONLY EDIT THE CODE BELOW 

def strategy(self):
    """
    An advanced adaptive rebalancing strategy that incorporates multiple statistical measures:

    1. For each symbol:
       - Get the order book snapshot and compute the mid-price.
       - Maintain a rolling history of mid-prices to compute volatility, Sharpe ratio, and a simple moving average (SMA).
       - Calculate percentage returns and use them to derive the Sharpe ratio.
       - Compute the SMA and the deviation of the current price from this average (a mean reversion indicator).
       - Calculate the order book imbalance to capture buying/selling pressure.
    2. Dynamically adjust order size based on volatility.
    3. Adjust limit order price offsets based on volatility.
    4. Incorporate inventory management: only trade if your current position deviates from the target (flat).
    5. Place orders based on combined signals (Sharpe ratio, imbalance, and mean reversion).
    """
    lookback = 3                # Number of prices in the rolling window
    sharpe_threshold = 0.00001

    # Ensure persistent price history exists for each symbol
    self.__dict__.setdefault("price_history", {symbol: [] for symbol in self.symbols})

    for symbol in self.symbols:
        snapshot = self.get_order_book_snapshot(symbol)
        if not snapshot.get('bids') or not snapshot.get('asks'):
            continue  # Skip if market data is insufficient

        best_bid = snapshot['bids'][0][0]
        print(best_bid)
        best_ask = snapshot['asks'][0][0]
        mid_price = (best_bid + best_ask) / 2.0

        # Update the rolling price history
        self.price_history[symbol].append(mid_price)
        if len(self.price_history[symbol]) > lookback:
            self.price_history[symbol].pop(0)
        
        # Proceed only if we have enough history
        if len(self.price_history[symbol]) < lookback:
            continue
        
        prices = np.array(self.price_history[symbol])
        returns = np.diff(prices) / prices[:-1]
        
        # Calculate expected return and volatility (with an epsilon to avoid division by zero)
        expected_return = np.mean(returns)
        volatility = np.std(returns) + 1e-6
        sharpe_ratio = expected_return / volatility

        # Compute simple moving average (SMA) and its deviation
        sma = np.mean(prices)
        deviation = (mid_price - sma) / sma

        # Calculate order book imbalance:
        # Positive imbalance implies higher bid volume (buy pressure), negative implies sell pressure.
        total_bid_vol = sum(bid[1] for bid in snapshot['bids'])
        total_ask_vol = sum(ask[1] for ask in snapshot['asks'])
        imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol + 1e-6)

        # Dynamically adjust order size based on volatility:
        base_order_size = 10
        size_adjustment_factor = 1 / volatility  # Inverse relation: lower volatility → larger orders
        order_size = int(base_order_size * size_adjustment_factor)
        order_size = max(1, min(order_size, 100))  # Clamp order size between 1 and 100

        # Adjust limit order offset based on volatility:
        # Higher volatility leads to a wider offset to account for rapid price movements.
        price_offset = volatility * mid_price  # This factor can be tuned based on backtests

        # Risk management: Check current inventory for the symbol; aim for a flat (zero) position.
        current_position = self.portfolio.get(symbol, 0)
        target_position = 0  # Flat position as a risk target

        # --------------------------
        # Trading Decisions
        # --------------------------
        
        # (1) Trend-following signal using Sharpe ratio and imbalance:
        #     - If the Sharpe ratio is strongly positive (uptrend) and there is buy pressure,
        #       and if your position is below the target, then place a buy order.
        if sharpe_ratio > sharpe_threshold and imbalance >= 0 and current_position <= target_position:
            buy_price = best_bid + price_offset  # Aggressively buy by offering a slightly higher price than the best bid
            self.create_limit_order(price=buy_price, size=order_size, side='buy', symbol=symbol)

        #     - Conversely, if the Sharpe ratio is strongly negative (downtrend) and there is sell pressure,
        #       and if your position is above the target, then place a sell order.
        elif sharpe_ratio < -sharpe_threshold and imbalance <= 0 and current_position >= target_position:
            sell_price = best_ask - price_offset  # Aggressively sell by offering a slightly lower price than the best ask
            self.create_limit_order(price=sell_price, size=order_size, side='sell', symbol=symbol)

        # (2) Mean reversion signal:
        #     - If the current price is significantly above its SMA, consider selling (if you have excess long exposure).
        if deviation > 0.01 and current_position >= target_position:
            self.create_limit_order(price=best_ask - price_offset, size=order_size, side='sell', symbol=symbol)
        #     - If the current price is significantly below its SMA, consider buying (if you are underexposed).
        elif deviation < -0.01 and current_position <= target_position:
            self.create_limit_order(price=best_bid + price_offset, size=order_size, side='buy', symbol=symbol)

    def create_limit_order(self, price, size, side, symbol):
        # Update portfolio based on order
        if side == 'buy':
            self.portfolio[symbol] += size
        elif side == 'sell':
            self.portfolio[symbol] -= size
        print(f"Limit Order: {side.upper()} {size} of {symbol} at {price:.4f}")

    def get_portfolio(self):
        return self.portfolio