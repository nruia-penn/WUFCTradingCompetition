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