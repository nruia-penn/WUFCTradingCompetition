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


        self.symbols: List[str] = ["NVR", "CPMD", "MFH", "ANG", "TVW"]
        self.order_size = 10  # Default order size
        self.default_spread = 0.01  # 1% spread for bid/ask adjustments
        self.max_volatility_threshold = 5  # Maximum volatility before reducing order size
        self.max_position_size = 500  # Limit per symbol to prevent excessive shorting
        
        # Track active orders and volatility memory
        self.active_orders = {}  # {order_id: symbol}
        self.volatility_memory = {symbol: [] for symbol in self.symbols}

        self.daily_returns = []
        self.previous_balance = balance

        
## ONLY EDIT THE CODE BELOW 


    def cancel_old_orders(self, symbol: str):
            """
            Cancels all outdated orders to free up capital.
            :param symbol: The trading symbol whose orders need to be refreshed.
            """
            print(f"[DEBUG] Cancelling old orders for {symbol}...")
            orders_to_cancel = list(self.active_orders.keys())
            for order_id in orders_to_cancel:
                success = self.remove_order(order_id, symbol)
                if success:
                    del self.active_orders[order_id]
                    print(f"[DEBUG] Cancelled Order {order_id} for {symbol}")

    def strategy(self):
        current_balance = self.get_balance
        if self.previous_balance > 0:
            daily_return = (current_balance - self.previous_balance) / self.previous_balance
            self.daily_returns.append(daily_return)
        self.previous_balance = current_balance

        for symbol in self.symbols:
            snapshot = self.get_order_book_snapshot(symbol)
            bids = snapshot.get('bids', [])
            asks = snapshot.get('asks', [])

            if not bids or not asks:
                print(f"[DEBUG] No market data for {symbol}, skipping...")
                continue

            best_bid = bids[0][0]
            best_ask = asks[0][0]
            mid_price = (best_bid + best_ask) / 2

            # Track recent mid-prices to calculate volatility
            self.volatility_memory[symbol].append(mid_price)
            if len(self.volatility_memory[symbol]) > 20:
                self.volatility_memory[symbol].pop(0)

            volatility = np.std(self.volatility_memory[symbol]) if len(self.volatility_memory[symbol]) > 5 else 0.01
            
            dynamic_spread = max(0.01, volatility * 0.08)

            # Adjust order size based on volatility
            adjusted_order_size = max(1, self.order_size * (1 - min(volatility / self.max_volatility_threshold, 1)))

            # Cancel old orders before placing new ones
            self.cancel_old_orders(symbol)

            portfolio = self.get_portfolio
            current_position = portfolio.get(symbol, 0) if portfolio else 0

            # Prevent excessive shorting
            if current_position < -self.max_position_size:
                print(f"[WARNING] Skipping short trade for {symbol}, already over limit.")
                continue

            # Ensure orders are placed at a realistic spread
            new_bid = round(best_bid * (1 - dynamic_spread), 2)  # Buy slightly below best bid
            new_ask = round(best_ask * (1 + dynamic_spread), 2)  # Sell slightly above best ask

            # Place buy order
            bid_order_id = self.create_limit_order(price=new_bid, size=int(adjusted_order_size), side='buy', symbol=symbol)
            if bid_order_id:
                self.active_orders[bid_order_id] = symbol
                print(f"[DEBUG] Placed Buy Order: {symbol} @ {new_bid}, Size: {int(adjusted_order_size)}")

            # Place sell order
            ask_order_id = self.create_limit_order(price=new_ask, size=int(adjusted_order_size), side='sell', symbol=symbol)
            if ask_order_id:
                self.active_orders[ask_order_id] = symbol
                print(f"[DEBUG] Placed Sell Order: {symbol} @ {new_ask}, Size: {int(adjusted_order_size)}")