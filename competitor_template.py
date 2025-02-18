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
        
## ONLY EDIT THE CODE BELOW 

    def strategy(self):
        """
        An adaptive rebalancing strategy designed to improve risk-adjusted performance:
        
        1. For each symbol, get the current order book snapshot.
        2. Calculate the mid-price and estimate volatility using the relative spread.
        3. Adjust the order size based on the estimated volatility.
        4. Adjust the limit order offset based on volatility:
        - When volatility is low, use a larger offset (more aggressive pricing).
        - When volatility is high, use a tighter offset (more conservative pricing).
        5. Place a buy order if current holdings are below target or a sell order if above target.
        """
        print('got here')
        snapshot = self.get_order_book_snapshot("NVR")
        bids = snapshot.get('bids', [])
        asks = snapshot.get('asks', [])
        if not bids or not asks:
            return
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        if best_ask < 150:
            order_id = self.create_limit_order(price=149.5, size=10, side='buy', symbol="NVR")