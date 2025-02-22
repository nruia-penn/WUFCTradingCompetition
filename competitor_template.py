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

        self.symbols: List[str] = ["NVR", "CPMD", "MFH", "ANG", "TVW"]
        
    def strategy(self):
        if not hasattr(self, "price_history"):
            self.price_history = {s: [] for s in self.symbols}

        population_size = 8
        generations = 8
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

            risk_fraction = 0.01
            balance = self.get_balance()
            risk_per_trade = risk_fraction * balance
            
            mid_price = (best_bid + best_ask) / 2
            stop_loss_distance = volatility * mid_price
            if stop_loss_distance < 1e-6:
                stop_loss_distance = 1e-6
            
            optimal_size = risk_per_trade / stop_loss_distance
            
            scaling_factor = min(1.0, abs(signal) / sharpe_threshold)
            final_order_size = int(max(1, optimal_size * scaling_factor))
            
            if signal > sharpe_threshold:
                self.create_limit_order(price=best_bid + 0.01, size=final_order_size, side='buy', symbol=symbol)
            elif signal < -sharpe_threshold:
                self.create_limit_order(price=best_ask - 0.01, size=final_order_size, side='sell', symbol=symbol)