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
        super().__init__(
            participant_id=participant_id,
            balance=balance,
            order_book_manager=order_book_manager,
            order_queue_manager=order_queue_manager
        )

        self.symbols: List[str] = ["NVR", "CPMD", "MFH", "ANG", "TVW"]
        
    def strategy(self):
        # 1. Rolling price collection
        if not hasattr(self, "price_history"):
            self.price_history = {s: [] for s in self.symbols}

        # 2. GP parameters
        population_size = 8
        generations = 3
        lookback = 15
        sharpe_evaluate_window = 10
        sharpe_threshold = 0.0825
        
        # 3. Create GP population if missing
        if not hasattr(self, "gp_population"):
            self.gp_population = []
            for _ in range(population_size):
                individual = {
                    "weights": [random.uniform(-1, 1) for _ in range(3)],
                    "fitness": None
                }
                self.gp_population.append(individual)
        
        # 4. Best rule fallback
        if not hasattr(self, "best_rule"):
            self.best_rule = {"weights": [0, 0, 0], "fitness": -999}
        
        # Track how many times we've run GP
        if not hasattr(self, "gp_runs"):
            self.gp_runs = 0
        
        # Update mid-prices
        for symbol in self.symbols:
            snapshot = self.get_order_book_snapshot(symbol)
            # If no order book data, skip
            if not snapshot['bids'] or not snapshot['asks']:
                continue

            # Extract best bid/ask carefully, skipping None
            raw_best_bid = snapshot['bids'][0][0]
            raw_best_ask = snapshot['asks'][0][0]
            if raw_best_bid is None or raw_best_ask is None:
                # Skip if we can't get numeric prices
                continue

            mid_price = (raw_best_bid + raw_best_ask) / 2
            self.price_history[symbol].append(mid_price)
            # keep only 'lookback' prices
            if len(self.price_history[symbol]) > lookback:
                self.price_history[symbol].pop(0)

        # 5. Run GP if we haven't done it yet
        if self.gp_runs < 1:
            for _ in range(generations):
                # Evaluate fitness
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

                        # Build daily returns for this window
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
                
                # Sort population by fitness
                self.gp_population.sort(key=lambda x: x["fitness"], reverse=True)
                half = population_size // 2
                parents = self.gp_population[:half]
                children = []
                
                while len(children) < half:
                    p1, p2 = random.sample(parents, 2)
                    child_w = [(a + b)/2.0 for a, b in zip(p1["weights"], p2["weights"])]
                    # mutate one weight
                    idx = random.randint(0, 2)
                    child_w[idx] += random.uniform(-0.1, 0.1)
                    children.append({"weights": child_w, "fitness": None})
                
                self.gp_population = parents + children
            
            # After generations, pick best
            if self.gp_population:
                for ind in self.gp_population:
                    if ind["fitness"] is None:
                        ind["fitness"] = -999
                self.best_rule = max(self.gp_population, key=lambda x: x["fitness"])
            else:
                self.best_rule = {"weights": [0, 0, 0], "fitness": -999}
            
            self.gp_runs += 1
        
        # 6. Use best rule to place trades
        w0, w1, w2 = self.best_rule["weights"]
        
        for symbol in self.symbols:
            snapshot = self.get_order_book_snapshot(symbol)
            if not snapshot['bids'] or not snapshot['asks']:
                continue
            # If top-of-book is None, skip
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
