# liquidity_bot.py

import time
import random
import numpy as np
from typing import List, Optional, Dict
from enum import Enum

from Participant import Participant
from orderForTree import Order
from PriceGenerator import PriceGenerator


class LocalOrderStatus(Enum):
    IN_ORDER_BOOK = 1
    CANCELLED = 2
    FILLED = 3


class LiquidityBot(Participant):
    def __init__(
            self,
            participant_id: str,
            order_queue_manager,
            price_generator: PriceGenerator,
            order_book_manager,
            
            balance: float = 100000.0,
            symbols: Optional[List[str]] = None,
            average_interval: float = 2.0,
            interval_jitter: float = 0.5,
            mean_quantity: int = 80,
            quantity_std_dev: float = 18.0,
            market_order_probability: float = 0.33,
            base_spread: float = 0.5,
            levels: int = 3,
            level_spacing: float = 0.5,
            max_position: int = 1000,
            max_balance_use_fraction: float = 0.3,
            max_order_age: float = 30.0,
            stale_check_interval: float = 30.0,
            max_spread_width: float = 5.0,
            use_price_generator_duration: float = 4.0
    ):

        super().__init__(
            participant_id=participant_id,
            balance=balance,
            order_book_manager=order_book_manager,
            order_queue_manager=order_queue_manager
        )

        self.__symbols: List[str] = symbols if symbols is not None else list(price_generator.securities.keys())
        self.__average_interval: float = average_interval
        self.__interval_jitter: float = interval_jitter
        self.__mean_quantity: int = mean_quantity
        self.__quantity_std_dev: float = quantity_std_dev
        self.__market_order_probability: float = market_order_probability
        self.__base_spread: float = base_spread
        self.__levels: int = levels
        self.__level_spacing: float = level_spacing
        self.__max_position: int = max_position
        self.__max_balance_use_fraction: float = max_balance_use_fraction
        self.__max_order_age: float = max_order_age
        self.__stale_check_interval: float = stale_check_interval
        self.__max_spread_width: float = max_spread_width
        self.__use_price_generator_duration: float = use_price_generator_duration

        self.random_state = np.random.RandomState(seed=random.randint(0, 2 ** 32 - 1))

        self.__active_orders: Dict[str, List[Dict]] = {sym: [] for sym in self.__symbols}
        self.__last_stale_check: float = 0.0

        self.initial_prices = {}
        for sym in self.__symbols:
            p = price_generator.get_current_price(sym)
            self.initial_prices[sym] = p if p is not None else 1.0

        self.symbol_index = 0


    def strategy(self):
        """
        Main market-making strategy that:
          1. Cancels stale orders if enough time has passed.
          2. For each symbol, retrieves best bid/ask and places buy/sell limit orders 
             around the midpoint, adjusted by position and balance constraints.
        """
        if not self.__symbols:
            return

        symbol = self.__symbols[self.symbol_index % len(self.__symbols)]
        self.symbol_index += 1

        current_time = time.time()
        use_book_prices = (current_time - self.initial_prices.get(symbol, current_time) > self.__use_price_generator_duration)

        # Retrieve order book snapshot
        best_bid = self.get_orderbook_price(symbol, True)
        best_ask = self.get_orderbook_price(symbol, False)

        

        if use_book_prices and best_bid and best_ask:
            if best_bid > 0 and best_ask > 0:
                use_generator_price = False
            else:
                use_generator_price = True
        else:
            use_generator_price = True

        if self.random_state.rand() < self.__market_order_probability:
            self.place_random_market_order(symbol)
        else:
            if use_generator_price:
                current_price = self.get_order_book_snapshot(symbol).get('mid_price', self.initial_prices.get(symbol, 1.0))
                if current_price <= 0:
                    return
                self.place_liquidity_ladder_using_price(symbol, current_price)
            else:
                self.place_liquidity_ladder_using_book(symbol, best_bid, best_ask)

        self.sleep_random_interval()

    def refresh_stale_orders(self):
        """
        Cancels orders that have exceeded the maximum allowed age.
        """
        current_time = time.time()
        for sym in self.__symbols:
            new_list = []
            for odict in self.__active_orders[sym]:
                order_id = odict['id']
                placed_time = odict['time']
                status = odict['status']
                age = current_time - placed_time
                if status == LocalOrderStatus.IN_ORDER_BOOK and age > self.__max_order_age:
                    removed = self.remove_order(order_id, sym)
                    if removed:
                        odict['status'] = LocalOrderStatus.CANCELLED
                    else:
                        odict['status'] = LocalOrderStatus.CANCELLED
                else:
                    new_list.append(odict)
            self.__active_orders[sym] = new_list

    def place_liquidity_ladder_using_price(self, symbol: str, current_price: float):
        """
        Places a ladder of limit orders based on a generated price.
        """
        spread = self.adaptive_spread(symbol)
        spread = min(spread, self.__max_spread_width)
        for _ in range(self.__levels):
            buy_offset = self.random_state.exponential(scale=self.__level_spacing)
            sell_offset = self.random_state.exponential(scale=self.__level_spacing)
            buy_price = max(0.01, current_price - spread - buy_offset)
            sell_price = current_price + spread + sell_offset
            buy_price_rounded = round(buy_price, 2)
            sell_price_rounded = round(sell_price, 2)
            self.place_limit_order_with_risk_check(symbol, 'buy', buy_price_rounded)
            self.place_limit_order_with_risk_check(symbol, 'sell', sell_price_rounded)

    def place_liquidity_ladder_using_book(self, symbol: str, best_bid: float, best_ask: float):
        """
        Places a ladder of limit orders based on the order book's best bid and ask.
        """
        inside_spread = max(0.0001, best_ask - best_bid)
        stdev = 0.60
        for _ in range(self.__levels):
            buy_price = self.random_state.normal(loc=best_bid, scale=stdev)
            sell_price = self.random_state.normal(loc=best_ask, scale=stdev)
            buy_price = max(0.01, buy_price)
            sell_price = max(0.01, sell_price)
            buy_price_rounded = round(buy_price, 2)
            sell_price_rounded = round(sell_price, 2)
            self.place_limit_order_with_risk_check(symbol, 'buy', buy_price_rounded)
            self.place_limit_order_with_risk_check(symbol, 'sell', sell_price_rounded)

    def place_limit_order_with_risk_check(self, symbol: str, side: str, price: float):
        """
        Places a limit order after performing risk checks on quantity and balance.
        """
        quantity = self.dynamic_order_quantity(symbol)
        if quantity <= 0:
            return

        current_position = self.get_portfolio.get(symbol, 0)

        if side == 'buy' and (current_position + quantity) > self.__max_position:
            quantity = max(1, self.__max_position - current_position)
        elif side == 'sell' and (current_position - quantity) < -self.__max_position:
            quantity = max(1, current_position + self.__max_position)

        if side == 'buy':
            max_cost = price * quantity
            allowed_cost = self.get_balance * self.__max_balance_use_fraction
            if max_cost > allowed_cost:
                quantity = max(1, int(allowed_cost / price))

        if quantity <= 0:
            return

        order_id = self.create_limit_order(
            price=price,
            size=quantity,
            side=side,
            symbol=symbol
        )

        if order_id:
            self.track_order(symbol, order_id, LocalOrderStatus.IN_ORDER_BOOK)

    def place_random_market_order(self, symbol: str):
        """
        Places a random market order based on defined probability and quantity.
        """
        side = random.choice(['buy', 'sell'])
        quantity = self.dynamic_order_quantity(symbol)
        if quantity <= 0:
            return

        if side == 'buy':
            snapshot = self.get_order_book_snapshot(symbol)
            best_bid = snapshot.get('best_bid', self.initial_prices.get(symbol, 1.0))
            max_cost = best_bid * quantity
            allowed_cost = self.get_balance * self.__max_balance_use_fraction
            if max_cost > allowed_cost:
                quantity = max(1, int(allowed_cost / best_bid))

        if quantity <= 0:
            return

        order_id = self.create_market_order(
            size=quantity,
            side=side,
            symbol=symbol
        )


    def track_order(self, symbol: str, order_id: str, status: LocalOrderStatus):
        """
        Tracks the order by storing its ID and timestamp.
        """
        self.__active_orders[symbol].append({
            'id': order_id,
            'time': time.time(),
            'status': status
        })

    def sleep_random_interval(self):
        """
        Sleeps for a random interval based on exponential and normal distributions.
        """
        lam = 1.0 / max(0.1, self.__average_interval)
        sleep_time = self.random_state.exponential(scale=1 / lam)
        if self.__interval_jitter > 0:
            sleep_time += self.random_state.normal(loc=0, scale=self.__interval_jitter)
        time.sleep(max(0.5, sleep_time))

    def dynamic_order_quantity(self, symbol: str) -> int:
        """
        Calculates dynamic order quantity based on recent volatility.
        """
        vol = self.get_recent_volatility(symbol)
        base_q = self.__mean_quantity / (1.0 + vol)
        q = abs(self.random_state.normal(base_q, self.__quantity_std_dev))
        return max(1, int(q))

    def adaptive_spread(self, symbol: str) -> float:
        """
        Adapts spread based on recent volatility.
        """
        vol = self.get_recent_volatility(symbol)
        raw_spread = self.__base_spread * (1.0 + vol)
        return max(0.01, raw_spread)

    def get_recent_volatility(self, symbol: str) -> float:
        """
        Retrieves recent volatility for the given symbol.
        """
        return self.random_state.rand()
