from limitTreeNodes import LimitLevelTree
from limitTreeNodes import LimitLevel
import traceback
import threading

class LimitOrderBook:

    def __init__(self, symbol: str):
        self.bids = LimitLevelTree()      
        self.asks = LimitLevelTree()     
        self.best_bid = None             
        self.best_ask = None            

        self._price_levels = {}
        self._orders = {}
        self.symbol = symbol
        self.hasProcessed = 0
        self.price_floor = 0
        self.price_cap = 1000

        self.lock = threading.Lock()

    def top_level(self, askForBid: bool):
        if askForBid:
            return self.best_bid.orders.head if self.best_bid is not None else None
        else:
            return self.best_ask.orders.head if self.best_ask is not None else None

    def get_best_price(self, askForBid: bool):
        if askForBid:
            return self.best_bid.price if self.best_bid else None
        else:
            return self.best_ask.price if self.best_ask else None

    def process(self, order, quantity: int):
        if order.size == 0:
            self.remove(order)
        else:
            if order.order_id in self._orders:
                self.update(order, quantity)
            else:
                self.add(order)

    def update(self, order, size_diff):
        self._orders[order.order_id].size = order.size
        self._orders[order.order_id].parent_limit.size -= size_diff

    def remove(self, order):
       
        with self.lock:
            try:
                
                removed_order = self._orders.pop(order.order_id)
            except KeyError:
                return False

            
            level_key = (removed_order.price, removed_order.is_bid)
            try:
                limit_level = self._price_levels[level_key]
               
                limit_level.orders.remove(removed_order)

                
                if len(limit_level.orders) == 0:
                    self._price_levels.pop(level_key)
                    limit_level.remove()
                    if removed_order.is_bid:
                        if limit_level == self.best_bid:
                            self._update_best_bid()
                    else:
                        if limit_level == self.best_ask:
                            self._update_best_ask()
            except KeyError as e:
                print("Error removing price level from dictionary", e)
                traceback.print_exc()
                pass

    def add(self, order):
        if order.price < self.price_floor or order.price > self.price_cap:
            print(
                f"Ignoring order {order.order_id} with price {order.price} outside bounds [{self.price_floor}, {self.price_cap}]")
            return

        level_key = (order.price, order.is_bid)
        already_in_dict = level_key in self._price_levels

        try:
            if not already_in_dict:
                limit_level = LimitLevel(order)
                self._orders[order.order_id] = order
                self._price_levels[level_key] = limit_level

                if order.is_bid:
                    self.bids.insert(limit_level)
                    if (self.best_bid is None) or (limit_level.price > self.best_bid.price):
                        self.best_bid = limit_level
                else:
                    self.asks.insert(limit_level)
                    if (self.best_ask is None) or (limit_level.price < self.best_ask.price):
                        self.best_ask = limit_level
            else:
                self._orders[order.order_id] = order
                self._price_levels[level_key].append(order)
        except Exception as e:
            if already_in_dict:
                print("Exception while adding order to existing level:", e)
            else:
                self._orders[order.order_id] = order
                self._price_levels[level_key].append(order)

    def _update_best_bid(self):
        if self.bids.right_child is None:
            self.best_bid = None
            return None

        node = self.bids.right_child
        while node.right_child:
            node = node.right_child
        self.best_bid = node
        return self.best_bid.orders.head

    def _update_best_ask(self):
        if self.asks.right_child is None:
            self.best_ask = None
            return None

        node = self.asks.right_child
        while node.left_child:
            node = node.left_child
        self.best_ask = node
        return self.best_ask.orders.head

    def get_order_book(self, depth=None):
        try:
            bid_keys = []
            ask_keys = []
            for (price, is_bid) in self._price_levels.keys():
                if is_bid:
                    bid_keys.append(price)
                else:
                    ask_keys.append(price)

            bid_keys.sort(reverse=True)
            ask_keys.sort()

            best_bid_price = self.best_bid.price if self.best_bid else float('-inf')
            best_ask_price = self.best_ask.price if self.best_ask else float('inf')

            bids_all = [
                (float(price), self._price_levels[(price, True)].size)
                for price in bid_keys
                if price < best_ask_price and (price, True) in self._price_levels
            ]
            if depth is not None:
                bids_all = bids_all[:depth]

            asks_all = [
                (float(price), self._price_levels[(price, False)].size)
                for price in ask_keys
                if price > best_bid_price and (price, False) in self._price_levels
            ]
            if depth is not None:
                asks_all = asks_all[:depth]

            return {
                'bids': bids_all,
                'asks': asks_all,
            }
        except Exception:
            return {
                'bids': [],
                'asks': [],
            }
