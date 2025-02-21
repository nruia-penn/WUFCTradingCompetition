# exchange/orderbook_manager.py

from typing import Dict, List, Tuple
from limitOrderBook import LimitOrderBook
from OrderQueue import PerTickerOrderQueue



class OrderBookManager:
    def __init__(self, participant_manager):
        self.order_books: Dict[str, LimitOrderBook] = {}
        self.participant_manager = participant_manager
        self.orderQueue = PerTickerOrderQueue(self.order_books)


    def add_order_book(self, symbol: str):

        if symbol not in self.order_books:
            newOrderBook = LimitOrderBook(symbol)
            self.order_books[symbol] = newOrderBook
            self.orderQueue.add_order_book(symbol, newOrderBook, self.participant_manager)
            

    def get_order_book_snapshot(self, symbol: str, depth=None) -> Dict[str, List[Tuple[float, int, float]]]:

        if symbol in self.order_books:
            return self.order_books[symbol].get_order_book(depth=depth)
        else:
            return {'bids': [], 'asks': []}
        
    def get_best_price(self, symbol: str, isBid: bool):

        if symbol in self.order_books:
            return self.order_books[symbol].get_best_price(askForBid=isBid)
        else:
            return 0

    def get_all_order_books(self) -> Dict[str, Dict[str, List[Tuple[float, int, float]]]]:

        snapshots = {}
        for symbol, order_book in self.order_books.items():
            snapshots[symbol] = order_book.get_order_book()
        return snapshots


    def get_order_book(self, symbol: str):
        return self.order_books.get(symbol)

