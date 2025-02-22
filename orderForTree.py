import time
import uuid
from typing import Optional
from dataclasses import dataclass

@dataclass
class Order:


    order_id: str
    timestamp: float
    price: Optional[float]
    size: int
    is_bid: bool
    order_type: str
    participant_id: str
    symbol: str

    

    def __post_init__(self):
        #DLL attributes
        self.next_item = None
        self.previous_item = None
        self.root = None

    @property
    def parent_limit(self):
        return self.root.parent_limit



    def pop_from_list(self):

        if self.previous_item is None:

            self.root.head = self.next_item
            if self.next_item:
                self.next_item.previous_item = None

        elif self.next_item is None:

            self.root.tail = self.previous_item
            if self.previous_item:
                self.previous_item.next_item = None

        self.root.count -= 1
        self.parent_limit.size -= self.size

        return self.__repr__()
    
    @staticmethod
    def create_limit_order(price: float, size: int, side: str, participant_id: str, symbol: str) -> 'Order':
       
        if price <= 0:
            raise ValueError("Limit order price must be positive.")
        if size <= 0:
            raise ValueError("Order size must be positive.")
        if (side == 'buy'):
            is_bid = True
        elif (side == 'sell'):
            is_bid = False
        else:
            raise ValueError("Limit Order has to be of side 'buy' or 'sell'")
        
        return Order(
            order_id=str(uuid.uuid4()),
            timestamp=time.time(),
            price=price,
            size=size,
            is_bid=is_bid,
            order_type='limit',
            participant_id=participant_id,
            symbol=symbol
        )

    @staticmethod
    def create_market_order(size: int, side: str, participant_id: str, symbol: str) -> 'Order':

        if (side == 'buy'):
            is_bid = True
        elif (side == 'sell'):
            is_bid = False
        else:
            raise ValueError("Market Order has to be of side 'buy' or 'sell'")

        return Order(
            order_id=str(uuid.uuid4()),
            timestamp=time.time(),
            price=None,
            size=size,
            is_bid=is_bid,
            order_type='market',
            participant_id=participant_id,
            symbol=symbol
        )
    
    @staticmethod
    def create_cancel_order(order_id: str, participant_id: str, symbol: str):
        
        return Order(
            order_id=order_id,
            timestamp=time.time(),
            price=None,
            size=0,
            is_bid=True,
            order_type='cancel',
            participant_id=participant_id,
            symbol=symbol,
        )

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str((self.order_id, self.is_bid, self.price, self.size, self.timestamp))