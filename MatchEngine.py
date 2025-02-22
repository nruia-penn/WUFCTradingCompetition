from functools import partial
from orderForTree import Order
from limitOrderBook import LimitOrderBook
from ParticipantManager import ParticipantManager


class MatchEngine:


    def __init__(self, OrderBook: LimitOrderBook, pm: ParticipantManager):
        self.orderBook = OrderBook
        self.pm = pm

    # 
    def acceptLimitOrder(self, aggressiveOrder: Order):
        while aggressiveOrder.size > 0:
            opposingOrder : Order = self.orderBook.top_level(askForBid= not aggressiveOrder.is_bid)
            if opposingOrder is None:
                
                self.orderBook.process(aggressiveOrder, 0)
                break
            if (aggressiveOrder.price < opposingOrder.price if aggressiveOrder.is_bid else aggressiveOrder.price > opposingOrder.price):
                self.orderBook.process(aggressiveOrder, 0)
                break
            if (aggressiveOrder.participant_id == opposingOrder.participant_id):
                pass
            
            trade_quantity = min(aggressiveOrder.size, opposingOrder.size)
                
            if (aggressiveOrder.is_bid):
                buyer_balance = self.pm.get_participant_balance(aggressiveOrder.participant_id)
                buyOrder = aggressiveOrder
                sellOrder = opposingOrder
            else:
                buyer_balance = self.pm.get_participant_balance(opposingOrder.participant_id)
                sellOrder = aggressiveOrder
                buyOrder = opposingOrder
                
            total_cost = trade_quantity * buyOrder.price 
                
            processThisOne = True
                
            if buyer_balance < total_cost:
                trade_quantity = int(buyer_balance // buyOrder.price)
                if trade_quantity <= 0:
                    self.orderBook.remove(buyOrder)
                    self.orderBook.process(sellOrder, 0)
                    if not aggressiveOrder.is_bid:
                        processThisOne = False
                    else:
                        break
            
            if processThisOne:
                aggressiveOrder.size -= trade_quantity
                opposingOrder.size -= trade_quantity

                self.pm.send_execution_report(
                buyer_participant_id=buyOrder.participant_id,
                seller_participant_id=sellOrder.participant_id,
                trade_details={
                    'buyer_order_id': buyOrder.order_id,
                    'seller_order_id': sellOrder.order_id,
                    'symbol': buyOrder.symbol,
                    'buy_price': buyOrder.price, 
                    'sell_price': sellOrder.price, 
                    'quantity': trade_quantity
                })
                self.orderBook.process(aggressiveOrder, trade_quantity)
                self.orderBook.process(opposingOrder, trade_quantity)

    def acceptMarketOrder(self, aggressiveOrder: Order):
        while aggressiveOrder.size > 0 and self.orderBook.top_level(askForBid=aggressiveOrder.is_bid) is not None:
            opposingOrder : Order = self.orderBook.top_level(askForBid= not aggressiveOrder.is_bid)
            
            if opposingOrder is None:
                break
            if (aggressiveOrder.participant_id == opposingOrder.participant_id):
                break
            trade_quantity = min(aggressiveOrder.size, opposingOrder.size)
            trade_price = opposingOrder.price
            total_cost = trade_price * trade_quantity
            processThisOne = True
            if (aggressiveOrder.is_bid):
                buyOrder = aggressiveOrder
                sellOrder = opposingOrder
                buyBalance = self.pm.get_participant_balance(aggressiveOrder.participant_id)
                if buyBalance < total_cost:
                    partial_quantity = int(buyBalance // trade_price)
                    if partial_quantity <= 0:
                        break
                    trade_quantity = partial_quantity
                    total_cost = trade_price * trade_quantity
            else:
                buyOrder = opposingOrder
                sellOrder = aggressiveOrder
                buyBalance = self.pm.get_participant_balance(opposingOrder.participant_id)
                if buyBalance < total_cost:
                    partial_quantity = int(buyBalance // trade_price)
                    if partial_quantity <= 0:
                        self.orderBook.remove(buyOrder)
                        processThisOne = False
                    trade_quantity = partial_quantity
                    total_cost = trade_price * trade_quantity

            if processThisOne:
                opposingOrder.size -= trade_quantity
                aggressiveOrder.size -= trade_quantity

                self.pm.send_execution_report(
                buyer_participant_id=buyOrder.participant_id,
                seller_participant_id=sellOrder.participant_id,
                trade_details={
                    'buyer_order_id': buyOrder.order_id,
                    'seller_order_id': sellOrder.order_id,
                    'symbol': buyOrder.symbol,
                    'buy_price': trade_price,
                    'sell_price': trade_price,
                    'quantity': trade_quantity
                })

                self.orderBook.process(opposingOrder, trade_quantity)
            


    def acceptCancelOrder(self, order: Order):
        self.orderBook.process(order, 0)


