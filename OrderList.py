import threading

class OrderList:
    __slots__ = ['head', 'tail', 'parent_limit', 'count', '_lock']

    def __init__(self, parent_limit):
        self.head = None
        self.tail = None
        self.count = 0
        self.parent_limit = parent_limit
        self._lock = threading.Lock()

    def __len__(self):
        with self._lock:
            return self.count

    def append(self, order):
        with self._lock:
            order.root = self

            if self.head is None:
                self.head = order
                self.tail = order
            else:
                self.tail.next_item = order
                order.previous_item = self.tail
                self.tail = order
                self.parent_limit.size += order.size
            self.count += 1

    def remove(self, order):
        with self._lock:
            if order is self.head:
                self.head = order.next_item
            if order is self.tail:
                self.tail = order.previous_item
            if order.previous_item:
                order.previous_item.next_item = order.next_item
            if order.next_item:
                order.next_item.previous_item = order.previous_item
            self.count -= 1
            self.parent_limit.size -= order.size

