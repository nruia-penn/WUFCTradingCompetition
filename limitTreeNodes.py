from OrderList import OrderList
import traceback
import math


class LimitLevel:
    __slots__ = [
        'price', 'size', 'parent', 'left_child', 'right_child', 'orders',
        '_height'  # cached height
    ]

    def __init__(self, order):
        self.price = order.price
        self.size = order.size

        self.parent = None
        self.left_child = None
        self.right_child = None

        self._height = 1

        self.orders = OrderList(self)
        self.append(order)

    @property
    def is_root(self):
        return isinstance(self.parent, LimitLevelTree)

    @property
    def volume(self):
        return self.price * self.size

    @property
    def balance_factor(self):
        right_height = self.right_child._height if self.right_child else 0
        left_height = self.left_child._height if self.left_child else 0
        return right_height - left_height

    @property
    def grandpa(self):
        if self.parent and not isinstance(self.parent, LimitLevelTree):
            return self.parent.parent
        return None

    @property
    def height(self):
        return self._height

    def _update_height(self):
        left_height = self.left_child._height if self.left_child else 0
        right_height = self.right_child._height if self.right_child else 0
        self._height = max(left_height, right_height) + 1

    @property
    def min(self):
        node = self
        while node.left_child:
            node = node.left_child
        return node

    def append(self, order):
        self.orders.append(order)

    def _replace_node_in_parent(self, new_value=None):
        if self.is_root:
            self.parent.right_child = new_value
            if new_value:
                new_value.parent = self.parent
        else:
            if self == self.parent.left_child:
                self.parent.left_child = new_value
            else:
                self.parent.right_child = new_value

            self.parent._update_height()

            if new_value:
                new_value.parent = self.parent

    def remove(self):
        if self.left_child and self.right_child:
            successor = self.right_child.min
            self.price, self.size = successor.price, successor.size
            successor.remove()
            self.balance_grandpa()
        elif self.left_child:
            self._replace_node_in_parent(self.left_child)
        elif self.right_child:
            self._replace_node_in_parent(self.right_child)
        else:
            self._replace_node_in_parent(None)

    def balance_grandpa(self):
        if self.grandpa and isinstance(self.grandpa, LimitLevel):
            self.grandpa.balance()

    def balance(self):
        self._update_height()

        # Only trigger rebalancing if imbalance exceeds threshold (2)
        if self.balance_factor > 2:
            if self.right_child and self.right_child.balance_factor < 0:
                self._rl_case()
            else:
                self._rr_case()
        elif self.balance_factor < -2:
            if self.left_child and self.left_child.balance_factor > 0:
                self._lr_case()
            else:
                self._ll_case()

        self._update_height()

        if self.parent and not self.parent.is_root:
            self.parent.balance()

    def _ll_case(self):
        child = self.left_child
        # Re-link parent's pointer
        if self.is_root:
            self.parent.right_child = child
        elif self.parent.left_child == self:
            self.parent.left_child = child
        else:
            self.parent.right_child = child

        child.parent = self.parent
        self.left_child = child.right_child
        if self.left_child:
            self.left_child.parent = self
        child.right_child = self
        self.parent = child

        self._update_height()
        child._update_height()

    def _rr_case(self):
        child = self.right_child
        if self.is_root:
            self.parent.right_child = child
        elif self.parent.left_child == self:
            self.parent.left_child = child
        else:
            self.parent.right_child = child

        child.parent = self.parent
        self.right_child = child.left_child
        if self.right_child:
            self.right_child.parent = self
        child.left_child = self
        self.parent = child

        self._update_height()
        child._update_height()

    def _lr_case(self):
        # Left-right case: first rotate left at the left child, then right at self.
        child = self.left_child
        grand_child = child.right_child
        child.parent, grand_child.parent = grand_child, self
        child.right_child = grand_child.left_child
        if child.right_child:
            child.right_child.parent = child
        self.left_child, grand_child.left_child = grand_child, child
        self._ll_case()

    def _rl_case(self):
        # Right-left case: first rotate right at the right child, then left at self.
        child = self.right_child
        grand_child = child.left_child
        child.parent, grand_child.parent = grand_child, self
        child.left_child = grand_child.right_child
        if child.left_child:
            child.left_child.parent = child
        self.right_child, grand_child.right_child = grand_child, child
        self._rr_case()

    def __str__(self):
        if not self.is_root:
            s = f'Node Value: {self.price}\n'
            s += f'Left Child: {self.left_child.price if self.left_child else "None"}\n'
            s += f'Right Child: {self.right_child.price if self.right_child else "None"}\n\n'
        else:
            s = ''

        left_side_print = self.left_child.__str__() if self.left_child else ''
        right_side_print = self.right_child.__str__() if self.right_child else ''
        return s + left_side_print + right_side_print

    def __len__(self):
        return len(self.orders)


class LimitLevelTree:
    __slots__ = ['right_child']

    def __init__(self):
        self.right_child = None

    def insert(self, limit_level):
        if self.right_child is None:
            self.right_child = limit_level
            self.right_child.parent = self
            return

        current_node = self.right_child
        while True:
            if limit_level.price > current_node.price:
                if current_node.right_child is None:
                    current_node.right_child = limit_level
                    limit_level.parent = current_node
                    # Only balance if imbalance is significant.
                    if abs(current_node.balance_factor) > 2:
                        current_node.balance_grandpa()
                    break
                else:
                    current_node = current_node.right_child
            elif limit_level.price < current_node.price:
                if current_node.left_child is None:
                    current_node.left_child = limit_level
                    limit_level.parent = current_node
                    if abs(current_node.balance_factor) > 2:
                        current_node.balance_grandpa()
                    break
                else:
                    current_node = current_node.left_child
            else:
                raise ValueError(f"LimitLevel with price {limit_level.price} already exists.")

    def find(self, price):
        current_node = self.right_child
        while current_node:
            if current_node.price == price:
                return current_node
            elif price < current_node.price:
                current_node = current_node.left_child
            else:
                current_node = current_node.right_child
        return None
