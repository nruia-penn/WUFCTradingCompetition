import numpy as np

# Mock order book snapshots for testing
fake_order_book_snapshot = {
    'NVR': {'bids': [(100, 10), (99, 5)], 'asks': [(101, 10), (102, 5)]},
    'CPMD': {'bids': [(200, 15), (199, 10)], 'asks': [(201, 15), (202, 10)]},
    'MFH': {'bids': [(300, 20), (299, 10)], 'asks': [(301, 20), (302, 10)]},
    'ANG': {'bids': [(400, 25), (399, 15)], 'asks': [(401, 25), (402, 15)]},
    'TVW': {'bids': [(500, 30), (499, 20)], 'asks': [(501, 30), (502, 20)]},
}

class CompetitorBoilerplate:
    def __init__(self, name):
        self.name = name
        self.symbols = list(fake_order_book_snapshot.keys())
        self.portfolio = {symbol: 0 for symbol in self.symbols}  # Initialize portfolio
        self.price_history = {symbol: [] for symbol in self.symbols}  # Initialize price history

    def get_order_book_snapshot(self, symbol):
        return fake_order_book_snapshot[symbol]

    def create_limit_order(self, price, size, side, symbol):
        # Update portfolio based on order
        if side == 'buy':
            self.portfolio[symbol] += size
        elif side == 'sell':
            self.portfolio[symbol] -= size
        print(f"Limit Order: {side.upper()} {size} of {symbol} at {price:.4f}")

    def get_portfolio(self):
        return self.portfolio

    def strategy(self):
        """
        An advanced adaptive rebalancing strategy that incorporates multiple statistical measures:

        1. For each symbol:
           - Get the order book snapshot and compute the mid-price.
           - Maintain a rolling history of mid-prices to compute volatility, Sharpe ratio, and a simple moving average (SMA).
           - Calculate percentage returns and use them to derive the Sharpe ratio.
           - Compute the SMA and the deviation of the current price from this average (a mean reversion indicator).
           - Calculate the order book imbalance to capture buying/selling pressure.
        2. Dynamically adjust order size based on volatility.
        3. Adjust limit order price offsets based on volatility.
        4. Incorporate inventory management: only trade if your current position deviates from the target (flat).
        5. Place orders based on combined signals (Sharpe ratio, imbalance, and mean reversion).
        """
        lookback = 1.2              # Number of prices in the rolling window
        sharpe_threshold = 0.000000001

        # Ensure persistent price history exists for each symbol
        self.__dict__.setdefault("price_history", {symbol: [] for symbol in self.symbols})

        for symbol in self.symbols:
            snapshot = self.get_order_book_snapshot(symbol)
            if not snapshot.get('bids') or not snapshot.get('asks'):
                print(f"Skipping {symbol}: insufficient market data.")
                continue  # Skip if market data is insufficient

            best_bid = snapshot['bids'][0][0]
            best_ask = snapshot['asks'][0][0]
            mid_price = (best_bid + best_ask) / 2.0

            # Update the rolling price history
            self.price_history[symbol].append(mid_price)
            if len(self.price_history[symbol]) > lookback:
                self.price_history[symbol].pop(0)
            
            # Proceed only if we have enough history
            if len(self.price_history[symbol]) < lookback:
                print(f"Not enough price history for {symbol}. Current history length: {len(self.price_history[symbol])}")
                continue
            
            prices = np.array(self.price_history[symbol])
            returns = np.diff(prices) / prices[:-1]
            
            # Calculate expected return and volatility (with an epsilon to avoid division by zero)
            expected_return = np.mean(returns)
            volatility = np.std(returns) + 1e-6
            sharpe_ratio = expected_return / volatility

            # Compute simple moving average (SMA) and its deviation
            sma = np.mean(prices)
            deviation = (mid_price - sma) / sma

            # Calculate order book imbalance:
            # Positive imbalance implies higher bid volume (buy pressure), negative implies sell pressure.
            total_bid_vol = sum(bid[1] for bid in snapshot['bids'])
            total_ask_vol = sum(ask[1] for ask in snapshot['asks'])
            imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol + 1e-6)

            # Dynamically adjust order size based on volatility:
            base_order_size = 10
            size_adjustment_factor = 1 / volatility  # Inverse relation: lower volatility → larger orders
            order_size = int(base_order_size * size_adjustment_factor)
            order_size = max(1, min(order_size, 100))  # Clamp order size between 1 and 100

            # Adjust limit order offset based on volatility:
            # Higher volatility leads to a wider offset to account for rapid price movements.
            price_offset = volatility * mid_price  # This factor can be tuned based on backtests

            # Risk management: Check current inventory for the symbol; aim for a flat (zero) position.
            portfolio = self.get_portfolio()
            current_position = portfolio.get(symbol, 0)
            target_position = 0  # Flat position as a risk target

            # Debugging output
            print(f"Symbol: {symbol}, Sharpe Ratio: {sharpe_ratio}, Imbalance: {imbalance}, Current Position: {current_position}, Order Size: {order_size}")

            # --------------------------
            # Trading Decisions
            # --------------------------
            
            # (1) Trend-following signal using Sharpe ratio and imbalance:
            #     - If the Sharpe ratio is strongly positive (uptrend) and there is buy pressure,
            #       and if your position is below the target, then place a buy order.
            if sharpe_ratio > sharpe_threshold and imbalance >= 0 and current_position <= target_position:
                buy_price = best_bid + price_offset  # Aggressively buy by offering a slightly higher price than the best bid
                print(f"Placing buy order for {symbol} at {buy_price:.4f}")
                self.create_limit_order(price=buy_price, size=order_size, side='buy', symbol=symbol)

            #     - Conversely, if the Sharpe ratio is strongly negative (downtrend) and there is sell pressure,
            #       and if your position is above the target, then place a sell order.
            elif sharpe_ratio < -sharpe_threshold and imbalance <= 0 and current_position >= target_position:
                sell_price = best_ask - price_offset  # Aggressively sell by offering a slightly lower price than the best ask
                print(f"Placing sell order for {symbol} at {sell_price:.4f}")
                self.create_limit_order(price=sell_price, size=order_size, side='sell', symbol=symbol)

            # (2) Mean reversion signal:
            #     - If the current price is significantly above its SMA, consider selling (if you have excess long exposure).
            if deviation >= 0.01 and current_position >= target_position:
                print(f"Mean reversion sell signal for {symbol}.")
                self.create_limit_order(price=best_ask - price_offset, size=order_size, side='sell', symbol=symbol)
            #     - If the current price is significantly below its SMA, consider buying (if you are underexposed).
            elif deviation <= -0.01 and current_position <= target_position:
                print(f"Mean reversion buy signal for {symbol}.")
                self.create_limit_order(price=best_bid + price_offset, size=order_size, side='buy', symbol=symbol)

# A simple test harness to verify the strategy logic.
if __name__ == "__main__":
    # Create an instance of your competitor class.
    competitor = CompetitorBoilerplate("test_competitor")

    # Run the strategy multiple times to simulate a live environment (to build up price history).
    for i in range(15):
        print(f"\n--- Strategy run #{i+1} ---")
        competitor.strategy()
        print(f"Updated Portfolio: {competitor.get_portfolio()}")