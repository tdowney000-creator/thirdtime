import io
import os
from abc import ABC, abstractmethod

from alpaca.trading.client import TradingClient  # pyright: ignore[reportMissingImports]


def load_api_keys() -> tuple[str, str]:
    """Load API key and secret key from a local sks.txt file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sks_path = os.path.join(script_dir, "sks.txt")

    if not os.path.isfile(sks_path):
        raise FileNotFoundError(f"The file '{sks_path}' does not exist.")

    with io.open(sks_path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    if len(lines) < 2:
        raise ValueError("sks.txt must contain at least the API key and secret key on separate lines.")

    return lines[0], lines[1]


class Trading(ABC):
    """Abstract base class for trading operations."""

    @abstractmethod
    def connect(self):
        """Return a connected trading client."""
        raise NotImplementedError

    @abstractmethod
    def get_account(self):
        """Return the current trading account information."""
        raise NotImplementedError

    @abstractmethod
    def get_positions(self):
        """Return all current open positions."""
        raise NotImplementedError

    @abstractmethod
    def place_order(self, symbol: str, qty: int, side: str, order_type: str = "market"):
        """Place a market order for the given symbol."""
        raise NotImplementedError

    @abstractmethod
    def close_position(self, symbol: str):
        """Close an open position for the given symbol."""
        raise NotImplementedError


class AlpacaTrading(Trading):
    """Concrete Alpaca trading implementation."""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.client = TradingClient(api_key, secret_key, paper=paper)

    def connect(self):
        return self.client

    def get_account(self):
        return self.client.get_account()

    def get_positions(self):
        return self.client.get_all_positions()

    def place_order(self, symbol: str, qty: int, side: str, order_type: str = "market"):
        return self.client.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force="day",
        )

    def close_position(self, symbol: str):
        return self.client.close_position(symbol)


if __name__ == "__main__":
    API, SEC = load_api_keys()
    trading = AlpacaTrading(API, SEC, paper=True)
    account = trading.get_account()
    print(account)

