"""src/data package."""
try:
    from .database import DatabaseManager
    from .provider import DataProvider, YFinanceProvider
    __all__ = ['DatabaseManager', 'DataProvider', 'YFinanceProvider']
except ImportError:
    pass  # Kalau diimport secara absolute, skip relative import

