"""
manager.py — Risk Management berbasis ATR untuk IDX-Screener v2.

Fungsi utama:
- calculate_stop_loss(): stop loss dinamis berdasarkan volatilitas (ATR)
- calculate_position_size(): berapa % kapital yang aman per posisi
- check_liquidity(): apakah saham cukup likuid untuk trading

Filosofi: posisi sizing berbasis risiko (risk-based sizing), bukan fixed % kapital.
Contoh: kalau risk per trade = 1% kapital, dan jarak ke stop = 2%, maka posisi = 50% kapital.
"""

from typing import Optional

import pandas as pd


class RiskManager:
    """
    ATR-based risk management calculator.

    Args:
        atr_multiplier: Kelipatan ATR untuk stop loss (default 2.0 = volatility-based)
        min_avg_volume: Volume rata-rata minimum untuk dianggap likuid (default 500jt lembar)
        min_turnover_idr: Turnover harian minimum dalam IDR (default 1 Miliar)
    """

    def __init__(
        self,
        atr_multiplier: float = 2.0,
        min_avg_volume: float = 500_000,
        min_turnover_idr: float = 1_000_000_000,
    ) -> None:
        self.atr_multiplier     = atr_multiplier
        self.min_avg_volume     = min_avg_volume
        self.min_turnover_idr   = min_turnover_idr

    def calculate(self, df: pd.DataFrame, capital: float = 10_000_000) -> dict:
        """
        Hitung semua risk metrics dari DataFrame.

        Args:
            df: OHLCV + indicators DataFrame (baris terakhir = hari ini)
            capital: Total kapital trading dalam IDR (default 10 juta untuk ilustrasi)

        Returns:
            Dict berisi: entry, stop_loss, risk_pct, position_value, position_pct,
                         liquidity, liquidity_warning
        """
        result = {}

        # Entry price = close hari ini
        entry = float(df['Close'].iloc[-1]) if 'Close' in df.columns else None
        if entry is None or entry <= 0:
            return {'error': 'Close price tidak tersedia'}

        result['entry'] = entry

        # Stop loss
        stop_loss_result = self.calculate_stop_loss(df)
        result.update(stop_loss_result)

        # Position sizing (asumsi risk per trade = 1% kapital)
        if 'stop_loss' in stop_loss_result:
            sizing = self.calculate_position_size(
                capital=capital,
                risk_pct_of_capital=1.0,
                entry=entry,
                stop=stop_loss_result['stop_loss'],
            )
            result.update(sizing)

        # Liquidity check
        liquidity = self.check_liquidity(df)
        result['liquidity'] = liquidity['liquid']
        if not liquidity['liquid']:
            result['liquidity_warning'] = liquidity['warning']

        return result

    def calculate_stop_loss(self, df: pd.DataFrame) -> dict:
        """
        Hitung stop loss dinamis berdasarkan ATR.
        Stop Loss = Close - (ATR_14 × multiplier)

        ATR-based stop loss lebih baik dari fixed % karena:
        - Saham volatile → stop lebih lebar (biar tidak kena noise)
        - Saham stabil → stop lebih ketat

        Returns:
            Dict berisi: stop_loss, atr_14, risk_pct (jarak stop dari entry)
        """
        if df.empty:
            return {}

        close = float(df['Close'].iloc[-1])

        # Coba ambil ATR dari kolom pre-computed (lebih efisien)
        if 'atr_14' in df.columns and pd.notna(df['atr_14'].iloc[-1]):
            atr = float(df['atr_14'].iloc[-1])
        else:
            # Hitung ATR manual jika tidak ada di kolom
            atr = self._calculate_atr(df)

        if atr is None or atr <= 0:
            return {'stop_loss': None, 'atr_14': None, 'risk_pct': None}

        stop_loss = close - (atr * self.atr_multiplier)
        risk_pct  = (close - stop_loss) / close * 100

        return {
            'stop_loss': round(stop_loss, 2),
            'atr_14': round(atr, 2),
            'risk_pct': round(risk_pct, 2),
        }

    def calculate_position_size(
        self,
        capital: float,
        risk_pct_of_capital: float,
        entry: float,
        stop: float,
    ) -> dict:
        """
        Hitung ukuran posisi berdasarkan risiko per trade.

        Formula:
            risk_amount     = capital × risk_pct_of_capital / 100
            risk_per_share  = entry - stop
            shares          = risk_amount / risk_per_share
            position_value  = shares × entry

        Args:
            capital: Total kapital (IDR)
            risk_pct_of_capital: % kapital yang di-risk per trade (misal 1.0 = 1%)
            entry: Harga entry (Close)
            stop: Harga stop loss

        Returns:
            Dict berisi: position_value, position_pct (% of capital), shares, risk_amount
        """
        if entry <= 0 or stop <= 0 or entry <= stop:
            return {}

        risk_amount    = capital * risk_pct_of_capital / 100
        risk_per_share = entry - stop

        if risk_per_share <= 0:
            return {}

        shares         = risk_amount / risk_per_share
        position_value = shares * entry
        position_pct   = position_value / capital * 100

        # Cap position size maksimal 25% kapital (diversifikasi)
        if position_pct > 25:
            position_pct   = 25
            position_value = capital * 0.25
            shares         = position_value / entry

        return {
            'position_value': round(position_value),
            'position_pct': round(position_pct, 1),
            'shares': round(shares),
            'risk_amount': round(risk_amount),
        }

    def check_liquidity(self, df: pd.DataFrame) -> dict:
        """
        Cek apakah saham cukup likuid untuk di-trading.

        Saham illiquid berisiko:
        - Bid-ask spread besar (invisible cost)
        - Susah keluar dari posisi
        - Rentan digoreng (manipulasi harga)

        Returns:
            Dict: {'liquid': bool, 'avg_volume_20d': float, 'avg_turnover': float, 'warning': str|None}
        """
        if df.empty or 'Volume' not in df.columns or 'Close' not in df.columns:
            return {'liquid': False, 'warning': 'Data volume tidak tersedia'}

        # Gunakan 20 hari terakhir
        recent = df.tail(20)
        avg_volume  = float(recent['Volume'].mean())
        avg_turnover = float((recent['Volume'] * recent['Close']).mean())

        is_liquid   = avg_volume >= self.min_avg_volume and avg_turnover >= self.min_turnover_idr

        warning = None
        if not is_liquid:
            reasons = []
            if avg_volume < self.min_avg_volume:
                reasons.append(f"volume rata-rata {avg_volume:,.0f} < {self.min_avg_volume:,.0f}")
            if avg_turnover < self.min_turnover_idr:
                reasons.append(f"turnover Rp{avg_turnover/1e6:.0f}jt/hari < Rp{self.min_turnover_idr/1e9:.0f}M")
            warning = f"⚠️ Likuiditas rendah: {', '.join(reasons)}. Rawan gorengan, spread besar."

        return {
            'liquid': is_liquid,
            'avg_volume_20d': round(avg_volume),
            'avg_turnover_idr': round(avg_turnover),
            'warning': warning,
        }

    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> Optional[float]:
        """Hitung ATR manual dari OHLCV kalau kolom atr_14 tidak tersedia."""
        needed = {'High', 'Low', 'Close'}
        if not needed.issubset(df.columns) or len(df) < window:
            return None

        prev_close = df['Close'].shift(1)
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - prev_close).abs()
        tr3 = (df['Low'] - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean().iloc[-1]

        return float(atr) if pd.notna(atr) else None
