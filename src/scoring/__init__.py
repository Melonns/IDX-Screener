"""src/scoring package."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.engine import ScoringEngine
from scoring.config import SCORING_CONFIG

__all__ = ['ScoringEngine', 'SCORING_CONFIG']

