import sys
import os

# Add project root to sys.path so tests can import from intent_classifier, agent_state, etc.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
