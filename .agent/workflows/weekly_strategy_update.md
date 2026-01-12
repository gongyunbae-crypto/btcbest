---
description: Weekly routine to research and implement new Bitcoin trading strategies.
---

# Weekly Strategy Update Routine

Run this workflow once a week to keep the strategy library fresh.

1. **Research Phase**

   - Search for "latest bitcoin trading strategies python 2025" or specific indicators like "Hurst Exponent", "Kalman Filter", "Machine Learning trading".
   - Select 2-3 high-quality strategies with available logic or code.

2. **Implementation Phase**

   - Add the signal logic to `strategy.py`.
   - Ensure specific helper functions (e.g., `apply_new_strategy_X`) are created.

3. **Integration Phase**

   - Add the new strategies to the "Preset Library" in `app.py`.
   - Create a button that loads the strategy into the simulator.

4. **Verification Phase**
   - Run a quick backtest to ensure no syntax errors.
   - Update `walkthrough.md` with the new additions.
