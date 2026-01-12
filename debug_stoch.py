
import vectorbt as vbt
import pandas as pd
import numpy as np

# Create dummy data
price = pd.Series(np.random.normal(100, 10, 100))
high = price + 1
low = price - 1
close = price

try:
    stoch = vbt.STOCH.run(high, low, close)
    print("STOCH attributes:", dir(stoch))
    
    # Try common names
    try: print("k:", stoch.k.values[-1])
    except: print("No .k")
    
    try: print("slow_k:", stoch.slow_k.values[-1])
    except: print("No .slow_k")
    
    try: print("percent_k:", stoch.percent_k.values[-1])
    except: print("No .percent_k")

except Exception as e:
    print("Error:", e)
