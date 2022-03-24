from BarrierLookbackOptions import OptionPricer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



results = pd.DataFrame()

option_params = {'K': 100,
                 'B_up': 200}

dt = 0.0001

option_pricer = OptionPricer(100, 0.05, 0.4, 1, dt, antithetic=True)
deltas, d_l, d_u = option_pricer.calculate_delta_likelihood_ratio('UpAndOutCall', 50000, **option_params)
print(np.mean(deltas))
print (d_l, d_u)



prices = []
x = np.linspace(99,101, 5)

option_params = {'K': 100,
                 'B_up': 200}

# for  s in tqdm(x):
#     option_pricer = OptionPricer(s, 0.05, 0.1, 1, dt, antithetic=True)
#     payoffs, _, _ = option_pricer.calculate_price('UpAndOutCall', 10000, **option_params)
#     price = np.mean(payoffs)
#     prices.append(price)

# plt.plot(x, prices)
# plt.savefig("image.jpg")

# print ((prices[-1]-prices[0])/2)

