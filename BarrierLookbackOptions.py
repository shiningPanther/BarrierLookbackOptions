import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class OptionPricer:

    def __init__(self, x0, r, sigma, T, dt, antithetic=True):
        self.paths = []
        self.statistics = []
        self.x0 = x0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.dt = dt
        self.antithetic = antithetic
        self.possible_option_types = ['VanillaCall, VanillaPut, LookBackCall, UpAndOutCall']

    def simulate_paths(self, N):
        counter = 0
        while counter < N:
            paths = self._simulate_one_path()
            # We only store every 100th path in memory
            if counter % 100 == 0:
                self.paths.append(paths[0])
            self._gather_statistics(paths)
            counter = counter + 1 if not self.antithetic else counter + 2

    def calculate_price(self, option_type, **kwargs):
        if option_type not in self.possible_option_types:
            print('Invalid option type')
            return
        stats = pd.DataFrame(self.statistics)
        print(stats)
        if option_type == 'VanillaCall':
            pass





    def _simulate_one_path(self):

        def _simulate_GBM(normals):
            x = np.exp((self.r - self.sigma ** 2 / 2) * self.dt + self.sigma * normals)
            x = self.x0 * x.cumprod()
            return x

        normal_rv = np.random.normal(0, np.sqrt(self.dt), size=int(self.T / self.dt))
        x_list = [_simulate_GBM(normal_rv)]
        if not self.antithetic:
            return x_list
        normal_rv = -normal_rv
        x_list.append(_simulate_GBM(normal_rv))
        plt.plot(x_list[0])
        plt.plot(x_list[1])
        return x_list

    def _gather_statistics(self, paths):
        d = {}
        for path in paths:
            d['ST'] = path[-1]  # final value
            d['min'] = np.min(path)
            d['argmin'] = np.argmin(path)
            d['max'] = np.max(path)
            d['argmax'] = np.argmax(path)
            self.statistics.append(d)

option = OptionPricer(x0=100, r=0.05, sigma=0.4, T=1, dt=0.0001)
option.simulate_paths(1)

