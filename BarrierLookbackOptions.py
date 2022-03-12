import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class OptionPricer:

    def __init__(self, s0, r, sigma, T, dt, antithetic=True):
        self.paths = []
        self.s0 = s0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.dt = dt
        self.antithetic = antithetic
        self.possible_option_types = ['VanillaCall', 'VanillaPut', 'LookBackCall', 'UpAndOutCall']

    def calculate_price(self, option_type, N, **kwargs):
        if option_type not in self.possible_option_types:
            print('Invalid option type')
            return
        counter = 0
        payoff = 0
        while counter < N:
            paths = self._simulate_one_path()
            if counter % 100 == 0:
                self.paths.append(paths[0])
            for path in paths:
                payoff += self._get_payoff(path, option_type, **kwargs)
            counter = counter + 1 if not self.antithetic else counter + 2
        payoff /= N
        discount = np.exp(-self.r * self.T)
        return discount * payoff

    def calculate_delta(self, option_type, N, dS0, **kwargs):
        if option_type not in self.possible_option_types:
            print('Invalid option type')
            return
        counter = 0
        payoff_diff = 0
        while counter < N:
            paths = self._simulate_delta_path(dS0)
            for path in paths:
                payoff1 = self._get_payoff(path[0], option_type, **kwargs)
                payoff2 = self._get_payoff(path[1], option_type, **kwargs)
                payoff_diff += payoff1 - payoff2
            counter = counter + 1 if not self.antithetic else counter + 2
        payoff_diff = payoff_diff / N / dS0 / 2
        discount = np.exp(-self.r * self.T)
        return discount * payoff_diff

    def _get_payoff(self, path, option_type, **kwargs):
        if option_type == 'VanillaCall':
            return max(path[-1] - kwargs['K'], 0)
        if option_type == 'UpAndOutCall':
            if max(path) > kwargs['B_up']:
                return 0
            return max(path[-1] - kwargs['K'], 0)
        if option_type == 'LookBackCall':
            return max(max(path) - kwargs['K'], 0)

    def _simulate_delta_path(self, dS0):

        s0_1 = self.s0 + dS0
        s0_2 = self.s0 - dS0

        def _simulate_GBM(normals):
            s = np.exp((self.r - self.sigma ** 2 / 2) * self.dt + self.sigma * normals)
            s1 = s0_1 * s.cumprod()
            s2 = s0_2 * s.cumprod()
            return [s1, s2]

        normal_rv = np.random.normal(0, np.sqrt(self.dt), size=int(self.T / self.dt))
        s_list = [_simulate_GBM(normal_rv)]
        if not self.antithetic:
            return s_list
        s_list.append(_simulate_GBM(-normal_rv))
        return s_list

    def _simulate_one_path(self):

        def _simulate_GBM(normals):
            s = np.exp((self.r - self.sigma ** 2 / 2) * self.dt + self.sigma * normals)
            s = self.s0 * s.cumprod()
            return s

        normal_rv = np.random.normal(0, np.sqrt(self.dt), size=int(self.T / self.dt))
        s_list = [_simulate_GBM(normal_rv)]
        if not self.antithetic:
            return s_list
        s_list.append(_simulate_GBM(-normal_rv))
        return s_list


option = OptionPricer(s0=100, r=0.05, sigma=0.4, T=1, dt=0.001, antithetic=True)
N = 100_000
option_params = {'K': 100,
                 'B_up': 200,
                 'B_down': 80}

print(option.calculate_delta('UpAndOutCall', N, dS0=0.01, **option_params))
print(option.calculate_price('UpAndOutCall', N, **option_params))

