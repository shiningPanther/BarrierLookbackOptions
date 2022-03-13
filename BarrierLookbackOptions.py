import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool


class OptionPricer:

    def __init__(self, s0, r, sigma, T, dt, antithetic=True):
        self.paths = []
        self.s0 = s0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.discount = np.exp(-self.r * self.T)
        self.dt = dt
        self.antithetic = antithetic
        self.possible_option_types = ['VanillaCall', 'VanillaPut', 'LookBackCall', 'UpAndOutCall']

    def calculate_price(self, option_type, N, **kwargs):
        if option_type not in self.possible_option_types:
            print('Invalid option type')
            return
        counter = 0
        payoffs = np.zeros(N)
        while counter < N:
            paths = self._simulate_one_path()
            if counter % 100 == 0:
                self.paths.append(paths[0])
            for path in paths:
                payoffs[counter] = self._get_payoff(path, option_type, **kwargs)
                counter += 1
        payoffs *= self.discount
        ci_d, ci_u = self._bootstrap(payoffs)
        return payoffs.mean(), ci_d, ci_u

    def calculate_delta(self, option_type, N, dS0, **kwargs):
        if option_type not in self.possible_option_types:
            print('Invalid option type')
            return
        counter = 0
        payoff_diffs = np.zeros(N)
        while counter < N:
            paths = self._simulate_delta_path(dS0)
            for path in paths:
                payoff1 = self._get_payoff(path[0], option_type, **kwargs)
                payoff2 = self._get_payoff(path[1], option_type, **kwargs)
                payoff_diffs[counter] = payoff1 - payoff2
                counter += 1
        payoff_diffs /= (2*dS0)
        payoff_diffs *= self.discount
        ci_d, ci_u = self._bootstrap(payoff_diffs)
        return payoff_diffs.mean(), ci_d, ci_u

    def calculate_gamma(self, option_type, N, dS0, **kwargs):
        if option_type not in self.possible_option_types:
            print('Invalid option type')
            return
        counter = 0
        payoff_diffs = np.zeros(N)
        while counter < N:
            paths = self._simulate_gamma_path(dS0)
            for path in paths:
                payoff1 = self._get_payoff(path[0], option_type, **kwargs)
                payoff2 = self._get_payoff(path[1], option_type, **kwargs)
                payoff3 = self._get_payoff(path[2], option_type, **kwargs)
                payoff4 = self._get_payoff(path[3], option_type, **kwargs)
                payoff5 = self._get_payoff(path[4], option_type, **kwargs)
                payoff_diffs[counter] = -payoff1 + 16*payoff2 - 30*payoff3 + 16*payoff4 - payoff5
                counter += 1
        payoff_diffs /= (12*dS0**2)
        payoff_diffs *= self.discount
        ci_d, ci_u = self._bootstrap(payoff_diffs)
        return payoff_diffs.mean(), ci_d, ci_u

    def calculate_vega(self, option_type, N, dsigma, **kwargs):
        if option_type not in self.possible_option_types:
            print('Invalid option type')
            return
        counter = 0
        payoff_diffs = np.zeros(N)
        while counter < N:
            paths = self._simulate_vega_path(dsigma)
            for path in paths:
                payoff1 = self._get_payoff(path[0], option_type, **kwargs)
                payoff2 = self._get_payoff(path[1], option_type, **kwargs)
                payoff_diffs[counter] = payoff1 - payoff2
                counter += 1
        payoff_diffs /= (2*dsigma)
        payoff_diffs *= self.discount
        ci_d, ci_u = self._bootstrap(payoff_diffs)
        return payoff_diffs.mean(), ci_d, ci_u

    def _bootstrap(self, values):
        N = 10_000
        L = len(values)
        bootstraps = np.zeros(N)
        for i in range(N):
            bootstraps[i] = np.random.choice(values, size=L, replace=True).mean()
        return np.quantile(bootstraps, 0.05), np.quantile(bootstraps, 0.95)

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

    def _simulate_vega_path(self, dsigma):

        sigma_1 = self.sigma + dsigma
        sigma_2 = self.sigma - dsigma

        def _simulate_GBM(normals):
            s1 = np.exp((self.r - sigma_1 ** 2 / 2) * self.dt + sigma_1 * normals)
            s2 = np.exp((self.r - sigma_2 ** 2 / 2) * self.dt + sigma_2 * normals)
            s1 = self.s0 * s1.cumprod()
            s2 = self.s0 * s2.cumprod()
            return [s1, s2]

        normal_rv = np.random.normal(0, np.sqrt(self.dt), size=int(self.T / self.dt))
        s_list = [_simulate_GBM(normal_rv)]
        if not self.antithetic:
            return s_list
        s_list.append(_simulate_GBM(-normal_rv))
        return s_list

    def _simulate_gamma_path(self, dS0):

        s0_1 = self.s0 - 2 * dS0
        s0_2 = self.s0 - dS0
        s0_3 = self.s0
        s0_4 = self.s0 + dS0
        s0_5 = self.s0 + 2 * dS0

        def _simulate_GBM(normals):
            s = np.exp((self.r - self.sigma ** 2 / 2) * self.dt + self.sigma * normals)
            s1 = s0_1 * s.cumprod()
            s2 = s0_2 * s.cumprod()
            s3 = s0_3 * s.cumprod()
            s4 = s0_4 * s.cumprod()
            s5 = s0_5 * s.cumprod()
            return [s1, s2, s3, s4, s5]

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


option = OptionPricer(s0=100, r=0.05, sigma=0.4, T=1, dt=0.0001)
N = 10_000
option_params = {'K': 100,
                 'B_up': 200,
                 'B_down': 80}

prices = option.calculate_price('UpAndOutCall', N, **option_params)
deltas = option.calculate_delta('UpAndOutCall', N, dS0=1, **option_params)
gammas = option.calculate_gamma('UpAndOutCall', N, dS0=1, **option_params)
vegas = option.calculate_vega('UpAndOutCall', N, dsigma=0.01, **option_params)

print(f'The UpAndOut call price is {round(prices[0],2)} with 95% CI ({round(prices[1],2)}, {round(prices[2],2)})')
print(f'The UpAndOut delta is {round(deltas[0],2)} with 95% CI ({round(deltas[1],2)}, {round(deltas[2],2)})')
print(f'The UpAndOut gamma is {round(gammas[0],2)} with 95% CI ({round(gammas[1],2)}, {round(gammas[2],2)})')
print(f'The UpAndOut vega is {round(vegas[0],2)} with 95% CI ({round(vegas[1],2)}, {round(vegas[2],2)})')


