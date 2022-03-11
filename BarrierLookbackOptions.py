import numpy as np
import matplotlib.pyplot as plt


class OptionPricer:

    def __init__(self, x0, r, sigma, T, dt, antithetic=True):
        self.paths = []
        self.x0 = x0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.dt = dt
        self.antithetic = antithetic

    def simulate_paths(self, N):
        counter = 0
        while counter < N:
            paths = self._simulate_one_path()
            if counter % 100 == 0:
                self.paths.append(paths[0])
            counter = counter + 1 if not self.antithetic else counter + 2

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


option = OptionPricer(x0=100, r=0.05, sigma=0.4, T=1, dt=0.0001)
option.simulate_paths(1)

