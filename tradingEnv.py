import gym
import numpy as np
import pandas as pd
from gym import spaces


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data, initial_balance=10000, leverage=1, max_drawdown=0.2, spread=2, allowed_weekend=False, base_currency='USD', max_lot_size=1.0, window=10, **kwargs):
        super(TradingEnv, self).__init__()

        self.data = data
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.max_drawdown = max_drawdown
        self.spread = spread / 10000  # Convertir pips a unidades
        self.allowed_weekend = allowed_weekend
        self.base_currency = base_currency
        self.max_lot_size = max_lot_size
        self.window = window
        self.current_step = window
        self.lowest_balance = initial_balance

        # Espacio de acción discreto con dos opciones: 0, que significa no hacer nada, 1 que significa comprar, y 2 que significa vender
        # El segundo elemento de la acción es un valor continuo que representa el porcentaje del máximo tamaño de lote que se desea comprar o vender (de -1 a 1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)  # Espacio de acciones continuas con dos dimensiones
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(window, 7))  # Ajustar la dimensión para incluir la información de varios períodos

    def reset(self):
        self.current_step = self.window
        self.balance = self.initial_balance
        self.units_held = 0
        self.total_units_sold = 0
        self.total_units_bought = 0
        self.net_worth = self.initial_balance
        self.lowest_balance = self.initial_balance
        self.stop_loss = None

        return self._get_observation()

    def _get_observation(self):
        window_data = self.data.iloc[self.current_step - self.window:self.current_step]
        day_of_week = window_data.index.dayofweek.to_numpy()
        obs = np.column_stack([
            window_data['Open'].to_numpy(),
            window_data['High'].to_numpy(),
            window_data['Low'].to_numpy(),
            window_data['Close'].to_numpy(),
            window_data['Volume'].to_numpy(),
            np.full(self.window, self.balance),
            day_of_week
        ])

        return obs

    def step(self, action):
        current_price = self.data.iloc[self.current_step]['Close']
        day_of_week = self.data.index[self.current_step].dayofweek
        self.current_step += 1
        prev_net_worth = self.net_worth
        done = False

        if action[0] == 1:  # Comprar
            lot_size = action[1] * self.max_lot_size
            units_to_buy = int((self.balance * self.leverage * lot_size) // ((current_price + self.spread) * 1000))
            self.balance -= units_to_buy * (current_price + self.spread) * 1000
            self.units_held += units_to_buy
            self.total_units_bought += units_to_buy
            self.stop_loss = current_price * 0.95  # Establecer el stop-loss en el 5% por debajo del precio de compra

        elif action[0] == 2:  # Vender
            lot_size = action[1] * self.max_lot_size
            units_to_sell = int(self.units_held * lot_size)
            self.balance += units_to_sell * (current_price - self.spread) * 1000
            self.units_held -= units_to_sell
            self.total_units_sold += units_to_sell

        self.net_worth = self.balance + self.units_held * current_price * 1000
        self.lowest_balance = min(self.lowest_balance, self.balance)

        if self.stop_loss is not None and current_price <= self.stop_loss:  # Si se alcanza el stop-loss
            self.balance += self.units_held * (self.stop_loss - self.spread) * 1000
            self.units_held = 0

        if self.net_worth <= self.initial_balance * (1 - self.max_drawdown):
            done = True

        done |= self.current_step >= len(self.data) - 1
        reward = self.net_worth - prev_net_worth

        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Net Worth: {self.net_worth}, Balance: {self.balance}, Units Held: {self.units_held}, Stop Loss: {self.stop_loss}")

    def close(self):
        pass