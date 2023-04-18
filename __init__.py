from gym.envs.registration import register

register(
    id='TradingEnv-v1',
    entry_point='tradingEnv:TradingEnv',
)
