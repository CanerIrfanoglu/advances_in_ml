import numpy as np

# Generate tick imbalanced bars id of every tick by computing: get_bars_ids(get_b(ticks))
class Tick_Imbalance():
  def __init__(self, alpha_T, alpha_imbalance, expected_imbalance, expected_ticks):
    self.T = 0
    self.bar_id = 0
    self.imbalance = 0
    self.expected_tick_imbalance = expected_imbalance * expected_ticks

    self.alpha_ticks = alpha_T
    self.alpha_imbalance = alpha_imbalance
    self.EWMA_ticks = expected_ticks
    self.EWMA_imbalance = expected_imbalance


  def __get_EWMA(self, alpha, rt, EWMA_0):
    return alpha * rt + (1 - alpha) * EWMA_0

  def __get_b(self, price_change):
    if price_change == 0: return self.last_b
    self.last_b = abs(price_change)/price_change

    return self.last_b

  def get_b(self, price_ticks: np.ndarray) -> np.ndarray:
    price_change = (price_ticks[1:] - np.roll(price_ticks, 1)[1:]) / np.roll(price_ticks, 1)[1:]
    b = [self.__get_b(x) for x in price_change]

    return np.asarray(b)

  def get_bar_ids(self, b: np.ndarray) -> np.ndarray:
    bars_ids = []    
    for _imbalance in b:
      self.T += 1
      self.imbalance += _imbalance

      self.EWMA_imbalance = self.__get_EWMA(self.alpha_imbalance, _imbalance, self.EWMA_imbalance)

      bars_ids.append(self.bar_id)

      if abs(self.imbalance) >= self.expected_tick_imbalance:
        self.EWMA_ticks = self.__get_EWMA(self.alpha_ticks, self.T, self.EWMA_ticks)
        
        self.expected_tick_imbalance = self.EWMA_ticks * abs(self.EWMA_imbalance)

        self.T = 0
        self.imbalance = 0
        self.bar_id += 1
  
    return np.asarray(bars_ids)
