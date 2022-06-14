import pandas as pd
import numpy as np
from sklearn.metrics.scorer import make_scorer

def get_custom_scorer(scorer,greater_is_better):
  return make_scorer(scorer,greater_is_better)



def MAPE(y_pred,y_true):
  return np.mean(abs(y_pred-y_true)*100/y_true)
    
  
if __name__=="__main__":
  print(f"{__file__} called")
