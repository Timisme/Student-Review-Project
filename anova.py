import numpy as np 
import pandas as pd 
import statsmodels.api as sm 
from statsmodels.formula.api import ols

df = pd.DataFrame({
	'Drill_Speed': np.repeat(['125','200'],8),
	'Feed_Rate': np.tile(np.repeat(['0.015','0.03','0.045','0.06'], 2), 2),
	'thrust_force': [2.7, 2.78, 2.45, 2.49, 2.6, 2.72, 2.75, 2.86,
	2.83, 2.86, 2.85, 2.8, 2.86, 2.87, 2.94, 2.88],
	'block': np.tile(['1','2'], 8)})

model = ols('thrust_force ~ C(Drill_Speed) + C(Feed_Rate) + C(Drill_Speed):C(Feed_Rate) + C(block)', data= df).fit()
model = ols('thrust_force ~ C(Drill_Speed) + C(Feed_Rate) + C(Drill_Speed):C(Feed_Rate)', data= df).fit()

print(sm.stats.anova_lm(model, typ= 2))
