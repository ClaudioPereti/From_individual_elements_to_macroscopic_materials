# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
#Imports libraries and experiment's data
#score contain the metrics evaluated on test set sampled randomly at each run
#hosono contain the observed values and te predictions on hosono dataset
#supercon contain the predicted critical temperature for the material sampled during the train test split.
#The material not present on the test set are labelled with -1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

supercon = pd.read_csv('../data/experiments/dati_articolo/rgr_hosoko_supercon/supercon_tested_ready.csv')
hosono = pd.read_csv('../data/experiments/dati_articolo/rgr_hosoko_supercon/hosono_tested.csv')
score = pd.read_csv('../data/experiments/dati_articolo/rgr_hosoko_supercon/rgr_score_ready.csv')


# %% [markdown]
# # Average metrics of SuperCon's random sampled test set

# %%

score_dict = {metric : score[metric].mean() for metric in score.columns}
score_dict


# %% [markdown]
# # Average metrics of Hosono's test set

# %%

hosono_temperature_mask = hosono['Tc'] > 0
hosono_prediction = hosono.iloc[:,2:][hosono_temperature_mask]
hosono_critical_temperature = hosono['Tc'][hosono_temperature_mask]

rmse_hosono = [mean_squared_error(hosono_prediction.loc[:,prediction],hosono_critical_temperature,squared=False) for prediction  in hosono_prediction]
rmse_hosono = np.array(rmse_hosono)

fig,ax = plt.subplots()
pd.DataFrame(rmse_hosono).hist(ax = ax)
ax.set_title('RMSE Histogram')
ax.set_xlabel('Root mean squared error')
ax.set_ylabel('Count')

mse_hosono = [mean_squared_error(hosono_prediction.loc[:,prediction],hosono_critical_temperature) for prediction  in hosono_prediction]
mse_hosono = np.array(mse_hosono)

r2_hosono = [r2_score(hosono_prediction.loc[:,prediction],hosono_critical_temperature) for prediction  in hosono_prediction]
r2_hosono = np.array(r2_hosono)

print(f"Rmse: {rmse_hosono.mean():.1f} +- {rmse_hosono.std():.1f}\nMse: {mse_hosono.mean():.0f} +- {mse_hosono.std():.0f}\nR2: {r2_hosono.mean():.2f} +- {r2_hosono.std():.2f}")



# %% [markdown]
# # Select material tested at least a number (num) times

# %%
supercon_prediction = supercon.iloc[:,2:]

num_tested_material = (supercon_prediction > -1).sum(axis=1)
num = 10
print(f"Number of material tested at least {num}: {(num_tested_material > num).sum()}")


# %%
supercon_prediction_on_selected_material = supercon_prediction[num_tested_material > num]
supercon_prediction_on_selected_material = supercon_prediction_on_selected_material.replace(-1,np.nan)
supercon_temperature_on_selected_material = supercon[num_tested_material > num]['critical_temp']


supercon_average_prediction = supercon_prediction_on_selected_material.mean(axis = 1)
supercon_std_prediction = supercon_prediction_on_selected_material.std(axis = 1)

hosono_average_prediction = hosono_prediction.mean(axis=1)
hosono_std_prediction =  hosono_prediction.std(axis=1)



# %% [markdown]
# # Plot average prediction and obeserved critical temperature with error bar

# %%
plt.rcParams["figure.figsize"] = (8,8)
plt.errorbar(supercon_temperature_on_selected_material,supercon_average_prediction,yerr=supercon_std_prediction,fmt='o',markersize=7,capsize=3,label='SuperCon',)
plt.title('Critical Temperature')
plt.ylabel('Predicted critical temperature')
plt.xlabel('Observed critical temperature')
plt.errorbar(hosono_critical_temperature,hosono_average_prediction,yerr=hosono_std_prediction,fmt='d',markersize=7,capsize=3,label='Hosono',)
plt.plot([0,140],[0,140],'r--',linewidth=1.5,zorder = 3)
plt.legend()

# %%
