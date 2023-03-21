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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %% [markdown]
# Fixed score contain the metrics of the neural network trained with an engineered input with dimension 80

# %%
fixed_score = pd.read_csv('../data/experiments/dati_articolo/comparison/fixed_dim1.csv')
fixed_score.mean()['rmse']


# %% [markdown]
# different_score contain the metrics of deepsets trained with a latent space tuned like an hyperparameter

# %%
different_score = pd.read_csv('../data/experiments/dati_articolo/comparison/score_different_latent_dim.csv')#.drop(index=[0,3,5]).drop(labels='latent_dim',axis=1).reset_index(drop=True).astype(float)
different_score = different_score.tail(140)[(different_score.tail(140)['r2'].astype(float)>0)]
different_score_std = different_score.astype(float).groupby(by='latent_dim').std()
different_score_mean = different_score.astype(float).groupby(by='latent_dim').mean()

# %%

fig, ax = plt.subplots()
sns.scatterplot(data = different_score_mean,x='latent_dim',y='rmse',ax = ax,label='DeepSet')
sns.scatterplot(x=[80],y=fixed_score.mean()['rmse'],ax= ax,label='Engineered features')


# %%


fig = plt.figure()
plt.errorbar(x=different_score_mean.index,y=different_score_mean['rmse'],yerr = different_score_std['rmse'] ,fmt = 'o',markersize=8,capsize=2,label='Deep Set')

plt.errorbar(x=[80],y=fixed_score.mean()['rmse'],yerr=fixed_score.std()['rmse'],fmt='o',markersize=8,capsize=2,label='Engineered features')
fig.legend(loc='upper right',bbox_to_anchor=(0.88,0.88))
plt.xlabel('Latent dimension')
plt.ylabel('Root mean squared error')

