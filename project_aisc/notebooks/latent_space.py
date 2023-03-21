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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import mendeleev

# %%
molecular_representation = [pd.read_csv('../data/experiments/dati_articolo/latent_space/01_bidimensional_molecules_classification'+str(i)+'.csv') for i in range(60)]
for i in range(0,len(molecular_representation)):
    molecular_representation[i]['X1'] = list(map(lambda x: float(x.strip('[]')),molecular_representation[i]['X1'].values))
molecular_representation_sorted = []

for mol in molecular_representation:
    factor_X1 = 1

    if (mol['X1']>100).sum() > 5:
        factor_X1 = -1

    molecular_representation_sorted.append(mol[['X1','Critical temperature']] * [factor_X1,1])
    
molecular_representation_averaged = sum(molecular_representation_sorted)/len(molecular_representation_sorted)

# %% [markdown]
# # Histogram of Superconductors and Non superconductors in latent space for classifier

# %%
fig,ax = plt.subplots(figsize=(8,8))
sns.histplot(data=molecular_representation_averaged,x='X1',hue='Critical temperature',ax=ax)
plt.legend(labels=['Superconductor','Non Superconductor'])
ax.set_xlim([-30,10])
ax.set_xlabel(r'$X_1$',fontdict={'fontsize':14})

# %%
atomic_representation = [pd.read_csv('../data/experiments/dati_articolo/latent_space/01_bidimensional_atom_classification'+str(i)+'.csv') for i in range(50)]

atomic_representation_sorted = []
for index,atom in enumerate(atomic_representation):
    factor_X1 = 1
    mol = molecular_representation[index]
    if (mol['X1']>100).sum() > 5:
        factor_X1 = -1
    atomic_representation_sorted.append(atom[['X1',]] * [factor_X1,])
    
atomic_representation_averaged = sum(atomic_representation_sorted)/len(atomic_representation_sorted)

# %%
symbols = [mendeleev.element(i).symbol for i in range(1,atomic_representation_averaged.shape[0]+1)]
atom_list = [i for i in range(96)]
atomic_representation_averaged.index = symbols

gas_nobles = [1,9,17,35,53,85]
alkali_metals = [0,2,10,18,36,54,86]
earth_alkali_metals = [3,11,19,37,55,87]
transitional_elements = [i for i in range(20,30)] + [i for i in range(38,49)] + [56]+[i for i in range(71,81)]
halogens = [8,16,34,52,84]
reactive_nonmetals = [5,6,7,14,15,33]
metalloid = [30,48,49,80,81,82]
lanthanides = [i for i in range(57,72)]
actinides = [i for i in range(89,96)]
atomic_categories = ['Transition metals' for i in range(96)]
categories = [gas_nobles,alkali_metals,earth_alkali_metals,transitional_elements,halogens,reactive_nonmetals,metalloid,lanthanides,actinides]


for gas in actinides:
    atomic_categories[gas] = 'Actinides'
for gas in lanthanides:
    atomic_categories[gas] = 'Lanthanides'
for gas in metalloid:
    atomic_categories[gas] = 'Metalloid'
for gas in reactive_nonmetals:
    atomic_categories[gas] = 'Reactive nonmetals'
for gas in halogens:
    atomic_categories[gas] = 'Halogens'
for gas in earth_alkali_metals:
    atomic_categories[gas] = 'Alkaline earth metals'
for gas in alkali_metals:
    atomic_categories[gas] = 'Alkali metals'
for gas in gas_nobles:
    atomic_categories[gas] = 'Noble gases'
    
atomic_representation_averaged['Categories'] = atomic_categories

# %% [markdown]
# # Atomic & Molecular Latent Space for classifier

# %%
fig, axs = plt.subplots(1,2,figsize = (13,12))
sns.scatterplot(data=atomic_representation_averaged,y=atomic_representation_averaged.index,x='X1',style='Categories',hue='Categories',ax = axs[0])
axs[0].set_title('Atomic Latent Space',fontdict={'fontsize':18})
axs[0].set_xlabel(r'$X_1$',fontdict={'fontsize':14})
axs[0].set_ylabel('Atomic number',fontdict={'fontsize':14})
axs[0].set_yticks([])
axs[0].invert_yaxis()

for line in range(0,atomic_representation_averaged.shape[0]):
     axs[0].text(x = atomic_representation_averaged.iloc[line,0] + 0.02,
                 y = atomic_representation_averaged.index[line],
                 s = mendeleev.element(line+1).symbol,
                 fontsize = 8,
                )

sns.histplot(data=molecular_representation_averaged,x='X1',hue='Critical temperature',ax=axs[1])
axs[1].set_xlim([-30,10])
axs[1].set_title('Molecular Latent Space',fontdict={'fontsize':18})
axs[1].set_xlabel(r'$X_1$',fontdict={'fontsize':14})
axs[1].legend({'Supercontuctor':1,'Non Superconductor':0})
#fig.savefig('/home/claudio/art_cls_latent_space.png')

# %%
molecular_representation = [pd.read_csv('../data/experiments/dati_articolo/latent_space/01_reversed_bidimensional_molecules_regression'+str(i)+'.csv') for i in range(50)]
for i in range(0,len(molecular_representation)):
    molecular_representation[i]['X1'] = list(map(lambda x: float(x.strip('[]')),molecular_representation[i]['X1'].values))

molecular_representation_sorted = []

for mol in molecular_representation:
    factor_X1 = 1

    if (mol[mol['Critical temperature'] < 10.0]['X1'] > 0).sum() > 1000:
        factor_X1 = -1

    molecular_representation_sorted.append(mol[['X1','Critical temperature']] * [factor_X1,1])

molecular_representation_averaged = sum(molecular_representation_sorted)/len(molecular_representation_sorted)

# %%
#Critical Temperature contain the high critical temperature and the low ones with a threshold setted to 10
molecular_representation_averaged['Critical Temperature'] = (molecular_representation_averaged['Critical temperature']>10).astype(int)

# %% [markdown]
# # Histogram of Superconductors  with high  critical temperature and with low critical tempearture
#

# %%
fig,ax = plt.subplots()
sns.histplot(data=molecular_representation_averaged,x='X1',hue='Critical Temperature',ax=ax)
ax.set_xlim([-40,30])
ax.set_xlabel(r'$X_1$',fontdict={'fontsize':14})
ax.legend(labels = ['High temperature','Low temperature'])

# %%
atomic_representation = [pd.read_csv('../data/experiments/dati_articolo/latent_space/01_reversed_bidimensional_atom_regression'+str(i)+'.csv') for i in range(50)]
atomic_representation_sorted = []

for index,mol in enumerate(molecular_representation):
    factor_X1 = 1

    if (mol[mol['Critical temperature'] < 10.0]['X1']<0).sum() > 1000:
        factor_X1 = -1

    atomic_representation_sorted.append(atomic_representation[index]*factor_X1)

atomic_representation_averaged = sum(atomic_representation_sorted)/len(atomic_representation_sorted)
atomic_representation_averaged['Categories'] = atomic_categories

# %% [markdown]
# # Atomic & Molecular Latent Space for regressor

# %%

atomic_representation_averaged = pd.read_csv('../data/experiments/dati_articolo/rgr_avg_features.csv',index_col=0)
atomic_representation_averaged = -1* atomic_representation_averaged
atomic_representation_averaged['Categories'] = atomic_categories

# %%
fig, axs = plt.subplots(1,2,figsize = (13,12))
sns.scatterplot(data=atomic_representation_averaged,y=atomic_representation_averaged.index,x='X',style='Categories',hue='Categories',ax = axs[0])
axs[0].set_title('Atomic Latent Space',fontdict={'fontsize':18})
axs[0].set_xlabel(r'$X_1$',fontdict={'fontsize':14})
axs[0].set_ylabel('Atomic number',fontdict={'fontsize':14})
axs[0].set_yticks([])

for line in range(0,atomic_representation_averaged.shape[0]):
    axs[0].text(x = atomic_representation_averaged.iloc[line,0] + 0.02,
                y = atomic_representation_averaged.index[line],
                s = mendeleev.element(line+1).symbol,
                )
sns.histplot(data=molecular_representation_averaged,x='X1',hue='Critical Temperature',ax=axs[1])
axs[1].set_xlim([-50,20])
axs[1].legend(labels = ['High temperature','Low temperature'])
axs[1].set_title('Molecular Latent Space',fontdict={'fontsize':18})
axs[1].set_xlabel(r'$X_1$',fontdict={'fontsize':14})
axs[1].legend(['High temperature','Low temperature'])
#fig.savefig('/home/claudio/art_rgr_latent_space.png')

# %%
