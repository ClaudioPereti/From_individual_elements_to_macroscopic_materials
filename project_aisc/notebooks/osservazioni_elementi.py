# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#Load SueprCon data
supercon_path = "../data/raw/supercon.csv"
supercon = pd.read_csv(supercon_path)

# # Temperature critiche in SuperCon

critical_temperature = supercon['critical_temp']
plt.title("Istogramma temperature critiche SuperCon")
plt.ylabel("Count")
plt.xlabel("Temperature critiche")
critical_temperature.hist()

sns.boxplot(x = critical_temperature,whis=1.5)

print(f"median : {critical_temperature.median()}\nmin & max temperature :{critical_temperature.min(),critical_temperature.max()}\n25 & 75 percentile: {np.percentile(critical_temperature,[25,75])}")

# ## Commento referee pnas
#
# Further, it is surprising that Fig S3 shows that divalent elements Ca, Sr, and Ba contribute to increasing Tc more than Cu or Fe. This is counter to the established role that Cu and Fe are known to play in producing high Tc's. The trends reported in Figs. 5, S3 and S5 reaffirm previous concerns about the soundness of the training basis and its predictive capability. 

# # Elementi in SuperCon

supercon_elements = supercon.iloc[:,:-2]
# We inspect the distribution of elements on superconducting materials
plt.title("Distribuzione elementi in SuperCon")
plt.ylabel("Count")
supercon_elements.sum().sort_values(ascending=False).plot.bar(figsize=(20,20))

# Now we look at the percentage of elements that appear on a chemical formula at least one
plt.title("Distribuzione frequenza elementi che appaiono almeno una volta")
plt.ylabel("Frequenza")
((supercon_elements>0).sum()/supercon_elements.shape[0]).sort_values(ascending=False).plot.bar(figsize= (20,20))


most_common_elements = list(((supercon_elements>0).sum()/supercon_elements.shape[0]).sort_values(ascending=False).head(10).index)
supercon_elements[most_common_elements].corr().style.background_gradient()
# Correlazione pearson

# # Analisi Cu in SuperCon

# Materiali che hanno almeno un atomo di Cu
print(f"Materiali con almeno un Cu : {supercon[supercon['Cu'] > 0]['critical_temp'].count()}\nFrazione materiali con almeno un Cu : {supercon[supercon['Cu'] > 0]['critical_temp'].count()/supercon.shape[0]}")

plt.title('Istogramma materiali con Cu')
plt.ylabel('Count')
plt.xlabel('Temperature critiche')
supercon[supercon['Cu'] > 0]['critical_temp'].hist()

critical_temperature = supercon[supercon['Cu'] > 0]['critical_temp']
print(f"median : {critical_temperature.median()}\nmin & max temperature :{critical_temperature.min(),critical_temperature.max()}\n25 & 75 percentile: {np.percentile(critical_temperature,[25,75])}")

# # Analisi Fe in SuperCon

print(f"Materiali con almeno un Fe : {supercon[supercon['Fe'] > 0]['critical_temp'].count()}\nFrazione materiali con almeno un Fe : {supercon[supercon['Fe'] > 0]['critical_temp'].count()/supercon.shape[0]}")

plt.title('Istogramma materiali con Fe')
plt.ylabel('Count')
plt.xlabel('Temperature critiche')
supercon[supercon['Fe'] > 0]['critical_temp'].hist()

critical_temperature = supercon[supercon['Fe'] > 0]['critical_temp']
print(f"median : {critical_temperature.median()}\nmin & max temperature :{critical_temperature.min(),critical_temperature.max()}\n25 & 75 percentile: {np.percentile(critical_temperature,[25,75])}")

# # Analisi Ba in SuperCon

print(f"Materiali con almeno un Ba : {supercon[supercon['Ba'] > 0]['critical_temp'].count()}\nFrazione materiali con almeno un Ba : {supercon[supercon['Ba'] > 0]['critical_temp'].count()/supercon.shape[0]}")

plt.title('Istogramma materiali con Ba')
plt.ylabel('Count')
plt.xlabel('Temperature critiche')
supercon[supercon['Ba']>0]['critical_temp'].hist()

critical_temperature = supercon[supercon['Ba'] > 0]['critical_temp']
print(f"median : {critical_temperature.median()}\nmin & max temperature :{critical_temperature.min(),critical_temperature.max()}\n25 & 75 percentile: {np.percentile(critical_temperature,[25,75])}")

# Ba e Cu sono correlati; guardiamo quanto
print(f"Materiali che hanno Ba ma non Cu : {supercon[supercon['Ba']>0][supercon['Cu'] == 0].shape[0]}\nFrazione materiali che hanno Ba ma non Cu sul totale di materiali con Ba : {supercon[supercon['Ba']>0][supercon['Cu'] == 0].shape[0]/supercon[supercon['Ba']>0].shape[0]}")

# Ba e Fe sono correlati; guardiamo quanto
print(f"Materiali che hanno Ba ma non Fe : {supercon[supercon['Ba']>0][supercon['Fe'] == 0].shape[0]}\nFrazione materiali che hanno Ba ma non Fe sul totale di materiali con Ba : {supercon[supercon['Ba']>0][supercon['Fe'] == 0].shape[0]/supercon[supercon['Ba']>0].shape[0]}")

critical_temperature = supercon[supercon['Ba'] > 0][supercon['Cu'] == 0]['critical_temp']
print(f"median : {critical_temperature.median()}\nmin & max temperature :{critical_temperature.min(),critical_temperature.max()}\n25 & 75 percentile: {np.percentile(critical_temperature,[25,75])}")

# # Conclusioni sul Ba
#
# ## Distribuzione in SuperCon
# Il bario è particolarmente presente nel dataset SuperCon (4673 materiali) e tende significativamente a presentarsi in composti nei quali è presente anche rame. Infatti quasi il ~90% dei materiali contenenti bario contiene anche rame.
#
# ## Temperature Critiche
# I materiali a base di bario mostrano un'elevata temperatura critica media (mediana) pari a 68 K. Questa temperatura critica è maggiore dei materiali a base di rame (58.8 K) e di ferro (20.88 K). Il 50% dei materiali abita la regione di temperature critiche compresa fra 35 e 86 K con gli estremi che variano fra temperature molto basse (0.1) ed estremamente alte (143) 

# # Analilsi Ca in SuperCon

# +

print(f"Materiali con almeno un Ca : {supercon[supercon['Ca'] > 0]['critical_temp'].count()}\nFrazione materiali con almeno un Ca : {supercon[supercon['Ca'] > 0]['critical_temp'].count()/supercon.shape[0]}")
# -

plt.title('Istogramma materiali con Ca')
plt.ylabel('Count')
plt.xlabel('Temperature critiche')
supercon[supercon['Ca']>0]['critical_temp'].hist()

critical_temperature = supercon[supercon['Ca'] > 0]['critical_temp']
print(f"median : {critical_temperature.median()}\nmin & max temperature :{critical_temperature.min(),critical_temperature.max()}\n25 & 75 percentile: {np.percentile(critical_temperature,[25,75])}")

# Ca e Cu sono correlati; guardiamo quanto
print(f"Materiali che hanno Ca ma non Cu : {supercon[supercon['Ca']>0][supercon['Cu'] == 0].shape[0]}\nFrazione materiali che hanno Ca ma non Cu sul totale di materiali con Ca : {supercon[supercon['Ca']>0][supercon['Cu'] == 0].shape[0]/supercon[supercon['Ca']>0].shape[0]}")

# Ca e Fe sono correlati; guardiamo quanto
print(f"Materiali che hanno Ca ma non Fe : {supercon[supercon['Ca']>0][supercon['Fe'] == 0].shape[0]}\nFrazione materiali che hanno Ba ma non Cu sul totale di materiali con Ba : {supercon[supercon['Ca']>0][supercon['Fe'] == 0].shape[0]/supercon[supercon['Ba']>0].shape[0]}")

critical_temperature = supercon[supercon['Ca'] > 0][supercon['Cu'] == 0]['critical_temp']
print(f"median : {critical_temperature.median()}\nmin & max temperature :{critical_temperature.min(),critical_temperature.max()}\n25 & 75 percentile: {np.percentile(critical_temperature,[25,75])}")

# # Conclusioni sul Ca
#
# ## Distribuzione Ca in SuperCon
#
# Il calcio è uno fra gli elementi più presenti nei materiali appartenenti al dataset SuperCon ( 3343 materiali) e si associa fortemente al rame. Il ~90% dei materiali aventi Ca possiede anche atomi di rame al suo interno.
#
# ## Temperature Critiche
#
# I materiali a base di calcio mostrano una temperatura mediana (73 K) maggiore dei materiali a base di rame (58.8 K), di ferro (20.88 K) e di bario (68.8 K). I materiali a base di calcio mostrano una maggiore localizzazione attorno alla temperatura mediana rispetto ai materiali a base di bario come si deduce dalla distribuzione delle temperature critiche e dal fatto che il 50% dei materiali si trovi fra 41 e  88 K. La presenza di materiali a base di calcio spazia la regione di temperature critiche da 0.1 K fineo a 143 K. 

# # Analisi Sr in SuperCon

# +

print(f"Materiali con almeno un Sr : {supercon[supercon['Sr'] > 0]['critical_temp'].count()}\nFrazione materiali con almeno un Sr : {supercon[supercon['Sr'] > 0]['critical_temp'].count()/supercon.shape[0]}")
# -

plt.title('Istogramma materiali con Sr')
plt.ylabel('Count')
plt.xlabel('Temperature critiche')
supercon[supercon['Sr']>0]['critical_temp'].hist()

critical_temperature = supercon[supercon['Sr'] > 0]['critical_temp']
print(f"median : {critical_temperature.median()}\nmin & max temperature :{critical_temperature.min(),critical_temperature.max()}\n25 & 75 percentile: {np.percentile(critical_temperature,[25,75])}")

# Sr e Cu sono correlati; guardiamo quanto
print(f"Materiali che hanno Sr ma non Cu : {supercon[supercon['Sr']>0][supercon['Cu'] == 0].shape[0]}\nFrazione materiali che hanno Sr ma non Cu sul totale di materiali con Sr : {supercon[supercon['Sr']>0][supercon['Cu'] == 0].shape[0]/supercon[supercon['Sr']>0].shape[0]}")

# Sr e Fe sono correlati; guardiamo quanto
print(f"Materiali che hanno Sr ma non Fe : {supercon[supercon['Sr']>0][supercon['Fe'] == 0].shape[0]}\nFrazione materiali che hanno Sr ma non Fe sul totale di materiali con Sr : {supercon[supercon['Sr']>0][supercon['Fe'] == 0].shape[0]/supercon[supercon['Sr']>0].shape[0]}")

critical_temperature = supercon[supercon['Sr'] > 0][supercon['Cu'] == 0]['critical_temp']
print(f"median : {critical_temperature.median()}\nmin & max temperature :{critical_temperature.min(),critical_temperature.max()}\n25 & 75 percentile: {np.percentile(critical_temperature,[25,75])}")

# # Conclusioni sul Sr
#
# ## Distribuzione Sr in SuperCon
#
# Lo stronzio è molto presente nei materiali appartenenti al dataset SuperCon ( 3693 materiali) e si associa fortemente al rame. Il ~90% dei materiali aventi Sr possiede anche atomi di rame al suo interno.
#
# ## Temperature Critiche
#
# I materiali a base di stronzio mostrano una temperatura mediana (42 K) maggiore rispetto alla mediana del dataset SuperCon e del ferro ma inferiore al rame, bario e calcio . I materiali a base di stronzio mostrano una scarsa localizzazione attorno alla temperatura mediana, con uno spicatto accumulo sotto 42 ma lunghe code per alte temperature. Il 50% dei materiali si trovi fra 24 e 76.5 K. La presenza di materiali a base di stronzio spazia la regione di temperature critiche da 0.001 K fineo a 136 K.

only_ca = supercon[supercon['Ca']>0][supercon['Ba'] == 0].shape[0]
only_ba = supercon[supercon['Ba']>0][supercon['Ca'] == 0].shape[0]
only_both = supercon[supercon['Ca']>0][supercon['Ba'] > 0].shape[0]
print(f"Somma dei materiali aventi Bario e Calcio : {only_ca + only_ba + only_both}")

supercon[supercon['Cu']>0]['Cu'].mean()

supercon[supercon['Cu']>0][supercon['Sr']>0]['Sr'].mean()

supercon[supercon['Cu']>0][supercon['Ba']>0]['Ba'].mean()

supercon[supercon['Cu']>0][supercon['Ca']>0]['Ca'].mean()

# # Conclusioni Generali
#
# Non risulta sorprendente, considerato il dataset SuperCon usato per l'addestramento, che la rete associ ad elementi come il bario (Ba) ed il calcio (Ca) un maggiore contributo per l'incremento della temperatura critica. Questi non sono da considerare come un raggruppamento a se stante di materiali superconduttori ma come un sottogruppo dei cupriti caratterizzati da maggiori temperature critiche. Vale un discorso analogo per i materiali a base di stronzio che possono essere considerati come un sotto-raggruppamento. Tuttavia hanno una temperatura critica mediana inferiore a quella dei cupriti. La differenza che la rete attribuisce a questi due elementi (che è minore rispetto agli altri due) non è chiara.
# Possiamo avanzare l'ipotesi che questa differenza sia un riflesso della distribuzione degli elementi presenti in SuperCon. Per i cupriti, la presenza media di rame nei composti e di 2.5 atomi. Se prendiamo invece lo stronzio, questo è di soli 1.5, per il calcio di 1.14 mentre per il bario è di 1.7 atomi (presi come sottoinsiemi dei cupriti, anche in generale non varia significativamente).
# Essendo meno numerosi mediamente, questi elementi devono contribuire maggiormente per singolo contributo, ovvero per singolo atomo.
#


