"""Illustrate the actual first feature row; does not estimate position or performance."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
df=pd.read_csv(R/'project/hpc/test.csv')
row=df.iloc[0]
img=np.zeros((8,8))
for ix in range(8):
 for iy in range(8): img[7-iy,ix]=row[f'pe_{ix}_{iy}']
fig,ax=plt.subplots(figsize=(6.8,5.6),layout='constrained')
im=ax.imshow(img,cmap='viridis')
ax.set(xticks=range(8),yticks=range(8),yticklabels=list(reversed(range(8))),xlabel='Channel index ix',ylabel='Channel index iy',title='First supplied feature row · observed count map')
c=fig.colorbar(im,ax=ax);c.set_label('Photoelectrons')
fig.savefig(R/'docs/assets/light-map.png',dpi=160)
plt.close(fig)
