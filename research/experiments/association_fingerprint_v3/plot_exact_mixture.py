#!/usr/bin/env python3
"""Plot the frozen-core exact-mixture baseline from measured task means."""
import json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[3]
d=json.loads((ROOT/'research/results/association_fingerprint_v3/exact_mixture.json').read_text())
colors={'command_cosine':'#286fbb','action_bigram_cosine':'#218886'};labels={'command_cosine':'Command similarity','action_bigram_cosine':'Action bigrams'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(10.8,3.6),constrained_layout=True,sharex=True)
for col,(metric,title) in enumerate([('fmr','False merges'),('fsr','Missed links'),('ba','Balanced accuracy')]):
 ax=axes[col]
 for method in colors:
  series=sorted([r for r in d['results'] if r['method']==method],key=lambda r:r['agent_fraction']); x=[r['agent_fraction']*100 for r in series]; y=[r[metric]*100 for r in series]
  ax.plot(x,y,marker='o',markersize=4,color=colors[method],linewidth=2,label=labels[method])
  lows=[];highs=[]
  for r in series:
   count=round(r['agent_fraction']*32);v=np.array([arr[col] for arr in d['task_means'][f'{count}:{method}'].values()]);rng=np.random.default_rng(20261103);boot=v[rng.integers(0,len(v),(10000,len(v)))].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975])*100;lows.append(lo);highs.append(hi)
  ax.fill_between(x,lows,highs,color=colors[method],alpha=.11,linewidth=0)
 ax.set_title(f'{"abc"[col]}  {title}',loc='left',fontweight='bold',color='#18324d');ax.set_xlim(-3,103);ax.set_ylim(0,100);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Agent roles (%)');ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True)
 if col==0:ax.set_ylabel('Task-macro rate (%)')
handles,leglabels=axes[0].get_legend_handles_labels();fig.legend(handles,leglabels,loc='outside upper center',ncol=2,frameon=False)
for base in [ROOT/'research/paper/figures/exact-mixture-v3',ROOT/'site/assets/association-fingerprint-v3/exact-mixture-v3']:
 for ext in ('pdf','png','svg'):
  fig.savefig(str(base)+'.'+ext,dpi=180,bbox_inches='tight',facecolor='white')
print('Generated exact-mixture figure from per-task measurements.')
