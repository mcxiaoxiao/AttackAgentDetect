#!/usr/bin/env python3
import json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[3];D=json.loads((ROOT/'research/results/behavioral_fingerprint_v2/refinement/error_strata.json').read_text())
BLUE='#2676bd';TEAL='#238b8e';ORANGE='#ce7232';NAVY='#17324f';GRID='#d9e2ec'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'text.color':NAVY,'axes.labelcolor':NAVY,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(7.2,3.0),gridspec_kw={'width_ratios':[1,1.15]})
ax=axs[0]
for i,typ in enumerate(('HH','HA','AA')):
 r=next(x for x in D['summary'] if x['method']=='command_cosine' and x['metric']=='FSR' and x['type']==typ);v=r['value']*100;lo,hi=np.array(r['ci95'])*100
 ax.errorbar(i,v,yerr=[[v-lo],[hi-v]],fmt='o',color=BLUE,capsize=3);ax.text(i,v+4,f'{v:.1f}',ha='center',fontsize=7.5)
ax.set(xticks=[0,1,2],xticklabels=['HH','HA','AA'],ylim=(0,105),ylabel='Task-macro FSR (%)');ax.set_title('a  Execution mode changes missed links',loc='left',fontsize=8.5,weight='bold');ax.grid(axis='y',color=GRID,lw=.6)
ax=axs[1]
for i,typ in enumerate(('HH','HA','AA')):
 for met,shift,color,marker in [('FSR',.12,TEAL,'o'),('FMR',-.12,ORANGE,'s')]:
  r=next(x for x in D['paired_deltas'] if x['type']==typ and x['metric']==met);v=r['delta']*100;lo,hi=np.array(r['ci95'])*100
  ax.errorbar(v,2-i+shift,xerr=[[v-lo],[hi-v]],fmt=marker,color=color,capsize=3,label=met if i==0 else None)
ax.axvline(0,color=NAVY,lw=.8,ls='--');ax.set(yticks=[2,1,0],yticklabels=['HH','HA','AA'],xlabel='Change after B/S fusion (pp)',xlim=(-40,18));ax.set_title('b  Fingerprints shift the error trade-off',loc='left',fontsize=8.5,weight='bold');ax.grid(axis='x',color=GRID,lw=.6);ax.legend(frameon=False,fontsize=7,loc='lower left',ncol=2)
fig.subplots_adjust(left=.09,right=.98,top=.88,bottom=.20,wspace=.36)
for root in [ROOT/'research/paper/figures',ROOT/'site/assets/behavioral-fingerprint-v2']:
 for ext in ('pdf','svg','png'):fig.savefig(root/f'fingerprint-v2-error-strata.{ext}',bbox_inches='tight',dpi=240 if ext=='png' else None)
