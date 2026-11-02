#!/usr/bin/env python3
"""Render exploratory paired BA changes and the CTH operating trade-off."""
import json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[3];D=ROOT/'research/results/association_fingerprint_v3'
within=json.loads((D/'ba_tree_nested_multisource.json').read_text())['results']; held=json.loads((D/'ba_leave_source_out_purged.json').read_text())['results'];errs=json.loads((D/'ba_leave_source_out_purged_errors.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
fig,(ax,bx)=plt.subplots(1,2,figsize=(11.2,4.4),gridspec_kw={'width_ratios':[1.35,1]},constrained_layout=True)
labels=['CTH · task holdout','KYPO · participant holdout','Antieval · task holdout','CTH · source holdout','KYPO · source holdout','Antieval · source holdout']; entries=[within['cth'],within['kypo'],within['antieval'],held['cth'],held['kypo'],held['antieval']]
for i,(lab,r) in enumerate(zip(labels,entries)):
 y=5-i;d=r['delta_ba']*100;lo,hi=np.array(r['paired_cluster_ci95'])*100; color='#2872b9' if i<3 else '#218886'
 ax.errorbar(d,y,xerr=[[d-lo],[hi-d]],fmt='o',color=color,capsize=3,markersize=5,linewidth=1.5);ax.text(hi+.3,y,f'{d:+.1f}',va='center',fontsize=9,color='#19334e')
ax.axvline(0,color='#74889c',linestyle='--',linewidth=1);ax.set_yticks(range(5,-1,-1),labels);ax.set_xlim(-7,11);ax.set_xlabel('Paired BA change (percentage points)');ax.set_title('a  Multi-source association',loc='left',fontweight='bold',color='#18324d');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
metrics=errs['cth']['metrics'];names=['FMR','FSR','BA'];x=np.arange(3);width=.31
b=[metrics[k]['baseline']*100 for k in ('fmr','fsr','ba')];f=[metrics[k]['fingerprint']*100 for k in ('fmr','fsr','ba')]
for off,vals,color,label in [(-width/2,b,'#286fbb','Evidence'),(width/2,f,'#218886','Evidence + fingerprint')]:
 bars=bx.bar(x+off,vals,width,color=color,label=label)
 for bar,val in zip(bars,vals):bx.text(bar.get_x()+bar.get_width()/2,val+1,f'{val:.1f}',ha='center',va='bottom',fontsize=8)
bx.set_xticks(x,names);bx.set_ylim(0,100);bx.set_ylabel('Task-macro rate (%)');bx.set_title('b  CTH source holdout',loc='left',fontweight='bold',color='#18324d');bx.grid(axis='y',alpha=.2);bx.set_axisbelow(True);bx.legend(loc='upper right',frameon=False,fontsize=8)
for base in [ROOT/'research/paper/figures/association-ba-v3',ROOT/'site/assets/association-fingerprint-v3/association-ba-v3']:
 for ext in ('pdf','png','svg'):fig.savefig(str(base)+'.'+ext,dpi=180,bbox_inches='tight',facecolor='white')
print('Rendered six paired BA comparisons and the CTH error trade-off.')
