#!/usr/bin/env python3
"""One schematic atlas for the 24 defined B/S/T measurements, without results."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from paper_style import apply, save, PAGE_WIDTH, INK, BLUE, TEAL, MUTED, GRID

ROOT = Path(__file__).resolve().parents[3]
TIME = "#7a8798"


def main():
    apply()
    fig, axes = plt.subplots(8, 3, figsize=(PAGE_WIDTH, 7.35))
    fig.subplots_adjust(left=.012, right=.995, bottom=.02, top=.945, hspace=.10, wspace=.08)
    headings = [("B", "Local behavior", BLUE), ("S", "Sequence evolution", TEAL),
                ("T", "Observed timing", TIME)]
    titles = [
        ["Tool diversity", "Command entropy", "Dominant command", "Action switching",
         "Transition entropy", "Longest action run", "Action-pair reuse", "Adjacent repeats"],
        ["First-seen tools", "Revisit share", "Revisit lag", "A–B–A returns",
         "Lag-three returns", "Bigram reuse", "Trigram reuse", "Window distribution drift"],
        ["Typical log gap", "Gap interquartile range", "Burstiness", "Fast-gap share",
         "Long-pause share", "Serial gap dependence", "Execution duration", "Idle after completion"],
    ]
    formulas = [
        [r"$|U(C)|/n$", r"$-\sum_c p_c\log_2p_c$", r"$\max_c p_c$",
         r"$\sum_t\mathbf{1}[a_t\ne a_{t+1}]/m$", r"$H(A_{t+1}\mid A_t)$",
         r"$L_{\max}/n$", r"$1-|U(P_A)|/|P_A|$",
         r"$\sum_t\mathbf{1}[c_t=c_{t+1}]/(n-1)$"],
        [r"$\operatorname{mean}\mathbf{1}[c_t\notin\mathcal{H}_{t-1}]$",
         r"$\Pr(\Delta_t\geq2\mid\mathrm{repeated})$",
         r"$\operatorname{mean}\{\log(1+\Delta_t)/\log129\}$",
         r"$\Pr(a_t=a_{t+2}\mid a_t\ne a_{t+1})$",
         r"$\Pr(a_t=a_{t+3})$",
         r"$\operatorname{mean}\mathbf{1}[(c_{t-1},c_t)\in\mathcal{H}_2]$",
         r"$\operatorname{mean}\mathbf{1}[c_{t-2:t}\in\mathcal{H}_3]$",
         r"$\mathrm{JSD}(p_j,p_{j-1})$"],
        [r"$\operatorname{median}(\ell)$", r"$Q_{.75}(\ell)-Q_{.25}(\ell)$",
         r"$(\sigma_g-\mu_g)/(\sigma_g+\mu_g)$",
         r"$\operatorname{mean}\mathbf{1}[g\leq1\,\mathrm{s}]$",
         r"$\operatorname{mean}\mathbf{1}[g\geq10\,\mathrm{s}]$",
         r"$\operatorname{corr}(\ell_t,\ell_{t+1})$",
         r"$\operatorname{median}\log(1+d)$",
         r"$\operatorname{median}\log(1+i)$"],
    ]
    fills = [("#eff5fc", "#f5f8fc"), ("#eef8f4", "#f5faf8"), ("#f0f3f6", "#f7f8fa")]
    for col, (block, heading, color) in enumerate(headings):
        pos = axes[0, col].get_position()
        fig.text(pos.x0+.006, .973, f"{block}   {heading}", fontsize=9, weight="bold", color=color)
        for row in range(8):
            ax = axes[row, col]
            ax.set(xlim=(0, 10), ylim=(0, 1));ax.axis("off")
            group = 0 if row < 3 or row > 5 else 1
            ax.add_patch(Rectangle((0, 0), 10, 1, facecolor=fills[col][group], edgecolor="none", zorder=-5))
            ax.plot([0, 0], [0, 1], color=color, lw=1, alpha=.35)
            ax.text(.25, .87, f"{block}{row+1}  {titles[col][row]}", fontsize=7.1,
                    color=INK, va="center", weight="medium")
            ax.text(.25, .64, formulas[col][row], fontsize=6.15, color=color, va="center")

    def text(ax, x, y, value, color=INK, size=6.4, ha="center"):
        ax.text(x, y, value, fontsize=size, color=color, ha=ha, va="center")

    def tokens(ax, seq, color, focus=None, y=.28):
        xx = np.linspace(.8, 9.2, len(seq))
        for i, (xp, value) in enumerate(zip(xx, seq)):
            text(ax, xp, y, value, color if focus is None or i in focus else MUTED, size=7.1)
        return xx

    def arrow(ax, a, b, color, y=.28, style="->", connection=None):
        options = {"arrowstyle": style, "lw": .75, "color": color}
        if connection:options["connectionstyle"] = connection
        ax.annotate("", xy=(b, y), xytext=(a, y), arrowprops=options)

    def bracket(ax, a, b, color, y=.17, label=None):
        ax.plot([a, a, b, b], [y+.07, y, y, y+.07], color=color, lw=.75)
        if label:text(ax, (a+b)/2, y-.10, label, color, size=6)

    def hist(ax, values, color, xx=(1.4, 4.7, 8), base=.16):
        for x, value in zip(xx, values):
            ax.add_patch(Rectangle((x-.65, base), 1.3, .32*value/max(values), facecolor=color, alpha=.65, linewidth=0))
        for x, label in zip(xx, "abc"):text(ax, x, .075, label, MUTED, size=6)

    def timeline(ax, points, color=TIME):
        ax.hlines(.28, .6, 9.4, color=GRID, lw=.6)
        ax.vlines(points, .18, .38, color=color, lw=.85)

    # B: command distribution, action organization, and local repetition.
    ax=axes[0,0];xx=tokens(ax,list("abacabbc"),BLUE)
    for i in (0,1,3):ax.plot(xx[i],.14,"o",ms=2.5,color=BLUE)
    hist(axes[1,0],[4,2,2],BLUE)
    ax=axes[2,0]
    for i,v in enumerate("abacaaac"):
        ax.add_patch(Rectangle((.6+i*1.1,.20),.9,.29,facecolor=BLUE if v=="a" else GRID,alpha=.7,edgecolor="none"))
        text(ax,1.05+i*1.1,.345,v,"white" if v=="a" else MUTED,size=6.5)
    ax=axes[3,0];seq=list("ABAACC");xx=tokens(ax,seq,BLUE)
    for i in range(len(seq)-1):arrow(ax,xx[i]+.35,xx[i+1]-.35,BLUE if seq[i]!=seq[i+1] else GRID)
    ax=axes[4,0];text(ax,1,.28,"A",BLUE,size=7.2)
    for yy,lbl,p in [(.43,"A","¼"),(.28,"B","½"),(.13,"C","¼")]:
        ax.annotate("",xy=(6.8,yy),xytext=(1.7,.28),arrowprops={"arrowstyle":"->","lw":.65,"color":BLUE})
        text(ax,7.3,yy,lbl,BLUE,size=6.5);text(ax,8.6,yy,p,MUTED,size=6)
    ax=axes[5,0];xx=tokens(ax,list("ABAAAC"),BLUE);bracket(ax,xx[2]-.35,xx[4]+.35,BLUE)
    ax=axes[6,0];tokens(ax,["AB","…","AB"],BLUE);bracket(ax,.8,9.2,BLUE)
    ax=axes[7,0];seq=list("abbaac");xx=tokens(ax,seq,BLUE)
    for i in range(len(seq)-1):
        ax.plot([xx[i]+.35,xx[i+1]-.35],[.14,.14],lw=1.3 if seq[i]==seq[i+1] else .5,
                color=BLUE if seq[i]==seq[i+1] else GRID)

    # S: patterns span previous events or windows; glyphs do not impose stages.
    ax=axes[0,1];tokens(ax,["a","b","a","c"],TEAL);ax.plot(9.2,.13,"*",ms=4.5,color=TEAL);text(ax,7.6,.13,"new",TEAL,size=6)
    ax=axes[1,1];tokens(ax,["a","…","b","a"],TEAL,focus={3},y=.25);arrow(ax,.9,9.0,TEAL,y=.25,connection="arc3,rad=-.12")
    ax=axes[2,1];tokens(ax,["a","…","…","a"],TEAL);bracket(ax,.8,9.2,TEAL,y=.13,label="revisit lag Δₜ")
    ax=axes[3,1];tokens(ax,["A","B","A"],TEAL,y=.25);arrow(ax,.9,9.0,TEAL,y=.25,connection="arc3,rad=-.12")
    ax=axes[4,1];tokens(ax,["A","B","C","A"],TEAL);bracket(ax,.8,9.2,TEAL,y=.13,label="lag 3")
    ax=axes[5,1];tokens(ax,["ab","…","ab"],TEAL,focus={2});bracket(ax,.8,9.2,TEAL)
    ax=axes[6,1];tokens(ax,["abc","…","abc"],TEAL,focus={2});bracket(ax,.8,9.2,TEAL)
    ax=axes[7,1]
    for offset, vals in [(1,[4,1,2]),(6.3,[1,4,2])]:
        for j,v in enumerate(vals):ax.add_patch(Rectangle((offset+j*.8,.18),.55,.075*v,facecolor=TEAL,alpha=.65,linewidth=0))
    arrow(ax,4,5.6,TEAL,y=.33);text(ax,2.1,.08,r"$p_{j-1}$",MUTED,size=6);text(ax,7.4,.08,r"$p_j$",MUTED,size=6)

    # T: real timestamp or duration fields are required to instantiate a glyph.
    ax=axes[0,2];timeline(ax,[.8,1.7,3.3,5.2,6.4,8.7]);bracket(ax,3.3,5.2,TIME,y=.47,label="gap g")
    ax=axes[1,2];ax.hlines(.37,1.1,8.9,color=TIME,lw=.7)
    ax.add_patch(Rectangle((3,.25),3.5,.24,facecolor=TIME,alpha=.25,edgecolor=TIME,lw=.7))
    ax.vlines([1.1,4.4,8.9],.27,.47,color=TIME,lw=.75);bracket(ax,3,6.5,TIME,y=.15,label="IQR")
    ax=axes[2,2];timeline(ax,[.9,1.15,1.4,1.7,5,8.4,8.65,8.9])
    ax=axes[3,2];timeline(ax,[.8,1.3,4.5,4.9,8.5]);bracket(ax,.8,1.3,TIME,y=.45)
    ax=axes[4,2];timeline(ax,[.8,1.3,7.8,8.4]);bracket(ax,1.3,7.8,TIME,y=.47,label="g ≥ 10 s")
    ax=axes[5,2]
    ax.plot([1,1,8.5],[.48,.12,.12],color=GRID,lw=.6)
    ax.scatter([2,2.5,3.8,4.4,5.8,7.4],[.17,.21,.22,.30,.35,.43],s=5,color=TIME)
    text(ax,9,.11,r"$\ell_t$",TIME,size=6);text(ax,.55,.45,r"$\ell_{t+1}$",TIME,size=6)
    ax=axes[6,2]
    for y,a,b in [(.38,1.0,5.0),(.16,2.6,8.5)]:
        ax.hlines(y,a,b,color=TIME,lw=4);ax.vlines([a,b],y-.07,y+.07,color=TIME,lw=.7)
    text(ax,6.5,.46,"d",TIME,size=6.5)
    ax=axes[7,2];ax.hlines(.28,1,3.1,color=TIME,lw=4);ax.hlines(.28,7.1,9.1,color=TIME,lw=4)
    ax.hlines(.28,3.1,7.1,color=GRID,lw=.6);bracket(ax,3.1,7.1,TIME,y=.50,label="idle")

    save(fig,"fingerprint-bst-dimensions",(ROOT/"research/paper/figures",ROOT/"site/assets/behavioral-fingerprint-v2"))


if __name__ == "__main__":main()
