#!/usr/bin/env python3
"""Package the existing manuscript and exactly the figures it references."""
import pathlib,re,shutil,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[3];PAPER=ROOT/'research/paper';SITE=ROOT/'site/assets/behavioral-fingerprint-v2'
tex=PAPER/'human-agent-fingerprint-overnight.tex';source=tex.read_text();figures=list(dict.fromkeys(re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',source)))
figure_count=len(re.findall(r'\\begin\{figure\}',source))
(SITE/'figures').mkdir(exist_ok=True,parents=True)
shutil.copy2(tex,SITE/tex.name)
readme='Current research source. Open human-agent-fingerprint-overnight.tex with the figures folder beside it. The document contains '+str(figure_count)+' research figures and includes '+str(len(figures))+' visual assets (including the project mark). Methods, predictions, metrics and protocols are versioned under research/experiments/behavioral_fingerprint_v2, research/experiments/association_fingerprint_v3 and their matching results directories.\nOwl mark: Selina Bauder / OpenMoji, CC BY-SA 4.0, unmodified. Source: https://openmoji.org/library/emoji-1F989/\n'
(SITE/'SOURCE-BUNDLE-README.txt').write_text(readme)
with zipfile.ZipFile(SITE/'human-agent-fingerprint-overnight-source.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
 z.write(tex,tex.name);z.writestr('README.txt',readme)
 for f in figures:
  p=PAPER/'figures'/f;assert p.exists(),p;shutil.copy2(p,SITE/'figures'/f);z.write(p,'figures/'+f)
print('Packaged',figure_count,'research figures and',len(figures),'visual assets with current TeX source')
