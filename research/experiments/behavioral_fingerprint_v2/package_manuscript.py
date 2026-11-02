#!/usr/bin/env python3
"""Package the existing manuscript and exactly the vector figures it references."""
import pathlib,re,shutil,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[3];PAPER=ROOT/'research/paper';SITE=ROOT/'site/assets/behavioral-fingerprint-v2'
tex=PAPER/'human-agent-fingerprint-overnight.tex';source=tex.read_text();figures=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',source)
(SITE/'figures').mkdir(exist_ok=True,parents=True)
shutil.copy2(tex,SITE/tex.name)
readme='Current research source. Open human-agent-fingerprint-overnight.tex with the figures folder beside it. The document references '+str(len(figures))+' PDF vector figures. Methods, predictions, metrics and protocols are versioned under research/experiments/behavioral_fingerprint_v2 and research/results/behavioral_fingerprint_v2 in the repository.\n'
(SITE/'SOURCE-BUNDLE-README.txt').write_text(readme)
with zipfile.ZipFile(SITE/'human-agent-fingerprint-overnight-source.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
 z.write(tex,tex.name);z.writestr('README.txt',readme)
 for f in figures:
  p=PAPER/'figures'/f;assert p.exists(),p;shutil.copy2(p,SITE/'figures'/f);z.write(p,'figures/'+f)
print('Packaged',len(figures),'figures with current TeX source')
