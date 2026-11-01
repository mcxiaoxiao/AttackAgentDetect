# Research

The current study evaluates behavior-based association of Human and Agent traces from public CTF tasks. Each pair is scored with command TF--IDF similarity and IDF-weighted containment of distinctive artifact names.

- [Method and evaluation report](reports/pairwise-trace-association.md)
- [Current paper](paper/main.tex)
- [CTF trace set and pair browser](../site/dataset.html)
- [Research site](https://mcxiaoxiao.github.io/AttackAgentDetect/)

The evaluation covers 97 traces and 40 task groups. A same-task label means the records belong to the same CTF challenge; it does not identify a shared operator, campaign, or real-world attack. Human terminal records and Agent tool-call records use different collection interfaces, so execution mode cannot be separated from capture format in this dataset.
