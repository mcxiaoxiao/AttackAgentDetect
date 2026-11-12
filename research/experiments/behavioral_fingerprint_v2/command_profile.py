"""Corpus-independent command-composition profile dimensions.

Unlike a TF-IDF vocabulary fitted on training traces, this profile uses a FIXED,
a-priori taxonomy of standard Unix/Linux command names plus a fixed naming
canonicalization (version/variant normalization). Every dimension is therefore
computable from a single log without reference to any corpus, so the profile is a
genuine fingerprint dimension that can be folded into the B/S/T schema as default
command-composition slots. The taxonomy is domain knowledge (standard utilities),
not a label-driven selection.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

# A-priori fixed taxonomy of standard command names (coreutils, shell, session-init
# tools, interpreters, build, network, system/package/service, editors, and common
# daemons/utilities). Defined independently of any dataset labels.
_TAXONOMY = """
7z ag alias apache2 apropos apt apt-get arp at awk base64 basename bash bc bg
bind blkid builtin cat chattr chgrp chmod chown chroot cksum clang clear cmake
cmp comm command compgen complete cp csplit curl cut dash date dd declare df
dig dirname dirs dmesg docker dpkg dpkg-query du echo ed egrep emacs enable
env envsubst eval exec exit expand expr false fc fg fgrep file find fmt fold
free ftp fuser g++ gcc gdb getent getopt getopts gettext git gawk grep groupadd
groups gunzip gzip hash head help hexdump hg history host hostname htop httpd
id ifconfig info iostat ip iptables java javac jobs journalctl join kill killall
kubectl last less let ln loadkeys local locale locale-check localectl login
logout ls lsattr lsblk lsmod lsof lua make man mariadb mariadb-admin md5sum
mesg mkdir mkfs mktemp modprobe more mount mv mysql mysqladmin nano ncat nc
netcat netstat nginx nice nl nologin nohup npm nslookup od pacman passwd paste
pathchk perl pgrep php php8.3 ping pkill pkttyagent polkit-agent-helper-1 pr
printenv printf ps pushd pwd python
python3 python3.12 Rscript read readlink readonly realpath redis-cli renice
reset return rev rg rm rmdir route rpm rsync run-parts runuser screen scp sed
seq service set sftp sha1sum sha256sum shift shopt shred shuf sleep snap sort
source split ss ssh stat strace strings stty su sudo svn sysctl systemctl
systemd systemd-tty-ask-password-agent tac tail tar tcpdump tee test timeout
tmux top touch tput tr traceroute trap tree true truncate tty type typeset
ulimit umask umount unalias unexpand uniq unset unzip uptime useradd userdel
usermod utempter vi vim vim.basic vmstat w wait wc wget whatis whereis which
who whoami x86_64-linux-gnu-strings xxd xz yes yum zip zsh
"""
COMMAND_PROFILE_TAXONOMY = tuple(sorted(set(_TAXONOMY.split())))
PROFILE_INDEX = {c: i for i, c in enumerate(COMMAND_PROFILE_TAXONOMY)}
NUM_PROFILE = len(COMMAND_PROFILE_TAXONOMY)

# Fixed naming canonicalization (standard Unix naming knowledge, not data-fitted).
_VARIANT_MAP = {
    "vim.basic": "vim", "vi.basic": "vi", "gawk": "awk", "mawk": "awk",
    "mariadb": "mysql", "mariadb-admin": "mysqladmin", "httpd": "apache2",
    "locale-check": "locale", "dpkg-query": "dpkg",
    "systemd-tty-ask-password-agent": "systemd",
    "x86_64-linux-gnu-strings": "strings",
}


def canonicalize_command(command: str) -> str:
    """Normalize version/variant executable names to a canonical base name."""
    c = str(command).lower()
    c = _VARIANT_MAP.get(c, c)
    c = re.sub(r"^x86_64-linux-gnu-", "", c)
    c = re.sub(r"\d+(\.\d+)*$", "", c)  # php8.3 -> php, python3.12 -> python
    c = _VARIANT_MAP.get(c, c)
    return c


def command_profile_vector(commands: Sequence[str], sublinear: bool = True) -> np.ndarray:
    """Fixed-length command-composition profile for one trace (NUM_PROFILE dims).

    Each slot is the (sublinear) term frequency of one canonical command from the
    fixed taxonomy. Commands outside the taxonomy contribute zero. No corpus or
    training vocabulary is consulted, so the vector profiles a single log.
    """
    cmds = [canonicalize_command(c) for c in commands]
    out = np.zeros(NUM_PROFILE, dtype=float)
    if not cmds:
        return out
    for token, count in Counter(cmds).items():
        if token in PROFILE_INDEX:
            out[PROFILE_INDEX[token]] = (1 + math.log(count)) if sublinear else count / len(cmds)
    return out


def command_presence_vector(commands: Sequence[str]) -> np.ndarray:
    """Binary tool-repertoire profile for one trace (NUM_PROFILE dims).

    Each slot is 1 iff the canonical command appears at least once, else 0.
    Presence/absence of a tool is more stable across unseen actors than its
    frequency (which co-varies with workflow and trace length), so this binary
    repertoire transfers better under source-group holdout while remaining a
    fixed, single-log, corpus-independent fingerprint dimension.
    """
    out = np.zeros(NUM_PROFILE, dtype=float)
    for c in commands:
        token = canonicalize_command(c)
        if token in PROFILE_INDEX:
            out[PROFILE_INDEX[token]] = 1.0
    return out