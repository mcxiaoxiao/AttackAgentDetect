"""Command-to-category mapping for behavioral fingerprint extension.

Maps executable names to coarse semantic categories for command-aware
behavioral features (B_cat, S_cat, T_cat).
"""

# Command category definitions
# Each category groups commands with similar operational semantics
COMMAND_CATEGORIES = {
    # System/privilege operations
    "system": {
        "sudo", "su", "chmod", "chown", "chgrp",
        "apt-get", "apt", "yum", "dnf", "pacman",
        "mount", "umount", "fdisk", "mkfs",
        "systemctl", "service", "init",
        "useradd", "userdel", "passwd", "groupadd",
    },

    # Navigation/directory operations
    "navigation": {
        "cd", "ls", "pwd", "find", "locate", "tree",
        "dir", "dirs", "pushd", "popd",
    },

    # Inspection/read operations
    "inspection": {
        "cat", "grep", "head", "tail", "less", "more",
        "wc", "diff", "cmp", "file", "stat", "lstat",
        "strings", "hexdump", "xxd", "od",
        "man", "info", "help", "whatis", "apropos",
    },

    # Editing operations
    "editing": {
        "vim", "vi", "nano", "emacs", "ed",
        "sed", "awk", "perl", "python", "python3",
        "ruby", "node", "bash", "sh", "zsh",
    },

    # Network operations
    "network": {
        "curl", "wget", "ssh", "scp", "rsync",
        "ping", "nmap", "netstat", "ss", "ip",
        "ifconfig", "route", "iptables",
        "ftp", "sftp", "telnet", "nc", "ncat",
    },

    # Process management
    "process": {
        "ps", "top", "htop", "kill", "killall",
        "bg", "fg", "jobs", "nohup", "nice", "renice",
        "wait", "sleep", "timeout",
    },

    # File operations
    "file_ops": {
        "cp", "mv", "rm", "mkdir", "rmdir", "touch",
        "ln", "tar", "zip", "unzip", "gzip", "gunzip",
        "bzip2", "xz", "7z",
    },

    # Agent-specific tools (AIT/AECID dataset)
    "agent_tools": {
        "inspect_shell", "run_command", "submit",
        "execute_command", "read_file", "write_file",
    },
}

# Build reverse mapping: command -> category
COMMAND_TO_CATEGORY = {}
for category, commands in COMMAND_CATEGORIES.items():
    for cmd in commands:
        COMMAND_TO_CATEGORY[cmd] = category

# Category list (ordered for consistent indexing)
CATEGORY_LIST = [
    "system",
    "navigation",
    "inspection",
    "editing",
    "network",
    "process",
    "file_ops",
    "agent_tools",
]

# Number of categories (excluding "unknown")
NUM_CATEGORIES = len(CATEGORY_LIST)


def categorize_command(command: str) -> str:
    """Map a command name to its category.

    Args:
        command: Executable basename (lowercase).

    Returns:
        Category string from CATEGORY_LIST, or "unknown" if unmapped.
    """
    return COMMAND_TO_CATEGORY.get(command, "unknown")


def categorize_sequence(commands: list[str]) -> list[str]:
    """Map a command sequence to category sequence.

    Args:
        commands: List of executable basenames.

    Returns:
        List of category strings (same length as input).
    """
    return [categorize_command(cmd) for cmd in commands]
