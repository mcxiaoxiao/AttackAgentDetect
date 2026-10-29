# Linux auditd example

`minimal.audit` is a hand-written format fixture, not research data. It contains two `execve` events recorded as `SYSCALL` and `EXECVE` pairs. The pairs are intentionally out of order so the example checks event-ID joining as well as command extraction. Nothing in the log is executed.

Run it from the repository root:

```bash
python -m attackagent.jsonl_cli normalize \
  --input examples/auditd/minimal.audit \
  --output /tmp/normalized.jsonl
```

The output contains one session trace with commands `curl` and `grep`, coarse actions `network` and `search`, source event references, and parser-quality counters. The parser joins records by audit event ID, decodes printable hexadecimal arguments, removes exact duplicate records, groups valid audit sessions, and leaves events with unset sessions separate.

The output is evidence normalization, not attribution. It does not infer whether a human, script, or model selected the command. The current action mapping is intentionally coarse; Sysmon, network flow, and application logs need separate adapters. See `attackagent/auditd.py` for the supported fields and tests.
