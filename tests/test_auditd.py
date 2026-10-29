import unittest

from attackagent.auditd import load_auditd


def record(record_type, serial, timestamp, **fields):
    values = " ".join(f'{key}="{value}"' if " " in str(value) else f"{key}={value}" for key, value in fields.items())
    return f"type={record_type} msg=audit({timestamp}:{serial}) {values}"


class AuditdParserTest(unittest.TestCase):
    def test_joins_out_of_order_records_decodes_hex_deduplicates_and_splits_idle_sessions(self):
        exec_one = record("EXECVE", 10, "1000.000", argc=2, a0="2F62696E2F6375726C", a1="68747470733A2F2F78")
        syscall_one = record("SYSCALL", 10, "1000.000", arch="c000003e", syscall=59, success="yes", pid=10, ppid=1, auid=1000, ses=7, comm="curl", exe="/bin/curl")
        syscall_two = record("SYSCALL", 11, "1010.000", arch="c000003e", syscall=2, success="yes", pid=11, ppid=10, auid=1000, ses=7, comm="cat", exe="/bin/cat")
        syscall_three = record("SYSCALL", 12, "1400.000", arch="c000003e", syscall=59, success="yes", pid=12, ppid=1, auid=1000, ses=7, comm="python3", exe="/usr/bin/python3")
        traces, stats = load_auditd(
            "\n".join([exec_one, syscall_two, syscall_one, syscall_one, syscall_three]),
            host="sensor-a",
            return_stats=True,
        )
        self.assertEqual(len(traces), 2)
        traces.sort(key=lambda row: len(row.commands), reverse=True)
        self.assertEqual(traces[0].commands, ("curl", "cat"))
        self.assertEqual(traces[0].source_format, "linux-auditd-v1")
        self.assertEqual(traces[0].event_refs, ("sensor-a/audit(1000.000:10)", "sensor-a/audit(1010.000:11)"))
        self.assertEqual(traces[0].quality["duplicate_lines"], 1)
        self.assertEqual(stats["duplicate_lines"], 1)
        self.assertEqual(stats["joined_events"], 3)
        self.assertIn("network", traces[0].actions)

    def test_unset_sessions_remain_event_level_and_incomplete_records_are_counted(self):
        one = record("SYSCALL", 21, "2000.0", arch="c000003e", syscall=0, ses=4294967295, auid=4294967295, pid=1, comm="cat", exe="/bin/cat")
        two = record("SYSCALL", 22, "2001.0", arch="c000003e", syscall=0, ses=4294967295, auid=4294967295, pid=1, comm="cat", exe="/bin/cat")
        no_syscall = record("EXECVE", 23, "2002.0", argc=1, a0="2F62696E2F7368")
        traces = load_auditd("\n".join([one, two, no_syscall]))
        self.assertEqual(len(traces), 3)
        self.assertTrue(all(len(row.event_refs) == 1 for row in traces))
        shell = next(row for row in traces if row.commands == ("sh",))
        self.assertEqual(shell.quality["missing_syscall"], 1)

    def test_malformed_lines_are_counted_without_crashing(self):
        traces, stats = load_auditd("not an audit event\ntype=SYSCALL pid=3", return_stats=True)
        self.assertEqual(traces, [])
        self.assertEqual(stats["malformed_lines"], 2)


if __name__ == "__main__":
    unittest.main()
