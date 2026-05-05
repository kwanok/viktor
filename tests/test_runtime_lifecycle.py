from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone

from tests.workspace import workspace_ctx
from viktor_dgmh.archive import init_workspace
from viktor_dgmh.runtime_lifecycle import SlackSupervisor, request_restart, runtime_status


class RuntimeLifecycleTests(unittest.TestCase):
    def test_restart_request_file_and_status(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)

            request = request_restart(root, "reload after validated code change", source="test", requested_by="U1")
            status = runtime_status(root)

            self.assertEqual(request.status, "pending")
            self.assertEqual(status["restart_request"]["request_id"], request.request_id)
            self.assertEqual(status["restart_request"]["reason"], "reload after validated code change")

    def test_supervisor_restarts_fake_child_on_request(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            supervisor = SlackSupervisor(
                root=root,
                command=[sys.executable, "-c", "import time; time.sleep(60)"],
                poll_interval_seconds=0.01,
                terminate_timeout_seconds=1.0,
            )
            try:
                supervisor.start_child()
                first_pid = supervisor.process.pid
                request_restart(root, "test restart", source="test")

                supervisor.tick()

                self.assertIsNotNone(supervisor.process)
                self.assertNotEqual(supervisor.process.pid, first_pid)
                status = runtime_status(root)
                self.assertEqual(status["restart_request"]["status"], "handled")
                self.assertEqual(status["daemon_state"]["status"], "running")
            finally:
                supervisor.stop_child()

    def test_supervisor_blocks_restart_loop(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            supervisor = SlackSupervisor(
                root=root,
                command=[sys.executable, "-c", "import time; time.sleep(60)"],
                poll_interval_seconds=0.01,
                terminate_timeout_seconds=1.0,
            )
            try:
                supervisor.start_child()
                supervisor.restart_history = [datetime.now(timezone.utc) for _ in range(3)]
                request_restart(root, "too many restarts", source="test")

                supervisor.tick()

                status = runtime_status(root)
                self.assertEqual(status["restart_request"]["status"], "blocked")
                self.assertEqual(status["daemon_state"]["status"], "blocked")
            finally:
                supervisor.stop_child()


if __name__ == "__main__":
    unittest.main()
