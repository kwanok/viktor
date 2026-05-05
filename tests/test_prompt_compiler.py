from __future__ import annotations

import unittest

from tests.workspace import workspace_ctx
from viktor_dgmh.archive import init_workspace
from viktor_dgmh.memory import append_preference
from viktor_dgmh.models import PreferenceSignal
from viktor_dgmh.prompt_compiler import compile_system_prompt


class PromptCompilerTests(unittest.TestCase):
    def test_compiler_combines_self_model_task_prompt_and_ephemeral_preferences(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            append_preference(
                root,
                PreferenceSignal(
                    signal_id="s1",
                    session_id="chat1",
                    source_event_id="p1",
                    kind="reflection_identity",
                    polarity="negative",
                    strength=0.95,
                    context="identity",
                    target="self_model",
                    text="Identity questions should be answered as Viktor only.",
                    preferred_text="나는 빅토르야.",
                ),
            )

            prompt = compile_system_prompt(root)

            self.assertIn("Self model:", prompt)
            self.assertIn("Public name: Viktor", prompt)
            self.assertIn("Default tone with this user: banmal", prompt)
            self.assertIn("Forbidden outward self-descriptions", prompt)
            self.assertIn("Behavior prompt:", prompt)
            self.assertIn("Recent learned user preferences", prompt)
            self.assertIn("나는 빅토르야.", prompt)

    def test_compiler_can_omit_ephemeral_preferences_for_archive_evaluation(self) -> None:
        with workspace_ctx() as root:
            init_workspace(root)
            append_preference(
                root,
                PreferenceSignal(
                    signal_id="s1",
                    session_id="chat1",
                    source_event_id="p1",
                    kind="reflection_identity",
                    polarity="negative",
                    strength=0.95,
                    context="identity",
                    target="self_model",
                    text="Temporary preference.",
                ),
            )

            prompt = compile_system_prompt(root, include_ephemeral=False)

            self.assertIn("Self model:", prompt)
            self.assertNotIn("Temporary preference.", prompt)


if __name__ == "__main__":
    unittest.main()
