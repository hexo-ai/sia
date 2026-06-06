"""Characterization: lock the exact text of the meta and feedback prompts.

These prompts are the product's core business logic and are written verbatim to
disk (meta_agent_prompt.txt / feedback_agent_prompt.txt). The refactor must not
change a single character of them on the default path.
"""

from golden_master import assert_golden

from sia.orchestrator import TaskFiles, build_feedback_prompt, build_meta_prompt
from sia.providers import load_provider

# Fixed, path-free inputs so the prompt text is fully deterministic.
TASK_FILES = TaskFiles(
    sample_task_descriptions="SAMPLE DESCRIPTIONS BODY",
    reference_target_agent_py="print('reference target agent')",
    sample_agent_execution={"messages": [{"role": "user", "content": "hi"}]},
    task_md="# Example Task\nSolve the example problem precisely.",
)


def test_meta_prompt_golden():
    prompt = build_meta_prompt(
        task_files=TASK_FILES,
        task_model="claude-haiku-4-5-20251001",
        working_dir="/WORK/run_1/gen_1",
    )
    assert_golden("meta_prompt.txt", prompt)


def test_meta_prompt_anthropic_provider_is_byte_identical():
    """An explicit anthropic provider must not change the default prompt text."""
    with_provider = build_meta_prompt(
        task_files=TASK_FILES,
        task_model="claude-haiku-4-5-20251001",
        working_dir="/WORK/run_1/gen_1",
        provider=load_provider("anthropic"),
    )
    assert_golden("meta_prompt.txt", with_provider)


def test_meta_prompt_openai_provider_golden():
    """OpenAI-compatible providers prepend the client-setup block (new golden)."""
    prompt = build_meta_prompt(
        task_files=TASK_FILES,
        task_model="moonshotai/Kimi-K2.6",
        working_dir="/WORK/run_1/gen_1",
        provider=load_provider("nebius"),
    )
    assert_golden("meta_prompt_openai.txt", prompt)


def test_feedback_prompt_golden():
    prompt = build_feedback_prompt(
        current_gen=2,
        max_gen=3,
        task_files=TASK_FILES,
        agent_py="print('current target agent gen 2')",
        task="# Example Task\nSolve the example problem precisely.",
        execution_status="SUCCESS: example status block",
        execution_section="EXECUTION SECTION BODY",
        run_dir="/RUN/run_1",
        next_gen_dir="/RUN/run_1/gen_3",
        previous_gens="1",
        task_model="claude-haiku-4-5-20251001",
    )
    assert_golden("feedback_prompt.txt", prompt)


def test_feedback_prompt_includes_knowledge_digest_when_provided():
    prompt = build_feedback_prompt(
        current_gen=2,
        max_gen=3,
        task_files=TASK_FILES,
        agent_py="print('current target agent gen 2')",
        task="# Example Task\nSolve the example problem precisely.",
        execution_status="SUCCESS: example status block",
        execution_section="EXECUTION SECTION BODY",
        run_dir="/RUN/run_1",
        next_gen_dir="/RUN/run_1/gen_3",
        previous_gens="1",
        task_model="claude-haiku-4-5-20251001",
        knowledge_digest="experiment:gen_2 tested hypothesis:validation",
    )

    assert "EXPERIMENT KNOWLEDGE GRAPH DIGEST" in prompt
    assert "experiment:gen_2 tested hypothesis:validation" in prompt
    assert "Include sections named Hypothesis, Evidence, Planned Change, Expected Impact, and Risk" in prompt
