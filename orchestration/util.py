import logging
import os
from datetime import datetime
from typing import Literal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Backend type definition
AgentBackend = Literal["claude", "openhands"]


async def run_agent_claude(model_name, max_turns, prompt, agent_working_directory):
    """Run agent using Claude Code SDK

    Note: Claude Code automatically saves trajectories to ~/.claude/projects/
    """
    from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

    logger.info("=" * 80)
    logger.info(f"Starting agent execution with {model_name} model")
    logger.info(f"Working directory: {agent_working_directory}")
    logger.info(f"Max turns: {max_turns}")
    logger.info("=" * 80)

    turn_count = 0
    start_time = datetime.now()

    try:
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                cwd=agent_working_directory,
                allowed_tools=["Bash", "Read", "Write", "Edit", "Glob"],
                permission_mode="bypassPermissions",
                max_turns=max_turns,
                model=model_name,
            ),
        ):
            logged_content = False

            if hasattr(message, 'content') and message.content:
                for block in message.content:
                    # Log agent text responses
                    if hasattr(block, "text") and block.text:
                        if not logged_content:
                            turn_count += 1
                            logger.info(f"\n{'─' * 80}")
                            logger.info(f"TURN {turn_count}: Agent Response")
                            logger.info(f"{'─' * 80}")
                            logged_content = True
                        logger.info(f"{block.text}")

                    # Log tool calls
                    elif hasattr(block, "name"):
                        if not logged_content:
                            turn_count += 1
                            logger.info(f"\n{'─' * 80}")
                            logger.info(f"TURN {turn_count}: Tool Execution")
                            logger.info(f"{'─' * 80}")
                            logged_content = True

                        logger.info(f"🔧 Tool: {block.name}")
                        if hasattr(block, "input") and block.input:
                            # Pretty print tool input
                            import json
                            try:
                                input_str = json.dumps(block.input, indent=2)
                                logger.info(f"   Input: {input_str}")
                            except:
                                logger.info(f"   Input: {block.input}")

                    # Log tool results
                    elif hasattr(block, "type") and block.type == "tool_result":
                        if hasattr(block, "content"):
                            result = block.content if isinstance(block.content, str) else str(block.content)
                            # Truncate very long outputs
                            if len(result) > 500:
                                result = result[:500] + f"\n... (truncated, {len(result)} total chars)"
                            logger.info(f"   ✓ Result: {result}")

            # Log final result
            if isinstance(message, ResultMessage):
                elapsed_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"\n{'=' * 80}")
                logger.info(f"EXECUTION COMPLETE")
                logger.info(f"{'=' * 80}")
                logger.info(f"Total turns: {turn_count}")
                logger.info(f"Execution time: {elapsed_time:.2f} seconds")
                logger.info(f"Final result: {message.result}")
                logger.info(f"{'=' * 80}")

    except Exception as e:
        logger.error(f"\n{'!' * 80}")
        logger.error(f"ERROR: {str(e)}")
        logger.error(f"{'!' * 80}", exc_info=True)
        raise


async def run_agent_openhands(model_name, max_turns, prompt, agent_working_directory):
    """Run agent using OpenHands SDK"""
    try:
        from openhands.sdk import LLM, Agent, Conversation, Tool
        from openhands.tools.terminal import TerminalTool
        from openhands.tools.file_editor import FileEditorTool
    except ImportError:
        logger.error("OpenHands SDK not installed. Install with: pip install openhands-ai")
        raise

    logger.info("=" * 80)
    logger.info(f"Starting OpenHands agent execution with {model_name} model")
    logger.info(f"Working directory: {agent_working_directory}")
    logger.info(f"Max turns: {max_turns}")
    logger.info("=" * 80)

    turn_count = 0
    start_time = datetime.now()

    try:
        # Determine API key based on model provider
        api_key = None
        if "claude" in model_name.lower() or "anthropic" in model_name.lower():
            api_key = os.getenv("ANTHROPIC_API_KEY")
        elif "gemini" in model_name.lower() or "google" in model_name.lower():
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        elif "gpt" in model_name.lower() or "openai" in model_name.lower():
            api_key = os.getenv("OPENAI_API_KEY")
        else:
            # Fallback to generic LLM_API_KEY
            api_key = os.getenv("LLM_API_KEY")

        if not api_key:
            logger.warning(f"No API key found for model {model_name}. Using LLM_API_KEY if available.")
            api_key = os.getenv("LLM_API_KEY")

        # Create LLM instance
        llm = LLM(
            model=model_name,
            api_key=api_key,
        )

        # Create agent with available tools
        agent = Agent(
            llm=llm,
            tools=[
                Tool(name=TerminalTool.name),
                Tool(name=FileEditorTool.name),
            ],
        )

        # Create conversation with workspace and persistence
        # Trajectory will be saved in: agent_working_directory/openhands_trajectory/
        trajectory_dir = os.path.join(agent_working_directory, "openhands_trajectory")

        conversation = Conversation(
            agent=agent,
            workspace=agent_working_directory,
            persistence_dir=trajectory_dir
        )

        # Send the task prompt
        logger.info(f"\n{'─' * 80}")
        logger.info(f"TURN {turn_count + 1}: Sending prompt to agent")
        logger.info(f"{'─' * 80}")
        conversation.send_message(prompt)

        # Run the agent
        logger.info(f"Running agent (max turns: {max_turns})...")
        logger.info(f"  → Trajectory will be saved to: {trajectory_dir}")
        result = conversation.run()

        # Log completion
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n{'=' * 80}")
        logger.info(f"EXECUTION COMPLETE")
        logger.info(f"{'=' * 80}")
        logger.info(f"Execution time: {elapsed_time:.2f} seconds")
        logger.info(f"Final result: {result}")
        logger.info(f"  ✓ Trajectory saved to: {trajectory_dir}")
        logger.info(f"{'=' * 80}")

    except Exception as e:
        logger.error(f"\n{'!' * 80}")
        logger.error(f"ERROR: {str(e)}")
        logger.error(f"{'!' * 80}", exc_info=True)
        raise


async def run_agent(
    model_name: str,
    max_turns: str,
    prompt: str,
    agent_working_directory: str,
    backend: AgentBackend = "claude"
):
    """
    Run an agent with the specified backend.

    Args:
        model_name: The model to use (format depends on backend)
        max_turns: Maximum number of turns for the agent
        prompt: The task prompt to send to the agent
        agent_working_directory: Working directory for the agent
        backend: Which agent backend to use ("claude" or "openhands")

    Examples:
        # Claude backend with Claude models
        await run_agent("haiku", 20, prompt, "/path/to/dir", backend="claude")

        # OpenHands backend with Gemini
        await run_agent("gemini/gemini-3.1-pro-preview", 20, prompt, "/path/to/dir", backend="openhands")

        # OpenHands backend with GPT
        await run_agent("openai/gpt-4", 20, prompt, "/path/to/dir", backend="openhands")
    """
    logger.info(f"Using {backend} backend")

    if backend == "claude":
        await run_agent_claude(model_name, max_turns, prompt, agent_working_directory)
    elif backend == "openhands":
        await run_agent_openhands(model_name, max_turns, prompt, agent_working_directory)
    else:
        raise ValueError(f"Unknown backend: {backend}. Must be 'claude' or 'openhands'")
