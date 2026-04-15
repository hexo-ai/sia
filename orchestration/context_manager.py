"""
Context Manager for SIA (Self-Improving Agent) System

Manages the context.md file that tracks the evolution of agent generations,
including code changes, performance metrics, and insights across iterations.
"""

import os
import json
import re
import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


class ContextManager:
    """Manages context.md for tracking generation evolution in a run"""

    def __init__(self, run_directory: str, run_config: Dict[str, Any]):
        """
        Initialize the context manager.

        Args:
            run_directory: Path to the run directory (e.g., runs/run_1)
            run_config: Configuration dict with keys:
                - task_dir: Task directory path
                - meta_model: Meta-agent model name
                - task_model: Task-agent model name
                - backend: Backend type ('claude' or 'openhands')
                - max_gen: Maximum number of generations
        """
        self.run_dir = run_directory
        self.context_path = os.path.join(run_directory, "context.md")
        self.config = run_config
        self.generations = []
        self.meta_model = run_config.get('meta_model', 'haiku')
        self.backend = run_config.get('backend', 'claude')

    def initialize(self):
        """Create context.md with header information"""
        header = f"""# Run Context: {os.path.basename(self.run_dir)}

**Task**: {self.config.get('task_dir', 'N/A')}
**Meta Model**: {self.config.get('meta_model', 'N/A')}
**Task Model**: {self.config.get('task_model', 'N/A')}
**Backend**: {self.config.get('backend', 'N/A')}
**Started**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Max Generations**: {self.config.get('max_gen', 'N/A')}

---

"""
        with open(self.context_path, 'w', encoding='utf-8') as f:
            f.write(header)
        print(f"Initialized context.md at {self.context_path}")

    def _generate_llm_summary(self, gen_num: int, gen_data: Dict[str, Any], metrics: Dict[str, Any]) -> Optional[str]:
        """
        Call LLM to generate a summary of changes and improvements.

        Args:
            gen_num: Generation number
            gen_data: Generation data dictionary
            metrics: Extracted metrics for this generation

        Returns:
            Summary string or None if generation 1 or error
        """
        # Skip for generation 1 (no previous generation to compare)
        if gen_num == 1:
            return None

        try:
            # Read current generation's target_agent.py
            current_agent_path = gen_data['agent_path']
            current_agent_code = Path(current_agent_path).read_text(encoding='utf-8')

            # Read previous generation's target_agent.py
            prev_gen_dir = os.path.join(self.run_dir, f"gen_{gen_num - 1}")
            prev_agent_path = os.path.join(prev_gen_dir, "target_agent.py")
            prev_agent_code = Path(prev_agent_path).read_text(encoding='utf-8') if os.path.exists(prev_agent_path) else "Not available"

            # Read improvement.md from current generation
            improvement_path = gen_data.get('improvement_path')
            improvement_content = ""
            if improvement_path and os.path.exists(improvement_path):
                improvement_content = Path(improvement_path).read_text(encoding='utf-8')

            # Get previous generation's metrics
            prev_metrics = self.generations[-1]['metrics'] if self.generations else {}

            # Format metrics comparison
            metrics_comparison = self._format_metrics_comparison(prev_metrics, metrics)

            # Create prompt for LLM
            prompt = f"""You are analyzing the evolution of an AI agent across generations.

**TASK**: Provide a concise summary (2-4 sentences) of what changed between Generation {gen_num - 1} and Generation {gen_num}, focusing on:
1. Key code/structural changes made
2. The reasoning behind these changes (from improvement.md)
3. Impact on performance metrics (if any)

**IMPROVEMENT PLAN** (improvement.md from gen_{gen_num}):
```
{improvement_content if improvement_content else "No improvement.md found"}
```

**PREVIOUS AGENT CODE** (gen_{gen_num - 1}/target_agent.py):
```python
{prev_agent_code[:3000]}{"..." if len(prev_agent_code) > 3000 else ""}
```

**CURRENT AGENT CODE** (gen_{gen_num}/target_agent.py):
```python
{current_agent_code[:3000]}{"..." if len(current_agent_code) > 3000 else ""}
```

**METRICS COMPARISON**:
{metrics_comparison}

**YOUR SUMMARY** (2-4 sentences, be specific about what changed and why):
"""

            # Create a temporary directory for LLM execution
            with tempfile.TemporaryDirectory() as temp_dir:
                # Import run_agent here to avoid circular imports
                from util import run_agent

                # Run the LLM to generate summary
                summary_result = []

                async def get_summary():
                    # We'll capture the output by writing to a file
                    summary_file = os.path.join(temp_dir, "summary.txt")

                    # Modify prompt to ask LLM to write summary to file
                    file_prompt = prompt + f"\n\nWrite your 2-4 sentence summary to the file: {summary_file}"

                    await run_agent(
                        model_name=self.meta_model,
                        max_turns="5",
                        prompt=file_prompt,
                        agent_working_directory=temp_dir,
                        backend=self.backend
                    )

                    # Read the summary from file
                    if os.path.exists(summary_file):
                        return Path(summary_file).read_text(encoding='utf-8').strip()
                    return None

                # Run async function
                summary = asyncio.run(get_summary())

                if summary:
                    print(f"Generated LLM summary for Generation {gen_num}")
                    return summary
                else:
                    print(f"Warning: Could not generate LLM summary for Generation {gen_num}")
                    return None

        except Exception as e:
            print(f"Warning: Error generating LLM summary: {e}")
            return None

    def _format_metrics_comparison(self, prev_metrics: Dict[str, Any], current_metrics: Dict[str, Any]) -> str:
        """Format a comparison of metrics between generations"""
        if not prev_metrics and not current_metrics:
            return "No metrics available for comparison"

        lines = []
        all_keys = set(list(prev_metrics.keys()) + list(current_metrics.keys()))

        for key in sorted(all_keys):
            prev_val = prev_metrics.get(key, "N/A")
            curr_val = current_metrics.get(key, "N/A")

            # Try to calculate delta if both are numeric
            delta_str = ""
            try:
                if prev_val != "N/A" and curr_val != "N/A":
                    prev_num = float(str(prev_val).rstrip('%'))
                    curr_num = float(str(curr_val).rstrip('%'))
                    delta = curr_num - prev_num
                    delta_str = f" ({delta:+.2f})"
            except:
                pass

            lines.append(f"- {key}: {prev_val} → {curr_val}{delta_str}")

        return "\n".join(lines) if lines else "No metrics to compare"

    def add_generation(self, gen_num: int, gen_data: Dict[str, Any]):
        """
        Append a generation entry to context.md.

        Args:
            gen_num: Generation number (1-indexed)
            gen_data: Dictionary with keys:
                - success: bool, whether execution succeeded
                - timestamp: str, execution timestamp
                - duration: float, execution duration in seconds
                - agent_path: str, path to target_agent.py
                - gen_dir: str, path to generation directory
                - improvement_path: Optional[str], path to improvement.md
                - execution_type: str, 'Single' or 'Multi-trajectory'
        """
        # Extract agent stats
        agent_stats = self._get_agent_stats(gen_data['agent_path'])

        # Calculate deltas if previous generation exists
        deltas = {}
        if self.generations:
            prev_stats = self.generations[-1]['agent_stats']
            deltas = {
                'size_pct': ((agent_stats['size'] - prev_stats['size']) / prev_stats['size'] * 100),
                'lines_delta': agent_stats['lines'] - prev_stats['lines']
            }

        # Extract metrics
        metrics = self._extract_metrics(gen_data['gen_dir'])

        # Extract insights from improvement.md (if exists)
        insights = []
        if gen_data.get('improvement_path') and os.path.exists(gen_data['improvement_path']):
            insights = self._extract_insights(gen_data['improvement_path'])

        # Generate LLM summary of changes and improvements
        llm_summary = self._generate_llm_summary(gen_num, gen_data, metrics)

        # Format entry
        entry = self._format_generation_entry(
            gen_num, gen_data, agent_stats, deltas, metrics, insights, llm_summary
        )

        # Append to file
        with open(self.context_path, 'a', encoding='utf-8') as f:
            f.write(entry + "\n---\n\n")

        # Store for delta calculations and summary
        self.generations.append({
            'gen_num': gen_num,
            'agent_stats': agent_stats,
            'metrics': metrics,
            'success': gen_data.get('success', True)
        })

        print(f"Added Generation {gen_num} to context.md")

    def finalize(self):
        """Add summary statistics at the end of context.md"""
        if not self.generations:
            return

        first_gen = self.generations[0]
        last_gen = self.generations[-1]

        # Find best generation by primary metric (accuracy)
        best_gen = None
        best_metric = -float('inf')
        for g in self.generations:
            accuracy = g['metrics'].get('accuracy')
            if accuracy is not None:
                if isinstance(accuracy, str):
                    # Handle percentage strings like "48.99%"
                    try:
                        accuracy = float(accuracy.rstrip('%'))
                    except:
                        continue
                if accuracy > best_metric:
                    best_metric = accuracy
                    best_gen = g

        # Calculate evolution
        evolution_text = "N/A"
        if first_gen['metrics'].get('accuracy') is not None and last_gen['metrics'].get('accuracy') is not None:
            first_acc = first_gen['metrics']['accuracy']
            last_acc = last_gen['metrics']['accuracy']

            # Handle percentage strings
            if isinstance(first_acc, str):
                first_acc = float(first_acc.rstrip('%'))
            if isinstance(last_acc, str):
                last_acc = float(last_acc.rstrip('%'))

            gain = last_acc - first_acc
            evolution_text = f"{first_acc:.2f}% → {last_acc:.2f}% ({gain:+.2f}%)"

        # Build summary
        summary = f"""## Summary Statistics

**Total Generations**: {len(self.generations)}
**Successful Executions**: {sum(1 for g in self.generations if g.get('success', True))}
**Best Performance**: Generation {best_gen['gen_num'] if best_gen else 'N/A'} ({best_metric:.2f}% accuracy)

**Evolution**:
- {evolution_text}

**Code Growth**:
- Initial: {first_gen['agent_stats']['lines']} lines ({first_gen['agent_stats']['size']:,} bytes)
- Final: {last_gen['agent_stats']['lines']} lines ({last_gen['agent_stats']['size']:,} bytes)
- Growth: {last_gen['agent_stats']['lines'] - first_gen['agent_stats']['lines']} lines ({last_gen['agent_stats']['size'] - first_gen['agent_stats']['size']:+,} bytes)
"""

        with open(self.context_path, 'a', encoding='utf-8') as f:
            f.write(summary)

        print(f"Finalized context.md with summary statistics")

    def _get_agent_stats(self, agent_path: str) -> Dict[str, int]:
        """Get file statistics for target_agent.py"""
        try:
            with open(agent_path, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
            size = os.path.getsize(agent_path)
            return {'size': size, 'lines': lines}
        except Exception as e:
            print(f"Warning: Could not get agent stats: {e}")
            return {'size': 0, 'lines': 0}

    def _extract_metrics(self, gen_dir: str) -> Dict[str, Any]:
        """Extract performance metrics from various sources"""
        metrics = {}

        # Priority 1: results.json
        results_path = os.path.join(gen_dir, 'results.json')
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'accuracy' in data:
                        metrics['accuracy'] = data['accuracy']
                    if 'correct' in data:
                        metrics['correct'] = data['correct']
                    if 'total_questions' in data:
                        metrics['total'] = data['total_questions']
                    if 'total_tokens' in data:
                        metrics['total_tokens'] = data['total_tokens']
            except Exception as e:
                print(f"Warning: Could not parse results.json: {e}")

        # Priority 2: detailed_results.json
        detailed_results_path = os.path.join(gen_dir, 'detailed_results.json')
        if os.path.exists(detailed_results_path) and not metrics:
            try:
                with open(detailed_results_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'overall_accuracy' in data:
                        metrics['accuracy'] = data['overall_accuracy']
            except Exception as e:
                print(f"Warning: Could not parse detailed_results.json: {e}")

        # Priority 3: Parse stdout
        stdout_path = os.path.join(gen_dir, 'target_agent_stdout.log')
        if os.path.exists(stdout_path) and not metrics:
            metrics.update(self._parse_stdout_metrics(stdout_path))

        return metrics

    def _parse_stdout_metrics(self, stdout_path: str) -> Dict[str, Any]:
        """Parse metrics from stdout log using regex patterns"""
        metrics = {}
        try:
            with open(stdout_path, 'r', encoding='utf-8') as f:
                content = f.read()

                # Look for common patterns
                patterns = {
                    'accuracy': [
                        r'accuracy[:\s=]+(\d+\.?\d*)\s*%?',
                        r'final\s+accuracy[:\s=]+(\d+\.?\d*)\s*%?',
                        r'test\s+accuracy[:\s=]+(\d+\.?\d*)\s*%?',
                    ],
                    'validation': [
                        r'validation[:\s=]+(\d+\.?\d*)',
                        r'val[:\s=]+(\d+\.?\d*)',
                    ],
                    'correct': [
                        r'(\d+)\s*/\s*\d+\s+correct',
                        r'correct[:\s=]+(\d+)',
                    ],
                    'total': [
                        r'\d+\s*/\s*(\d+)\s+(?:questions|samples|total)',
                    ],
                }

                for metric_name, pattern_list in patterns.items():
                    for pattern in pattern_list:
                        match = re.search(pattern, content, re.IGNORECASE)
                        if match:
                            try:
                                value = float(match.group(1))
                                metrics[metric_name] = value
                                break
                            except:
                                continue
        except Exception as e:
            print(f"Warning: Could not parse stdout metrics: {e}")

        return metrics

    def _extract_insights(self, improvement_path: str) -> List[str]:
        """Extract key points from improvement.md"""
        insights = []
        try:
            with open(improvement_path, 'r', encoding='utf-8') as f:
                content = f.read()

                # Look for bullet points or numbered items in improvement.md
                # Pattern 1: Lines starting with - or *
                bullet_pattern = r'^[-*]\s+(.+)$'
                bullets = re.findall(bullet_pattern, content, re.MULTILINE)

                # Pattern 2: Numbered items
                numbered_pattern = r'^\d+\.\s+(.+)$'
                numbered = re.findall(numbered_pattern, content, re.MULTILINE)

                # Combine and take first 5 meaningful insights
                all_insights = bullets + numbered

                # Filter out very short or header-like items
                meaningful_insights = [
                    insight.strip()
                    for insight in all_insights
                    if len(insight.strip()) > 20 and not insight.strip().endswith(':')
                ]

                insights = meaningful_insights[:5]
        except Exception as e:
            print(f"Warning: Could not extract insights: {e}")

        return insights

    def _format_generation_entry(
        self,
        gen_num: int,
        gen_data: Dict[str, Any],
        stats: Dict[str, int],
        deltas: Dict[str, float],
        metrics: Dict[str, Any],
        insights: List[str],
        llm_summary: Optional[str] = None
    ) -> str:
        """Format markdown entry for a generation"""

        status = "✓ SUCCESS" if gen_data.get('success', True) else "✗ FAILED"

        entry = f"""## Generation {gen_num}

**Status**: {status}
**Timestamp**: {gen_data.get('timestamp', 'N/A')}
**Duration**: {gen_data.get('duration', 0):.1f}s

### Target Agent Changes
"""

        if gen_num == 1:
            entry += f"""- Initial agent created by meta-agent
- File size: {stats['size']:,} bytes
- Lines of code: {stats['lines']}
"""
        else:
            delta_size = deltas.get('size_pct', 0)
            delta_size_str = f"+{delta_size:.1f}%" if delta_size > 0 else f"{delta_size:.1f}%"
            delta_lines = deltas.get('lines_delta', 0)
            delta_lines_str = f"+{delta_lines}" if delta_lines > 0 else f"{delta_lines}"

            entry += f"""- Modified by feedback agent
- File size: {stats['size']:,} bytes ({delta_size_str})
- Lines: {stats['lines']} ({delta_lines_str} lines)
"""
            if insights:
                entry += "- Key changes from improvement.md:\n"
                for insight in insights[:3]:
                    # Truncate very long insights
                    insight_text = insight[:200] + "..." if len(insight) > 200 else insight
                    entry += f"  * {insight_text}\n"

        # Add LLM-generated summary if available
        if llm_summary:
            entry += f"""
### Evolution Summary (LLM Analysis)
{llm_summary}
"""

        entry += f"""
### Execution Summary
- Execution status: {status}
- Output format: {gen_data.get('execution_type', 'Unknown')}

### Performance Metrics
"""

        if metrics:
            for key, value in metrics.items():
                if isinstance(value, float):
                    entry += f"- {key}: {value:.2f}\n"
                else:
                    entry += f"- {key}: {value}\n"
        else:
            entry += "- No structured metrics found\n"

        # Show changes vs previous generation
        if gen_num > 1 and self.generations:
            prev_metrics = self.generations[-1]['metrics']
            changes = []

            for key in metrics:
                if key in prev_metrics:
                    current = metrics[key]
                    previous = prev_metrics[key]

                    # Handle percentage strings
                    if isinstance(current, str):
                        try:
                            current = float(current.rstrip('%'))
                        except:
                            continue
                    if isinstance(previous, str):
                        try:
                            previous = float(previous.rstrip('%'))
                        except:
                            continue

                    # Calculate delta
                    if isinstance(current, (int, float)) and isinstance(previous, (int, float)):
                        delta = current - previous
                        changes.append(f"- {key}: {delta:+.2f}")

            if changes:
                entry += "\n### Changes vs Previous Generation\n"
                entry += "\n".join(changes) + "\n"

        return entry
