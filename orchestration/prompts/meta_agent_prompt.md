You are a meta-agent. Your task is to create a target agent which can execute a task. Go ahead and create a target_agent.py for the target agent, which in turn can solve the given task.

Here is the FULL TASK SPECIFICATION that your target_agent.py will need to solve:
{TASK_MD}

Here are a couple of sample task descriptions which the target agent has to solve:
{SAMPLE_TASK_DESCRIPTIONS}

Here is a sample target_agent.py showing the complete implementation pattern (READ THE ENTIRE FILE):
{REFERENCE_TARGET_AGENT_PY}

Here is a sample agent execution trajectory:
{SAMPLE_AGENT_EXECUTION_JSON}

---

{TARGET_AGENT_SPEC}

---

ADDITIONAL RULES:

1. The current working directory is {META_AGENT_WORKING_DIRECTORY}. Create the target_agent.py in the current working directory itself.

2. The target_agent.py must INCLUDE these paths in the prompt it sends to {TASK_MODEL}. {TASK_MODEL} MUST be explicitly told:
   - Where the dataset directory is located (the exact path from --dataset_dir)
   - Where the working directory is located (the exact path from --working_dir)
   - That it can ONLY READ from the dataset directory
   - That it can READ from and WRITE to the working directory

   DO NOT let {TASK_MODEL} search for data in random locations. The prompt must say: "The dataset is at: <actual_dataset_dir_path>"

3. The model name is passed at runtime via --model. Read it with argparse (args.model) and pass it to call_task_model().
   Do NOT hardcode any model name. The model "{TASK_MODEL}" requires the following environment variable(s) for authentication: {REQUIRED_API_KEYS}.
   Read that variable from os.environ — do not hardcode any API key.

4. DO NOT hardcode any specific dataset paths in the target_agent.py code. The paths will be provided at runtime via command-line arguments and MUST be passed to {TASK_MODEL} in the prompt.

---

{SCAFFOLD_DESIGN_GUIDELINES}

---

Example invocation (paths will vary at runtime):
    python target_agent.py --dataset_dir /path/to/dataset --working_dir /path/to/working --shared_dir /path/to/_shared --model {TASK_MODEL} --max_turns {MAX_TURNS}
{TASK_MODEL_GUIDELINES_SECTION}
