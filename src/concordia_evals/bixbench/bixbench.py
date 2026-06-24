"""BixBench: Evaluating Language Models on Bioinformatics Data Analysis.

BixBench is a benchmark for evaluating LLM agents on bioinformatics data
analysis tasks. Agents are given biological datasets and must create Jupyter
notebooks to analyze the data and answer open-ended research questions.

Paper: https://arxiv.org/abs/2502.09839
Dataset: https://huggingface.co/datasets/futurehouse/BixBench
"""

import json
import logging
import os
import shutil
import string
from pathlib import Path
from textwrap import dedent
from typing import Any, Literal

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, get_model
from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import (
    Generate,
    Solver,
    TaskState,
    basic_agent,
    chain,
    solver,
    system_message,
)
from inspect_ai.tool import Tool, tool
from inspect_ai.util import SandboxEnvironmentSpec, sandbox
from platformdirs import user_cache_dir

BIXBENCH_CACHE_PATH = Path(user_cache_dir("concordia-evals"))

FILE_DIR = Path(__file__).parent.parent.parent.parent
DEFAULT_DATA_DIR = BIXBENCH_CACHE_PATH / "bixbench" / "data"
COMPOSE_FILES_DIR = BIXBENCH_CACHE_PATH / "bixbench" / "compose_files"
DEFAULT_IMAGE_NAME = "jingedawang/bixbench-inspect-env:latest"
DEFAULT_MAX_MESSAGES = 80
MAX_OUTPUT_CHARS_PER_CELL = 2000
MAX_TOTAL_DIFF_CHARS = 10000
MAX_VIEW_CELLS = 100
CELL_TIMEOUT_SECONDS = 120
KERNEL_CONN_FILE = "/workspace/.bixbench_kernel.json"
KERNEL_LOG_FILE = "/workspace/.bixbench_kernel.log"
RUNNER_PATH = "/workspace/.bixbench_runner.py"
BEFORE_OUTPUTS_PATH = "/workspace/.nb_before_outputs.json"
HF_REPO_ID = "futurehouse/BixBench"
BIXBENCH_DATASET_REVISION = "f8cc3bdcc6357c88b8c3648306522b9c422dc95a"

# Python script deployed into the sandbox. It manages a persistent IPython kernel
# and executes only the requested slice of the notebook.
RUNNER_SCRIPT_TEMPLATE = """
import json
import os
import queue
import subprocess
import sys
import time
import traceback

import nbformat
from jupyter_client import BlockingKernelClient
from nbformat.v4 import new_output

NOTEBOOK_PATH = "/workspace/notebook.ipynb"
BEFORE_OUTPUTS_PATH = "/workspace/.nb_before_outputs.json"
CONN_FILE = "/workspace/.bixbench_kernel.json"
KERNEL_LOG = "/workspace/.bixbench_kernel.log"
PER_CELL_TIMEOUT = $PER_CELL_TIMEOUT
MAX_PER_CELL = $MAX_PER_CELL
MAX_TOTAL = $MAX_TOTAL


def output_text(cell):
    parts = []
    for o in cell.get("outputs", []):
        ot = o.get("output_type")
        if ot == "stream":
            parts.append(o.get("text", ""))
        elif ot in ("execute_result", "display_data"):
            data = o.get("data", {}).get("text/plain", "")
            if data:
                parts.append(str(data))
        elif ot == "error":
            parts.append("\\n".join(o.get("traceback", [])))
    return "\\n".join(parts)


def ensure_kernel():
    if os.path.exists(CONN_FILE):
        kc = BlockingKernelClient(connection_file=CONN_FILE)
        kc.load_connection_file()
        kc.start_channels()
        try:
            kc.wait_for_ready(timeout=5)
            kc.execute("1+1", silent=True, store_history=False)
            kc.get_shell_msg(timeout=5)
            kc.stop_channels()
            return True
        except Exception:
            kc.stop_channels()
        try:
            os.remove(CONN_FILE)
        except OSError:
            pass
    if os.path.exists(CONN_FILE):
        os.remove(CONN_FILE)
    proc = subprocess.Popen(
        [sys.executable, "-m", "ipykernel_launcher", "-f", CONN_FILE],
        stdout=open(KERNEL_LOG, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    for _ in range(60):
        time.sleep(0.5)
        if os.path.exists(CONN_FILE):
            try:
                kc = BlockingKernelClient(connection_file=CONN_FILE)
                kc.load_connection_file()
                kc.start_channels()
                kc.wait_for_ready(timeout=5)
                kc.execute("1+1", silent=True, store_history=False)
                kc.get_shell_msg(timeout=5)
                kc.stop_channels()
                return True
            except Exception:
                pass
    return False


def restart_kernel():
    if not os.path.exists(CONN_FILE):
        return ensure_kernel()
    kc = BlockingKernelClient(connection_file=CONN_FILE)
    kc.load_connection_file()
    kc.start_channels()
    try:
        kc.wait_for_ready(timeout=10)
        kc.shutdown(restart=True)
        try:
            kc.get_shell_msg(timeout=10)
        except queue.Empty:
            pass
        kc.wait_for_ready(timeout=30)
        kc.stop_channels()
    except Exception:
        kc.stop_channels()
        return ensure_kernel()
    return True


def run_cell(kc, code, timeout):
    deadline = time.monotonic() + timeout
    msg_id = kc.execute(code, store_history=False)
    outputs = []
    timed_out = False
    while True:
        try:
            remaining = max(0.1, deadline - time.monotonic())
            msg = kc.get_iopub_msg(timeout=remaining)
        except queue.Empty:
            outputs.append({
                "output_type": "error",
                "ename": "TimeoutError",
                "evalue": f"Cell timed out after {timeout}s",
                "traceback": [f"TimeoutError: Cell execution exceeded {timeout} seconds"],
            })
            timed_out = True
            break
        parent = msg.get("parent_header", {})
        if parent.get("msg_id") != msg_id:
            continue
        msg_type = msg["header"]["msg_type"]
        content = msg["content"]
        if msg_type == "status" and content.get("execution_state") == "idle":
            break
        if msg_type == "stream":
            if (
                outputs
                and outputs[-1].get("output_type") == "stream"
                and outputs[-1].get("name") == content["name"]
            ):
                outputs[-1]["text"] += content["text"]
            else:
                outputs.append({
                    "output_type": "stream",
                    "name": content["name"],
                    "text": content["text"],
                })
        elif msg_type == "display_data":
            outputs.append({
                "output_type": "display_data",
                "data": content.get("data", {}),
                "metadata": content.get("metadata", {}),
            })
        elif msg_type == "execute_result":
            outputs.append({
                "output_type": "execute_result",
                "data": content.get("data", {}),
                "metadata": content.get("metadata", {}),
                "execution_count": content.get("execution_count"),
            })
        elif msg_type == "error":
            outputs.append({
                "output_type": "error",
                "ename": content["ename"],
                "evalue": content["evalue"],
                "traceback": content["traceback"],
            })
    try:
        kc.get_shell_msg(timeout=max(0.1, deadline - time.monotonic()))
    except queue.Empty:
        pass
    return outputs, timed_out


def main():
    cell_index_str = sys.argv[1]
    mode = sys.argv[2]
    passed_idx = json.loads(cell_index_str)

    if not ensure_kernel():
        print("Failed to start or connect to the IPython kernel.")
        sys.exit(0)

    if mode in ("full", "from_start"):
        if not restart_kernel():
            print("Failed to restart the IPython kernel.")
            sys.exit(0)

    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    target_idx = passed_idx if passed_idx is not None else len(nb.cells) - 1

    if mode == "cell_only":
        indices = [target_idx]
    elif mode == "from_start":
        indices = list(range(0, target_idx + 1))
    elif mode == "to_end":
        indices = list(range(target_idx, len(nb.cells)))
    else:
        indices = list(range(len(nb.cells)))

    try:
        with open(BEFORE_OUTPUTS_PATH) as f:
            before_outputs = json.load(f)
    except Exception:
        before_outputs = []

    kc = BlockingKernelClient(connection_file=CONN_FILE)
    kc.load_connection_file()
    kc.start_channels()
    restart_after = False
    try:
        kc.wait_for_ready(timeout=10)
        for idx in indices:
            if idx < 0 or idx >= len(nb.cells):
                continue
            cell = nb.cells[idx]
            if cell.cell_type != "code":
                continue
            outputs, timed_out = run_cell(kc, cell.source, PER_CELL_TIMEOUT)
            converted = []
            for o in outputs:
                ot = o["output_type"]
                if ot == "stream":
                    converted.append(new_output(output_type="stream", name=o["name"], text=o["text"]))
                elif ot == "display_data":
                    converted.append(new_output(output_type="display_data", data=o["data"], metadata=o.get("metadata", {})))
                elif ot == "execute_result":
                    converted.append(new_output(output_type="execute_result", data=o["data"], metadata=o.get("metadata", {}), execution_count=o.get("execution_count")))
                elif ot == "error":
                    converted.append(new_output(output_type="error", ename=o["ename"], evalue=o["evalue"], traceback=o["traceback"]))
                else:
                    converted.append(o)
            nb.cells[idx].outputs = converted
            if timed_out:
                restart_after = True
    finally:
        kc.stop_channels()

    nbformat.write(nb, NOTEBOOK_PATH)

    if restart_after:
        restart_kernel()

    lines = []
    total = 0
    changed_count = 0
    for i, cell in enumerate(nb.cells):
        out = output_text(cell)
        changed = i < len(before_outputs) and out != before_outputs[i]
        is_target = i == target_idx
        if not changed and not is_target:
            continue
        changed_count += 1
        block_lines = [f"### Cell {i} ({'edited' if is_target else 'changed'}):"]
        if is_target and cell.cell_type == "code":
            block_lines.extend(["```python", cell.source, "```"])
        if out:
            truncated = out[:MAX_PER_CELL]
            if len(out) > MAX_PER_CELL:
                truncated += f"\\n... [{len(out) - MAX_PER_CELL} more characters omitted]"
            block_lines.append(f"### Output {i}:")
            block_lines.append(truncated)
        block = "\\n".join(block_lines)
        if total + len(block) > MAX_TOTAL and total > 0:
            lines.append("... [additional changed cells omitted to save context]")
            break
        lines.append(block)
        total += len(block)

    if not lines:
        print(f"Notebook executed. Target cell {target_idx} produced no new output.")
    else:
        print(f"Summary: {changed_count} cell(s) had new/changed output.\\n")
        print("\\n\\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Runner error: {e}")
        traceback.print_exc()
"""

logger = logging.getLogger(__name__)

OPEN_ENDED_EVAL_PROMPT = """
Here is a question, the correct answer to the question, and a proposed answer.
Question: {question}
Correct answer: {correct_answer}
Proposed answer: {proposed_answer}
You must respond with a binary score (0 or 1) for whether the proposed answer is equivalent to the correct answer.
Nothing else is permitted.
"""

CAPSULE_SYSTEM_PROMPT_OPEN = """
You are an expert bioinformatician and seasoned biological data scientist.
Your task is to create a comprehensive Jupyter notebook named 'notebook.ipynb' that analyzes data to answer an open-ended question.
The notebook should contain all necessary artifacts (plots, tables, print outputs) to fully answer this question, structured in a way that another model could use to derive the answer.
"""

CHAIN_OF_THOUGHT_AGNOSTIC = """
Follow these steps to create your notebook, using chain-of-thought reasoning at each stage:

1. Load Data and Perform Descriptive Statistics:
<analysis_planning>
- Identify which data files are most relevant to resolving the task.
- Plan how to load these files efficiently in {language}.
- List the specific descriptive statistics you plan to use (e.g., summary(), str(), head()).
- Consider potential issues like missing data or unexpected formats. How will you handle each?
- Plan how to present this information clearly in the notebook.
- Write down key statistics you expect to see and how you'll interpret them.
- Consider potential data quality issues and how you'll address them.
</analysis_planning>
Execute your plan to load data and perform descriptive statistics.

2. Develop Analysis Plan:
<analysis_planning>
- Break down each task into testable components. List these components.
- For each component, list appropriate statistical tests or visualizations.
- Consider alternative approaches for each component and justify your choices.
- Identify potential confounding factors and how to address them.
- Plan the sequence of your analysis steps, explaining the rationale for each.
- Consider how this analysis plan will be documented in the notebook.
- List potential statistical assumptions for your chosen methods and how you'll test them.
- Think about how your analysis plan addresses your original task.
</analysis_planning>
Write out your analysis plan as comments in the notebook.

3. Execute Analysis Plan:
<analysis_planning>
- For each step in your analysis plan, list the {language} or bash functions and libraries you'll use.
- Think about how to structure your code for readability and efficiency.
- Plan how to document your code with clear comments.
- Consider how to present results clearly, using tables or visualizations where appropriate.
- Ensure that all outputs are clearly labeled and explained in the context of the task.
- Plan how you'll interpret each result in relation to the original task.
- Consider potential unexpected results and how you'll handle them.
</analysis_planning>
Execute your analysis plan, creating new cells as needed.

4. Conclude and Submit Answer:
<thought_process>
- Reflect on how your results relate to the original task.
- Consider any limitations or uncertainties in your analysis.
- Plan a concise summary of your findings.
- Think about how to phrase your conclusion as clear statements.
- Ensure that the notebook contains all necessary information for another model to derive these answers.
- Consider any additional insights or patterns you've noticed during the analysis.
- Think about potential follow-up questions or areas for further investigation.
</thought_process>
"""

SUBMIT_ANSWER_OPEN = """
[Use the submit tool to submit your final answer as a single string with your short answer]
IMPORTANT: Wrap your answer in XML tags <answer> </answer>
Example output:
```
submit("<answer>Your concise answer to the question</answer>")
```
Remember, the final notebook should contain all necessary artifacts (plots, tables, print outputs) to solve the task provided.
"""

GENERAL_NOTEBOOK_GUIDELINES = """
General Guidelines:
- Write small to medium-sized cells for easier debugging.
- Edit existing cells by their index number when fixing bugs, rather than creating new ones.
- Check dataframe shapes before printing. Use head() for large dataframes.
- Ensure each cell executes successfully before moving to the next.
- Common packages are already installed. Do NOT install new packages unless execution fails with a missing-package error.
- If you must install a package, prefer pip/mamba for Python and keep R package installs minimal. Some R packages (e.g. 'brant') are not available or compile slowly and will hang the notebook.
- The notebook runs a Python kernel with the rpy2 extension loaded.
- Cell 0 has already loaded the rpy2 extension. Do not overwrite Cell 0; start your analysis in Cell 1 or later.
- All cells are by default Python cells.
- You can run R code in a cell by putting %%R on the first line of the cell.
- You can run bash commands in a cell by putting %%bash on the first line of the cell.
- You can only create code cells, no markdown cells.
- The edit_cell tool only returns the source of the edited/appended cell and any outputs that changed during the last execution. Use the view_notebook tool when you need to see the full notebook.
- When calling edit_cell, choose execution_mode wisely:
  - cell_only: run only the edited/appended cell (fastest; safe only when earlier cells are already loaded in the persistent kernel, e.g. when appending a new cell after a successful full run).
  - from_start: restart the kernel and run from cell 0 up to the edited cell (useful when debugging an early cell).
  - to_end: run from the edited cell to the end of the notebook (useful after fixing an early cell and wanting to refresh downstream results).
  - full: restart the kernel and run the entire notebook (safest when the notebook structure or imports changed).
"""

R_SPECIFIC_GUIDELINES = """
R-Specific Guidelines (for cells starting with %%R):
1. Start every R cell with %%R on the first line.
2. The following R packages are already installed and should be preferred: base, stats, MASS, tidyverse (dplyr, tidyr, ggplot2, readr, stringr, forcats, purrr, tibble, lubridate), Bioconductor (DESeq2, clusterProfiler, org.Hs.eg.db, EnhancedVolcano, SummarizedExperiment, apeglm, GenomicRanges).
3. Load packages using this format to minimize verbose output:
   ```r
   if (!requireNamespace("package_name", quietly = TRUE)) {{
     install.packages("package_name")
   }}
   suppressPackageStartupMessages(library(package_name))
   ```
4. Use the tidyverse wherever possible: dplyr, tidyr, ggplot2, readr, stringr, forcats, purrr, tibble, and lubridate.
4. Use explicit namespace qualification for functions. For example, use dplyr::select() instead of select().
5. For data operations, suppress messages about column name repairs:
   ```r
   variable_name <- read_excel("<fpath>.csv", col_names = FALSE, .name_repair = "minimal")
   ```
6. When transferring data between Python and R with rpy2, you can use the -i and -o flags:
   ```r
   %%R -i python_df -o r_result
   r_result <- some_r_function(python_df)
   ```
"""

AVOID_IMAGES = """
AVOID USING PLOTS/IMAGES. USE TABLES AND PRINT OUTPUTS INSTEAD AS MUCH AS POSSIBLE.
"""

BASH_TOOL_USAGE = """
If you need to use Busco, you can use it through udocker as follows:

```bash
# BUSCO Guidelines:
# 1. Set up the required directory structure for BUSCO:
mkdir busco_downloads
mkdir busco_downloads/lineages
mv <lineage_db> busco_downloads/lineages  # Move your downloaded lineage database

# 2. Run BUSCO analysis on protein files:
for protein_file in *.<protein_extension> ; do
  output_name=$(echo "$protein_file" | sed "s/.<protein_extension>$/.busco/g")
  udocker --allow-root run -u $(id -u) \
    -v /content/:/busco_wd \
    ezlabgva/busco:v5.8.0_cv1 \
    busco -i $protein_file \
    -m prot \
    --offline \
    -o $output_name \
    -l <lineage_name>
done

# Note: Replace the following placeholders:
# - <lineage_db>: Your downloaded BUSCO lineage database directory
# - <protein_extension>: Your protein file extension (e.g., faa, fasta)
# - <lineage_name>: Name of the BUSCO lineage to use (e.g., eukaryota_odb10)
```

You can also use mafft, clipkit, fastqc, iqtree, metaeuk, perl, phykit through the command line.
"""


def _get_compose_file(
    question_id: str,
    capsule_dir: Path,
    docker_image_name: str,
    force_rebuild: bool = False,
) -> str:
    """Generate a Docker Compose file for a BixBench sample.

    The capsule data is mounted read-only as /workspace_template. On container
    startup the contents are copied into /workspace, which lives inside the
    container's own writable layer. This guarantees that every sample run
    starts from a pristine workspace, preventing contamination between:
    - different questions sharing a capsule,
    - different models evaluating the same question,
    - repeated runs of the same question.

    Args:
        question_id: Unique identifier for the question.
        capsule_dir: Path to the prepared capsule data directory.
        docker_image_name: Name of the Docker image to use.
        force_rebuild: Whether to force a rebuild of the Docker image.

    Returns:
        Path to the generated Docker Compose file.
    """
    import docker  # type: ignore
    from docker.errors import ImageNotFound  # type: ignore

    image_compose_file = COMPOSE_FILES_DIR / f"{question_id}.yaml"
    image_compose_file.parent.mkdir(parents=True, exist_ok=True)

    with image_compose_file.open(mode="w+") as f:
        f.write(
            dedent(
                f"""\
            services:
              default:
                image: {docker_image_name}
                platform: linux/amd64
                volumes:
                  - {capsule_dir}:/workspace_template:ro
                working_dir: /workspace
                environment:
                  - PATH=/opt/conda/envs/bixbench/bin:/opt/conda/bin:/usr/local/bin:/usr/bin:/bin
                x-local: true
                entrypoint:
                  - /bin/sh
                  - -c
                  - >
                    rm -rf /workspace &&
                    mkdir -p /workspace &&
                    cp -a /workspace_template/. /workspace/ &&
                    exec tail -f /dev/null
            """
            )
        )

        image_exists = False
        try:
            docker.from_env().images.get(docker_image_name)
            image_exists = True
        except ImageNotFound:
            try:
                logger.info(
                    "Image %s not found locally, attempting to pull from registry.",
                    docker_image_name,
                )
                docker.from_env().images.pull(docker_image_name)
                image_exists = True
            except Exception as exc:
                logger.warning(
                    "Failed to pull image %s: %s. Will build from Dockerfile.",
                    docker_image_name,
                    exc,
                )

        if force_rebuild or not image_exists:
            f.write(
                f"""    build:
      context: {FILE_DIR}
      network: host"""
            )

    return str(image_compose_file)


def _prepare_capsule_data(
    question: dict[str, Any],
    data_dir: Path = DEFAULT_DATA_DIR,
    hf_repo_id: str = HF_REPO_ID,
    hf_revision: str = BIXBENCH_DATASET_REVISION,
) -> Path:
    """Download and prepare capsule data for a BixBench question.

    This downloads the ZIP file from HuggingFace, extracts it, moves the
    contents of the Data/ folder to the root, removes any Notebook/ folders
    and .ipynb files, and returns the path to the prepared directory.

    Note: the returned directory is mounted read-only inside the Docker
    container (as /workspace_template) and copied into a fresh /workspace on
    every container startup, so multiple questions/models/runs that share this
    capsule cannot contaminate each other.

    Args:
        question: Dictionary containing question data from the dataset.
        data_dir: Directory to store prepared data.
        hf_repo_id: HuggingFace dataset repository ID.
        hf_revision: HuggingFace dataset revision (commit hash) to use.

    Returns:
        Path to the prepared capsule directory.
    """
    from huggingface_hub import hf_hub_download

    zip_filename: str = question["data_folder"]
    capsule_dir = data_dir / zip_filename.replace(".zip", "")

    # Return early if already prepared
    if capsule_dir.exists() and any(capsule_dir.iterdir()):
        logger.debug("Capsule data already prepared for %s", question["question_id"])
        return capsule_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / zip_filename

    # Download if not present
    if not zip_path.exists():
        logger.info("Downloading %s for %s", zip_filename, question["question_id"])
        hf_hub_download(
            repo_id=hf_repo_id,
            filename=zip_filename,
            local_dir=data_dir,
            repo_type="dataset",
            revision=hf_revision,
        )

    # Extract
    logger.info("Extracting %s for %s", zip_filename, question["question_id"])
    shutil.unpack_archive(zip_path, capsule_dir)

    # Move Data/ contents to root
    data_folder = next(
        (p for p in capsule_dir.rglob("*") if p.is_dir() and "Data" in p.name),
        None,
    )
    if data_folder is not None:
        for item in data_folder.iterdir():
            dest = capsule_dir / item.name
            if dest.exists():
                if item.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            if item.is_file():
                shutil.copy2(item, dest)
            elif item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
        shutil.rmtree(data_folder)
    else:
        logger.warning(
            "No Data folder found in %s for %s",
            zip_filename,
            question["question_id"],
        )

    # Remove Notebook/ folder if present
    notebook_folder = next(
        (p for p in capsule_dir.rglob("*") if p.is_dir() and "Notebook" in p.name),
        None,
    )
    if notebook_folder is not None:
        shutil.rmtree(notebook_folder)

    # Remove any .ipynb files
    for ipynb_file in capsule_dir.glob("*.ipynb"):
        ipynb_file.unlink()

    return capsule_dir


@tool(name="edit_cell")
def _edit_cell() -> Tool:
    """Tool for editing a Jupyter notebook cell and executing part of the notebook.

    A persistent IPython kernel is kept alive inside the sandbox, so the agent can
    choose to run only the edited cell instead of re-running the whole notebook.
    """

    async def execute(
        contents: str,
        cell_index: int | None = None,
        execution_mode: Literal["cell_only", "from_start", "to_end", "full"] = "full",
    ) -> str:
        """Edit a code cell and execute a slice of the notebook.

        Args:
            contents: The code to insert into the cell.
            cell_index: The index of the cell to edit. If None, appends a new cell.
            execution_mode: Which cells to execute:
                - cell_only: run only the edited/appended cell (fastest; requires
                  earlier cells to already be loaded in the persistent kernel).
                - from_start: restart the kernel and run cells 0 through target.
                - to_end: run the target cell through the end of the notebook.
                - full: restart the kernel and run the whole notebook (safest).

        Returns:
            A concise diff: the edited/appended cell source and any changed outputs.
        """
        idx_str = "None" if cell_index is None else str(cell_index)
        edit_script = f"""
import nbformat
nb = nbformat.read("/workspace/notebook.ipynb", as_version=4)
idx = {idx_str}
if idx is None or idx >= len(nb.cells):
    nb.cells.append(nbformat.v4.new_code_cell(source={repr(contents)}))
    action = f"Appended new cell (#{{len(nb.cells) - 1}})."
else:
    nb.cells[idx].source = {repr(contents)}
    action = f"Edited cell #{{idx}}."
nbformat.write(nb, "/workspace/notebook.ipynb")
print(action)
"""
        edit_result = await sandbox().exec(["python", "-c", edit_script])
        if not edit_result.success:
            return f"Failed to edit notebook: {edit_result.stderr}"

        # Snapshot outputs before execution so we can report only what changed.
        snapshot_script = """
import json, nbformat
nb = nbformat.read("/workspace/notebook.ipynb", as_version=4)

def _output_text(cell):
    parts = []
    for o in cell.get("outputs", []):
        ot = o.get("output_type")
        if ot == "stream":
            parts.append(o.get("text", ""))
        elif ot in ("execute_result", "display_data"):
            data = o.get("data", {}).get("text/plain", "")
            if data:
                parts.append(str(data))
        elif ot == "error":
            parts.append("\\n".join(o.get("traceback", [])))
    return "\\n".join(parts)

with open("/workspace/.nb_before_outputs.json", "w") as f:
    json.dump([_output_text(c) for c in nb.cells], f)
"""
        await sandbox().exec(["python", "-c", snapshot_script])

        runner_result = await sandbox().exec(
            ["python", RUNNER_PATH, json.dumps(cell_index), execution_mode]
        )
        if not runner_result.success:
            return f"Notebook execution failed:\n{runner_result.stderr}"

        return (
            runner_result.stdout
            or runner_result.stderr
            or "Notebook executed. No output to display."
        )

    return execute


@tool(name="view_notebook")
def _view_notebook() -> Tool:
    """Tool for viewing the full current notebook."""

    async def execute() -> str:
        """Return a readable representation of the whole notebook.

        Returns:
            Markdown representation of the notebook (truncated if very large).
        """
        view_script_lines = [
            "import nbformat",
            f"MAX_SOURCE = {MAX_OUTPUT_CHARS_PER_CELL}",
            f"MAX_OUT = {MAX_OUTPUT_CHARS_PER_CELL}",
            f"MAX_CELLS = {MAX_VIEW_CELLS}",
            'nb = nbformat.read("/workspace/notebook.ipynb", as_version=4)',
            'lines = [f"Total cells: {len(nb.cells)}"]',
            "for i, cell in enumerate(nb.cells[:MAX_CELLS]):",
            '    lines.append(f"### Cell {i}:")',
            '    if cell.cell_type == "code":',
            "        src = cell.source",
            "        if len(src) > MAX_SOURCE:",
            '            src = src[:MAX_SOURCE] + "\\n... [source truncated]"',
            '        lines.extend(["```python", src, "```"])',
            '        for output in cell.get("outputs", []):',
            "            text = ''",
            '            if output.output_type == "stream":',
            '                text = output.get("text", "")',
            '            elif output.output_type in ("execute_result", "display_data"):',
            '                data = output.get("data", {}).get("text/plain", "")',
            "                if data:",
            "                    text = str(data)",
            '            elif output.output_type == "error":',
            '                text = "\\n".join(output.traceback)',
            "            if text:",
            "                if len(text) > MAX_OUT:",
            '                    text = text[:MAX_OUT] + "\\n... [output truncated]"',
            '                lines.append(f"### Output {i}:")',
            "                lines.append(text)",
            '    elif cell.cell_type == "markdown":',
            "        lines.append(cell.source)",
            "if len(nb.cells) > MAX_CELLS:",
            '    lines.append(f"... [{len(nb.cells) - MAX_CELLS} cells omitted]")',
            'print("\\n".join(lines))',
        ]
        view_script = "\n".join(view_script_lines)
        result = await sandbox().exec(["python", "-c", view_script])
        return result.stdout or "Notebook view produced no output."

    return execute


@tool()
def _list_workdir() -> Tool:
    """Tool for listing the contents of the working directory."""

    async def execute() -> str:
        """List the contents of the working directory.

        Returns:
            A listing of files and directories in /workspace.
        """
        result = await sandbox().exec(["ls", "-la", "/workspace"])
        return result.stdout or ""

    return execute


@solver
def _init_environment() -> Solver:
    """Initialize the notebook environment in the sandbox.

    The notebook uses a Python kernel but loads the rpy2 IPython extension so
    that subsequent cells can run R code via the %%R cell magic. This matches
    the setup used in the original BixBench paper.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        init_script = """
import nbformat
nb = nbformat.v4.new_notebook()
nb.metadata.kernelspec = {
    "display_name": "Python 3 (ipykernel)",
    "language": "python",
    "name": "python3",
}
# Load the rpy2 extension so R cells (%%R) can be used alongside Python cells.
# Include a clear comment so the agent does not overwrite this cell.
nb.cells.append(
    nbformat.v4.new_code_cell(
        source='# Cell 0: pre-loaded by the environment. DO NOT overwrite this cell.\\n'
               '%load_ext rpy2.ipython\\n'
               'print("rpy2 extension loaded.")'
    )
)
nbformat.write(nb, "/workspace/notebook.ipynb")
print("Initialized notebook.ipynb with rpy2 extension")
"""
        result = await sandbox().exec(["python", "-c", init_script])
        if not result.success:
            logger.warning("Failed to initialize notebook: %s", result.stderr)

        # Deploy the persistent-kernel runner into the sandbox.
        runner_script = string.Template(RUNNER_SCRIPT_TEMPLATE).substitute(
            MAX_PER_CELL=str(MAX_OUTPUT_CHARS_PER_CELL),
            MAX_TOTAL=str(MAX_TOTAL_DIFF_CHARS),
            PER_CELL_TIMEOUT=str(CELL_TIMEOUT_SECONDS),
        )
        await sandbox().write_file(RUNNER_PATH, runner_script)
        return state

    return solve


def default_solver(max_messages: int = DEFAULT_MAX_MESSAGES) -> Solver:
    """Return the default solver for BixBench.

    Args:
        max_messages: Maximum number of messages allowed.

    Returns:
        A solver chain that initializes the environment and runs the agent.
    """
    return chain(
        _init_environment(),
        basic_agent(
            init=[system_message(CAPSULE_SYSTEM_PROMPT_OPEN)],
            tools=[
                _edit_cell(),
                _view_notebook(),
                _list_workdir(),
            ],
            max_messages=max_messages,
        ),
    )


@scorer(
    metrics={
        "accuracy": [accuracy(), stderr()],
    }
)
def bixbench_scorer(judge_model: str | None = None) -> Scorer:
    """Scorer for BixBench open-ended questions.

    Uses an LLM to judge whether the proposed answer is equivalent to the
    correct answer, returning a binary score (0 or 1).

    Args:
        judge_model: Model to use for grading. Defaults to the
            INSPECT_EVALS_JUDGE_MODEL environment variable, or openai/gpt-4o if
            neither is set.
    """
    if judge_model is None:
        judge_model = os.environ.get("INSPECT_EVALS_JUDGE_MODEL", "openai/gpt-4o")

    async def score(state: TaskState, target: Target) -> Score:
        prompt = OPEN_ENDED_EVAL_PROMPT.format(
            question=state.metadata["question"],
            correct_answer=target.text,
            proposed_answer=state.output.completion,
        )

        model = get_model(judge_model)
        result = await model.generate([ChatMessageUser(content=prompt)])

        score_text = result.completion.strip()
        # Extract just the first digit if there's extra text
        first_digit = None
        for char in score_text:
            if char in "01":
                first_digit = char
                break

        if first_digit == "1":
            correct = True
        elif first_digit == "0":
            correct = False
        else:
            # Fallback: try to parse more leniently
            correct = "1" in score_text and score_text.count("1") > score_text.count("0")

        return Score(
            value={"accuracy": 1 if correct else 0},
            explanation=f"Grading model response: {score_text}",
            answer=state.output.completion,
        )

    return score


@task
def bixbench(
    split: str = "train",
    solver: Solver | None = None,
    scorer: Scorer | list[Scorer] | None = None,
    epochs: int = 1,
    max_messages: int = DEFAULT_MAX_MESSAGES,
    docker_image_name: str | None = None,
    force_rebuild: bool = False,
    data_dir: Path | str | None = None,
    limit: int | None = None,
    avoid_images: bool = True,
    judge_model: str | None = None,
) -> Task:
    """Returns a Task representing a BixBench evaluation.

    BixBench evaluates LLM agents on bioinformatics data analysis tasks.
    Agents must analyze biological datasets using Jupyter notebooks and answer
    open-ended research questions.

    Args:
        split: Dataset split to use (default: 'train').
        solver: Custom solver to use. If None, uses the default solver.
        scorer: Custom scorer to use. If None, uses the default LLM-based scorer.
        epochs: Number of times to repeat each sample.
        max_messages: Maximum number of messages for the default solver.
        docker_image_name: Docker image name for the sandbox. Will be built
            from the included Dockerfile if not present.
        force_rebuild: Whether to force rebuild the Docker image.
        data_dir: Directory for downloaded capsule data. If None, uses
            the default cache directory.
        limit: Maximum number of samples to include.
        avoid_images: Whether to instruct the agent to avoid using plots/images
            and use tables/print outputs instead (default: True).
        judge_model: Model to use for grading open-ended answers
            (default: openai/gpt-4o).

    Returns:
        An Inspect Task configured for BixBench evaluation.
    """
    from datasets import load_dataset

    data_dir_path = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR

    logger.info("Loading BixBench dataset (split=%s)...", split)
    dataset = load_dataset(HF_REPO_ID, split=split, revision=BIXBENCH_DATASET_REVISION)

    samples: list[Sample] = []
    for item in dataset:
        # Prepare capsule data on disk
        capsule_dir = _prepare_capsule_data(item, data_dir_path)

        # Build the task prompt matching the original BixBench OPEN_PROMPT_TEMPLATE
        language = "python"
        task_prompt = (
            "Here is the question you need to address:\n\n"
            f"<question>\n{item['question']}\n</question>\n\n"
            + CHAIN_OF_THOUGHT_AGNOSTIC.format(language=language)
            + SUBMIT_ANSWER_OPEN
            + GENERAL_NOTEBOOK_GUIDELINES
            + R_SPECIFIC_GUIDELINES
        )
        if avoid_images:
            task_prompt += AVOID_IMAGES

        samples.append(
            Sample(
                input=task_prompt,
                id=item["question_id"],
                target=item["ideal"],
                sandbox=SandboxEnvironmentSpec(
                    type="docker",
                    config=_get_compose_file(
                        question_id=item["question_id"],
                        capsule_dir=capsule_dir,
                        docker_image_name=docker_image_name or DEFAULT_IMAGE_NAME,
                        force_rebuild=force_rebuild,
                    ),
                ),
                metadata={
                    "ideal": item["ideal"],
                    "distractors": item["distractors"],
                    "question": item["question"],
                    "capsule_uuid": item.get("capsule_uuid", ""),
                },
            )
        )

        if limit is not None and len(samples) >= limit:
            break

    logger.info("Created %d samples for BixBench", len(samples))

    return Task(
        dataset=MemoryDataset(samples=samples),
        solver=solver or default_solver(max_messages),
        scorer=scorer or bixbench_scorer(judge_model=judge_model),
        epochs=epochs,
    )
