import os

try:  # Support executing as part of the src package or standalone
    from .utils import read_file
except ImportError:  # pragma: no cover - fallback when run via top-level import
    from utils import read_file


"""
Construct Prompt

Design principles: 
- To evaluate base model performance on KernelBench, we use the simplest prompt possible to guide model output to generated desired output format.
- However, we do not do extensive prompt engineering or few-shot example in the LLM to steer behaviour. 
"""

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")


def get_arch_definition_from_file(arch_path):
    arch_src = read_file(arch_path)
    return get_arch_definition(arch_src)


def get_arch_definition(arch_src):
    """
    Construct torch definition from original torch nn.Module definition
    """
    prompt = f"Here is a pytorch defintion of a neural network architecture in the file model.py: ```{arch_src}```\n"
    return prompt


############################################
# CUDA Prompt
############################################
PROBLEM_STATEMENT = """You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups. \n
    You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
"""
PROBLEM_INSTRUCTION = """
Optimize the architecture named Model with custom CUDA operators and emit a drop-in replacement called ModelNew. Follow this contract exactly:
1. Reply with a single Markdown code block labeled `python` and no additional prose before or after it.
2. Begin the block with these imports exactly once: `import torch`, `import torch.nn as nn`, `from torch.utils.cpp_extension import load_inline`.
3. Define at least one CUDA kernel string plus a `functions` dictionary that calls `load_inline` to compile it, and expose Python wrapper functions that launch the kernels with correct grid and block dimensions.
4. Implement a complete `ModelNew` class whose `__init__` signature matches `Model` and whose `forward` uses the wrapper functions. Preserve all tensor shapes returned by the original model.
5. Do not include unit tests, benchmarking harnesses, placeholder comments, or explanatory text. Only runnable production code is allowed.
6. Never reference symbols you did not define or import within the block.
"""


def prompt_generate_custom_cuda(
    arc_src: str, example_arch_src: str, example_new_arch_src: str
) -> str:
    prompt = PROBLEM_STATEMENT

    if example_arch_src != "" and example_new_arch_src != "":
        prompt += f"""
        Here's an example to show you the syntax of inline embedding custom CUDA operators in torch: The example given architecture is: \n
        ``` \n
        {example_arch_src}
        ``` \n
        The example new arch with custom CUDA kernels looks like this: 
        ```
        {example_new_arch_src}
        ``` \n
        """

    prompt += f"""
    You are given the following architecture: \n
    ```
    {arc_src}
    ```
    """
    prompt += PROBLEM_INSTRUCTION
    return prompt


PROBLEM_STATEMENT_CLEANED = """You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups.\n\nYou have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
"""
PROBLEM_INSTRUCTION_CLEANED = """
Optimize the architecture named Model with custom CUDA operators and emit a drop-in replacement called ModelNew. Follow this contract exactly:
1. Reply with a single Markdown code block labeled `python` and no additional prose before or after it.
2. Begin the block with these imports exactly once: `import torch`, `import torch.nn as nn`, `from torch.utils.cpp_extension import load_inline`.
3. Define at least one CUDA kernel string plus a `functions` dictionary that calls `load_inline` to compile it, and expose Python wrapper functions that launch the kernels with correct grid and block dimensions.
4. Implement a complete `ModelNew` class whose `__init__` signature matches `Model` and whose `forward` uses the wrapper functions. Preserve all tensor shapes returned by the original model.
5. Do not include unit tests, benchmarking harnesses, placeholder comments, or explanatory text. Only runnable production code is allowed.
6. Never reference symbols you did not define or import within the block.
"""


PROBLEM_STATEMENT_TRITON = """You write custom GPU kernels in the Triton language to replace the PyTorch operators in the given architecture to achieve speedups.\n
You may mix Triton kernels with standard PyTorch operations, but ensure every custom kernel is implemented with `@triton.jit` and wrapped so that `ModelNew` can call it seamlessly.\n"""

PROBLEM_INSTRUCTION_TRITON = """
Optimize the architecture named Model with custom Triton kernels and output a drop-in replacement called ModelNew. When you respond:
1. Emit exactly one Markdown code block labeled `python` and nothing else.
2. Start with these imports in order: `import torch`, `import torch.nn as nn`, `import triton`, `import triton.language as tl`.
3. Implement each custom kernel with `@triton.jit`, provide launch-time wrappers that compute grid sizes and strides, and ensure the wrappers are invoked from ModelNew.
4. Keep ModelNew's constructor signature identical to Model's and preserve all output tensor shapes.
5. Avoid tests, benchmarking, or explanatory text; only include runnable library code.
6. Do not use dynamic decorators such as `triton.autotune`; return a single deterministic kernel implementation per operation.
7. Reference only symbols defined or imported inside the code block.
"""

def prompt_generate_custom_cuda_fewshot_and_template(ref_arch_src: str, shots: list) -> str:
    """
    Generate a prompt with specified few-shot examples following a template 

    shots: list of few-shot examples to include in the prompt
    Avaliable few shot options to start with: 
    - ex_add: pointwise addition
    - ex_fuse_gelu: fused gelu
    - ex_mnist2: fused convolutions and relus (DEPRECATED)
    - ex_tiled_matmul: tiled matrix multiplication
    - ex_flash_attn: simple flash attention
    """
    prompt = PROBLEM_STATEMENT_CLEANED

    # k = 1
    example_add = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_ex_add.py")
    )
    example_add_new = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_new_ex_add.py")
    )
    example_add_desc = "This given architecture is for a pointwise addition: "

    # k = 2
    example_fuse_gelu = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_ex_fuse_gelu.py")
    )
    example_fuse_gelu_new = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_new_ex_fuse_gelu.py")
    )
    example_fuse_gelu_desc = "This given architecture is for a fused gelu: "

    # k = 3 (DEPRECATED)
    example_mnist2 = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_ex_mnist2.py")
    )
    example_mnist2_new = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_new_ex_mnist2.py")
    )
    exmaple_mnist2_desc = "This given architecture is for a model with fused convolutions and relus: "

    # k = 4
    example_tiled_matmul = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_ex_tiled_matmul.py")
    )
    example_tiled_matmul_new = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_new_ex_tiled_matmul.py")
    )
    example_tiled_matmul_desc = "This given architecture is for a model with tiled matrix multiplication: "

    # k = 5
    example_flash_attn = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_ex_flash_attn.py")
    )
    example_flash_attn_new = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_new_ex_flash_attn.py")
    )
    example_flash_attn_desc = "This given architecture is for a model with simple io-aware implementation of attention, also known as flash attention: "

    examples = []
    for s in shots:
        if s not in ["ex_add", "ex_fuse_gelu", "ex_mnist2", "ex_tiled_matmul", "ex_flash_attn"]:
            raise ValueError(f"Invalid shot: {s}")
        elif s == "ex_add":
            examples.append((example_add, example_add_new, example_add_desc))
        elif s == "ex_fuse_gelu":
            examples.append((example_fuse_gelu, example_fuse_gelu_new, example_fuse_gelu_desc))
        elif s == "ex_mnist2": # DEPRECATED
            raise ValueError("ex_mnist2 is deprecated")
            examples.append((example_mnist2, example_mnist2_new, exmaple_mnist2_desc))
        elif s == "ex_tiled_matmul":
            examples.append((example_tiled_matmul, example_tiled_matmul_new, example_tiled_matmul_desc))
        elif s == "ex_flash_attn":
            examples.append((example_flash_attn, example_flash_attn_new, example_flash_attn_desc))
    

    for i, tup in enumerate(examples):
        base, kernel, desc = tup

        prompt += f"""
Example {i+1}:\n\n
Here is an example architecture:\n\n
```
{base}
```\n
{PROBLEM_INSTRUCTION_CLEANED} \n
Here is an optimized verison with custom CUDA kernels: \n
```
{kernel}
```\n\n
"""

# should we put task here?
    prompt += f"""
Task:\n\n
Here is an example architecture:\n\n
```
{ref_arch_src}
```\n
"""
    prompt += PROBLEM_INSTRUCTION_CLEANED
    return prompt

def prompt_generate_ex_with_CoT_template(ref_arch_src: str, cot_example: str) -> str:
    """
    Generate a prompt with a CoT example following a template 
    Avaliable CoT examples: 
    - ex_fuse_gelu: fused gelu
    - ex_mnist2: fused convolutions and relus
    - ex_tiled_matmul: tiled matrix multiplication
    """

    # I updated this to allow CoT. Also explicilty state think step by step.
    PROBLEM_INSTRUCTION_COT = """
Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Do not output testing code. 
In the end, make sure the final code block contains code for output architecture ModelNew with cuda code.\n
Let's think step by step.\n
""" 

    prompt = PROBLEM_STATEMENT_CLEANED
    
    assert cot_example in ["ex_fuse_gelu", "ex_mnist2", "ex_tiled_matmul"]

    # k = 2
    example_fuse_gelu = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_ex_fuse_gelu.py")
    )
    example_fuse_gelu_cot = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/cot/model_cot_fuse_gelu.py")
    )
    example_fuse_gelu_new = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_new_ex_fuse_gelu.py")
    )
    example_fuse_gelu_desc = "This given architecture is for a fused gelu: "

    # k = 3
    example_mnist2 = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_ex_mnist2.py")
    )
    example_mnist2_cot = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/cot/model_cot_mnist2.py")
    )
    example_mnist2_new = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_new_ex_mnist2.py")
    )
    exmaple_mnist2_desc = "This given architecture is for a model with fused convolutions and relus: "

    # k = 4
    example_tiled_matmul = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_ex_tiled_matmul.py")
    )
    example_tiled_matmul_cot = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/cot/model_cot_tiled_matmul.py")
    )
    example_tiled_matmul_new = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_new_ex_tiled_matmul.py")
    )
    example_tiled_matmul_desc = "This given architecture is for a model with tiled matrix multiplication: "
    
    match cot_example:
        case "ex_fuse_gelu":
            base = example_fuse_gelu
            cot = example_fuse_gelu_cot
            kernel = example_fuse_gelu_new
            desc = example_fuse_gelu_desc
        case "ex_mnist2":
            base = example_mnist2
            cot = example_mnist2_cot
            kernel = example_mnist2_new
            desc = exmaple_mnist2_desc
        case "ex_tiled_matmul":
            base = example_tiled_matmul
            cot = example_tiled_matmul_cot
            kernel = example_tiled_matmul_new
            desc = example_tiled_matmul_desc
        case _:
            raise ValueError(f"Invalid CoT example: {cot_example} not found in CoT examples")

    # construct example with 
    # NOTE: we only do one example with CoT for now
    # 1. ref_src problem -> 2. Instruction -> 3. CoT -> 4. Solution
    prompt += f"""
Here is an example architecture:\n\n
```
{base}
```\n
{PROBLEM_INSTRUCTION_COT} \n
{cot} \n
```
{kernel}
```\n\n
"""

# show task to solve
    prompt += f"""
Task:\n\n
Here is an example architecture:\n\n
```
{ref_arch_src}
```\n
"""
    prompt += PROBLEM_INSTRUCTION_COT

    return prompt



def prompt_generate_custom_cuda_from_file_one_example(ref_arch_src, example_ind=1):
    """
    Deprecated: use prompt_generate_custom_cuda_from_prompt_template instead
    Keep this around for background compatibility
    NOTE: Anne to clean this up
    Check example_ind for prompt templates
    """
    # arch = get_arch_definition_from_file(arch_path)
    arch = ref_arch_src
    # These are strictly defined for now

    example_arch_path = os.path.join(
        REPO_TOP_PATH, f"src/prompts/model_ex_{example_ind}.py"
    )
    example_new_arch_path = os.path.join(
        REPO_TOP_PATH, f"src/prompts/model_new_ex_{example_ind}.py"
    )

    if not os.path.exists(example_arch_path):
        raise FileNotFoundError(
            f"Example architecture file not found: {example_arch_path}"
        )
    if not os.path.exists(example_new_arch_path):
        raise FileNotFoundError(
            f"Example new architecture file not found: {example_new_arch_path}"
        )

    example_arch = read_file(example_arch_path)
    example_new_arch = read_file(example_new_arch_path)

    return prompt_generate_custom_cuda(arch, example_arch, example_new_arch)


def prompt_generate_custom_cuda_from_prompt_template(ref_arch_src: str) -> str:
    """
    Using prompt example (an element-wise addition) for prompt templates
    The most basic form of example just to show LLM the task and the expected output format
    """
    arch = ref_arch_src
    # These are strictly defined for now

    # path to prompt template, show an example of Model (torch specifications) and ModelNew (torch + custom CUDA kernels)
    example_arch_path = os.path.join(
        REPO_TOP_PATH, f"src/prompts/model_ex_add.py"
    )
    example_new_arch_path = os.path.join(
        REPO_TOP_PATH, f"src/prompts/model_new_ex_add.py"
    )

    if not os.path.exists(example_arch_path):
        raise FileNotFoundError(
            f"Example architecture file not found: {example_arch_path}"
        )
    if not os.path.exists(example_new_arch_path):
        raise FileNotFoundError(
            f"Example new architecture file not found: {example_new_arch_path}"
        )

    example_arch = read_file(example_arch_path)
    example_new_arch = read_file(example_new_arch_path)

    return prompt_generate_custom_cuda(arch, example_arch, example_new_arch)


def prompt_generate_custom_triton_from_template(ref_arch_src: str) -> str:
    """Generate a Triton-focused prompt mirroring the CUDA template flow."""

    prompt = PROBLEM_STATEMENT_TRITON
    example_arch_path = os.path.join(
        REPO_TOP_PATH, "src/prompts/model_ex_add.py"
    )
    example_triton_path = os.path.join(
        REPO_TOP_PATH, "src/prompts/model_new_ex_add_triton.py"
    )
    if os.path.exists(example_arch_path) and os.path.exists(example_triton_path):
        example_arch = read_file(example_arch_path)
        example_triton = read_file(example_triton_path)
        prompt += "\nHere is an example that meets the contract:\n```\n"
        prompt += example_arch
        prompt += "\n```\nbecomes\n```\n"
        prompt += example_triton
        prompt += "\n```\n"
    prompt += "\nYou are given the following architecture:\n\n```\n"
    prompt += ref_arch_src
    prompt += "\n```\n"
    prompt += PROBLEM_INSTRUCTION_TRITON
    prompt += "\nEnsure ModelNew calls the Triton kernels and returns the same tensor shapes as the reference Model."
    return prompt


def prompt_generate_prompt_with_hardware_info_from_template(ref_arch_src: str, gpu_name: str) -> str:
    """
    Similar to prompt_generate_custom_cuda_from_prompt_template, 
    but with hardware information for the given GPU
    """

    arch = ref_arch_src
    # These are strictly defined for now

    # path to prompt template, show an example of Model (torch specifications) and ModelNew (torch + custom CUDA kernels)
    example_arch_path = os.path.join(
        REPO_TOP_PATH, f"src/prompts/model_ex_add.py"
    )
    example_new_arch_path = os.path.join(
        REPO_TOP_PATH, f"src/prompts/model_new_ex_add.py"
    )

    gpu_spec_file_path = os.path.join(REPO_TOP_PATH, f"src/prompts/hardware/gpu_specs.py")

    example_arch = read_file(example_arch_path)
    example_new_arch = read_file(example_new_arch_path)
    gpu_spec_info = read_file(gpu_spec_file_path)

    return prompt_generate_prompt_with_hardware_info(
                                        ref_arch_src=arch, 
                                        gpu_name=gpu_name, 
                                        example_arch_src=example_arch, 
                                        example_new_arch_src=example_new_arch, 
                                        gpu_spec_info_src=gpu_spec_info
                                        )
    


def prompt_generate_prompt_with_hardware_info(ref_arch_src: str, 
                                              gpu_name: str, 
                                              example_arch_src: str, 
                                              example_new_arch_src: str, 
                                              gpu_spec_info_src: str) -> str:
    """
    Generate a prompt with hardware information for the given GPU
    gpu_spec_info_src: str of the gpu spec src file
    """

    # Create a dictionary to store the local namespace
    local_dict = {}
    
    # Execute the GPU spec file in the local namespace
    exec(gpu_spec_info_src, {}, local_dict)
    
    # Get the required variables from the local namespace
    GPU_SPEC_INFO = local_dict.get('GPU_SPEC_INFO')
    GPU_DEFINITIONS = local_dict.get('GPU_DEFINITIONS')
    GPU_BEST_PRACTICES = local_dict.get('GPU_BEST_PRACTICES')
    
    if not GPU_SPEC_INFO or not GPU_DEFINITIONS or not GPU_BEST_PRACTICES:
        raise ValueError("GPU_SPEC_INFO or GPU_DEFINITIONS or GPU_BEST_PRACTICES not found in gpu_spec_info_src")

    assert gpu_name in GPU_SPEC_INFO, f"GPU name {gpu_name} not found in GPU_SPEC_INFO"

    prompt = PROBLEM_STATEMENT

    if example_arch_src != "" and example_new_arch_src != "":
        prompt += f"""
        Here's an example to show you the syntax of inline embedding custom CUDA operators in torch: The example given architecture is: \n
        ``` \n
        {example_arch_src}
        ``` \n
        The example new arch with custom CUDA kernels looks like this: 
        ```
        {example_new_arch_src}
        ``` \n
        """
    
    curr_gpu_spec_info = GPU_SPEC_INFO[gpu_name]

    gpu_architecture = curr_gpu_spec_info.get("GPU Architecture")
    prompt += f"""
    Here is some information about the underlying hardware that you should keep in mind. \n\n
The GPU that will run the kernel is NVIDIA {gpu_name}, {gpu_architecture} architecture.\n\n"""
    
    for key, value in curr_gpu_spec_info.items():
        if key == "GPU Architecture":
            continue
        prompt += f"""- We have {value} of {key}.\n"""
    
    
    prompt += f"""\n\n
Here are some concepts about the GPU architecture that could be helpful: \n\n"""
    for key, value in GPU_DEFINITIONS.items():
        prompt += f"""- {key}: {value}\n"""

    prompt += f"""\n\n
Here are some best practices for writing CUDA kernels on GPU: \n\n"""
    for best_practice in GPU_BEST_PRACTICES:
        prompt += f"""- {best_practice}\n"""


    prompt += f"""
    You are given the following architecture: \n
    ```
    {ref_arch_src}
    ```
    """
    

    prompt += PROBLEM_INSTRUCTION
    return prompt


FORMATTER_SYSTEM_PROMPT = (
    "You are a GPU kernel formatting assistant. Always output a single runnable "
    "KernelBench solution as a python fenced code block that obeys the ModelNew contract."
)


def build_formatter_messages(
    language: str,
    reference_architecture: str,
    original_prompt: str,
    raw_completion: str | None,
) -> list[dict[str, str]]:
    """Construct messages for a formatting LLM that enforces KernelBench contracts."""
    contract = PROBLEM_INSTRUCTION if language == "cuda" else PROBLEM_INSTRUCTION_TRITON
    completion_text = raw_completion or ""
    user_content = f"""Rewrite the raw completion so it satisfies the KernelBench contract.

Contract requirements:
{contract.strip()}

Reference architecture:
```
{reference_architecture.strip()}
```

Original prompt:
```
{original_prompt.strip()}
```

Raw completion:
```
{completion_text.strip()}
```

Output exactly one ```python fenced code block that implements ModelNew per the contract.
Do not add commentary, tests, or extra markdown."""
    return [
        {"role": "system", "content": FORMATTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def prompt_fix_compile(ref_arch_src, custom_cuda, metadata):
    prompt = PROBLEM_STATEMENT
    prompt += f"""
    With the following architecture:
    ```
    {ref_arch_src}
    ```
    You generated the following solution and it failed to compile:
    ```
    {custom_cuda}
    ```
    Here's the metadata of the compilation error:
    ```
    {metadata}
    ```
    
    Please fix the compilation error in the new model code. Please output the corrected code in codeblocks.
    """
    return prompt


def prompt_fix_correctness(ref_arch_src, custom_cuda, metadata):
    prompt = PROBLEM_STATEMENT
    prompt += f"""
    With the following architecture:
    ```
    {ref_arch_src}
    ```
    You generated the following solution and it failed correctness:
    ```
    {custom_cuda}
    ```
    Here's the metadata of the correctness error:
    ```
    {metadata}
    ```
    Please consider how your custom CUDA kernels are implemented, how it is different from the reference implementation, and fix the correctness error in the new model code. Please output the corrected code in codeblocks.
    """
    return prompt

def main():
    gpu_name = "L40S"


    ref_arch_src = read_file(os.path.join(KERNEL_BENCH_PATH, f"level1/19_ReLU.py"))
    assert len(ref_arch_src) > 0, "ref_arch_src is empty"
    prompt = prompt_generate_prompt_with_hardware_info_from_template(ref_arch_src, gpu_name)
    print(prompt)
    # Write prompt to temp file
    temp_file_path = os.path.join(REPO_TOP_PATH, "scratch", "prompt_draft.txt")
    os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
    with open(temp_file_path, "w") as f:
        f.write(prompt)

if __name__ == "__main__":
    main()
