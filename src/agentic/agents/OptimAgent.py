import os
import json
from typing import Any, Dict, Iterable, List

from tqdm.auto import tqdm

from agentic.agents.reflexion_oneshot import Reflexion_Oneshot
from memories.Memory import MemoryClassMeta
from prompts import prompt_for_generation, prompt_for_reflection
from llm_apis.models.Base import BaseModel

try:
    from agentic.utils.utils import clear_code, extract_function_signatures, clear_json
    from agentic.dataloaders.ProblemState import ProblemState
except ImportError:  # pragma: no cover
    from utils.utils import clear_code, extract_function_signatures, clear_json
    from dataloaders.ProblemState import ProblemState


def _truncate_lines(lines: List[str], head: int = 3, tail: int = 3) -> List[str]:
    if len(lines) <= head + tail:
        return lines
    return lines[:head] + ["..."] + lines[-tail:]


def _format_messages(messages: Iterable[Dict[str, Any]]) -> str:
    lines = ["messages:"]
    for idx, message in enumerate(messages, start=1):
        role = message.get("role")
        lines.append(f"  [{idx}] role: {role}")
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text") or str(part)
                else:
                    text = str(part)
                part_lines = text.splitlines() or [""]
                lines.extend([f"    {line}" for line in _truncate_lines(part_lines)])
        else:
            text = str(content)
            content_lines = text.splitlines() or [""]
            lines.extend([f"    {line}" for line in _truncate_lines(content_lines)])
    return "\n".join(lines)


def _format_kwargs(kwargs: Dict[str, Any] | None) -> str:
    if not kwargs:
        return ""
    lines = ["kwargs:"]
    for key in sorted(kwargs):
        lines.append(f"  {key}: {kwargs[key]}")
    return "\n".join(lines)


def _stringify_response(response: Any) -> str:
    if isinstance(response, str):
        return "\n".join(_truncate_lines(response.splitlines()))
    if isinstance(response, dict):
        lines = []
        for key, value in response.items():
            lines.append(f"{key}: {value}")
        return "\n".join(_truncate_lines(lines))
    return str(response)


def _print_error(message: str) -> None:
    banner = "=" * 60
    print(f"\n{banner}\n!!! ERROR DETECTED !!!\n{message}\n{banner}\n")


def _record_history(mem, iteration: int, stage: str, payload: Dict[str, Any]) -> None:
    history = getattr(mem, "history", None)
    if history is None:
        history = []
        setattr(mem, "history", history)
    entry = {"iteration": iteration, "stage": stage}
    entry.update(payload)
    history.append(entry)


class OptimAgent(Reflexion_Oneshot):
    def __init__(self, model, dataset, corpus_path, max_perf_debug_num=5, mem_file=None):
        super().__init__(model, dataset, corpus_path, mem_file)
        self.max_perf_debug_num = max_perf_debug_num
        self._active_iteration = 0
        self._iteration_count = 0
        self.progress_desc = "Agentic Iterations"

    def memory_init(self, mem_file=None):
        """
        Args:
            mem_file: previous stored memories, which can be loaded to continue run
        """
        class Memory(metaclass=MemoryClassMeta, field_names=["ps", 
                                                             "call_err_msg", 
                                                             "exe_err_msg",
                                                             "reflection", 
                                                             "function_signatures", 
                                                             "oneshot", 
                                                             "perf_candidates",
                                                             "perf_strategy",
                                                             "raw_code",
                                                             "call_candidate",
                                                             "exe_candidate",
                                                             "perf_debug_num",
                                                             "pass_call", 
                                                             "pass_exe",
                                                             "pass_perf"]):
            pass
        
        if mem_file is not None:
            assert mem_file.endswith(".json"), f"expect a json file, but got {mem_file} instead"
            with open(mem_file, "r") as f:
                input_mems = json.load(f)
            assert len(input_mems) == len(self.dataset), f"expect {len(self.dataset)} samples, but got {len(input_mems)} instead"

        for ps in self.dataset.problem_states:
            if ps.label:
                fs_mem = extract_function_signatures(ps.label)
            else:
                fs_mem = None
            raw_code = [ps.solution] if ps.solution else [""]
            if mem_file is None:
                os_mem = self.instruction_retriever.query(ps.instruction)[0]
                tmp_mem = Memory(ps=ps, 
                                call_err_msg=None,
                                exe_err_msg=None, 
                                reflection=None, 
                                function_signatures=fs_mem, 
                                oneshot=os_mem["code"], 
                                perf_candidates=[],
                                perf_strategy=None,
                                raw_code=raw_code,
                                call_candidate=None,
                                exe_candidate=None,
                                perf_debug_num=0,
                                pass_call=False,
                                pass_exe=False,
                                pass_perf=False,
                                )
            else:
                input_mem = input_mems[ps.filename]
                tmp_mem = Memory(
                    ps=ps,
                    call_err_msg=input_mem["call_err_msg"],
                    exe_err_msg=input_mem["exe_err_msg"], 
                    reflection=input_mem["reflection"], 
                    function_signatures=fs_mem, 
                    oneshot=input_mem["oneshot"], 
                    perf_candidates=input_mem["perf_candidates"],
                    perf_strategy=input_mem["perf_strategy"],
                    raw_code=raw_code,
                    call_candidate=input_mem["call_candidate"],
                    exe_candidate=input_mem["exe_candidate"],
                    perf_debug_num=input_mem["perf_debug_num"],
                    pass_call=input_mem["pass_call"],
                    pass_exe=input_mem["pass_exe"],
                    pass_perf=input_mem["pass_perf"],
                )

            self.memories.append(tmp_mem)
            setattr(tmp_mem, "history", [])
    
    def write_memories(self, file_path):
        output_dict = {}
        with open(file_path, "w") as f:
            for mem in self.memories:
                output = {
                    "call_err_msg": str(mem.call_err_msg),
                    "exe_err_msg": str(mem.exe_err_msg),
                    "reflection": mem.reflection, 
                    "oneshot": mem.oneshot, 
                    "perf_candidates": [list(cand) for cand in mem.perf_candidates],
                    "perf_strategy": mem.perf_strategy,
                    "call_candidate": mem.call_candidate,
                    "exe_candidate": mem.exe_candidate,
                    "perf_debug_num": mem.perf_debug_num,
                    "pass_call": mem.pass_call, 
                    "pass_exe": mem.pass_exe,
                    "pass_perf": mem.pass_perf,
                    "ms": mem.ms if hasattr(mem, 'ms') else None,
                    "efficiency": mem.efficiency if hasattr(mem, 'efficiency') else None
                }
                output_dict[mem.ps.filename] = output
            json.dump(output_dict, f)
    
    def run(self, output_path=None, multi_thread=True, datalen=None, iteration_num=0, temperature=0, ancestor_num=2, start_idx=0, gpu_id=0, start_iter=0):
        """
        Args:
            output_path: the folder to store the final result
            multi_thread: whether use multithreading for generating
            datalen: for debug, to specify how many data from the dataset you want to use
            iteration_num: how many iterations you want to run
            temperature: LLM temperature
            ancestor_num: how many samples you want to add in the prompt when optimize the code
            start_idx: start idx of the data rows
            gpu_id: which gpu you want to use when you test the scripts
            start_iter: which iteration you want to start with. useful when you load previous result and memory
        """
        assert ancestor_num >= 0, f"expect ancestor_num to be larger than 0, but got {ancestor_num}"
        data_len = datalen if datalen else len(self.dataset)
        # force sequential execution so logs are easier to follow
        multi_thread = False
        self._iteration_count = iteration_num
        iteration_range = range(start_iter, start_iter + iteration_num)
        with tqdm(
            total=iteration_num,
            desc=self.progress_desc,
            unit="iter",
            leave=False,
            dynamic_ncols=True,
        ) as iter_bar:
            for iter in iteration_range:
                self._active_iteration = iter
                if output_path is not None:
                    root, extension = os.path.splitext(output_path)
                    iter_path = f"{root}_{iter}{extension}"
                    mem_output_path = f"{root}_mem_{iter}.json"
                with tqdm(
                    total=data_len,
                    desc="Solutions",
                    unit="kernel",
                    leave=False,
                    dynamic_ncols=True,
                ) as solution_bar:
                    for mem in self.memories[start_idx:(start_idx + data_len)]:
                        self.generate_solution(mem, temperature=temperature)
                        solution_bar.update(1)

            if output_path is None or (hasattr(self.dataset, 'rocm_tests') and self.dataset.rocm_tests):
                tmp_dir = "tmp"
                exe_dir = "pass_exe"
                perf_result_dir = "perf_results"
                perf_log_dir = "perf_logs"
            else:
                root, extension = os.path.splitext(output_path)
                tmp_dir = f"{root}_tmp"
                exe_dir = f"{root}_pass_exe"
                perf_result_dir = f"{root}_perf_results"
                perf_log_dir = f"{root}_perf_logs"
            with tqdm(
                total=data_len,
                desc="Execution",
                unit="kernel",
                leave=False,
                dynamic_ncols=True,
            ) as execution_bar:
                for mem in self.memories[start_idx:(start_idx + data_len)]:
                    try:
                        (
                            pass_call,
                            pass_exe,
                            call_stdout,
                            call_stderr,
                            exe_stdout,
                            exe_stderr,
                        ) = self.dataset.test_opt_correctness(mem.raw_code[0], mem.ps.filename, tmp_dir, exe_dir=exe_dir)
                    except Exception as e:
                        _print_error(f"failed to test the code for {mem.ps.filename}: {e}")
                        mem.call_err_msg = f"failed to test the code due to: {e}"
                        mem.exe_err_msg = f"failed to test the code due to: {e}"
                        _record_history(
                            mem,
                            self._active_iteration,
                            "execution",
                            {
                                "call_pass": False,
                                "exe_pass": False,
                                "error": str(e),
                            },
                        )
                        execution_bar.update(1)
                        continue

                    if not pass_call:
                        mem.call_err_msg = call_stderr
                        mem.exe_err_msg = exe_stderr
                    elif pass_call and not pass_exe:
                        mem.pass_call = True
                        mem.exe_err_msg = None if exe_stderr == "None" else exe_stderr
                        mem.call_candidate = mem.raw_code[0]
                    else:
                        mem.pass_call = True
                        mem.pass_exe = True
                        mem.exe_candidate = mem.raw_code[0]

                    _record_history(
                        mem,
                        self._active_iteration,
                        "execution",
                        {
                            "call_pass": pass_call,
                            "exe_pass": pass_exe,
                            "call_stdout": call_stdout,
                            "call_stderr": call_stderr,
                            "exe_stdout": exe_stdout,
                            "exe_stderr": exe_stderr,
                        },
                    )
                    execution_bar.update(1)

            if os.path.isdir(exe_dir) and os.listdir(exe_dir):
                perf_results_dict = {}

                if hasattr(self.dataset, 'rocm_tests') and self.dataset.rocm_tests:
                    perf_results_dict = self.dataset.run_perf_evaluation(
                        exec_folder=exe_dir,
                        gen_perf_folder=perf_result_dir,
                    )
                else:
                    script_dir = os.path.join(tmp_dir, "perf_gen")
                    self.dataset.write_perf_file(
                        input_folder_path=exe_dir,
                        results_path=perf_result_dir,
                        tmp_dir=script_dir,
                    )
                    self.dataset.run_perf_scripts(
                        gpu_id=gpu_id,
                        script_dir=script_dir,
                        log_dir=perf_log_dir,
                    )

                for mem in self.memories[start_idx:(start_idx + data_len)]:
                    if not mem.pass_exe:
                        continue

                    ms = None
                    efficiency = None

                    if hasattr(self.dataset, 'rocm_tests') and self.dataset.rocm_tests:
                        passed_mems = [mem for mem in self.memories[start_idx:(start_idx + data_len)] if mem.pass_exe]
                        perf_results_list = list(perf_results_dict.values())
                        if len(passed_mems) == len(perf_results_list):
                            for mem_item, perf_data in zip(passed_mems, perf_results_list):
                                ms = perf_data.get("ms")
                                efficiency = perf_data.get("efficiency")
                                if ms is not None and efficiency is not None:
                                    mem_item.pass_perf = True
                                    mem_item.raw_code.extend([ms, efficiency])
                                    mem_item.ms = ms
                                    mem_item.efficiency = efficiency
                                    _record_history(
                                        mem_item,
                                        self._active_iteration,
                                        "performance",
                                        {
                                            "pass_perf": True,
                                            "latency_ms": ms,
                                            "efficiency": efficiency,
                                        },
                                    )
                                else:
                                    mem_item.pass_perf = False
                                    mem_item.ms = None
                                    mem_item.efficiency = None
                                    _record_history(
                                        mem_item,
                                        self._active_iteration,
                                        "performance",
                                        {
                                            "pass_perf": False,
                                            "details": "incomplete performance data",
                                        },
                                    )
                    else:
                        path_gen = os.path.join(perf_result_dir, mem.ps.filename[:-3] + ".json")
                        if not os.path.exists(path_gen):
                            _record_history(
                                mem,
                                self._active_iteration,
                                "performance",
                                {
                                    "pass_perf": False,
                                    "details": "performance artifact missing",
                                },
                            )
                            continue
                        try:
                            _, efficiency, ms = self.dataset.calculate(path_gen, path_ref=None)
                            mem.pass_perf = True
                            mem.ms = ms
                            mem.efficiency = efficiency
                            mem.raw_code.extend([ms, efficiency])
                            _record_history(
                                mem,
                                self._active_iteration,
                                "performance",
                                {
                                    "pass_perf": True,
                                    "latency_ms": ms,
                                    "efficiency": efficiency,
                                },
                            )
                        except Exception as e:
                            _print_error(f"TritonBench performance calculation failed for {mem.ps.filename}: {e}")
                            mem.pass_perf = False
                            _record_history(
                                mem,
                                self._active_iteration,
                                "performance",
                                {
                                    "pass_perf": False,
                                    "error": str(e),
                                },
                            )
                            continue

            with tqdm(
                total=data_len,
                desc="Reflections",
                unit="kernel",
                leave=False,
                dynamic_ncols=True,
            ) as reflection_bar:
                for mem in self.memories[start_idx:(start_idx + data_len)]:
                    self.generate_reflexion(mem, temperature=temperature)
                    reflection_bar.update(1)

            for mem in self.memories[start_idx:(start_idx + data_len)]:
                if not mem.pass_perf:
                    continue
                if len(mem.perf_candidates) < ancestor_num:
                    mem.raw_code.append(mem.reflection)
                    if len(mem.raw_code) < 4:
                        mem.pass_perf = False
                        continue
                    mem.perf_candidates.append(tuple(mem.raw_code))
                    mem.perf_candidates = sorted(mem.perf_candidates, key=lambda x: x[1], reverse=True)
                elif mem.perf_candidates[0][1] > mem.raw_code[1]:
                    mem.raw_code.append(mem.reflection)
                    mem.perf_candidates[0] = tuple(mem.raw_code)
                    mem.perf_candidates = sorted(mem.perf_candidates, key=lambda x: x[1], reverse=True)

            for mem in self.memories[start_idx:(start_idx + data_len)]:
                if len(mem.perf_candidates) > 0:
                    mem.ps.solution = mem.perf_candidates[-1][0]
                elif mem.exe_candidate is not None:
                    mem.ps.solution = mem.exe_candidate
                elif mem.call_candidate is not None:
                    mem.ps.solution = mem.call_candidate
                else:
                    mem.ps.solution = mem.raw_code[0]

            if output_path is not None:
                self.dataset.write_file(iter_path, start_idx=start_idx, datalen=data_len)
                self.write_memories(mem_output_path)

            os.system(f'rm -rf {exe_dir}')
            os.system(f'rm -rf {perf_result_dir}')
            os.system(f'rm -rf {perf_log_dir}')

            iter_bar.update(1)
    
    def generate_solution(self, mem, temperature=0):

        tab = "\n"
        fss_text = "".join(f"* {sig}{tab}" for sig in mem.function_signatures)
        text = prompt_for_generation.prompt.format(
            instruction=mem.ps.instruction,
            function_signatures=fss_text
        )

        # for the one that has perf_candidates, and the code generated in this round pass_exe, we need to generate a new code
        # for the one that has perf_candidates, but the code generated in this round not pass_exe, if the debug_num has exceeds the man_debug_num, then generate a new code
        # otherwise, go to debug
        if len(mem.perf_candidates) > 0 and (mem.pass_exe or (not mem.pass_exe and mem.perf_debug_num >= self.max_perf_debug_num)):
            mem.perf_debug_num = 0

            text += """There are some reference codes(NO.1, NO.2 and so on). The reference codes are arranged in ascending order based on their performance, where lower latencies and higher efficiencies indicate better performance. According to their performance(latency in ms and efficiency in TFLOPS or GB/s) and the corresponding analysis, you need to generate a new code with better performance. You should maintain code correctness during optimization."""

            text +="\nYou can use optimization strategies such as Memory access efficiency, Hardware resource utilization, IR analysis, Assembly analysis, Kernel occupancy, TorchInductor with Triton tuning knobs and Auto-tunable kernel configurations and environment variables."

            for i, cand in enumerate(mem.perf_candidates):
                text += f"\nreference code: {cand[0]}"
                text += f"\nOriginal latency(ms): {cand[1]}"
                text += f"\noriginal efficiency(TFLOPS, GB/s): {cand[2]}"
                text += f"\nAnalysis: {cand[3]}"
            
            text += "\nAnalyze and compare all optimization strategies based on correct codes and give a better strategy motivated by them. Generate a better optimization code based on the better strategy ."
            text += "\nThink before writing the optimization and no more explanation is required after the thinking."
            text += "\nYou should not suggest changes to the name of the function and parameter names, counts, or order."
        else:
            if not mem.raw_code or mem.raw_code[0] == "":
                text += f"\nHere is an example snippet of code: {mem.oneshot}"
            else:
                one_shot = self.code_retriever.query(mem.raw_code[0])[0]["code"]
                text += f"\nHere is an example snippet of code: {one_shot}"
                text += f"\nPrevious attempt implementation:{mem.raw_code[0]}"
                
                if not mem.pass_call:
                    text += f"\nTest messages for previous attempt:{mem.call_err_msg}"
                    text += f"\nTest messages for correctness check of previous attempt:{mem.exe_err_msg}"
                
                elif not mem.pass_exe:
                    text += "\nThe previous attempt implementation can be run successfully."
                    text += f"\nTest messages for correctness check of previous attempt:{mem.exe_err_msg}"
                
                if len(mem.perf_candidates) > 0:
                    mem.perf_debug_num += 1
            
            
            if mem.reflection:
                text += f"\nReflection on previous attempt:{mem.reflection}"

        text += "\nOutput your answer in json format, with the format as follows: {\"thought\": \"\", \"code\": \"\"}. Please strictly output in JSON format."
        text += "\nGenerate the correct and optimized code without explanation, which we can run directly in the \"code\" field."

        msg = [
            {"role": "user", "content": text},
        ]

        kwargs = {"temperature": temperature, "max_tokens": 15000}
        formatted_messages = _format_messages(msg)
        formatted_kwargs = _format_kwargs(kwargs)
        response_text = ""
        response_obj: Any
        error_text: str | None = None

        try:
            response_obj = self.model.generate(msg, temperature=temperature, max_tokens=15000)
        except Exception as err:
            error_text = str(err)
            _print_error(f"failed to call LLM for {mem.ps.filename}: {error_text}")
            response_obj = {"code": ""}

        response_text = _stringify_response(response_obj)

        try:
            parsed = clear_json(response_text)
            mem.raw_code = [clear_code(parsed["code"])]
        except Exception:
            _print_error(f"failed to extract code for {mem.ps.filename}")
            raw_code = response_text.split("\"code\":")[-1]
            raw_code = raw_code.split("}")[0]
            mem.raw_code = [clear_code(raw_code)]
        # finally:
        
        if mem.raw_code[0] is None or mem.raw_code is None:
            print(f"raw code for {mem.ps.filename} is None")
            mem.raw_code = [""]

        mem.pass_call = False
        mem.pass_exe = False
        mem.pass_perf = False

        history_payload = {
            "request_messages": msg,
            "request_text": formatted_messages,
            "kwargs": kwargs,
            "kwargs_text": formatted_kwargs,
            "response_text": response_text,
        }
        if error_text:
            history_payload["error"] = error_text
        _record_history(mem, self._active_iteration, "solution", history_payload)

        return
    
    def generate_reflexion(self, mem, temperature):
        if mem.pass_perf:
            reflect_txt = prompt_for_reflection.prompt_ga.format(
                problem=mem.ps.instruction,
                code=mem.raw_code[0],
                latency=mem.raw_code[1],
                efficiency=mem.raw_code[2]
            )
        elif mem.pass_call and mem.pass_exe:
            reflect_txt = prompt_for_reflection.prompt_ga.format(
                problem=mem.ps.instruction,
                code=mem.raw_code[0],
                latency="",
                efficiency=""
            )
        elif mem.pass_call:
            reflect_txt = prompt_for_reflection.prompt_exe.format(
                problem=mem.ps.instruction,
                solution=mem.raw_code[0],
                call_test_result="succeed",
                exe_test_result=mem.exe_err_msg
            )
        else:
            reflect_txt = prompt_for_reflection.prompt.format(
                problem=mem.ps.instruction,
                solution=mem.raw_code[0],
                test_result=mem.call_err_msg
            )
        
        reflect_msg = [
            {
                "role": "user",
                "content": reflect_txt
            }
        ]
        kwargs = {"temperature": temperature}
        formatted_messages = _format_messages(reflect_msg)
        formatted_kwargs = _format_kwargs(kwargs)
        response_obj: Any
        error_text: str | None = None

        try:
            response_obj = self.model.generate(reflect_msg, temperature=temperature)
        except Exception as err:
            error_text = str(err)
            _print_error(f"failed to call reflection LLM for {mem.ps.filename}: {error_text}")
            response_obj = ""

        response_text = _stringify_response(response_obj)

        mem.reflection = response_obj
        history_payload = {
            "request_messages": reflect_msg,
            "request_text": formatted_messages,
            "kwargs": kwargs,
            "kwargs_text": formatted_kwargs,
            "response_text": response_text,
        }
        if error_text:
            history_payload["error"] = error_text
        _record_history(mem, self._active_iteration, "reflection", history_payload)
