import math
from typing import List, Tuple, Optional
import re

import time 
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import ts_downsample, plot_ts_with_anomalies, extract_anomalies, map_anomalies_to_original

class ChatTSAnalyzer:
    """
    ChatTS-14B 时序分析器（推理版）
    - 单卡推理（推荐 4-bit 量化以控制显存）
    - 支持长序列按滑窗推理并合并
    - 仅使用模型支持的生成参数（去掉 temperature/top_k）

    依赖:
      pip install "transformers>=4.43" accelerate bitsandbytes torch --extra-index-url https://download.pytorch.org/whl/cu118
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        load_in_4bit: bool = True,
        attn_implementation: str = "eager",   # 2080Ti/V100 等 pre-Ampere 建议用 eager
        torch_dtype: torch.dtype = torch.float16,
    ):
        """
        Args:
            model_path: 本地或HF路径，比如 "/home/data1/llm_models/bytedance-research/ChatTS-14B"
            device:    统一放到同一张卡上，例如 "cuda:0"
            load_in_4bit: 是否使用 bitsandbytes 4-bit 量化
            attn_implementation: 'eager' / 'sdpa' / 'flash_attention_2'（老卡用 'eager'）
            torch_dtype: 建议 fp16（2080Ti 不支持 bfloat16）
        """
        self.model_path = model_path
        self.device = torch.device(device)
        self.compute_dtype = torch_dtype  # 保存计算 dtype 用于输入转换
        
        m = re.match(r"cuda:(\d+)", device)
        device_index = int(m.group(1)) if m else 0

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            
            # 4bit 情况：直接让 HF 把模型放到指定卡上，别再 .to()
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                quantization_config=bnb_config,
                low_cpu_mem_usage=True,
                attn_implementation=attn_implementation,
                device_map={"": device_index},   # 👈 整模型在指定 index 上
            )
        else:
            # 非量化：正常 from_pretrained + .to()
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                quantization_config=bnb_config,
                low_cpu_mem_usage=True,
                attn_implementation=attn_implementation,
            ).to(self.device)
        

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, tokenizer=self.tokenizer
        )

        # 某些分支可能没有 pad_token_id，兜底到 eos
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

    # ------------------
    # 内部工具
    # ------------------
    def _build_prompt(
        self,
        timeseries_len: int,
        system_prompt: str,
        task_prompt_tpl: str,
    ) -> str:
        user_prompt = task_prompt_tpl.format(ts_len=timeseries_len)
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>"
            f"<|im_start|>user\n{user_prompt}<|im_end|><|im_start|>assistant\n"
        )
        return prompt

    def _prepare_inputs(
        self,
        prompt: str,
        timeseries: np.ndarray,
    ):
        # processor 会为 ChatTS 同时处理 text & timeseries
        inputs = self.processor(
            text=[prompt],
            timeseries=[timeseries],
            padding=True,
            return_tensors="pt",
        )

        # 使用初始化时保存的计算 dtype
        # 对于量化模型，这是 bnb_4bit_compute_dtype；对于非量化模型，这是 torch_dtype
        model_dtype = self.compute_dtype

        # 把所有张量移到同一设备，并且：
        #    - 只对"浮点张量"转换 dtype（如 timeseries 相关的张量）
        #    - 保留 input_ids / attention_mask 这些整型不动
        for k, v in inputs.items():
            if torch.is_tensor(v):
                v = v.to(self.device)
                if v.is_floating_point():
                    v = v.to(model_dtype)
                inputs[k] = v

        return inputs

    

    def _generate(
        self,
        inputs,
        max_new_tokens: int = 1024,
        top_p: float = 0.9,
    ) -> str:
        """
        仅传递模型支持的生成参数：
          - 去掉 temperature、top_k（之前已被模型忽略并警告）
          - 只保留 top_p & do_sample/use_cache/max_new_tokens
        """
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
        # 当 top_p < 1.0 时启用采样；=1.0 时走贪心
        if top_p < 1.0:
            gen_kwargs.update(dict(do_sample=True, top_p=top_p))
        else:
            gen_kwargs.update(dict(do_sample=False))

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # 切除前缀 prompt tokens，得到干净回答
        text = self.tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True
        )
        return text.strip()

    def _run_one_window(
        self,
        timeseries: np.ndarray,
        max_new_tokens: int,
        top_p: float,
        system_prompt: str,
        task_prompt_tpl: str,
    ) -> str:
        prompt = self._build_prompt(
            timeseries_len=len(timeseries),
            system_prompt=system_prompt,
            task_prompt_tpl=task_prompt_tpl,
        )
        inputs = self._prepare_inputs(prompt, timeseries)
        return self._generate(inputs, max_new_tokens=max_new_tokens, top_p=top_p)

    @staticmethod
    def _make_windows(
        n: int,
        window_len: int,
        overlap: float,
    ) -> List[Tuple[int, int]]:
        """
        返回一组 [start, end) 索引窗口。
        overlap: 0 ~ <1 ，例如 0.25 表示每窗重叠 25%
        """
        assert 0 <= overlap < 1, "overlap 需在 [0,1) 之间"
        if n <= window_len:
            return [(0, n)]
        stride = max(1, int(window_len * (1 - overlap)))
        starts = list(range(0, max(1, n - window_len + 1), stride))
        if starts[-1] + window_len < n:
            starts.append(n - window_len)
        return [(s, min(n, s + window_len)) for s in starts]

    # ------------------
    # 对外主入口
    # ------------------
    def analyze(
        self,
        timeseries: np.ndarray,
        max_new_tokens: int = 1024,
        window_len: Optional[int] = None,
        overlap: float = 0.25,
        per_window_new_tokens: Optional[int] = None,
        top_p: float = 1,
        system_prompt: str = "You are a helpful assistant.",
        task_prompt_tpl: str = (
            "I have a time series length of {ts_len}: <ts><ts/>. "
            "Please analyze the local changes in this time series."
        ),
        clear_cuda_cache_each_window: bool = False,
        header_each_window: bool = True,
    ) -> str:
        """
        Args:
            timeseries: 一维 numpy 数组
            max_new_tokens: 单窗/整段的最大生成长度（过大容易 OOM，建议 512~2048）
            window_len: 若为 None 或 len(ts) <= window_len，则整段推理；否则滑窗
            overlap: 滑窗重叠比例（0~<1）
            per_window_new_tokens: 每个窗口单独的 max_new_tokens（默认自动按总上限分配）
            top_p: nucleus sampling；=1 时不采样（贪心）
            clear_cuda_cache_each_window: 每窗后清理缓存以减少碎片
            header_each_window: 输出里给每个窗加一个头部行，标注区间与序号
        """
        assert timeseries.ndim == 1, "timeseries 需要是一维数组"

        # 情况1：整段直接跑
        if window_len is None or len(timeseries) <= window_len:
            return self._run_one_window(
                timeseries=timeseries,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                system_prompt=system_prompt,
                task_prompt_tpl=task_prompt_tpl,
            )

        # 情况2：滑窗
        windows = self._make_windows(n=len(timeseries), window_len=window_len, overlap=overlap)
        num_windows = len(windows)

        # 默认给每个窗分配一个较合理的输出上限
        pnt = per_window_new_tokens or max(
            128, min(1024, max_new_tokens // max(1, num_windows))
        )

        pieces: List[str] = []
        for i, (s, e) in enumerate(windows, 1):
            seg = timeseries[s:e]
            try:
                txt = self._run_one_window(
                    timeseries=seg,
                    max_new_tokens=pnt,
                    top_p=top_p,
                    system_prompt=system_prompt,
                    task_prompt_tpl=task_prompt_tpl,
                )
            except torch.cuda.OutOfMemoryError:
                # 简单的退避策略：尝试把该窗的max_new_tokens减半再跑一次
                torch.cuda.empty_cache()
                fallback_tokens = max(64, pnt // 2)
                txt = self._run_one_window(
                    timeseries=seg,
                    max_new_tokens=fallback_tokens,
                    top_p=top_p,
                    system_prompt=system_prompt,
                    task_prompt_tpl=task_prompt_tpl,
                )

            if header_each_window:
                pieces.append(f"[Window {i}/{num_windows}: {s}-{e}]\n{txt}")
            else:
                pieces.append(txt)

            if clear_cuda_cache_each_window:
                torch.cuda.empty_cache()

        # 简单拼接（需要更强合并可再加“一次总结合并”步骤）
        return "\n\n".join(pieces)

# ------------------
# 使用示例
# ------------------
if __name__ == "__main__":
    import os

    # 可选：减少碎片（新版本 PyTorch 支持）
    # os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # 准备数据
    # ts = np.random.randn(400_000).astype(np.float32)
    
    # 使用脚本所在目录作为基准路径，避免工作目录问题
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'Data', 'PI_20412.PV.csv')
    df = pd.read_csv(csv_path, index_col=0)
    
    n =768
    downsampled_data, time_index, position_index = ts_downsample(df['PI_20412.PV'],  n_out=n)
    downsampled_ts = downsampled_data.values
        
    analyzer = ChatTSAnalyzer(
        model_path="/home/data1/llm_models/bytedance-research/ChatTS-14B",
        device="cuda:1",
        load_in_4bit=True,           # 22GB 显存建议 True
        attn_implementation="eager", # 2080Ti 建议 eager
        # torch_dtype=torch.float16,
        torch_dtype=torch.bfloat16
    )
    
    prompt = (
    "我有一个长度为 {ts_len} 的时间序列：<ts><ts/>。"
    "请识别该时间序列中所有异常或异常片段。"
    "对于每个异常，请描述以下内容：\n"
    "- 异常发生的索引区间（闭区间，起止索引均为整数）\n"
    "- 偏离的幅度或模式特征（保留两位小数）\n"
    "- 可能的原因（例如：突然跳变、趋势漂移、异常点、噪声突增等）\n"
    "从全局数据的视角找出具有明显统计显著性的异常（例如接近 0 的极端值），忽略正常的周期性波动。\n"
    "\n"
    "输出要求：仅输出一个名为 anomalies 的 JSON 数组（不要输出其它文字、不要加代码块标记）严格按照示例格式,必须用中文回复。"
    "数组中每个元素包含字段：range（形如 [start, end] 的数组）、amp（数值）、label（字符串）、detail（字符串）、"
    "color（字符串）、extreme（可选，'min'|'max'|'auto'）。\n"
    "\n"
    "示例（仅作格式参考，不要照抄数值）：\n"
    "anomalies = [\n"
    "    {{\n"
    "        \"range\": [137, 139],\n"
    "        \"amp\": 1.91,\n"
    "        \"label\": \"Downward spike\",\n"
    "        \"detail\": \"Drops from ~1.91 to ~0.00 then recovers; possible transient interference or system failure.\",\n"
    "        \"color\": \"red\",\n"
    "        \"extreme\": \"min\"\n"
    "    }}\n"
    "]\n"
    )
    
    st = time.time()
    text = analyzer.analyze(downsampled_ts, 
                            max_new_tokens=1024,
                            top_p=1,
                            task_prompt_tpl = prompt)
    et = time.time()
    print(et-st)
    print(text)
    
    anomalies = extract_anomalies(text)
    print(len(anomalies), anomalies[:1])
    
    # 将异常索引映射到原始数据（使用 position_index）
    mapped_anomalies = map_anomalies_to_original(anomalies, position_index)
    print("映射后的异常（原始数据索引）:", mapped_anomalies[:1])
    
    # 使用原始数据和映射后的异常进行绘图
    original_ts = df['PI_20412.PV'].values
    fig, ax, notes = plot_ts_with_anomalies(
        original_ts, mapped_anomalies,
        number_style="plain",      # ① ② ③…
        notes_loc="bottom",         # 说明放在下方（不遮挡图）
        marker_fontsize=22
    )
    print(notes)  # 如果你也想在控制台打印出来
    results_dir = os.path.join(script_dir, 'Results')
    os.makedirs(results_dir, exist_ok=True)  # 确保 Results 目录存在
    plt.savefig(os.path.join(results_dir, 'ChatTS_anomalies.png'))

    # 1) 短序列/能放下：整段推理
    # text = analyzer.analyze(ts, max_new_tokens=1024)

    # 2) 长序列：滑窗推理
    # text = analyzer.analyze(
    #     ts,
    #     window_len=50_000,            # 根据显存/吞吐自己调
    #     overlap=0.25,
    #     max_new_tokens=2000,         # 总上限
    #     per_window_new_tokens=256,   # 每窗上限（不传则自动均分）
    #     top_p=0.9,
    #     clear_cuda_cache_each_window=True,
    # )
    # print(text[:2000], "...\n[TRUNCATED]")