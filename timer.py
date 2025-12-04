import torch
from typing import Callable, List, Optional, Tuple, Union
from transformers import AutoModelForCausalLM


TensorLike = Union[torch.Tensor, "np.ndarray"]  # 队友可传 numpy，再内部转成 tensor
DetectorFunc = Callable[[torch.Tensor], List[dict]]


class TimerAnomalyPipeline:
    """
    TIMER/SunDial 推理 + 残差异常检测一体化管线。

    设计目标：
    1. 一次加载模型，在多次预测/检测中复用；
    2. 支持 GPU/CPU 自由切换；
    3. 提供端到端接口：输入历史序列 → 返回预测、残差、异常区间。
    """

    def __init__(
        self,
        model_path: str,
        device: Union[str, dict] = "cpu",
        trust_remote_code: bool = True,
    ) -> None:
        """
        Args:
            model_path: 预训练权重路径或 HF Repo id。
            device: 
                - 单卡/CPU: "cpu" 或 "cuda:0" 等字符串
                - 多卡: "auto" 让 Hugging Face 自动分配，或传入 device_map 字典手动指定
            trust_remote_code: 是否加载自定义模型代码。
        """
        self.device_str = device
        self.is_multi_gpu = device == "auto" or (isinstance(device, dict))
        
        if self.is_multi_gpu:
            # 多卡模式：使用 device_map
            device_map = device if isinstance(device, dict) else "auto"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                device_map=device_map,
            )
            # 多卡时，输入数据通常放在第一张 GPU（cuda:0）
            self.input_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            # 单卡/CPU 模式：使用 .to(device)
            self.device = torch.device(device)
            self.input_device = self.device
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=trust_remote_code
            ).to(self.device)
        
        self.model.eval()

    # ------------------------------------------------------------------
    # 基础：预测
    # ------------------------------------------------------------------
    def generate_forecast(
        self,
        lookback_length: int,
        forecast_length: int,
        *,
        input_seqs: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """
        直接调用模型生成预测。

        Args:
            lookback_length: 输入窗口长度。
            forecast_length: 需要生成的未来长度，对应 max_new_tokens。
            input_seqs: 可选，形状 (batch_size, lookback_length)。若为 None，则使用随机
                正态分布占位；真实使用时建议手动传入。
            batch_size: 未提供 input_seqs 时使用的批大小。
            num_samples: 多样本生成个数（模型支持时生效）。

        Returns:
            torch.Tensor: 原始 generate 输出（根据模型实现可能包含样本维）。
        """
        if input_seqs is None:
            sequences = torch.randn(batch_size, lookback_length, device=self.input_device)
        else:
            sequences = input_seqs.to(self.input_device)

        gen_kwargs = dict(max_new_tokens=forecast_length)
        if num_samples is not None:
            gen_kwargs["num_samples"] = num_samples

        with torch.inference_mode():
            outputs = self.model.generate(sequences, **gen_kwargs)
        return outputs

    # ------------------------------------------------------------------
    # 工具：滚动残差
    # ------------------------------------------------------------------
    def rolling_forecast_residuals(
        self,
        data: TensorLike,
        *,
        lookback_length: int,
        streaming: bool = True,
        forecast_horizon: int = 1,
        reset_interval: int = 256,
        num_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """
        对一维序列做逐点前视预测，返回真实值与预测值的残差序列。

        Args:
            data: 1D 数据（torch.Tensor 或 numpy.ndarray）。
            lookback_length: 滚动窗口长度，要求 < len(data)。
            streaming: True 表示启用“累加 + 周期重置”的流式模式（适合在线场景）；
                False 表示每次直接切片最近窗口（适合离线批处理，避免追加开销）。
            forecast_horizon: 预测步长（未来生成的步数），需 ≥1。
            reset_interval: 每跑多少步重置一次上下文（使用真实窗口），避免误差积累。
                仅在 streaming=True 时生效。
            num_samples: 若模型支持多样本，则对样本取均值作为预测。

        Returns:
            torch.Tensor: 残差矩阵，形状 (len(data) - lookback_length - forecast_horizon + 1, forecast_horizon)。
                若 forecast_horizon=1，则自动 squeeze 成 1D，长度为 len(data) - lookback_length。
        """
        import numpy as np  # 局部导入，避免非必要依赖

        if isinstance(data, np.ndarray):
            series = torch.as_tensor(data, dtype=torch.float32, device=self.input_device)
        else:
            series = data.to(self.input_device).float()

        assert series.dim() == 1, "data 必须是一维序列"
        T = int(series.shape[0])
        assert T > lookback_length, "data 长度需大于 lookback_length"
        assert forecast_horizon >= 1, "forecast_horizon 需 ≥ 1"
        assert lookback_length + forecast_horizon <= T, "窗口+步长不能超过序列长度"

        residuals: List[torch.Tensor] = []
        if streaming:
            context = series[:lookback_length].clone()
            steps_since_reset = 0
        end_t = T - forecast_horizon

        with torch.inference_mode():
            for t in range(lookback_length, end_t + 1):
                if streaming:
                    if steps_since_reset >= reset_interval:
                        context = series[t - lookback_length:t].clone()
                        steps_since_reset = 0
                    seq = context[-lookback_length:].unsqueeze(0)
                else:
                    window = series[t - lookback_length:t]
                    seq = window.unsqueeze(0)
                
                # 检查输入序列是否包含异常值或全零
                seq_std = seq.std().item()
                seq_mean = seq.mean().item()
                
                if torch.any(~torch.isfinite(seq)):
                    print(f"[ERROR] t={t}: 输入序列 seq 包含 NaN/Inf, shape={seq.shape}, "
                          f"seq stats: min={seq.min().item():.6f}, max={seq.max().item():.6f}, "
                          f"has_nan={torch.any(torch.isnan(seq)).item()}, "
                          f"has_inf={torch.any(torch.isinf(seq)).item()}")
                    # 如果输入有 NaN，模型输出肯定也是 NaN，跳过这次预测
                    y_hat_block = torch.full((forecast_horizon,), float('nan'), 
                                            device=seq.device, dtype=seq.dtype)
                elif seq_std < 1e-8:  # 输入序列方差为0（全零或常数）
                    # 对于全零或常数输入，使用输入序列的最后一个值作为预测
                    print(f"[WARNING] t={t}: 输入序列方差过小 (std={seq_std:.6f}), 使用回退策略")
                    last_value = seq[0, -1].item() if seq.shape[1] > 0 else seq_mean
                    y_hat_block = torch.full((forecast_horizon,), last_value,
                                            device=seq.device, dtype=seq.dtype)
                else:
                    gen_kwargs = dict(max_new_tokens=forecast_horizon)
                    if num_samples is not None:
                        gen_kwargs["num_samples"] = num_samples
                    pred = self.model.generate(seq, **gen_kwargs)
                    
                    # 检查模型输出是否包含异常值
                    if torch.any(~torch.isfinite(pred)):
                        print(f"[ERROR] t={t}: 模型 generate 输出包含 NaN/Inf, "
                              f"pred shape={pred.shape}, seq shape={seq.shape}, "
                              f"seq stats: min={seq.min().item():.6f}, max={seq.max().item():.6f}, "
                              f"seq mean={seq_mean:.6f}, seq std={seq_std:.6f}")
                        # 回退策略：使用输入序列的最后一个值作为预测
                        last_value = seq[0, -1].item()
                        y_hat_block = torch.full((forecast_horizon,), last_value,
                                                device=seq.device, dtype=seq.dtype)
                        print(f"[WARNING] t={t}: 使用回退策略，预测值={last_value:.6f}")
                    else:
                        y_hat_block = self._extract_forecast(pred, forecast_horizon)

                y_true_block = series[t : t + forecast_horizon]
                
                # 检查最终结果
                if torch.any(~torch.isfinite(y_hat_block)):
                    print(f"[WARNING] t={t}: y_hat_block 包含 NaN/Inf, shape={y_hat_block.shape}, horizon={forecast_horizon}")
                if torch.any(~torch.isfinite(y_true_block)):
                    print(f"[WARNING] t={t}: y_true_block 包含 NaN/Inf")
                
                residual_block = (y_true_block - y_hat_block).detach()
                if torch.any(~torch.isfinite(residual_block)):
                    print(f"[WARNING] t={t}: 残差块包含 NaN/Inf, "
                          f"y_true_block={y_true_block}, y_hat_block={y_hat_block}")
                residuals.append(residual_block)

                if streaming:
                    context = torch.cat([context, series[t].view(1)], dim=0)
                    steps_since_reset += 1

        res_tensor = torch.stack(residuals, dim=0)
        if forecast_horizon == 1:
            return res_tensor.squeeze(-1)
        return res_tensor

    def rolling_predictive_scores(
        self,
        data: TensorLike,
        *,
        lookback_length: int,
        streaming: bool = True,
        forecast_horizon: int = 1,
        reset_interval: int = 256,
        num_samples: int = 10,
        score_type: str = "neg_loglik",
        tail_prob_eps: float = 1e-6,
        min_std: float = 1e-4,
    ) -> torch.Tensor:
        """
        使用模型的多样本预测来估计概率分布，输出负对数似然或尾概率得分。

        Args:
            data: 1D 序列（torch.Tensor 或 numpy.ndarray）。
            lookback_length: 滚动窗口长度。
            streaming: 是否复用上下文。
            forecast_horizon: 预测步长。
            reset_interval: streaming=True 时的上下文重置周期。
            num_samples: 每次预测生成的样本数，需 ≥ 2。
            score_type: "neg_loglik"（默认）或 "tail_prob"。
            tail_prob_eps: 计算 tail score 时的最小概率，避免 log(0)。
            min_std: 样本标准差的最小值，避免退化高斯。

        Returns:
            torch.Tensor: 得分矩阵，形状与 residuals 相同，数值越大表示越异常。
        """
        import numpy as np

        assert num_samples >= 2, "score_type 依赖预测分布，需要 num_samples ≥ 2"
        score_type = score_type.lower()
        assert score_type in {"neg_loglik", "tail_prob"}, "score_type 仅支持 neg_loglik/tail_prob"

        if isinstance(data, np.ndarray):
            series = torch.as_tensor(data, dtype=torch.float32, device=self.input_device)
        else:
            series = data.to(self.input_device).float()

        assert series.dim() == 1, "data 必须是一维序列"
        T = int(series.shape[0])
        assert T > lookback_length, "data 长度需大于 lookback_length"
        assert forecast_horizon >= 1, "forecast_horizon 需 ≥ 1"
        assert lookback_length + forecast_horizon <= T, "窗口+步长不能超过序列长度"

        scores: List[torch.Tensor] = []
        if streaming:
            context = series[:lookback_length].clone()
            steps_since_reset = 0
        end_t = T - forecast_horizon

        with torch.inference_mode():
            for t in range(lookback_length, end_t + 1):
                if streaming:
                    if steps_since_reset >= reset_interval:
                        context = series[t - lookback_length:t].clone()
                        steps_since_reset = 0
                    seq = context[-lookback_length:].unsqueeze(0)
                else:
                    window = series[t - lookback_length:t]
                    seq = window.unsqueeze(0)

                gen_kwargs = dict(max_new_tokens=forecast_horizon, num_samples=num_samples)
                pred = self.model.generate(seq, **gen_kwargs)
                samples = self._extract_sample_matrix(pred, forecast_horizon)

                if samples.shape[0] < 2:
                    print(f"[WARNING] t={t}: 样本数 < 2，score_type={score_type} 退化为单样本残差")
                    y_hat_block = samples.mean(dim=0)
                    y_true_block = series[t : t + forecast_horizon]
                    scores.append(torch.abs(y_true_block - y_hat_block))
                else:
                    sample_mean = samples.mean(dim=0)
                    sample_std = samples.std(dim=0, unbiased=False).clamp_min(min_std)
                    dist = torch.distributions.Normal(sample_mean, sample_std)
                    y_true_block = series[t : t + forecast_horizon]

                    if score_type == "neg_loglik":
                        score_block = -dist.log_prob(y_true_block)
                    else:
                        cdf_vals = dist.cdf(y_true_block)
                        two_tail = 2.0 * torch.minimum(cdf_vals, 1.0 - cdf_vals)
                        safe_tail = torch.clamp(two_tail, min=tail_prob_eps)
                        score_block = -torch.log(safe_tail)

                    if torch.any(~torch.isfinite(score_block)):
                        print(f"[WARNING] t={t}: score_block 包含 NaN/Inf，"
                              f"mean={sample_mean}, std={sample_std}, y_true={y_true_block}")
                    scores.append(score_block.detach())

                if streaming:
                    context = torch.cat([context, series[t].view(1)], dim=0)
                    steps_since_reset += 1

        score_tensor = torch.stack(scores, dim=0)
        if forecast_horizon == 1:
            return score_tensor.squeeze(-1)
        return score_tensor

    # ------------------------------------------------------------------
    # 工具：残差 → 异常区间
    # ------------------------------------------------------------------
    def detect_anomalies_from_residuals(
        self,
        residuals: torch.Tensor,
        *,
        method: str = "mad",
        residual_step: int = 0,
        threshold_k: float = 3.5,
        min_run: int = 1,
        custom_detector: Optional[DetectorFunc] = None,
    ) -> List[dict]:
        """
        使用指定策略对残差做异常检测，并输出连续区间。

        Args:
            residuals: 1D 残差张量。
            method: "mad"（默认）、"sigma"（3-sigma）或 "custom"。
            residual_step: 若残差为 2D（多步预测），选择第几步的残差参与检测。
            threshold_k: 阈值系数。对于 "mad" 是多少个 MAD；"sigma" 是多少个 sigma。
            min_run: 至少连续多少个点才算异常区间。
            custom_detector: 当 method="custom" 时，提供的回调函数，入参为 1D 残差
                （torch.Tensor，已在 CPU），返回格式需与本函数一致（列表，元素包含 "range"）。

        Returns:
            List[dict]: [{ "range": (start, end), "score": float }, ...]，索引与残差坐标一致。
        """
        if residuals.dim() == 2:
            assert 0 <= residual_step < residuals.shape[1], "residual_step 越界"
            r = residuals[:, residual_step].float()
        else:
            r = residuals.float()
        assert r.dim() == 1, "residuals 必须是一维或二维张量"
        method = method.lower()

        if method == "custom":
            if custom_detector is None:
                raise ValueError("method='custom' 需要提供 custom_detector 回调。")
            residual_cpu = r.detach().cpu()
            result = custom_detector(residual_cpu)
            if not isinstance(result, list):
                raise ValueError("custom_detector 需返回 List[dict]。")
            return result

        if method == "mad":
            center = torch.median(r)
            scale = torch.median(torch.abs(r - center))
            if scale <= 1e-8:
                scale = torch.std(r)
        elif method == "sigma":
            center = torch.mean(r)
            scale = torch.std(r)
        else:
            raise ValueError("method 仅支持 'mad'、'sigma' 或 'custom'")

        if scale <= 0 or not torch.isfinite(scale):
            scale = torch.tensor(1.0, device=r.device)

        z = torch.abs(r - center) / scale
        mask = (z > threshold_k)

        ranges: List[dict] = []
        start: Optional[int] = None
        for idx in range(len(mask)):
            if mask[idx]:
                if start is None:
                    start = idx
            else:
                if start is not None and idx - start >= min_run:
                    score = float(z[start:idx].max().item())
                    ranges.append({"range": (start, idx - 1), "score": score})
                start = None
        if start is not None and len(mask) - start >= min_run:
            score = float(z[start:].max().item())
            ranges.append({"range": (start, len(mask) - 1), "score": score})
        return ranges

    def diagnose_residuals(
        self,
        residuals: torch.Tensor,
        *,
        residual_step: int = 0,
        threshold_k: float = 3.5,
    ) -> dict:
        """
        诊断残差分布，帮助理解异常检测结果。
        
        Returns:
            dict: 包含残差统计信息、阈值分析等。
        """
        import numpy as np
        
        if residuals.dim() == 2:
            r = residuals[:, residual_step].float()
        else:
            r = residuals.float()
        
        r_np = r.detach().cpu().numpy() if isinstance(r, torch.Tensor) else r
        
        med = np.median(r_np)
        mad = np.median(np.abs(r_np - med))
        std = np.std(r_np)
        scale = mad if mad > 1e-8 else std
        if scale <= 0:
            scale = 1.0
        
        z_scores = np.abs(r_np - med) / scale
        threshold = threshold_k
        
        return {
            "残差统计": {
                "中位数": float(med),
                "MAD": float(mad),
                "标准差": float(std),
                "缩放因子": float(scale),
                "最小值": float(np.min(r_np)),
                "最大值": float(np.max(r_np)),
                "P25": float(np.percentile(r_np, 25)),
                "P75": float(np.percentile(r_np, 75)),
                "P95": float(np.percentile(r_np, 95)),
                "P99": float(np.percentile(r_np, 99)),
            },
            "阈值分析": {
                "当前阈值系数": threshold_k,
                "实际阈值（绝对值）": float(threshold_k * scale),
                "超过阈值的点数": int(np.sum(z_scores > threshold)),
                "最大 Z-score": float(np.max(z_scores)),
                "Z-score > 3.0 的点数": int(np.sum(z_scores > 3.0)),
                "Z-score > 4.0 的点数": int(np.sum(z_scores > 4.0)),
            },
            "建议": {
                "如果误报多": "考虑提高 threshold_k（如 3.5 → 4.0 或更高）",
                "如果漏报多": "考虑降低 threshold_k（如 3.5 → 3.0）",
                "如果异常后误报": "考虑使用更稳健的阈值（如使用 P95/P99 而非 MAD）",
            }
        }

    # ------------------------------------------------------------------
    # 端到端：检测入口
    # ------------------------------------------------------------------
    def detect_series(
        self,
        data: TensorLike,
        *,
        lookback_length: int,
        streaming: bool = True,
        reset_interval: int = 256,
        num_samples: Optional[int] = None,
        forecast_horizon: int = 1,
        residual_step: int = 0,
        method: str = "mad",
        threshold_k: float = 3.5,
        min_run: int = 1,
        custom_detector: Optional[DetectorFunc] = None,
        score_mode: str = "residual",
        tail_prob_eps: float = 1e-6,
        min_std: float = 1e-4,
    ) -> Tuple[torch.Tensor, List[dict]]:
        """
        端到端异常检测：输入历史序列 → 输出残差和异常区间。

        Args:
            data: 1D 序列。
            lookback_length: 初始窗口长度。
            streaming: 是否使用流式模式（传给 rolling_forecast_residuals）。
            reset_interval: 滚动预测时的上下文重置周期。
            num_samples: 多样本预测个数（可选）。
            forecast_horizon: 预测步长（生成未来长度）。
            residual_step: 当 forecast_horizon>1 时，用第几步的残差做检测。
            method: 残差检测策略，"mad" / "sigma" / "custom"。
            threshold_k: MAD / Sigma 阈值系数。
            min_run: 最小连续长度。
            custom_detector: 自定义检测器（method="custom" 时必填）。
            score_mode: "residual"（默认）、"neg_loglik" 或 "tail_prob"。
            tail_prob_eps: score_mode="tail_prob" 时的概率下限。
            min_std: 估计分布标准差的下限，避免退化。

        Returns:
            (residuals, intervals)
                residuals: torch.Tensor，形状 (len(data) - lookback_length - forecast_horizon + 1, forecast_horizon)；
                    当 forecast_horizon=1 时为一维张量，长度 len(data) - lookback_length。
                intervals: List[dict]，残差坐标系下的异常区间。
        """
        score_mode = score_mode.lower()
        if score_mode == "residual":
            residuals = self.rolling_forecast_residuals(
                data,
                lookback_length=lookback_length,
                streaming=streaming,
                reset_interval=reset_interval,
                num_samples=num_samples,
                forecast_horizon=forecast_horizon,
            )
        elif score_mode in {"neg_loglik", "tail_prob"}:
            if num_samples is None or num_samples < 2:
                raise ValueError("score_mode 为 neg_loglik/tail_prob 时需设置 num_samples ≥ 2")
            residuals = self.rolling_predictive_scores(
                data,
                lookback_length=lookback_length,
                streaming=streaming,
                reset_interval=reset_interval,
                num_samples=num_samples,
                forecast_horizon=forecast_horizon,
                score_type=score_mode,
                tail_prob_eps=tail_prob_eps,
                min_std=min_std,
            )
        else:
            raise ValueError("score_mode 仅支持 residual / neg_loglik / tail_prob")

        intervals = self.detect_anomalies_from_residuals(
            residuals,
            method=method,
            residual_step=residual_step,
            threshold_k=threshold_k,
            min_run=min_run,
            custom_detector=custom_detector,
        )
        return residuals, intervals

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _extract_forecast(self, generated: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        从 generate 输出中抽取最近 horizon 步的预测值。
        兼容 [B, Lout]、[B, S, Lout]、[S, B, Lout] 等形状。
        """
        out = generated
        
        # 检查输入是否包含 NaN/Inf
        if torch.any(~torch.isfinite(out)):
            print(f"[WARNING] _extract_forecast: generated 包含 NaN/Inf, shape={out.shape}")
        
        if out.dim() == 3:
            if out.shape[1] > 1:  # [B, S, L]
                out = out.mean(dim=1)
            else:                 # [S, B, L]
                out = out.mean(dim=0)
        
        if out.dim() == 2:
            if out.shape[0] == 0:
                print(f"[ERROR] _extract_forecast: out.shape[0]==0, shape={out.shape}")
                return torch.full((horizon,), float('nan'), device=out.device, dtype=out.dtype)
            if out.shape[1] < horizon:
                print(f"[WARNING] _extract_forecast: out.shape[1]={out.shape[1]} < horizon={horizon}, "
                      f"使用全部 {out.shape[1]} 个值并用 NaN 填充")
                result = out[0, :].clone()
                padding = torch.full((horizon - out.shape[1],), float('nan'), 
                                    device=out.device, dtype=out.dtype)
                return torch.cat([result, padding])
            return out[0, -horizon:]
        
        flat = out.flatten()
        if len(flat) < horizon:
            print(f"[WARNING] _extract_forecast: flat长度={len(flat)} < horizon={horizon}, "
                  f"使用全部 {len(flat)} 个值并用 NaN 填充")
            padding = torch.full((horizon - len(flat),), float('nan'), 
                               device=flat.device, dtype=flat.dtype)
            return torch.cat([flat, padding])
        
        result = flat[-horizon:]
        if torch.any(~torch.isfinite(result)):
            print(f"[WARNING] _extract_forecast: 提取的结果包含 NaN/Inf, "
                  f"flat stats: min={flat.min().item():.6f}, max={flat.max().item():.6f}, "
                  f"flat[-horizon:]={result}")
        return result

    def _extract_sample_matrix(self, generated: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        将 generate 输出整理为 [num_samples, horizon] 的样本矩阵。
        """
        out = generated.detach()

        if out.dim() == 3:
            b, s, l = out.shape
            samples = out.reshape(b * s, l)
        elif out.dim() == 2:
            samples = out
        elif out.dim() == 1:
            samples = out.unsqueeze(0)
        else:
            raise ValueError(f"不支持的 generate 输出形状: {out.shape}")

        seq_len = samples.shape[-1]
        if seq_len < horizon:
            print(f"[WARNING] _extract_sample_matrix: 输出长度 {seq_len} < horizon={horizon}，使用末尾数值填充")
            repeat_val = samples[:, -1:].repeat(1, max(horizon - seq_len, 1))
            samples = torch.cat([samples, repeat_val], dim=-1)

        return samples[:, -horizon:]


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # ---------------------------
    # 示例：TIMER 单步预测
    # ---------------------------
    # 单卡/CPU 模式
    pipeline = TimerAnomalyPipeline(
        model_path="/home/data1/llm_models/thuml/timer-base-84m",
        device="cuda:0",  # 单卡：使用 "cuda:0" 或 "cpu"
    )
    fake_series = torch.randn(2880, dtype=torch.float32)
    forecast = pipeline.generate_forecast(
        lookback_length=2880,
        forecast_length=96,
        input_seqs=fake_series.unsqueeze(0),
    )
    print("TIMER forecast shape:", tuple(forecast.shape))

    # ---------------------------
    # 示例：端到端异常检测
    # ---------------------------
    # 多卡模式：使用 "auto" 自动分配，或手动指定 device_map
    sundial_pipeline = TimerAnomalyPipeline(
        model_path="/home/data1/llm_models/thuml/sundial-base-128m",
        device="auto",  # 多卡：使用 "auto" 自动分配，或传入 device_map 字典
        # 手动指定示例：
        # device={"": 0, "transformer.layers.0": 0, "transformer.layers.1": 1, ...}
    )
    long_series = np.sin(np.linspace(0, 200, 4000)).astype(np.float32)
    long_series[2500:2510] += 3.0  # 注入简单异常

    residuals, intervals = sundial_pipeline.detect_series(
        long_series,
        lookback_length=288,
        reset_interval=128,
        num_samples=10,
        forecast_horizon=3,
        residual_step=0,
        threshold_k=3.0,
        min_run=2,
    )
    # 也可以尝试 method="sigma" 或自定义检测器，例如：
    # residuals, intervals = sundial_pipeline.detect_series(
    #     long_series,
    #     lookback_length=288,
    #     method="sigma",
    #     threshold_k=3.5,
    # )
    print("Residuals length:", residuals.shape[0])
    print("Detected intervals:", intervals)
    
    # 诊断残差分布（帮助理解误报原因）
    import json
    diag = sundial_pipeline.diagnose_residuals(residuals, residual_step=0, threshold_k=3.0)
    print("\n=== 残差诊断 ===")
    print(json.dumps(diag, indent=2, ensure_ascii=False))
    
    # 可视化结果
    from ts_ad.utils import plot_timer_anomalies
    fig, ax, notes = plot_timer_anomalies(
        original_series=long_series,
        residuals=residuals,
        intervals=intervals,
        lookback_length=288,
        forecast_horizon=3,
        residual_step=0,
        marker_fontsize=20,
        info_fontsize=14,
    )
    print("\n异常说明：\n", notes)
    plt.show()  # 或 fig.savefig("anomalies.png")
