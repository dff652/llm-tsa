

from tsdownsample import MinMaxLTTBDownsampler, M4Downsampler
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap


def ts_downsample(data, downsampler='m4', n_out=100000):
    """
    Downsample time series data
    :param data: pd.Series
    :param downsampler: str
    :return: numpy.array, numpy.array
    """

    if downsampler == 'm4':
        s_ds = M4Downsampler().downsample(data, n_out=n_out)
    elif downsampler == 'minmax':
        s_ds = MinMaxLTTBDownsampler().downsample(data, n_out=n_out)

    downsampled_data = data.iloc[s_ds]
    downsampled_time = data.index[s_ds]

    return downsampled_data, downsampled_time



def extract_anomalies(text: str):
    """
    从包含 'anomalies = [...]' 的字符串中提取并解析 anomalies 数组。
    如果数组被截断，将尽力截到最后一个完整对象并补全右方括号。
    返回: list[dict]
    """
    # 1) 找到 'anomalies = [' 的起点
    m = re.search(r'anomalies\s*=\s*\[', text)
    if not m:
        raise ValueError("No 'anomalies = [' found in text")
    start = m.end() - 1  # 指向 '['

    # 2) 逐字符扫描，找到匹配的 ']'；期间忽略字符串内的括号
    depth = 0
    in_str = False
    esc = False
    end = None
    for i, ch in enumerate(text[start:], start=start):
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    end = i
                    break

    raw = text[start:end+1] if end is not None else text[start:]  # 可能截断

    # 3) 如果截断：裁到最后一个完整的 '}' 并补 ']'
    if end is None:
        last_closing_brace = raw.rfind('}')
        if last_closing_brace == -1:
            raise ValueError("Found anomalies '[', but no complete object to parse.")
        raw = raw[:last_closing_brace+1] + ']'

    # 4) 解析为 JSON
    try:
        anomalies = json.loads(raw)
    except json.JSONDecodeError as e:
        # 常见修复：去掉尾随逗号
        raw_fixed = re.sub(r',\s*]', ']', raw)
        anomalies = json.loads(raw_fixed)

    if not isinstance(anomalies, list):
        raise ValueError("Parsed anomalies is not a list.")
    return anomalies


def map_anomalies_to_original(anomalies, time_index):
    """
    将下采样序列上的异常索引映射到原始 DataFrame 索引。
    
    Args:
        anomalies: 模型返回的异常列表，每个元素包含 "range": [start, end]
        time_index: ts_downsample 返回的 time_index（原始数据的索引）
    
    Returns:
        映射后的异常列表，range 变为原始数据索引
    """
    import numpy as np
    
    # 将 time_index 转换为可索引的数组
    if hasattr(time_index, 'values'):
        idx_array = time_index.values
    else:
        idx_array = np.asarray(time_index)
    
    mapped = []
    for a in anomalies:
        ds_start, ds_end = a["range"]
        
        # 边界检查
        ds_start = max(0, min(ds_start, len(idx_array) - 1))
        ds_end = max(0, min(ds_end, len(idx_array) - 1))
        
        # 通过 time_index 映射到原始索引
        orig_start = int(idx_array[ds_start])
        orig_end = int(idx_array[ds_end])
        
        mapped_anomaly = a.copy()
        mapped_anomaly["range"] = [orig_start, orig_end]
        mapped_anomaly["downsampled_range"] = [a["range"][0], a["range"][1]]  # 保留原下采样索引
        mapped.append(mapped_anomaly)
    
    return mapped


def plot_ts_with_anomalies(
    ts,
    anomalies,
    baseline_std=None,
    periodic_amp=None,
    figsize=(20, 8),
    dpi=200,
    title="Time series with anomaly annotations (numbered)",
    marker_fontsize=13,     # 图中编号字号
    info_fontsize=12,       # 左上角信息框字号
    legend_fontsize=12,
    tick_fontsize=12,
    number_style="circle",  # "circle" 用 ①②③…；"plain" 用 1,2,3…
    wrap_chars=110,         # 画布外说明的换行宽度
    notes_loc="bottom",     # "bottom" 把说明放在图下方；"right" 放在右侧
):
    """
    返回 (fig, ax, notes_str)
    - fig, ax: 绘图句柄
    - notes_str: 画布外的详细文字（你也可以 print(notes_str) 或保存到文件）
    """
    ts = np.asarray(ts)
    x = np.arange(len(ts))

    # ---- 背景统计（可外部覆盖） ----
    def _robust_baseline_std(arr):
        med = np.median(arr); mad = np.median(np.abs(arr - med))
        sigma = 1.4826 * mad
        return float(sigma if np.isfinite(sigma) and sigma > 0 else np.std(arr))

    def _estimate_periodic_amp(arr):
        n = len(arr)
        if n < 8: return 0.0
        t = np.arange(n); p = np.polyfit(t, arr, 1)
        detr = arr - (p[0]*t + p[1])
        F = np.abs(np.fft.rfft(detr)); F[0] = 0
        if len(F) <= 2: return 0.0
        amp = (2.0/n) * F[np.argmax(F)]
        return float(amp if amp >= 2*_robust_baseline_std(detr) else 0.0)

    if baseline_std is None: baseline_std = _robust_baseline_std(ts)
    if periodic_amp is None: periodic_amp = _estimate_periodic_amp(ts)

    # ---- 帮助：编号样式 ----
    def num_token(i):
        if number_style == "circle":
            # ① 从 unicode 0x2460 开始（支持 1..20）
            base = 0x2460
            return chr(base + (i - 1)) if 1 <= i <= 20 else f"({i})"
        return str(i)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(x, ts, label="Time series", color="blue")
    ax.grid(alpha=0.3)

    # 顶部信息框
    info = (
        f"{len(anomalies)} statistically significant anomalies detected\n"
        f"- Baseline std ≈ {baseline_std:.2f}\n"
        f"- Periodic amplitude ≈ {periodic_amp:.2f}"
    )
    ax.text(0.02, 0.97, info, transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", alpha=0.2), fontsize=info_fontsize)

    # 右上角图例
    ax.legend(loc="upper right", fontsize=legend_fontsize, framealpha=0.9)

    # ---- 绘制异常（图中只放编号 + 箭头），正文放画布外 ----
    notes_lines = []  # 收集外部说明文字
    for i, a in enumerate(anomalies, start=1):
        s, e = a["range"]
        c = a.get("color", None)
        label = a.get("label", f"Anomaly {i}")
        amp = a.get("amp", None)
        detail = a.get("detail", "")
        extreme = a.get("extreme", "auto")

        # 区间与边界
        ax.axvspan(s, e, alpha=0.15, color=c, label=f"Anomaly {i}: {label} ({s}-{e})")
        ax.axvline(s, linestyle="--", color=c); ax.axvline(e, linestyle="--", color=c)

        # 代表点
        seg = ts[s:e+1]
        if extreme == "min":
            local_idx = s + int(np.argmin(seg))
        elif extreme == "max":
            local_idx = s + int(np.argmax(seg))
        else:
            seg_mean = float(np.mean(seg))
            local_idx = s + int(np.argmax(np.abs(seg - seg_mean)))
        local_val = ts[local_idx]
        ax.scatter([local_idx], [local_val], color=c, edgecolor="black", zorder=5)

        # 仅显示编号（不放长文本）
        token = num_token(i)
        ax.annotate(
            token,
            xy=(local_idx, local_val), xycoords="data",
            xytext=(e + 8, local_val + 0.25), textcoords="data",
            arrowprops=dict(arrowstyle="->", color=c,
                            connectionstyle="arc3,rad=-0.15", shrinkA=2, shrinkB=2),
            color=c, fontsize=marker_fontsize,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", alpha=0.75)
        )

        # 外部说明文本
        line = f"{token} {label} ({s}-{e})"
        if amp is not None: line += f"  |  Amplitude ≈ {amp:.2f}"
        if detail: line += f"  |  {detail}"
        notes_lines.append(textwrap.fill(line, width=wrap_chars))

    # ---- 把正文放到画布外（图下方或右侧） ----
    notes_str = "\n\n".join(notes_lines)
    # if notes_loc == "bottom":
    #     # 为底部文本预留空间，再用 fig.text
    #     fig.subplots_adjust(bottom=0.32)  # 视文本多少可再调
    #     fig.text(
    #         0.02, 0.02, notes_str,
    #         ha="left", va="bottom", fontsize=marker_fontsize, color="black",
    #         bbox=dict(boxstyle="round,pad=0.4", alpha=0.08)
    #     )
    # elif notes_loc == "right":
    #     fig.subplots_adjust(right=0.76)
    #     fig.text(
    #         0.78, 0.5, notes_str,
    #         ha="left", va="center", fontsize=marker_fontsize, color="black",
    #         bbox=dict(boxstyle="round,pad=0.4", alpha=0.08)
    #     )

    ax.set_title(title)
    ax.set_xlabel("Index"); ax.set_ylabel("Value")
    ax.tick_params(labelsize=tick_fontsize)
    fig.tight_layout()
    return fig, ax, notes_str


def plot_timer_anomalies(
    original_series,
    residuals,
    intervals,
    lookback_length,
    forecast_horizon=1,
    residual_step=0,
    figsize=(20, 8),
    dpi=200,
    title="Time series with TIMER-based anomaly detection",
    marker_fontsize=18,
    info_fontsize=14,
    legend_fontsize=12,
    tick_fontsize=12,
    number_style="plain",
    wrap_chars=110,
    notes_loc="bottom",
):
    """
    可视化 TIMER/SunDial 异常检测结果。
    
    该函数将 timer.detect_series() 返回的残差和异常区间转换为可视化格式，
    并处理索引偏移（残差坐标系 → 原始序列坐标系）。
    
    Args:
        original_series: 原始一维时间序列（numpy array 或 torch.Tensor）。
        residuals: timer.detect_series() 返回的残差张量。
        intervals: timer.detect_series() 返回的异常区间列表，格式 [{"range": (start, end), "score": float}, ...]。
        lookback_length: 预测时使用的窗口长度（用于索引偏移转换）。
        forecast_horizon: 预测步长（用于索引偏移计算）。
        residual_step: 若残差为 2D，选择第几步的残差用于可视化。
        figsize, dpi, title, marker_fontsize 等: 绘图参数，传递给 plot_ts_with_anomalies。
        number_style: "plain" 或 "circle"，编号样式。
        wrap_chars: 说明文字换行宽度。
        notes_loc: "bottom" 或 "right"，说明文字位置（当前未启用，返回 notes_str 供外部使用）。
    
    Returns:
        (fig, ax, notes_str): 同 plot_ts_with_anomalies。
    """
    import torch
    
    # 转换原始序列为 numpy
    if isinstance(original_series, torch.Tensor):
        ts = original_series.detach().cpu().numpy()
    else:
        ts = np.asarray(original_series)
    
    # 提取残差（若为 2D，选择指定步）
    if isinstance(residuals, torch.Tensor):
        res = residuals.detach().cpu().numpy()
    else:
        res = np.asarray(residuals)
    
    if res.ndim == 2:
        assert 0 <= residual_step < res.shape[1], f"residual_step {residual_step} 越界（残差形状 {res.shape}）"
        res_1d = res[:, residual_step]
    else:
        res_1d = res
    
    # 计算索引偏移：残差索引 0 对应原始序列索引 lookback_length
    # 若 forecast_horizon > 1，残差长度 = len(ts) - lookback_length - forecast_horizon + 1
    # 为简化，统一偏移 lookback_length（实际可能还需要考虑 forecast_horizon，但通常用第一步残差）
    offset = lookback_length
    
    # 转换异常区间：从残差坐标系转换到原始序列坐标系
    formatted_anomalies = []
    for i, interval in enumerate(intervals):
        res_start, res_end = interval["range"]
        score = interval.get("score", 0.0)
        
        # 转换到原始序列索引
        orig_start = res_start + offset
        orig_end = res_end + offset
        
        # 确保不越界
        orig_start = max(0, min(orig_start, len(ts) - 1))
        orig_end = max(0, min(orig_end, len(ts) - 1))
        
        # 计算该区间的残差幅度（用于显示）
        if res_start < len(res_1d) and res_end < len(res_1d):
            seg_res = res_1d[res_start : res_end + 1]
            amp = float(np.abs(seg_res).max()) if len(seg_res) > 0 else 0.0
        else:
            amp = 0.0
        
        # 格式化异常信息
        formatted_anomalies.append({
            "range": (orig_start, orig_end),
            "score": score,
            "amp": amp,
            "label": f"异常 {i+1}",
            "detail": f"残差幅度 ≈ {amp:.3f}，异常分数 {score:.2f}",
            "color": "red",  # 默认红色，可后续扩展支持多色
        })
    
    # 调用通用绘图函数
    return plot_ts_with_anomalies(
        ts,
        formatted_anomalies,
        baseline_std=None,  # 自动计算
        periodic_amp=None,  # 自动计算
        figsize=figsize,
        dpi=dpi,
        title=title,
        marker_fontsize=marker_fontsize,
        info_fontsize=info_fontsize,
        legend_fontsize=legend_fontsize,
        tick_fontsize=tick_fontsize,
        number_style=number_style,
        wrap_chars=wrap_chars,
        notes_loc=notes_loc,
    )
