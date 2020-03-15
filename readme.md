# Griffin-lim based on pytorch stft and istft

## 依赖
- pytorch
- torchaudio
- cupy
- librosa

## 文档说明
- `examples/`: 样例音频
    - `ground_truth.wav`: 原始音频
    - `librosa_cpu.wav`: 基于 `librosa` 库实现的 Griffin-lim 算法所合成音频
    - `pytorch_cuda.wav`: 基于 `pytorch` 库实现的 Griffin-lim 算法所合成音频
    - `mel_tensor.pt`: 由 `ground_truth.wav` 转换得到的梅尔频谱，用于作为 `griffin-lim` 算法的输入
- `libs/`: 
    - `audio_processing.py`: Griffin-lim 算法实现主体
    - `mel2wav_params.py`: 超参数集合
    - `utils.py`: 提供了一些工具函数

## 接口

```python
from libs.audio_processing import vocoder_griffin_lim

def vocoder_griffin_lim(mel_spec, mel_len, mel_params, gl_type="cuda"):
    """
    Construct mel spectrograms to audio by griffin_lim methods.
    Support for cuda and cpu computation.

    Parameters
    ------------
    mel_spec: torch type tensor [B, n_mel_channels, mel_len]
    mel_len: torch type tensor [B]
    mel_params: 
    gl_type: target device to excute griffin-lim function, choice between "cpu" and "cuda". cpu version is implemented by librosa, cuda version is implemented by pytorch.

    Returns
    ---------
    wavs: A list contains wav signal matrix. It's type depends on gl_type
    """
```

### 设计思路
- 使用 cupy 支持复数的计算
- 使用 pytorch 与 torchaudio 支持 stft 与 istft 计算
- 使用 dlpack 实现 cupy 与 pytorch 间的张量复用，节省内存拷贝与数据转换的时间

## 性能对比

| 硬件 | 时间（s） | 加速比 |
| --- | --- | --- |
| Intel i7-7700 | 1.13 | 1 |
| GTX 1070Ti | 0.088 | 12.84|

## 其它
当前为了方便进行 librosa 与 pytorch 的对比，cpu 版本的 griffin-lim 算法使用了 `librosa+numpy` 实现，而 cuda 版本的 griffin-lim 算法使用 `pytorch+cupy` 实现。但是 `pytorch+numpy` 的策略也是可以实现 cpu 版本的 griffin-lim 算法。