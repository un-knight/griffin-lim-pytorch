import torch
import torchaudio
import numpy as np
import cupy as cp
import librosa

from .utils import (
    t2c,
    c2t
)

filterbank = None
def get_filterbank(mel_params):
    """
    Generating filterbank from librosa.filters.mel

    Parameters
    ----------
    mel_params: mel spectrom parameters
    device: type of torch.device (default: None)

    Returns
    ------
    filterbank: pytorch tesnsor
    """
    global filterbank

    if filterbank is None:
        filterbank = librosa.filters.mel(
            sr=mel_params.sampling_rate,
            n_fft=mel_params.n_fft,
            n_mels=mel_params.n_mel_channels,
            fmin=mel_params.mel_fmin,
            fmax=mel_params.mel_fmax)
        filterbank = torch.from_numpy(filterbank)
    return filterbank

def griffin_lim_np(magnitudes, mel_params):
    """
    Griffin-Lim algorithm to convert magnitude spectrograms to audio signals
    """
    phase = np.exp(2j * np.pi * np.random.rand(*magnitudes.shape).astype(np.float32))
    complex_spec = magnitudes * phase
    signal = librosa.istft(
                        stft_matrix=complex_spec, 
                        hop_length=mel_params.hop_length,
                        win_length=mel_params.win_length)
    if not np.isfinite(signal).all():
        print("WARNING: audio was not finite, skipping audio saving")
        return np.array([0])

    for _ in range(mel_params.griffin_lim_iters):
        stft_matrix = librosa.stft(
                        y=signal, 
                        n_fft=mel_params.n_fft,
                        hop_length=mel_params.hop_length,
                        win_length=mel_params.win_length)
        phase = np.exp(1.j * np.angle(stft_matrix))
        complex_spec = magnitudes * phase
        signal = librosa.istft(
                        stft_matrix=complex_spec, 
                        hop_length=mel_params.hop_length,
                        win_length=mel_params.win_length)
    return signal

def griffin_lim_th(magnitudes, mel_params):
    """
    Griffin-Lim algorithm to convert magnitude spectrograms to audio signals.
    Implemented with pytorch and cupy.

    Parameters
    ------------
    magnitudes: input magnitudes, pytorch tensor
    mel_params: mel spectrom parameters

    Returns
    -----------
    signal: synthesis wav signal with 
    """
    phase = cp.exp(2j * cp.pi * cp.random.rand(*magnitudes.shape, dtype=cp.float32))
    # complex_spec = magnitudes * phase
    def mul_phase(magnitudes, phase):
        """
        Parameters
        ------------
        magnitudes: pytorch tensor
        phase: cupy ndarray
        """
        phase_real = c2t(cp.real(phase))
        phase_imag = c2t(cp.imag(phase))
        
        input_real = magnitudes * phase_real
        input_imag = magnitudes * phase_imag

        # a virtual complex solution, since pytorch doesn't support complex type
        complex_spec = torch.cat([input_real.unsqueeze(dim=-1), input_imag.unsqueeze(dim=-1)], dim=-1)
        return complex_spec
    
    # add window to mitigate high frequence signal impact
    window = torch.hann_window(window_length=mel_params.win_length, device=magnitudes.device)
    complex_spec = mul_phase(magnitudes, phase)
    signal = torchaudio.functional.istft(
                        stft_matrix=complex_spec, 
                        n_fft=mel_params.n_fft,
                        hop_length=mel_params.hop_length,
                        win_length=mel_params.win_length,
                        window=window)
    if not torch.isfinite(signal).all():
        print("WARNING: audio was not finite, skipping audio saving")
        return torch.zeros(1)

    for _ in range(mel_params.griffin_lim_iters):
        stft_matrix = torch.stft(
                        signal, 
                        n_fft=mel_params.n_fft,
                        hop_length=mel_params.hop_length,
                        win_length=mel_params.win_length)
        phase = t2c(torchaudio.functional.angle(stft_matrix))
        phase = cp.exp(1j * phase)
        complex_spec = mul_phase(magnitudes, phase)
        signal = torchaudio.functional.istft(
                        stft_matrix=complex_spec,
                        n_fft=mel_params.n_fft,
                        hop_length=mel_params.hop_length,
                        win_length=mel_params.win_length,
                        window=window)
    return signal

# @benchmark(iters=10)
def vocoder_griffin_lim(mel_spec, mel_len, mel_params, gl_type="cuda"):
    """
    Construct mel spectrograms to audio by griffin_lim methods.
    Support for cuda and cpu computation.

    Parameters
    ------------
    mel_spec: torch type tensor [B, n_mel_channels, mel_len]
    mel_len: torch type tensor [B]
    mel_params: 
    gl_type: target device to excute griffin-lim function, choice between "cpu" and "cuda"

    Returns
    ---------
    wavs: A list contains wav signal matrix. It's type depends on gl_type
    """
    # print("Running Griffin-Lim")

    wavs = []
    if mel_params.signal_normalization:
        mel_spec = _denomalize(mel_spec, mel_params)

    mel_spec = mel_spec.permute(0, 2, 1).contiguous()
    # for dynamic range decompression
    mel = torch.exp(mel_spec)
    filterbank = get_filterbank(mel_params).to(mel.device)
    # inverted mel spectrograms into linear frequency spectrograms
    magnitudes = (torch.matmul(mel, filterbank) * mel_params.griffin_lim_mag_scale) ** mel_params.griffin_lim_power
    magnitudes = magnitudes.permute(0, 2, 1)
    
    griffin_lim_fn = None
    if gl_type == "cuda":
        griffin_lim_fn = griffin_lim_th
    else:
        griffin_lim_fn = griffin_lim_np
        magnitudes = magnitudes.cpu().numpy()
    
    for j, sample in enumerate(magnitudes):
        sample = sample[:, :mel_len[j]]
        wav = griffin_lim_fn(
            sample, 
            mel_params
            )
        wavs.append(wav)
    return wavs

def _denomalize(mel_spec, mel_params):
    if mel_params.allow_clipping_in_normalization:
        if mel_params.symmetric_mels:
            return (((torch.clamp(mel_spec, -mel_params.max_abs_value,
				mel_params.max_abs_value) + mel_params.max_abs_value) * -mel_params.min_level_db / (2 * mel_params.max_abs_value))
				+ mel_params.min_level_db)
        else:
            return ((torch.clamp(mel_spec, 0, mel_params.max_abs_value) * -mel_params.min_level_db / mel_params.max_abs_value) + mel_params.min_level_db)

    if mel_params.symmetric_mels:
        return (((mel_spec + mel_params.max_abs_value) * -mel_params.min_level_db / (2 * mel_params.max_abs_value)) + mel_params.min_level_db)
    else:
        return ((mel_spec * -mel_params.min_level_db / mel_params.max_abs_value) + mel_params.min_level_db)