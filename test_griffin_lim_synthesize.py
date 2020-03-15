import os
import time

import torch
from scipy.io.wavfile import write

from libs.mel2wav_params import m2w_params
from libs.audio_processing import vocoder_griffin_lim


def save_wav(wavs, audio_path=None):
    for i, wav in enumerate(wavs):
        # wav = wav * m2w_params.max_wav_value
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()
        write(audio_path, m2w_params.sampling_rate, wav)

def load_mel(filename):
    return torch.load(filename)


def main():
    # load ground-truth mel tensor
    filename = "./examples/mel_tensor.pt"

    mel = load_mel(filename)
    # mel2wav
    iters = 100
    mel = mel.to("cuda")
    st = time.perf_counter()
    for _ in range(iters):
        wavs = vocoder_griffin_lim(mel, [mel.size(2)]*mel.size(0), m2w_params, gl_type="cuda")
    print("cuda time: {}s".format((time.perf_counter()-st)/iters))

    st = time.perf_counter()
    for _ in range(iters):
        wavs = vocoder_griffin_lim(mel, [mel.size(2)]*mel.size(0), m2w_params, gl_type="cpu")
    print("cpu time: {}s".format((time.perf_counter()-st)/iters))

    path, ori_filename = os.path.split(filename)
    ori_filename, _ = os.path.splitext(ori_filename)
    th_result_path = os.path.join(path, "{}_gen_griffinlim.wav".format(ori_filename))
    save_wav(wavs, audio_path=th_result_path)


if __name__ == "__main__":
    main()