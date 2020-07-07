class Mel2WavParams(object):

    def __init__(self, **kwargs):
        super(Mel2WavParams, self).__init__()
        self.__kwargs = kwargs

    def __getattr__(self, name):
        try:
            super(Mel2WavParams, self).__getattribute__(name)
        except AttributeError:
            try:
                return self.__kwargs[name]
            except KeyError:
                raise KeyError("Unrecognize attribute {}".format(name))

    def __dir__(self):
        origin_dir = super(Mel2WavParams, self).__dir__()
        return list(self.__kwargs.keys()) + origin_dir

m2w_params = Mel2WavParams(
    max_wav_value = 32768, # Maximum audiowave value, 1 for mandarin, 32768 for english
    sampling_rate = 22050, # Sampling rate
    filter_length = 1024,
    hop_length = 256,
    win_length = 1024,
    mel_fmin = 0.0,
    mel_fmax = 8000.0,
    n_mel_channels = 160, # mel channels

    griffin_lim_mag_scale = 1024,
    griffin_lim_power = 1.2,
    griffin_lim_iters = 50,
    n_fft = 1024,

    # Mel and Linear spectrograms normalization/scaling and clipping
    signal_normalization = False,
    allow_clipping_in_normalization = True, #Only relevant if mel_normalization = True
    symmetric_mels = True, #Whether to scale the data to be symmetric around 0
    max_abs_value = 4., #max absolute value of data. If symmetric, data will be [-max, max] else [0, max]
    # normalize_for_wavenet = True, #whether to rescale to [0, 1] for wavenet.

    # #Limits
    min_level_db = -120,

    # dummy params
    text_cleaners = None,
    load_mel_from_disk = False,
)
