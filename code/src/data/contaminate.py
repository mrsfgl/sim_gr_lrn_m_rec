
import numpy as np


def contaminate_signal(X, noise_rate=10, noise_type='AWGN', missing_ratio=0):
    ''' Contaminates data with AWGN and random missing elements.

    Parameters:
        X: np.array(), double
            Original data tensor.

        noise_rate: double.
            For 'AWGN', target SNR in dB. For 'gross', the rate of
            corrupted elements.

        noise_type: string
            Type of noise to be added to data. Default: 'AWGN'
            Currently supports 'AWGN' and gross noise ('gross')

        missing_ratio: double,
            Ratio of missing elements to tensor size. Should be in [0,1].

    Outputs:
        Y: Masked Array. np.ma()
            Noisy tensor with missing elements.
    '''
    # Generate noise
    sizes = X.shape
    Y = X.copy()
    if noise_type == 'AWGN':
        signal_power = np.linalg.norm(X)**2/X.size
        signal_dB = 10 * np.log10(signal_power)
        noise_db = signal_dB - noise_rate
        noise_power = 10 ** (noise_db / 10)
        noise = np.sqrt(noise_power)*np.random.standard_normal(sizes)
        Y = Y + noise
    elif noise_type == 'gross':
        vec_ind = np.nonzero(np.random.binomial(1, noise_rate, size=X.size))[0]
        (
            Y[np.unravel_index(vec_ind, sizes, 'F')]
        ) = np.random.uniform(low=X.min(), high=X.max(), size=vec_ind.size)

    # Create mask
    vec_mask = np.random.uniform(size=X.size)-missing_ratio < 0
    mask = vec_mask.reshape(sizes)
    return np.ma.array(Y, mask=mask)
