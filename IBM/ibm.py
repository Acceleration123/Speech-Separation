"""
This is the code to generate IBM(Ideal Binary Mask) for speaker separation task.
In this implementation, case of two speakers is considered and speech signals of both speakers are known.
"""
import torch
import soundfile as sf
from loss.loss_func import SDRLoss


def stft_encoder(
        wav,
        n_fft=512,
        hop_length=256,
        window=torch.hann_window(512).pow(0.5)
):
    device = wav.device
    wav_tf = torch.stft(wav, n_fft=n_fft, hop_length=hop_length, window=window.to(device), return_complex=True)

    return wav_tf  # (F, T)


def stft_decoder(
        wav_tf,
        n_fft=512,
        hop_length=256,
        window=torch.hann_window(512).pow(0.5)
):
    device = wav_tf.device
    wav = torch.istft(wav_tf, n_fft=n_fft, hop_length=hop_length, window=window.to(device))

    return wav  # (T,)


def generate_ibm(spk1, spk2):
    """
    spk1 & spk2: (T,)  waveform
    """
    spk1_tf_mag = stft_encoder(spk1).abs()  # (F, T)
    spk2_tf_mag = stft_encoder(spk2).abs()  # (F, T)

    F, T = spk1_tf_mag.shape

    ibm_spk1 = torch.zeros(F, T)
    ibm_spk1_index = spk1_tf_mag > spk2_tf_mag   # (F, T)
    ibm_spk1[ibm_spk1_index] = 1.0

    ibm_spk2 = torch.ones(F, T) - ibm_spk1  # (F, T)

    return ibm_spk1, ibm_spk2


if __name__ == '__main__':
    mix, fs = sf.read('mix.wav', dtype='float32')
    spk1, _ = sf.read('spk1.wav', dtype='float32')
    spk2, _ = sf.read('spk2.wav', dtype='float32')

    mix = torch.tensor(mix)  # (T,)
    spk1 = torch.tensor(spk1)  # (T,)
    spk2 = torch.tensor(spk2)  # (T,)

    ibm_spk1, ibm_spk2 = generate_ibm(spk1, spk2)
    mix_tf = stft_encoder(mix)  # (F, T)

    # Apply IBM to spk1 and spk2
    spk1_tf_ibm = mix_tf * ibm_spk1
    spk2_tf_ibm = mix_tf * ibm_spk2

    spk1_ibm_est = stft_decoder(spk1_tf_ibm)
    spk2_ibm_est = stft_decoder(spk2_tf_ibm)

    sf.write('spk1_ibm_est.wav', spk1_ibm_est.cpu().detach().numpy(), samplerate=fs)
    sf.write('spk2_ibm_est.wav', spk2_ibm_est.cpu().detach().numpy(), samplerate=fs)

    # Calculate SDR
    loss_func = SDRLoss()
    est = torch.stack([spk1_ibm_est, spk2_ibm_est], dim=0)  # (2, T)
    ref = torch.stack([spk1, spk2], dim=0)  # (2, T)
    sdr_loss = loss_func(est, ref)
    print(f"SDR:{-sdr_loss}")
