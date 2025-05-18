import torch
import torch.nn as nn
from einops import rearrange


def stft_encoder(
        wav,
        n_fft=512,
        hop_length=256,
        window=torch.hann_window(512).pow(0.5)
):
    device = wav.device
    wav_tf = torch.stft(wav, n_fft=n_fft, hop_length=hop_length, window=window.to(device), return_complex=True)

    return wav_tf  # (B, F, T)


class DeepClustering(nn.Module):
    def __init__(
            self,
            n_freqs=257,
            hidden_size=600,
            num_layers=2,
            embedding_size=40,
            bidirectional=True,
    ):
        super(DeepClustering, self).__init__()
        self.embedding_size = embedding_size

        self.lstm = nn.LSTM(
            input_size=n_freqs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.fc = nn.Linear(
            in_features=2 * hidden_size if bidirectional else hidden_size,
            out_features=n_freqs * embedding_size
        )

        self.act = nn.Tanh()

    def forward(self, x):
        """
        input: x:(B, T)  waveform
        """
        x_spec = stft_encoder(x)
        x_spec = x_spec.abs().permute(0, 2, 1)  # (B, T, F)
        x_spec, _ = self.lstm(x_spec)   # (B, T, 2 * H)
        x_spec = self.fc(x_spec)   # (B, T, F * E)
        x_spec = rearrange(x_spec, 'b t (f e) -> b (t f) e', e=self.embedding_size)  # (B, T * F, E)
        x_spec = self.act(x_spec)  # (B, T * F, E)

        return x_spec


if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    model = DeepClustering().eval()
    flops, params = get_model_complexity_info(model, (16000,), as_strings=True,
                                              print_per_layer_stat=True, verbose=True)
    print(flops, params)
