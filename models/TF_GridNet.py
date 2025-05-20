import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
from ptflops import get_model_complexity_info


class STFT_Encoder(nn.Module):
    def __init__(
            self,
            n_fft=512,
            hop_length=256,
            window=torch.hann_window(512).pow(0.5)
    ):
        super(STFT_Encoder, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window

    def forward(self, mix):
        # mix:(B, T)
        device = mix.device
        mix_stft = torch.stft(mix, n_fft=self.n_fft, hop_length=self.hop_length,
                              window=self.window.to(device), return_complex=False)  # (B, F, T, 2)

        return mix_stft


class STFT_Decoder(nn.Module):
    def __init__(
            self,
            n_fft=512,
            hop_length=256,
            window=torch.hann_window(512).pow(0.5)
    ):
        super(STFT_Decoder, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window

    def forward(self, wav_stft):
        # wav_stft:(B, F, T, 2)
        device = wav_stft.device
        wav = torch.istft(wav_stft, n_fft=self.n_fft, hop_length=self.hop_length,
                          window=self.window.to(device))  # (B, T)

        return wav


class GridNetBlock(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        emb_dim,
        emb_ks,
        emb_hs,
        n_freqs,
        hidden_channels,
        n_head=4,
        approx_qk_dim=1024,
        # activation="prelu",
        eps=1e-5,
    ):
        super().__init__()

        in_channels = emb_dim * emb_ks

        self.intra_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        self.intra_linear = nn.ConvTranspose1d(
            hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
        )

        self.inter_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        self.inter_linear = nn.ConvTranspose1d(
            hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
        )

        E = math.ceil(
            approx_qk_dim * 1.0 / n_freqs
        )  # approx_qk_dim is only approximate
        assert emb_dim % n_head == 0
        for ii in range(n_head):
            self.add_module(
                "attn_conv_Q_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, 1),
                    nn.PReLU(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                )
            )
            self.add_module(
                "attn_conv_K_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, 1),
                    nn.PReLU(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                )
            )
            self.add_module(
                "attn_conv_V_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, emb_dim // n_head, 1),
                    nn.PReLU(),
                    LayerNormalization4DCF((emb_dim // n_head, n_freqs), eps=eps),
                )
            )
        self.add_module(
            "attn_concat_proj",
            nn.Sequential(
                nn.Conv2d(emb_dim, emb_dim, 1),
                nn.PReLU(),
                LayerNormalization4DCF((emb_dim, n_freqs), eps=eps),
            ),
        )

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def forward(self, x):
        """GridNetBlock Forward.

        Args:
            x: [B, C, T, Q]
            out: [B, C, T, Q]
        """
        B, C, old_T, old_Q = x.shape
        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        Q = math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        x = F.pad(x, (0, Q - old_Q, 0, T - old_T))

        # intra RNN
        input_ = x
        intra_rnn = self.intra_norm(input_)  # [B, C, T, Q]
        intra_rnn = (
            intra_rnn.transpose(1, 2).contiguous().view(B * T, C, Q)
        )  # [BT, C, Q]
        intra_rnn = F.unfold(
            intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BT, C*emb_ks, -1]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*emb_ks]
        intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, -1, H]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
        intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]
        intra_rnn = intra_rnn.view([B, T, C, Q])
        intra_rnn = intra_rnn.transpose(1, 2).contiguous()  # [B, C, T, Q]
        intra_rnn = intra_rnn + input_  # [B, C, T, Q]

        # inter RNN
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)  # [B, C, T, F]
        inter_rnn = (
            inter_rnn.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)
        )  # [BF, C, T]
        inter_rnn = F.unfold(
            inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BF, C*emb_ks, -1]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, -1, C*emb_ks]
        inter_rnn, _ = self.inter_rnn(inter_rnn)  # [BF, -1, H]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, H, -1]
        inter_rnn = self.inter_linear(inter_rnn)  # [BF, C, T]
        inter_rnn = inter_rnn.view([B, Q, C, T])
        inter_rnn = inter_rnn.permute(0, 2, 3, 1).contiguous()  # [B, C, T, Q]
        inter_rnn = inter_rnn + input_  # [B, C, T, Q]

        # attention
        inter_rnn = inter_rnn[..., :old_T, :old_Q]
        batch = inter_rnn

        all_Q, all_K, all_V = [], [], []
        for ii in range(self.n_head):
            all_Q.append(self["attn_conv_Q_%d" % ii](batch))  # [B, C, T, Q]
            all_K.append(self["attn_conv_K_%d" % ii](batch))  # [B, C, T, Q]
            all_V.append(self["attn_conv_V_%d" % ii](batch))  # [B, C, T, Q]

        Q = torch.cat(all_Q, dim=0)  # [B', C, T, Q]
        K = torch.cat(all_K, dim=0)  # [B', C, T, Q]
        V = torch.cat(all_V, dim=0)  # [B', C, T, Q]

        Q = Q.transpose(1, 2)
        Q = Q.flatten(start_dim=2)  # [B', T, C*Q]
        K = K.transpose(1, 2)
        K = K.flatten(start_dim=2)  # [B', T, C*Q]
        V = V.transpose(1, 2)  # [B', T, C, Q]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*Q]
        emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)  # [B', T, T]
        attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*Q]

        V = V.reshape(old_shape)  # [B', T, C, Q]
        V = V.transpose(1, 2)  # [B', C, T, Q]
        emb_dim = V.shape[1]

        batch = V.view([self.n_head, B, emb_dim, old_T, -1])  # [n_head, B, C, T, Q])
        batch = batch.transpose(0, 1)  # [B, n_head, C, T, Q])
        batch = batch.contiguous().view(
            [B, self.n_head * emb_dim, old_T, -1]
        )  # [B, C, T, Q])
        batch = self["attn_concat_proj"](batch)  # [B, C, T, Q])

        out = batch + inter_rnn
        return out


class LayerNormalization4D(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        param_size = [1, input_dimension, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            _, C, _, _ = x.shape
            stat_dim = (1,)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B, 1, T, F]
        std_ = torch.sqrt(x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps)  # [B, 1, T, F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            stat_dim = (1, 3)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,1]
        std_ = torch.sqrt(x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps)  # [B, 1, T, F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class TF_GridNet(nn.Module):
    def __init__(
            self,
            num_tf_blks=6,
            emb_dim=32,
            hidden_size=96,
            kernel_size=4,
            stride=4,
            width=257
    ):
        super(TF_GridNet, self).__init__()
        self.num_blks = num_tf_blks

        self.stft_encoder = STFT_Encoder()

        self.conv = nn.Conv2d(in_channels=2, out_channels=emb_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.gln = nn.GroupNorm(num_groups=1, num_channels=emb_dim, eps=1e-8)

        self.tf_grid_blks = nn.ModuleList([])
        for i in range(self.num_blks):
            self.tf_grid_blks.append(GridNetBlock(emb_dim=emb_dim, emb_ks=kernel_size, emb_hs=stride,
                                                  n_freqs=width, hidden_channels=hidden_size))

        self.de_conv = nn.ConvTranspose2d(in_channels=emb_dim, out_channels=4, kernel_size=(3, 3),
                                          stride=(1, 1), padding=(1, 1))

        self.stft_decoder = STFT_Decoder()

    def forward(self, x):
        # x: (B, T)
        n_samples = x.shape[-1]
        x = self.stft_encoder(x)  # (B, F, T, 2)

        x = x.permute(0, 3, 2, 1)  # (B, 2, T, F)
        x = self.conv(x)  # (B, 32, T, F)
        x = self.gln(x)  # (B, 32, T, F)

        # TF-Grid blocks
        for i in range(self.num_blks):
            x = self.tf_grid_blks[i](x)  # (B, 32, T, F)

        x = self.de_conv(x)  # (B, 4, T, F)
        x = x.permute(0, 3, 2, 1)  # (B, F, T, 4)

        spk1_stft_real, spk1_stft_imag = x[:, :, :, 0], x[:, :, :, 1]
        spk2_stft_real, spk2_stft_imag = x[:, :, :, 2], x[:, :, :, 3]

        spk1_stft = spk1_stft_real + 1j * spk1_stft_imag
        spk2_stft = spk2_stft_real + 1j * spk2_stft_imag

        spk1_pred = self.stft_decoder(spk1_stft)  # (B, T)
        spk1_pred = F.pad(spk1_pred, (0, n_samples - spk1_pred.shape[-1]))  # (B, T)

        spk2_pred = self.stft_decoder(spk2_stft)  # (B, T)
        spk2_pred = F.pad(spk2_pred, (0, n_samples - spk2_pred.shape[-1]))  # (B, T)

        spk_pred = torch.stack([spk1_pred, spk2_pred], dim=1)  # (B, 2, T)

        return spk_pred  # (B, 2, T)


if __name__ == '__main__':
    x = torch.randn(1, 16000)
    model = TF_GridNet().eval()
    y = model(x)
    print(y.shape)

    flops, params = get_model_complexity_info(model, (16000,), as_strings=True,
                                              print_per_layer_stat=True, verbose=True)
    print(flops, params)

