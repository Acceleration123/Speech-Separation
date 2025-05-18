import torch
import fast_bss_eval
import numpy as np
import soundfile as sf
import torch.nn as nn


class SDRLoss(nn.Module):
    def __init__(
            self,
            filter_length=512,
            use_cg_iter=None,
            clamp_db=None,
            zero_mean=True,
            load_diag=None
    ):
        super(SDRLoss, self).__init__()
        self.filter_length = filter_length
        self.use_cg_iter = use_cg_iter
        self.clamp_db = clamp_db
        self.zero_mean = zero_mean
        self.load_diag = load_diag

    def forward(self, est, ref):
        """
        est: (B, 2, T)  output of the neural network
        ref: (B, 2, T)  ground truth
        """
        sdr_loss = fast_bss_eval.sdr_loss(est=est, ref=ref, filter_length=self.filter_length,
                                          use_cg_iter=self.use_cg_iter, zero_mean=self.zero_mean,
                                          clamp_db=self.clamp_db, load_diag=self.load_diag, pairwise=False)

        return torch.mean(sdr_loss)  # negative sdr loss


if __name__ == '__main__':
    # test the SDRLoss
    sdr_loss_func = SDRLoss()

    spk1, _ = sf.read('spk1.wav', dtype='float32')  # (T,)
    spk2, _ = sf.read('spk2.wav', dtype='float32')  # (T,)
    ref = np.stack([spk1, spk2], axis=0)  # (2, T)
    ref = torch.tensor(ref).unsqueeze(0)  # (1, 2, T)
    est = torch.randn(2, len(spk1)).unsqueeze(0)  # (1, 2, T)

    sdr_loss = sdr_loss_func(est, ref)
    print(sdr_loss)






