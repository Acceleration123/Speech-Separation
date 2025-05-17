import random
import librosa
import numpy as np
import soundfile as sf
from torch.utils import data

TRAIN_MIX_DATABASE = "D:\\Users\\LibriMix_Mine\\train\\mix"
VALID_MIX_DATABASE = "D:\\Users\\LibriMix_Mine\\valid\\mix"


class dataset(data.Dataset):
    def __init__(
            self,
            num_tot=60000,
            num_per_epoch=10000,
            train=True
    ):
        super(dataset, self).__init__()
        self.train_mix_database = sorted(librosa.util.find_files(TRAIN_MIX_DATABASE, ext="wav"))[:num_tot]
        self.valid_mix_database = sorted(librosa.util.find_files(VALID_MIX_DATABASE, ext="wav"))
        self.num_per_epoch = num_per_epoch
        self.train = train

    def __len__(self):
        if self.train:
            return self.num_per_epoch
        else:
            return len(self.valid_mix_database)

    def __getitem__(self, idx):
        if self.train:
            mix_database = random.sample(self.train_mix_database, self.num_per_epoch)
        else:
            mix_database = self.valid_mix_database

        mix, fs = sf.read(mix_database[idx], dtype='float32')  # (T,)
        spk1, _ = sf.read(mix_database[idx].replace("mix", "spk1"), dtype='float32')  # (T,)
        spk2, _ = sf.read(mix_database[idx].replace("mix", "spk2"), dtype='float32')  # (T,)
        label = np.stack([spk1, spk2], axis=0)  # (2, T)

        return mix, label


if __name__ == '__main__':
    # test TrainDataset
    train_dataset = dataset(num_per_epoch=10, train=True)
    train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    for i, (mix, label) in enumerate(train_loader):
        print(mix.shape, label.shape)

    # test ValidDataset
    valid_dataset = dataset(train=False)
    valid_loader = data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)
    for i, (mix, label) in enumerate(valid_loader):
        print(mix.shape, label.shape)

