import os
import random
import numpy as np
import soundfile as sf
import torch
from torch import nn
import torchaudio
from torch.utils import data
from torchaudio_augmentations import (
    RandomResizedCrop,
    RandomApply,
    PolarityInversion,
    Noise,
    Gain,
    HighLowPass,
    Delay,
    PitchShift,
    Reverb,
    Compose,
)

# https://music-classification.github.io/tutorial/part3_supervised/tutorial.html

CATEGORIES = ["metal", "glass", "ceramic", "plastic", "fibre"]
ADJECTIVES = {
    "metal": ("resonant and echoing", "metallic", "ringing"),
    "glass": ("tinkling",),
    "ceramic": ("clinking and rattling", "rattling"),
    "plastic": ("dull", "muffled"),
    "fibre": ("muted", "silent"),
}
SAMPLERATE = 22050


def adjust_audio_length(wav, num_samples=SAMPLERATE * 1):
    random_index = random.randint(0, len(wav) // 10)
    wav = wav[random_index : random_index + num_samples]
    wav = np.pad(wav, (0, -len(wav) + num_samples), "constant")
    return wav


class ImpactSoundDataset(data.Dataset):
    def __init__(self, data_path, ignore, split, num_samples, is_augmentation):
        self.data_path = data_path if data_path else ""
        self.split = split
        self.num_samples = num_samples
        self.is_augmentation = is_augmentation
        cats = []
        for cat in CATEGORIES:
            if ignore is not None and cat in ignore:
                continue
            cats.append(cat)
        self.cats = cats
        self._get_wavlist()
        if is_augmentation:
            self._get_augmentations()

    def _get_wavlist(self):
        wavlist = []
        for cat in self.cats:
            datapath = f"{self.data_path}/{cat}/{self.split}"
            filepaths = [
                f"{datapath}/{filename}"
                for filename in os.listdir(datapath)
                if filename.endswith(".wav")
            ]
            wavlist.extend(filepaths)
        self.wavlist = wavlist

    def _get_augmentations(self):
        transforms = [
            RandomResizedCrop(n_samples=self.num_samples),
            RandomApply([PolarityInversion()], p=0.8),
            RandomApply([Noise(min_snr=0.3, max_snr=0.5)], p=0.3),
            RandomApply([Gain()], p=0.2),
            RandomApply([HighLowPass(sample_rate=SAMPLERATE)], p=0.8),
            RandomApply([Delay(sample_rate=SAMPLERATE)], p=0.5),
            RandomApply(
                [PitchShift(n_samples=self.num_samples, sample_rate=SAMPLERATE)], p=0.4
            ),
            RandomApply([Reverb(sample_rate=SAMPLERATE)], p=0.3),
        ]
        self.augmentation = Compose(transforms=transforms)

    def _adjust_audio_length(self, wav):
        return adjust_audio_length(wav, num_samples=self.num_samples)

    def __getitem__(self, index):
        filepath = self.wavlist[index]

        # get genre
        cat = filepath.split("/")[-3]
        cat_index = self.cats.index(cat)

        # get audio
        wav, fs = sf.read(filepath)
        wav = wav[:, 0]

        # adjust audio length
        wav = self._adjust_audio_length(wav).astype("float32")

        # data augmentation
        if self.is_augmentation:
            wav = (
                self.augmentation(torch.from_numpy(wav).unsqueeze(0)).squeeze(0).numpy()
            )

        return wav, cat_index

    def __len__(self):
        return len(self.wavlist)


def get_dataloader(
    data_path=None,
    ignore=None,
    split="train",
    num_samples=SAMPLERATE * 1,
    batch_size=8,
    num_workers=0,
    is_augmentation=False,
):
    is_shuffle = True if (split == "train") else False
    data_loader = data.DataLoader(
        dataset=ImpactSoundDataset(
            data_path, ignore, split, num_samples, is_augmentation
        ),
        batch_size=batch_size,
        shuffle=is_shuffle,
        drop_last=False,
        num_workers=num_workers,
    )
    return data_loader


class Conv_2d(nn.Module):
    def __init__(
        self, input_channels, output_channels, shape=3, pooling=2, dropout=0.1
    ):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(
            input_channels, output_channels, shape, padding=shape // 2
        )
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(pooling)
        self.dropout = nn.Dropout(dropout)

    def forward(self, wav):
        out = self.conv(wav)
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dropout(out)
        return out


class CNN(nn.Module):
    def __init__(
        self,
        num_channels=16,
        sample_rate=22050,
        n_fft=1024,
        f_min=0.0,
        f_max=11025.0,
        num_mels=128,
        num_classes=4,
    ):
        super(CNN, self).__init__()

        # mel spectrogram
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=num_mels,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.input_bn = nn.BatchNorm2d(1)

        # convolutional layers
        self.layer1 = Conv_2d(1, num_channels, pooling=(3, 2))
        self.layer2 = Conv_2d(num_channels, num_channels * 2, pooling=(3, 2))
        self.layer3 = Conv_2d(num_channels * 2, num_channels * 2, pooling=(3, 3))
        self.layer4 = Conv_2d(num_channels * 2, num_channels * 2, pooling=(3, 3))

        # dense layers
        self.dense1 = nn.Linear(num_channels * 2, num_channels * 2)
        self.dense_bn = nn.BatchNorm1d(num_channels * 2)
        self.dense2 = nn.Linear(num_channels * 2, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, wav):
        # input Preprocessing
        out = self.melspec(wav)  # wav [8 x 22050] -> melspec [8 x 128 x 44]
        out = self.amplitude_to_db(out)

        # input batch normalization
        out = out.unsqueeze(1)  # shape (8, 1, 128, 44)
        out = self.input_bn(out)

        # convolutional layers
        out = self.layer1(out)  # shape (8, 16, 64, 22)
        out = self.layer2(out)  # shape (8, 32, 32, 11)
        out = self.layer3(out)
        out = self.layer4(out)

        # reshape. (batch_size, num_channels, 1, 1) -> (batch_size, num_channels)
        out = out.reshape(len(out), -1)

        # dense layers
        out = self.dense1(out)
        out = self.dense_bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.dense2(out)

        return out


SAVE_MODEL_PATH = "best_model.ckpt"


class SoundClassifier:
    def __init__(
        self,
        data_path=None,
        ignore=[
            "fibre",
        ],
        save_model_path=None,
        num_samples=SAMPLERATE * 1,
        device=None,
    ) -> None:
        self.data_path = data_path
        self.ignore = ignore
        if save_model_path is None:
            save_model_path = SAVE_MODEL_PATH
        self.num_samples = num_samples
        self.save_model_path = save_model_path
        model = CNN()
        self.model = model
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

    def offline_train(self, num_epochs=300, lr=0.001):
        from sklearn.metrics import accuracy_score, confusion_matrix

        device = self.device
        model = self.model
        model = model.to(device)
        train_loader = get_dataloader(
            data_path=self.data_path,
            ignore=self.ignore,
            split="train",
            is_augmentation=True,
        )
        for train_wav, train_cat in train_loader:
            print(train_wav.shape)
            print(train_cat)
            break

        # # valid_loader = get_dataloader(split='valid')
        test_loader = get_dataloader(data_path=self.data_path, ignore=self.ignore, split="test")
        for test_wav, test_cat in test_loader:
            print(test_wav.shape)
            print(test_cat)
            break

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        valid_losses = []
        num_epochs = num_epochs

        for epoch in range(num_epochs):
            losses = []

            # Train
            model.train()
            for wav, genre_index in train_loader:
                wav = wav.to(device)
                genre_index = genre_index.to(device)

                # Forward
                out = model(wav)
                loss = loss_function(out, genre_index)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            print(
                "Epoch: [%d/%d], Train loss: %.4f"
                % (epoch + 1, num_epochs, np.mean(losses))
            )

            # Validation
            model.eval()
            y_true = []
            y_pred = []
            losses = []
            for wav, cat_index in test_loader:
                wav = wav.to(device)
                cat_index = cat_index.to(device)

                b, c = wav.size()
                logits = model(wav)
                # logits = logits.view(b, c, -1).mean(dim=1)
                loss = loss_function(logits, cat_index)
                losses.append(loss.item())
                _, pred = torch.max(logits.data, 1)

                # append labels and predictions
                y_true.extend(cat_index.tolist())
                y_pred.extend(pred.tolist())
            accuracy = accuracy_score(y_true, y_pred)
            valid_loss = np.mean(losses)
            print(
                "Epoch: [%d/%d], Valid loss: %.4f, Valid accuracy: %.4f"
                % (epoch + 1, num_epochs, valid_loss, accuracy)
            )

            # Save model
            valid_losses.append(valid_loss.item())
            if np.argmin(valid_losses) == epoch:
                print("Saving the best model at %d epochs!" % epoch)
                torch.save(model.state_dict(), self.save_model_path)

    # Load the best model
    def offline_test(self, plot_on=False):
        from sklearn.metrics import accuracy_score, confusion_matrix

        model_parameters = torch.load(self.save_model_path)
        model = self.model
        device = self.device
        model = model.to(device)
        model.load_state_dict(model_parameters)
        # Run evaluation
        model.eval()
        y_true = []
        y_pred = []

        test_loader = get_dataloader(
            data_path=self.data_path, ignore=self.ignore, split="test"
        )
        with torch.no_grad():
            for wav, cat_index in test_loader:
                wav = wav.to(device)
                cat_index = cat_index.to(device)

                # reshape and aggregate chunk-level predictions
                b, c = wav.size()
                logits = model(wav)
                # logits = logits.view(b, c, -1).mean(dim=1)
                _, pred = torch.max(logits.data, 1)

                # append labels and predictions
                y_true.extend(cat_index.tolist())
                y_pred.extend(pred.tolist())

        accuracy = accuracy_score(y_true, y_pred)
        print("Accuracy: %.4f" % accuracy)
        if plot_on:
            import seaborn as sns
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(
                cm,
                annot=True,
                xticklabels=CATEGORIES,
                yticklabels=CATEGORIES,
                cmap="YlGnBu",
            )

        return

    def inference(self, sound_path):
        if sound_path == "None":
            top_materials = ("fibre", "plastic")
            top_probs = (1.0, 0.0)
            top_adjectives = [
                np.random.choice(ADJECTIVES[material]) for material in top_materials
            ]
            return top_probs, top_materials, top_adjectives
        model = self.model
        device = torch.device("cpu")
        model.to(device)
        model_paramters = torch.load(self.save_model_path, map_location=device)
        model.load_state_dict(model_paramters)
        model.eval()
        wav, fs = sf.read(sound_path)
        wav = wav[:, 0]
        wav = adjust_audio_length(wav, self.num_samples).astype("float32")
        # make a single sample to have a "batch axis" = 1
        wav = torch.from_numpy(wav).unsqueeze(0)
        with torch.no_grad():
            # wav = wav.to(self.device)
            # b, c = wav.size()
            logits = model(wav).squeeze()
            probs = logits.softmax(axis=0)
            # _, pred = torch.max(logits.data, 1)
            # material = CATEGORIES[pred]
            top_probs, top_indices = torch.topk(probs, k=2)

        top_probs = top_probs.tolist()
        top_materials = [CATEGORIES[index] for index in top_indices.tolist()]
        new_top_materials = []
        for m in top_materials:
            if m == "fibre":
                m = "unknown material becasue there is no obvious impact sound"
            new_top_materials.append(m)
        top_adjectives = [
            np.random.choice(ADJECTIVES[material]) for material in top_materials
        ]
        return top_probs, new_top_materials, top_adjectives
