from torch.utils.data import Dataset
import os
import random
import soundfile as sf
import numpy as np

class BirdcallDataset(Dataset):
    def __init__(self, df, likelihood=[0,1], duration=5.):
        super().__init__()
        self.df = df
        self.likelihood = likelihood
        self.duration = duration
        self.sr = 32000
        self.data_dir = 'data'
        self.len = len(self.df)

        # save some frequently used objects
        self.train_audio_dir = os.path.join(self.data_dir, 'train_audio')
        self.bird_list = sorted(os.listdir(self.train_audio_dir))
        self.num_birds = len(self.bird_list)

        # get list of list of available audio files from df
        self.available_audios = [
            [os.path.splitext(file)[0] + '.wav'
                for file in self.df.loc[df['ebird_code']==bird, 'filename']]
            for bird in self.bird_list
        ]

        # get available bird ids from df
        # some birds might not show up at all when df is small
        self.available_bird_ids = [
            idx for idx in range(self.num_birds) if self.available_audios[idx]
        ]

        # for now there's no further use of self.df
        del(self.df)

    def _get_random_interval(self, bird_id):
        """
        Get random audio of bird_id
        and pick random interval of length self.duration.
        Return audio array of length self.sr * self.duration.
        """
        audio_file = random.choice(self.available_audios[bird_id])
        audio_path = os.path.join(self.train_audio_dir,
                                  self.bird_list[bird_id], audio_file)
        y, _ = sf.read(audio_path)
        y = y.astype(np.float32)

        len_y = len(y)
        interval_length = int(self.sr * self.duration)
        if len_y < interval_length:
            # pad y with zeros
            new_y = np.zeros(interval_length, dtype=y.dtype)
            start = np.random.randint(interval_length - len_y)
            end = start + len_y
            new_y[start:end] = y
            y = new_y
        elif len_y > interval_length:
            # trim y
            start = np.random.randint(len_y - interval_length)
            end = start + interval_length
            y = y[start:end]

        return y

    def __len__(self):
        # rough length
        return self.len

    def __getitem__(self, index: int):
        # NOTE: index not used since this is randomized!

        # determine number of labels based on likelihood
        num_labels = random.choices(range(len(self.likelihood)), self.likelihood)[0]

        # get that many audio files
        audio_array = []
        few_hot_label = np.zeros(self.num_birds, np.float32)
        if num_labels == 0:
            # NOTE: currently NOT implemented... for now expect likelihood[0] == 0
            # TODO: load random audio from external data or generate noise
            raise NotImplementedError
        else:
            labels = random.sample(self.available_bird_ids, num_labels)
            for bird_idx in labels:
                few_hot_label[bird_idx] = 1
            audio_arrays = [self._get_random_interval(bird_idx) for bird_idx in labels]
            audio_array = np.stack(audio_arrays).mean(0)

        return audio_array, few_hot_label
