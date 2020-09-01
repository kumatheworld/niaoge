from torch.utils.data import Dataset
import os
import random
import soundfile as sf
import numpy as np

class BirdcallDataset(Dataset):
    def __init__(self, df, duration=5., likelihood=[0,1]):
        super().__init__()
        sr = 32000
        self.num_frames = int(sr * duration)
        self.likelihood = likelihood
        self.len = len(df)
        self.train_audio_dir = os.path.join('data', 'train_audio')
        self.bird_list = sorted(os.listdir(self.train_audio_dir))
        self.num_birds = len(self.bird_list)

        # get list of list of available audio files from df
        self.available_audios = [
            [os.path.splitext(file)[0] + '.wav'
                for file in df.loc[df['ebird_code']==bird, 'filename']]
            for bird in self.bird_list
        ]

        # get available bird ids from df
        # some birds might not show up at all when df is small
        self.available_bird_ids = [
            idx for idx in range(self.num_birds) if self.available_audios[idx]
        ]

    def _get_random_interval(self, bird_id):
        """
        Get random audio of bird_id pick random interval.
        Return audio array of length self.num_frames.
        """
        audio_file = random.choice(self.available_audios[bird_id])
        audio_path = os.path.join(self.train_audio_dir,
                                  self.bird_list[bird_id], audio_file)
        y, _ = sf.read(audio_path, dtype='float32')

        num_frames = len(y)
        if num_frames < self.num_frames:
            # pad y with zeros
            new_y = np.zeros(self.num_frames, dtype=y.dtype)
            start = np.random.randint(self.num_frames - num_frames)
            end = start + num_frames
            new_y[start:end] = y
            y = new_y
        elif num_frames > self.num_frames:
            # trim y
            start = np.random.randint(num_frames - self.num_frames)
            end = start + self.num_frames
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
