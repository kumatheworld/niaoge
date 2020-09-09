from torch.utils.data import Dataset
import os
import random
import soundfile as sf
import numpy as np
import pandas as pd
import warnings
import librosa

class TrainDataset(Dataset):
    def __init__(self, df, duration=5., likelihood=[0,1], noise=None):
        super().__init__()
        self.sr = 32000
        self.num_frames = int(self.sr * duration)
        self.likelihood = likelihood
        self.noise = noise

        self.len = len(df)
        self.train_audio_dir = os.path.join('data', 'train_audio')
        self.bird_list = sorted(os.listdir(self.train_audio_dir))
        self.num_birds = len(self.bird_list)

        # get list of list of available audio files from df
        self.available_audios = [
            [self._get_audio_info(bird_id, mp3)
                for mp3 in df.loc[df['ebird_code']==bird, 'filename']]
            for bird_id, bird in enumerate(self.bird_list)
        ]

        # get available bird ids from df
        # some birds might not show up at all when df is small
        self.available_bird_ids = [
            idx for idx in range(self.num_birds) if self.available_audios[idx]
        ]

    def _get_audio_info(self, bird_id, mp3_file):
        audio_file = os.path.splitext(mp3_file)[0] + '.wav'
        audio_path = os.path.join(self.train_audio_dir,
                                  self.bird_list[bird_id], audio_file)
        y, sr = sf.read(audio_path)
        assert sr == self.sr, f'sample rate {self.sr} expected but {sr} found'
        num_frames = len(y)

        audio_info = {
            'audio_path': audio_path,
            'num_frames': num_frames
        }
        return audio_info

    def _get_random_interval(self, bird_id):
        """
        Get random audio of bird_id pick random interval.
        Return audio array of length self.num_frames.
        """
        audio_info = random.choice(self.available_audios[bird_id])
        audio_path = audio_info['audio_path']
        num_frames = audio_info['num_frames']

        if num_frames < self.num_frames:
            # pad with zeros
            y = np.zeros(self.num_frames, dtype='float32')
            start = np.random.randint(self.num_frames - num_frames + 1)
            end = start + num_frames
            y[start:end] = sf.read(audio_path, dtype='float32')[0]
        else:
            # trim
            start = np.random.randint(num_frames - self.num_frames + 1)
            end = start + self.num_frames
            y = sf.read(audio_path, dtype='float32',
                        frames=self.num_frames, start=start)[0]

        return y

    def __len__(self):
        # rough length
        return self.len

    def __getitem__(self, index):
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

        if self.noise:
            audio_array += self.noise * np.random.randn(self.num_frames)

        return audio_array, few_hot_label

class TestDataset(Dataset):
    def __init__(self):
        super().__init__()
        data_dir = 'data'
        self.sr = 32000

        csv_path = os.path.join(data_dir, 'example_test_audio_summary.csv')
        self.df = pd.read_csv(csv_path)
        self.len = len(self.df)

        df_last = self.df.drop_duplicates(subset=['filename'], keep='last')
        durations = df_last['seconds'].values
        audio_dir = os.path.join(data_dir, 'example_test_audio')
        audio_arrays = [
            self._get_audio_array(os.path.join(audio_dir, audio_file), duration)
            for audio_file, duration in zip(os.listdir(audio_dir), durations)
        ]
        self.data = np.concatenate(audio_arrays).reshape(self.len, 5 * self.sr)

        train_audio_dir = os.path.join(data_dir, 'train_audio')
        self.bird_list = sorted(os.listdir(train_audio_dir))
        self.num_birds = len(self.bird_list)
        self.bird2id = {bird: idx for idx, bird in enumerate(self.bird_list)}
        birds_lists = self.df['birds'].fillna('').values
        self.label = np.stack([self._get_few_hot(birds) for birds in birds_lists])

    def _get_audio_array(self, audio_path, duration):
        warnings.simplefilter('ignore')
        y, _ = librosa.load(audio_path, sr=self.sr, duration=duration)
        warnings.resetwarnings()
        return y

    def _get_few_hot(self, birds):
        few_hot = np.zeros(self.num_birds)
        for bird in birds.split():
            if bird in self.bird_list:
                few_hot[self.bird2id[bird]] = 1
        return few_hot

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data[index], self.label[index]
