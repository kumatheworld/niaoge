import os
import pandas as pd
import soundfile as sf

def get_num_frames(train_audio_dir, bird, mp3_file):
    audio_file = os.path.splitext(mp3_file)[0] + '.wav'
    audio_path = os.path.join(train_audio_dir, bird, audio_file)

    y, sr = sf.read(audio_path)
    assert sr == 32000, f"sample rate 32000 expected but {sr} found"

    num_frames = len(y)
    return num_frames

def add_num_frames():
    data_dir = 'data'
    train_csv = os.path.join(data_dir, 'train.csv')
    train_audio_dir = os.path.join(data_dir, 'train_audio')

    df = pd.read_csv(train_csv)
    num_frames_list = [
        get_num_frames(train_audio_dir, row['ebird_code'], row['filename'])
        for _, row in df.iterrows()
    ]

    df['num_frames'] = num_frames_list
    df.to_csv(train_csv)

if __name__ == '__main__':
    add_num_frames()
