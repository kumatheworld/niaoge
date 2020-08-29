### Prepare
* Download all data from https://www.kaggle.com/c/birdsong-recognition/data and put them under `data/`.
* Run the following to resample training audio files. It will take long!
  ```bash
  for audio in data/train_audio/*/*; do
    tmp=tmp.mp3
    ffmpeg -i $audio -ar 32000 $tmp
    mv $tmp $audio
  done
  ```
* Download a whatever PANN pretrained model from [here](https://zenodo.org/record/3987831#.X0j3PdMzblw) and put it under `audioset_tagging_cnn/`. Modify `configs/default.yaml` accordingly. If your downloaded model is `Wavegram_Logmel_Cnn14_mAP=0.439.pth`, you don't have to edit the config file.
* Install packages by ```pip install -r requirements.txt```.

### Train
* Run ```python train.py```.
* Watch training progress by ```tensorboard --logdir runs```.

### Test
* Will be out soon!

### TODO
* Get more juice out of data!
    * Roughly estimate a bird singing or not during a given interval
    * Augmentation
    * Respect co-occurrence of birds
    * Current audio mix/padding correct?
