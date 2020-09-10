### Prepare
* Download all data from https://www.kaggle.com/c/birdsong-recognition/data and put them under `data/`.
* Run the following to resample and convert training audio files. It will take about an hour.
  ```bash
  for audio in data/train_audio/*/*.mp3; do
    ffmpeg -i $audio -ar 32000 -ac 1 ${audio%.mp3}.wav
  done
  ```
* Run ```python preprocess.py```. This will add a new column ```num_frames``` to ```data/train.csv```.
* Download a whatever PANN pretrained model from https://zenodo.org/record/3987831#.X0j3PdMzblw and put it under `audioset_tagging_cnn/`. Modify `configs/default.yaml` accordingly. If your downloaded model is `Wavegram_Logmel_Cnn14_mAP=0.439.pth`, you don't have to edit the config file.
* Install packages by ```pip install -r requirements.txt```.

### Train
* Run ```python train.py```.
  * This reads ```configs/default.yaml``` by default. You can tweek hyperparameters either by editing ```configs/default.yaml``` or by creating a new YAML file ```configs/xxx.yaml``` and execute ```python train.py --config xxx``` instead.
* Watch training progress by ```tensorboard --logdir runs```.

### Test
* Run ```python test.py``` to check out how different thresholds affect the F1-score.
  * Actually, the test data is such a garbage that you'll end up seeing score monotonically increase w.r.t. threshold unless your model is very-well trained!

### TODO
* Add nocall data?
