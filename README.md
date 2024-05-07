# Speech Command Recognition

Authors:
- [Mikołaj Gałkowski](https://www.github.com/galkowskim)
- [Hubert Bujakowski](https://www.github.com/hbujakow)

This project was conducted as part of a deep learning course, focusing on speech command recognition using recurrent neural network (RNN) architectures. The objective was to develop models capable of accurately classifying spoken commands from audio samples. The dataset used for training and evaluation consisted of speech command recordings, categorized into various classes representing different spoken commands.

# Training

1. Set up the training environment and install required packaged defined in the `requirements.txt` file.
2. Navigate to the `experiments` directory.
3. Each model type has its own directory containing configuration files for training. The configuration files define the model architecture, training parameters.

- For LSTM, GRU, and RNN models, navigate to the `LSTM` directory. And training is run using:
```bash
python lstm.py --SEED <seed>
```

- For Whisper and AST models, navigate to the `whisper` and `AST` directories, respectively. And training is run using:
```bash
python main_trainer.py
```


# Model checkpoints

Model checkpoints are available in the `LSTM/results` and `whisper/checkpoints` directories. The checkpoints for the AST are not available due to their large size, however we include all configuration files used for training, which can be used to reproduce the results.

## Results for RNN, LSTM, and GRU based models

All experiments were run 3 times with different random seeds. The results are presented as the average accuracy and standard deviation.

| Model Name | Num. Layers | Hidden Size | Avg. Accuracy |
|------------|-------------|-------------|---------------|
| GRUModel   | 4           | 256         | **0.837 ± 0.004** |
| LSTMModel  | 4           | 256         | **0.846 ± 0.004** |
| RNNModel   | 4           | 16          | **0.255 ± 0.038** |

## Results for Whisper and AST Models

| Model Name | Pretrained | Avg. Accuracy |
|------------|------------|---------------|
| AST        | ✓          | **0.865 ± 0.000** |
| AST        | X          | 0.680 ± 0.004 |
| Whisper    | ✓          | **0.845 ± 0.002** |
| Whisper    | X          | 0.673 ± 0.008 |


From these results, we observe that LSTM and GRU models with 4 recurrent layers and a hidden size of 256 achieve the highest accuracy in classifying speech commands. Furthermore, pretrained AST demonstrates promising performance, emphasizing the potential of transfer learning in speech recognition tasks.

## Results with models combined with silence detection model

All models struggled with detecting silence (this can be seen in the confusion matrices available in `LSTM/results`, `whisper/checkpoints`, `AST/checkpoints`), which was a separate class in the dataset. To improve the results, we combined the models with a silence detection model. The results are presented only for best performing models with seed 1.


| Model Name | Pretrained | Accuracy | Accuracy with Silence Detection Model |
|------------|------------|----------|---------------------------------------|
| LSTMModel  | X          | 0.846    | **0.887**                            |
| AST        | ✓          | 0.865    | **0.971**                            |
| Whisper    | ✓          | 0.845    | **0.950**                            |