# Image Captioning with ConvNeXt and a Transformer

A PyTorch research notebook that generates image captions with a pretrained ConvNeXt Small image encoder and a custom Transformer decoder. It covers Flickr8k data preparation, model training, hyperparameter experiments, checkpointing, and greedy caption generation.

**[Open the notebook](Caption%20Generation%20Project.ipynb)** · **[Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)** · **[License](LICENSE)**

## Contents

- [Architecture](#architecture)
- [Experiments](#experiments)
- [Saved results](#saved-results)
- [Run on Kaggle](#run-on-kaggle)
- [Outputs](#outputs)
- [Evaluation note](#evaluation-note)
- [Regression checks](#regression-checks)
- [License](#license)

## Architecture

| Component | Implementation |
| --- | --- |
| Images | Resize and pad to 224 × 224 while preserving aspect ratio; apply ImageNet normalization |
| Encoder | ConvNeXt Small with ImageNet weights; pooling and classifier removed |
| Visual features | Flatten 768-channel spatial features into a sequence for cross-attention |
| Decoder | Custom Transformer with sinusoidal positional encoding |
| Text | GPT-2 tokenizer with added beginning, unknown, and padding tokens |
| Training | Teacher forcing, cross-entropy loss with padding ignored, AdamW (learning rate 1e-5 for the encoder, 1e-4 for the decoder), and cosine learning-rate schedules |
| Generation | Greedy token selection with the same causal mask as training, until the end token or length limit |

GPT-2 supplies the tokenizer only. The notebook trains its own caption decoder rather than loading a pretrained GPT-2 language model.

## Experiments

Unique image filenames are split into approximately 80% training, 10% validation, and 10% test data with `random_state=42`.

The optional experiment section compares decoder depths of 5, 6, and 7 layers and dropout values of 0.3, 0.4, and 0.5 on subsets of 250 training and 250 validation images.

The final training configuration uses 7 decoder layers, 8 attention heads, a 768-dimensional embedding, a 2048-dimensional feedforward layer, dropout of 0.3, batch size 8, and 2 epochs. These are configured settings, not a claim of optimal performance.

## Saved results

Copied from the outputs saved in the notebook (Kaggle run, May 2024). The run predates the encoder learning-rate, generation-mask and checkpoint-loading fixes, so rerun the notebook before comparing new results with these.

Splits: 8,091 captioned images, split into 6,472 training, 809 validation, and 810 test images.

Hyperparameter experiment (250 training and 250 validation images, 2 epochs), validation loss after epoch 2:

| Decoder layers | Dropout 0.3 | Dropout 0.4 | Dropout 0.5 |
| --- | --- | --- | --- |
| 5 | 5.396 | 5.485 | 5.743 |
| 6 | 5.308 | 5.520 | 5.824 |
| 7 | 5.383 | 5.548 | 5.998 |

The lowest saved validation loss was 6 layers with dropout 0.3; the final run uses 7 layers with dropout 0.3.

Final run (7 layers, dropout 0.3, full training and validation splits):

| Epoch | Training loss | Validation loss |
| --- | --- | --- |
| 1 | 4.547 | 3.730 |
| 2 | 3.632 | 3.319 |

The notebook also shows 5 test images with their reference and generated captions. No ROUGE-L score is saved (see [Evaluation note](#evaluation-note)).

## Run on Kaggle

1. Import `Caption Generation Project.ipynb` into a Kaggle notebook.
2. Attach the [Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k).
3. Enable a GPU and internet access for the initial pretrained-weight and tokenizer downloads.
4. Confirm the dataset paths in **Set Constants**:

   ```python
   IMAGES_PATH = "/kaggle/input/flickr8k/Images"
   CAPTIONS_PATH = "/kaggle/input/flickr8k/captions.txt"
   SPLIT_DIR = "/kaggle/working"
   ```

5. Run the preparation and model-definition cells, then train the model. The **Hyper Parameter Optimization** cell is optional; the later configuration cell initializes the model and optimizers independently.
6. Load the trained checkpoint and run the prediction cells to inspect captions. The checkpoint is loaded onto the selected device, so a GPU-trained checkpoint also loads on a CPU-only machine.

### Dependencies

The notebook imports PyTorch, torchvision, Transformers, torchmetrics, TensorBoard, NumPy, pandas, scikit-learn, matplotlib, Pillow, and torchviz. Its installation cell installs only `torchviz`; the remaining packages must be available in the environment. For a local environment, `python -m pip install -r requirements.txt` installs them unpinned; the torchviz graph cell also needs the Graphviz system package.

The saved outputs came from a Kaggle notebook with a Tesla T4 GPU, Python 3.10, and torch 2.1.2 (from the saved install output and notebook metadata).

For local Jupyter use, obtain the dataset separately and update the dataset, split, and checkpoint paths. CUDA is selected when available; otherwise the model uses the CPU. The repository does not include a dependency lockfile or a verified current environment.

## Outputs

- `train.txt`, `val.txt`, and `test.txt`: image splits.
- `cap_gen_exp/`: TensorBoard logs and training checkpoints.
- `cap_gen_exp/hpo/`: optional hyperparameter experiment logs.
- Prediction examples: 5 test images with reference and generated captions.

The loading cell builds the checkpoint path from the training settings, which is `cap_gen_exp/model_epoch_2_dropout_0.3_layers_7.pt` relative to the working directory (`/kaggle/working` on Kaggle) by default. Train first or provide a compatible checkpoint; the dataset and trained checkpoints are not included. These generated files are listed in `.gitignore`.

## Evaluation note

ROUGE-L evaluation uses the accumulated reference captions aligned with the generated predictions. The previous saved ROUGE-L output was cleared because it used the final loop's reference instead of the full reference list; rerun evaluation before reporting a score. Training samples captions randomly; validation and test access use the first reference for each image consistently. This is a deterministic single-reference evaluation, not a multi-reference benchmark.

Saved outputs document an experimental run, including imperfect captions. They are not a validated benchmark.

## Regression checks

Run the lightweight evaluation checks without downloading models or datasets:

```bash
python -m unittest discover -s tests -v
```

Captions are truncated with BOS/EOS preserved and padded to a fixed batch length. Causal and padding masks both use boolean blocked positions, so padding is not converted into a positive attention bias.

The checks cover score alignment, stable evaluation references, fixed caption lengths, and causal/padding mask semantics. Source checks also confirm that every code cell parses, the encoder optimizers use the encoder learning rate, generation passes the causal mask, and the checkpoint loads with `map_location`. They use NumPy tensor doubles, pandas, and small Pillow images. Actual PyTorch runtime behavior, caption quality, and full training have not been validated by these checks. The read-only GitHub Actions workflow runs the same checks without model downloads.

## License

[MIT](LICENSE). Dataset and pretrained-model assets have their own terms.
