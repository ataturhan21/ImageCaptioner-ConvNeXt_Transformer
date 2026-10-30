# Image Captioning with ConvNeXt and a Transformer

A PyTorch research notebook that generates image captions with a pretrained ConvNeXt Small image encoder and a custom Transformer decoder. It covers Flickr8k data preparation, model training, hyperparameter experiments, checkpointing, and greedy caption generation.

**[Open the notebook](Caption%20Generation%20Project.ipynb)** · **[Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)** · **[License](LICENSE)**

## Architecture

| Component | Implementation |
| --- | --- |
| Images | Resize and pad to 224 × 224 while preserving aspect ratio; apply ImageNet normalization |
| Encoder | ConvNeXt Small with ImageNet weights; pooling and classifier removed |
| Visual features | Flatten 768-channel spatial features into a sequence for cross-attention |
| Decoder | Custom Transformer with sinusoidal positional encoding |
| Text | GPT-2 tokenizer with added beginning, unknown, and padding tokens |
| Training | Teacher forcing, cross-entropy loss with padding ignored, AdamW, and cosine learning-rate schedules |
| Generation | Greedy token selection until the end token or length limit |

GPT-2 supplies the tokenizer only. The notebook trains its own caption decoder rather than loading a pretrained GPT-2 language model.

## Experiments

Unique image filenames are split into approximately 80% training, 10% validation, and 10% test data with `random_state=42`.

The optional experiment section compares decoder depths of 5, 6, and 7 layers and dropout values of 0.3, 0.4, and 0.5 on subsets of 250 training and 250 validation images.

The final training configuration uses 7 decoder layers, 8 attention heads, a 768-dimensional embedding, a 2048-dimensional feedforward layer, dropout of 0.3, batch size 8, and 2 epochs. These are configured settings, not a claim of optimal performance.

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
6. Load the trained checkpoint and run the prediction cells to inspect captions.

### Dependencies

The notebook imports PyTorch, torchvision, Transformers, torchmetrics, TensorBoard, NumPy, pandas, scikit-learn, matplotlib, Pillow, and torchviz. Its installation cell installs only `torchviz`; the remaining packages must be available in the environment.

For local Jupyter use, obtain the dataset separately and update the dataset, split, and checkpoint paths. CUDA is selected when available; otherwise the model uses the CPU. The repository does not include a dependency lockfile or a verified current environment.

## Outputs

- `train.txt`, `val.txt`, and `test.txt`: image splits.
- `cap_gen_exp/`: TensorBoard logs and training checkpoints.
- `cap_gen_exp/hpo/`: optional hyperparameter experiment logs.
- Prediction examples: 5 test images with reference and generated captions.

The default loading cell expects `/kaggle/working/cap_gen_exp/model_epoch_2_dropout_0.3_layers_7.pt`. Train first or provide a compatible checkpoint; the dataset and trained checkpoints are not included.

## Evaluation note

The ROUGE-L evaluation cell currently passes the final loop's `references` value instead of the accumulated `pred_results["references"]`. Correct the reference alignment before reporting a dataset score. Caption references are also sampled randomly during dataset access, so rigorous evaluation needs a fixed reference policy.

Saved outputs document an experimental run, including imperfect captions. They are not a validated benchmark.

## License

[GNU General Public License v3](LICENSE). Dataset and pretrained-model assets have their own terms.
