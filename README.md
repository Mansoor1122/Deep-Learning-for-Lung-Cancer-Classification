# Explainable Lung Cancer Classification Using EfficientNetB0

This project explores transfer learning for classifying chest CT images into four categories. It compares initial and modified EfficientNetB0 training configurations and uses Grad-CAM to visualize regions influencing model predictions.

## Dataset

The notebook downloads the Chest CT-Scan Images dataset using KaggleHub.

- **Dataset identifier:** `mohamedhanyyy/chest-ctscan-images`
- **Source:** https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images
- **Image type:** Chest CT images
- **Total images recorded in the notebook:** 1,000

### Classes and Image Counts

| Class | Training | Validation | Test | Total |
|-------|---------:|-----------:|-----:|------:|
| Adenocarcinoma | 195 | 23 | 120 | 338 |
| Large cell carcinoma | 115 | 21 | 51 | 187 |
| Normal | 148 | 13 | 54 | 215 |
| Squamous cell carcinoma | 155 | 15 | 90 | 260 |
| **Total** | **613** | **72** | **315** | **1,000** |

The notebook uses the dataset's supplied training, validation, and test folders. It does not establish whether these splits are independent at the patient level.

Training and validation folder names include additional anatomical and staging information, while test folder names use shorter class names.

## Project Workflow

1. Download and inspect the dataset.
2. Explore class distributions, sample images, and pixel intensity statistics.
3. Prepare image generators and apply training augmentation.
4. Train initial and modified EfficientNetB0 configurations.
5. Evaluate predictions using classification reports and confusion matrices.
6. Generate Grad-CAM visualizations for selected test images.

## Model

The classifier uses an EfficientNetB0 backbone initialized with ImageNet weights.

The classification head contains:

- Global average pooling.
- Dropout.
- A four-unit softmax output layer.

The notebook includes repeated training and evaluation sections. In the final modified configuration, the backbone is trainable.

### Training Configuration

| Setting | Modified configuration |
|---------|------------------------|
| Input size | 350 × 350 × 3 |
| Optimizer | Adam |
| Initial learning rate | 0.0009 |
| Loss | Categorical cross-entropy |
| Dropout | 0.4 |
| Maximum epochs | 32 |
| Training batch size | 5 |
| Validation batch size | 32 |
| Test batch size | 8 |
| Early stopping patience | 9 |

Callbacks include model checkpointing, early stopping, learning-rate reduction, and TensorBoard logging.

## Preprocessing and Augmentation

The image generators rescale pixel values by dividing by 255.

Training augmentation includes:

- Horizontal flipping.
- Zooming and shearing.
- Horizontal and vertical shifts.
- Small rotations.
- Nearest-neighbor filling of transformed regions.

Validation and test images are rescaled without training augmentation. Test-image shuffling is disabled for evaluation.

## Recorded Results

The notebook contains outputs from multiple training executions. These are separate recorded runs, rather than a single reproducible benchmark.

| Recorded execution | Test accuracy |
|--------------------|--------------:|
| Earlier initial model evaluation | 39.37% |
| Earlier modified model evaluation | 91.43% |
| Later initial model retraining | 16.19% |
| Final modified model retraining | 83.17% |

The variation indicates that results need to be confirmed using a consistent execution, fixed random seeds, and preserved model checkpoints.

### Final Recorded Modified Model

The final evaluation uses 315 test images.

| Metric | Value |
|--------|------:|
| Accuracy | 83.17% |
| Loss | 0.4830 |
| Macro precision | 0.8598 |
| Macro recall | 0.8613 |
| Macro F1-score | 0.8394 |
| Weighted F1-score | 0.8329 |

### Per-Class Performance

| Class | Precision | Recall | F1-score | Test images |
|-------|----------:|-------:|---------:|------------:|
| Adenocarcinoma | 0.8293 | 0.8500 | 0.8395 | 120 |
| Large cell carcinoma | 0.6098 | 0.9804 | 0.7519 | 51 |
| Normal | 1.0000 | 0.9815 | 0.9907 | 54 |
| Squamous cell carcinoma | 1.0000 | 0.6333 | 0.7755 | 90 |

These values are taken from saved notebook outputs and have not been independently reproduced for this README.

## Explainability with Grad-CAM

The notebook includes Grad-CAM analysis targeting EfficientNetB0's `top_conv` layer.

It computes gradients for a selected class, weights the convolutional feature maps, and overlays the resulting heatmap on the input image.

These visualizations help inspect regions influencing a prediction. They do not establish that highlighted regions correspond to clinically verified tumors.

## Requirements

Main dependencies include:

- TensorFlow / Keras
- KaggleHub
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Pillow
- OpenCV
- Keras Tuner

Exact dependency versions are not pinned.

## Running the Notebook

1. Open `lung-cancer-classification.ipynb` in Google Colab.
2. Select a GPU runtime.
3. Install missing dependencies:

       pip install tensorflow kagglehub numpy matplotlib seaborn scikit-learn pillow opencv-python keras-tuner

4. Run the KaggleHub download cell.
5. Use the paths constructed from the downloaded dataset directory rather than the older `../input/` paths.
6. Run data exploration and preprocessing.
7. Choose the training configuration to reproduce.
8. Train the model or load its matching saved checkpoint.
9. Run evaluation and Grad-CAM analysis.

Some later cells repeat earlier setup and retrain models. Review these sections before running the entire notebook, as they may overwrite existing checkpoints.

## Reproducibility Notes

- Saved outputs come from multiple training runs.
- Some training logs contain input-exhaustion warnings.
- Explicit validation steps use only part of the validation generator per epoch.
- Input scaling should be checked against the preprocessing included in the installed EfficientNetB0 implementation.
- A clean rerun is needed before treating the reported scores as a final benchmark.

## Resources and Acknowledgments

- **Dataset:** https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images
- **Helper functions referenced in the notebook:** https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/extras/helper_functions.py

Dataset ownership and reused helper-code credits remain with their respective creators.

## Author

Mansoor Ahmad

## Intended Use

This project is intended for research and education. It has not been validated for clinical diagnosis.
