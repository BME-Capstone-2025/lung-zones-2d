# custom 2d lung zone detection

## Folder Structure: 

project/
├── data/
│   ├── raw/                 # Original data (images, annotations, etc.)
│   └── processed/           # Preprocessed/augmented data ready for training
├── models/
│   ├── checkpoints/         # Model checkpoints during training
│   └── saved_model/         # Final saved models (e.g., TensorFlow SavedModel or TFLite)
├── notebooks/               # Jupyter notebooks for exploration, visualization, and debugging
├── src/
│   ├── __init__.py          # Makes src a package
│   ├── data_loader.py       # Custom data loading and preprocessing functions (using tf.data)
│   ├── model.py             # Model definition (e.g., based on MobileNetV2 plus custom FC layers)
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation/testing script
│   └── utils.py             # Utility functions (e.g., visualization, metric computation)
├── tests/                   # Unit tests for your modules
│   └── test_model.py
├── configs/                 # Configuration files (e.g., hyperparameters, paths)
│   └── config.yaml
├── requirements.txt         # List of dependencies
└── README.md                # Project overview, setup instructions, usage, etc.