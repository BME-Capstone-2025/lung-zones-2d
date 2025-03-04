import os
import tensorflow as tf
from model import get_model
from data_loader import get_dataset  # Assuming your data loader code is in data_loader.py

def main():
    # -----------------------------
    # Configurations for model, training, and data.
    # -----------------------------
    model_config = {
        "model_type": "mobilenetv2",      # Change this to try other models (e.g. "resnet50")
        "input_shape": (224, 224, 3),
        "num_keypoints": 8,               # 8 keypoints → 16 outputs (x and y for each)
        "base_trainable": False,          # Freeze the pre-trained base
    }
    
    training_config = {
        "epochs": ,
        "learning_rate": 1e-4,
        "batch_size": 32,
        "checkpoint_dir": "./models/checkpoints",
        "logs_dir": "./logs",
    }
    
    data_config = {
        "base_dir": "C:/Users/micha/Desktop/BME461",  # Update to your dataset's base directory
        "image_size": (224, 224),
        # You can pass an augmentation function if desired, e.g. augment_fn=apply_chain_augmentations
        "augment_fn": None,
    }
    
    # -----------------------------
    # Create and compile the model.
    # -----------------------------
    model = get_model(model_config)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=training_config["learning_rate"]),
        loss="mean_squared_error",
        metrics=["mae"]
    )
    
    # -----------------------------
    # Prepare datasets.
    # -----------------------------
    train_dataset = get_dataset(
        base_dir=data_config["base_dir"],
        batch_size=training_config["batch_size"],
        image_size=data_config["image_size"],
        shuffle=True,
        augment_fn=data_config["augment_fn"]
    )
    
    val_dataset = get_dataset(
        base_dir=data_config["base_dir"], # TODO: Make a seperate test dataset directory
        batch_size=training_config["batch_size"],
        image_size=data_config["image_size"],
        shuffle=False,
        augment_fn=None  # No augmentation for validation.
    )
    
    # -----------------------------
    # Setup callbacks.
    # -----------------------------
    os.makedirs(training_config["checkpoint_dir"], exist_ok=True)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(training_config["checkpoint_dir"], "model_epoch_{epoch:02d}.h5"),
        save_best_only=True,
        monitor="val_loss",
        verbose=1
    )
    
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=training_config["logs_dir"])

    # -----------------------------
    # Train the model.
    # -----------------------------
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=training_config["epochs"],
        callbacks=[checkpoint_cb, tensorboard_cb]
    )
    
if __name__ == "__main__":
    main()
