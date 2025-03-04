import tensorflow as tf

def build_mobilenetv2_model(input_shape, num_keypoints, base_trainable=False):
    """
    Builds a keypoint detection model using MobileNetV2 as the base.

    Args:
        input_shape (tuple): Shape of input images, e.g. (224, 224, 3).
        num_keypoints (int): Number of keypoints to detect.
        base_trainable (bool): If False, freeze the MobileNetV2 base.

    Returns:
        tf.keras.Model: The compiled model.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights='imagenet'
    )
    base_model.trainable = base_trainable

    # Build custom head on top of the base model.
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_keypoints * 2, activation='linear')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    return model

def build_resnet50_model(input_shape, num_keypoints, base_trainable=False):
    """
    Example alternative: Build a model using ResNet50 as the base.
    """
    base_model = tf.keras.applications.ResNet50(
        input_shape=input_shape, include_top=False, weights='imagenet'
    )
    base_model.trainable = base_trainable

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_keypoints * 2, activation='linear')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    return model

def get_model(model_config):
    """
    Factory method to build and return a model based on configuration.

    Args:
        model_config (dict): Configuration dictionary with keys:
            - model_type: e.g. 'mobilenetv2' or 'resnet50'
            - input_shape: e.g. (224, 224, 3)
            - num_keypoints: Number of keypoints (each with x and y coordinates)
            - base_trainable: Boolean, whether the base should be trainable

    Returns:
        tf.keras.Model: The constructed model.
    """
    model_type = model_config.get("model_type", "mobilenetv2").lower()
    if model_type == "mobilenetv2":
        return build_mobilenetv2_model(
            input_shape=model_config["input_shape"],
            num_keypoints=model_config["num_keypoints"],
            base_trainable=model_config.get("base_trainable", False)
        )
    elif model_type == "resnet50":
        return build_resnet50_model(
            input_shape=model_config["input_shape"],
            num_keypoints=model_config["num_keypoints"],
            base_trainable=model_config.get("base_trainable", False)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
