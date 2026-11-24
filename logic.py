"""Анализ изображения через обученную Keras-модель."""
import json
import re
from functools import lru_cache
from pathlib import Path
import h5py
import keras
import numpy as np
from keras.src.legacy.saving import saving_utils
from PIL import Image, ImageOps

np.set_printoptions(suppress=True)

MODEL_PATH = Path(__file__).with_name("keras_model.h5")
LABELS_PATH = Path(__file__).with_name("labels.txt")
IMAGE_SIZE = (224, 224)


class _CompatDepthwiseConv2D(keras.layers.DepthwiseConv2D):
    """Старый Keras писал groups=1, Keras 3 этот аргумент не принимает."""

    @classmethod
    def from_config(cls, config):
        config.pop("groups", None)
        return super().from_config(config)


def _find_functional_config(config):
    """Достаёт конфиг сверточной части из вложенной модели Teachable Machine."""
    if isinstance(config, dict):
        if config.get("class_name") == "Functional":
            return config
        for value in config.values():
            found = _find_functional_config(value)
            if found is not None:
                return found
    elif isinstance(config, list):
        for item in config:
            found = _find_functional_config(item)
            if found is not None:
                return found
    return None


def _assign_weights(h5file, group_name, model):
    """Копирует веса из H5-группы в слои с теми же именами."""
    group = h5file["model_weights"][group_name]
    if isinstance(model, (list, tuple)):
        layers = model
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        layers = [model]
    for layer in layers:
        if layer.name not in group:
            continue
        layer_group = group[layer.name]
        weights = []
        for variable in layer.weights:
            key = variable.name.split("/")[-1].split(":")[0]
            if isinstance(layer, keras.layers.DepthwiseConv2D) and key == "kernel":
                key = "depthwise_kernel"  # имя веса в старом Keras
            if key not in layer_group:
                key += ":0"
            weights.append(np.asarray(layer_group[key]))
        if weights:
            layer.set_weights(weights)


@lru_cache(maxsize=1)
def _get_model_and_labels():
    """Модель и подписи классов загружаются один раз, при первом вызове."""
    class_names = [
        line.strip().split(maxsplit=1)[-1]
        for line in LABELS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    with h5py.File(MODEL_PATH, "r") as h5file:
        model_config = json.loads(h5file.attrs["model_config"])
        feature_config = _find_functional_config(model_config)
        if feature_config is None:
            raise ValueError("В файле модели не найдена свёрточная часть")
        feature_extractor = saving_utils.model_from_config(
            feature_config,
            custom_objects={"DepthwiseConv2D": _CompatDepthwiseConv2D},
        )

        inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
        x = feature_extractor(inputs)
        if isinstance(x, (list, tuple)):  # старый Functional возвращает список
            x = x[0]
        x = keras.layers.GlobalAveragePooling2D(
            name="global_average_pooling2d_GlobalAveragePooling2D1"
        )(x)
        x = keras.layers.Dense(100, activation="relu", name="dense_Dense1")(x)
        outputs = keras.layers.Dense(
            len(class_names), activation="softmax", use_bias=False, name="dense_Dense2"
        )(x)
        model = keras.Model(inputs, outputs)

        _assign_weights(h5file, "sequential_1", feature_extractor)
        dense_layers = [model.get_layer("dense_Dense1"), model.get_layer("dense_Dense2")]
        _assign_weights(h5file, "sequential_3", dense_layers)
    return model, class_names


def analyze_image(image_path: str) -> tuple[str, float]:
    """Возвращает (класс, уверенность от 0 до 1) для изображения."""
    model, class_names = _get_model_and_labels()

    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, IMAGE_SIZE, Image.Resampling.LANCZOS)
    image_array = np.asarray(image, dtype=np.float32)
    normalized = (image_array / 127.5) - 1.0

    prediction = model.predict(np.expand_dims(normalized, axis=0), verbose=0)[0]
    index = int(np.argmax(prediction))
    return class_names[index], float(prediction[index])


def detect_object(image_path: str) -> list[tuple[str, float]]:
    """Результат анализа в формате, ожидаемом main.py: процент уверенности."""
    class_name, confidence = analyze_image(image_path)
    return [(class_name, confidence * 100)]


def graphic_percent(name: str, percent: float) -> str:
    """Форматирует одну строку ответа для main.py."""
    return f"{name}: {percent:.2f}%"


GREEN_NORM = 40  # Норма озелененой местности в %
LOW_CONFIDENCE = 55  # ниже этой уверенности совет может быть ненадёжным


def get_recommendation(percent: float, objects: list[tuple[str, float]]) -> str:
    """Совет или похвала по результату анализа фотографии."""
    if not objects:
        return "Не удалось проанализировать фото, попробуйте ещё раз."

    name, confidence = objects[0]

    if confidence < LOW_CONFIDENCE:
        return "Снимок неоднозначный, попробуйте снять территорию ещё раз."

    match = re.search(r"(\d+(?:[.,]\d+)?)\s*%", name)
    if match:
        green = float(match.group(1).replace(",", "."))
        if green >= GREEN_NORM:
            return (
                f"Зелени {percent:.2f}% — это выше нормы в {GREEN_NORM}%! "
                "Территория достаточно озеленена, так держать 🌳"
            )
        return (
            f"Зелени {percent:.2f}%, а норма — от {GREEN_NORM}% 🌱. "
            "Добавьте деревья и кустарники вдоль дорог, газоны и маленькие скверы."
        )

    name = name.lower()
    if "мало" in name or "недостаточно" in name:
        return (
            f"Зелени мало. Норма — от {GREEN_NORM}% территории. "
            "Добавьте деревья и кустарники. Также можно добавить газон."
        )
    if "достаточно" in name or "озеленен" in name:
        return "Хорошая работа! Территория достаточно озеленена. Так держать!"

    return "Зелень есть, но проверьте, хватает ли теневых деревьев и газонов"
