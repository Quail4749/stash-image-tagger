from pathlib import Path
from typing import Final

import huggingface_hub  # type: ignore
import numpy as np
import onnxruntime as ort  # type: ignore
import pyvips  # type: ignore
import urllib3
import urllib3.util
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger
from stashapi.stashapp import StashInterface  # type: ignore

app = FastAPI()

# Load the ONNX model
MODEL_PATH: Final[str] = "SmilingWolf/wd-vit-tagger-v3"
MODEL_FILE: Final[str] = "model.onnx"
TAGS_FILE: Final[str] = "selected_tags.csv"

WORKING_DIR = Path(__file__).parent
MODELS_PATH: Path = WORKING_DIR / Path("models")

model_target_size = 224


def download_model() -> None:
    if not MODELS_PATH.exists():
        logger.info(f"Creating models directory at {MODELS_PATH}")
        MODELS_PATH.mkdir(parents=True)

    if not (MODELS_PATH / MODEL_FILE).exists():
        logger.info(f"Downloading model {MODEL_PATH} from Hugging Face Hub")
        huggingface_hub.hf_hub_download(
            MODEL_PATH,
            filename=MODEL_FILE,
            local_dir=MODELS_PATH,
        )
        logger.info(f"Model downloaded to {MODELS_PATH / MODEL_FILE}")
    if not (MODELS_PATH / TAGS_FILE).exists():
        logger.info(f"Downloading tags for {MODEL_PATH} from Hugging Face Hub")
        huggingface_hub.hf_hub_download(
            MODEL_PATH,
            filename=TAGS_FILE,
            local_dir=MODELS_PATH,
        )
        logger.info(f"Tags downloaded to {MODELS_PATH / TAGS_FILE}")


def prepare_image(image: pyvips.Image) -> np.ndarray:
    """
    Preprocess the image to the required input size and format for the model using pyvips.
    """
    target_size = model_target_size

    # Convert image to RGBA if not already
    if image.hasalpha():
        image = image.flatten(background=[255, 255, 255])
    else:
        image = image.bandjoin(255).flatten(background=[255, 255, 255])

    # Pad image to square
    image_shape = (image.width, image.height)
    max_dim = max(image_shape)
    pad_left = (max_dim - image_shape[0]) // 2
    pad_top = (max_dim - image_shape[1]) // 2

    padded_image = pyvips.Image.black(max_dim, max_dim).new_from_image([255, 255, 255])
    padded_image = padded_image.insert(image, pad_left, pad_top)

    # Resize
    if max_dim != target_size:
        padded_image = padded_image.resize(target_size / max_dim, kernel="bicubic")

    # Convert to numpy array and to float32
    image_array = image.numpy()

    # Convert RGB to BGR
    image_array = image_array[:, :, ::-1].astype(np.float32)

    return np.expand_dims(image_array, axis=0)


def postprocess_output(output: np.ndarray) -> dict:
    """
    Postprocess the model output to return a human-readable format.
    """
    # Assuming the model output is a probability distribution across labels
    labels = ["label1", "label2", "label3"]  # Replace with actual labels
    probabilities = output[0]
    return {label: float(prob) for label, prob in zip(labels, probabilities)}


@app.post("/tag-image")
async def tag_image(file: UploadFile = File(...)):
    """
    Endpoint to tag an uploaded image using the ONNX model.
    """
    try:
        image_data = await file.read()

        image = pyvips.Image.new_from_buffer(image_data, "")

        input_data = prepare_image(image)

        outputs = ort_session.run(None, {"input": input_data})

        tags = postprocess_output(outputs[0])

        return JSONResponse(content={"tags": tags})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post(
    "/update-stash-tags",
    description="Given a stash API key and URL, update the tags for all images in the stash using DanBooru Tags and the tags defined by the model.",
)
async def update_stash_tags(stash_api_key: str, stash_url: str):
    parsed_url = urllib3.util.parse_url(stash_url)
    scheme, host, port = parsed_url.scheme, parsed_url.host, parsed_url.port
    if not host:
        host = "localhost"
    if not scheme:
        scheme = "http"
    if not port:
        port = 9999
    if not stash_api_key:
        raise HTTPException(status_code=400, detail="API key not provided")

    stash_interface = StashInterface(
        {"ApiKey": stash_api_key, "Scheme": scheme, "Host": host, "Port": port}
    )


if __name__ == "__main__":
    import uvicorn

    download_model()

    uvicorn.run(app, host="0.0.0.0", port=6980)
