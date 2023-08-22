import os
import torch
from tqdm import tqdm
from tungstenkit import BaseIO, Field, Video, define_model
from typing import List

from video_utils import (
    clean_temp,
    create_temp,
    create_video,
    detect_fps,
    extract_frames,
    get_temp_frame_paths,
    restore_audio,
)
# flake8: noqa
import os
import tempfile
import warnings
import shutil

warnings.filterwarnings("ignore", module="torchvision", category=UserWarning)

import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from pathlib import Path
from tungstenkit import BaseIO, Field, Option, define_model

from realesrgan.utils import RealESRGANer


OUTPUT_PATH = "output.mp4"


class Input(BaseIO):
    input_video: Video = Field(description="Input video for upscaling")
    version: str = Option(
        description="RealESRGAN version",
        choices=[
            "General - RealESRGANplus",
            "General - v3",
            "Anime - anime6B",
            "AnimeVideo - v3",
        ],
        default="General - v3",
    )
    scale: float = Option(description="Rescaling factor", default=2, ge=1, le=4)
    tile: int = Option(
        description="Tile size. Default is 0, that is no tile. When encountering the out-of-GPU-memory issue, please specify it, e.g., 400 or 200",
        default=0,
        ge=0,
    )

class Output(BaseIO):
    output_video: Video


@define_model(
    input=Input,
    output=Output,
    batch_size=1,
    gpu=True,
    gpu_mem_gb=16,
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "python3-opencv"],
    python_packages=[
        "torch==1.7.1",
        "torchvision==0.8.2",
        "numpy==1.21.1",
        "lmdb==1.2.1",
        "opencv-python==4.5.3.56",
        "PyYAML==5.4.1",
        "tqdm==4.62.2",
        "yapf==0.31.0",
        "basicsr==1.4.2",
        "facexlib==0.2.5",
        "gfpgan==1.3.8",
        "basicsr==1.4.2",
        "tqdm",
    ],
)
class VideoUpscalerRealESRGAN:
    def choose_model(self, scale: float, version: str, tile: int = 0):
        half = True if torch.cuda.is_available() else False
        if version == "General - RealESRGANplus":
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
            model_path = "weights/RealESRGAN_x4plus.pth"
            self.upsampler = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=model,
                tile=tile,
                tile_pad=10,
                pre_pad=0,
                half=half,
            )
        elif version == "General - v3":
            model = SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=32,
                upscale=4,
                act_type="prelu",
            )
            model_path = "weights/realesr-general-x4v3.pth"
            self.upsampler = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=model,
                tile=tile,
                tile_pad=10,
                pre_pad=0,
                half=half,
            )
        elif version == "Anime - anime6B":
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=6,
                num_grow_ch=32,
                scale=4,
            )
            model_path = "weights/RealESRGAN_x4plus_anime_6B.pth"
            self.upsampler = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=model,
                tile=tile,
                tile_pad=10,
                pre_pad=0,
                half=half,
            )
        elif version == "AnimeVideo - v3":
            model = SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=16,
                upscale=4,
                act_type="prelu",
            )
            model_path = "weights/realesr-animevideov3.pth"
            self.upsampler = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=model,
                tile=tile,
                tile_pad=10,
                pre_pad=0,
                half=half,
            )


    def predict(self, inputs: List[Input]):
        input = inputs[0]

        target_path = str(input.input_video.path)
        tile = input.tile
        version = input.version
        scale = input.scale
        if tile <= 100 or tile is None:
            tile = 0

        if os.path.exists(OUTPUT_PATH):
            os.remove(OUTPUT_PATH)
        clean_temp(target_path)
        self.choose_model(scale, version, tile)

        # Preprocessing
        create_temp(target_path)
        fps = detect_fps(target_path)
        print("Extracting frames...")
        extract_frames(target_path, fps)
        temp_frame_paths = get_temp_frame_paths(target_path)
        if not temp_frame_paths:
            raise RuntimeError("Frames not found")

        # Inference
        self._run_inference(temp_frame_paths, scale=scale, tile=tile, version=version)

        # Postprocessing
        print("Creating video...")
        create_video(target_path)
        restore_audio(target_path, OUTPUT_PATH)

        clean_temp(target_path)

        return [Output(output_video=Video.from_path(OUTPUT_PATH))]

    def _run_inference(self, temp_frame_paths: List[str], *, scale: float, version: str, tile: int):
        for i in tqdm(range(len(temp_frame_paths)), desc="Upscaling"):
            self._infer(temp_frame_paths[i], scale=scale, version=version, tile=tile)

    def _infer(self, path: List[str], *, scale: float, version: str, tile: int):
        extension = os.path.splitext(os.path.basename(path))[1]
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = "RGBA"
        elif len(img.shape) == 2:
            img_mode = None
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_mode = None

        h, w = img.shape[0:2]
        if h < 300:
            img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

        self.choose_model(scale, version, tile)

        try:
            output, _ = self.upsampler.enhance(img, outscale=scale)
        except RuntimeError as error:
            print("Error", error)
            print(
                'If you encounter CUDA out of memory, try to set "tile" to a smaller size, e.g., 400.'
            )
            raise error

        if img_mode == "RGBA":  # RGBA images should be saved in png format
            extension = "png"

        out_path = Path(tempfile.mkdtemp()) / f"out{extension}"
        cv2.imwrite(str(out_path), output)
        os.remove(path)
        shutil.move(str(out_path), path)


if __name__ == "__main__":
    inp = Input(input_video=Video.from_path("a-better-tomorrow.mp4"))
    model = VideoUpscalerRealESRGAN()
    model.setup()
    model.predict([inp])
