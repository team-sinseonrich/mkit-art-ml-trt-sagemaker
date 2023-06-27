import torch
import os

from huggingface_hub import hf_hub_download
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from segment_anything import sam_model_registry, SamPredictor
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

from .config import cfg


__all__ = [
           "load_groundingdino",
           "load_sam",
           "load_instructpix2pix",
          ]


# GroundingDINO
def load_groundingdino():
    if not os.path.isfile(cfg.cache_config_file):
        cfg.cache_config_file = hf_hub_download(repo_id=cfg.ckpt_repo_id, 
                                                  filename=cfg.ckpt_config_filename,
                                                  cache_dir='../models')
    if not os.path.isfile(cfg.cache_file):
        cfg.cache_file = hf_hub_download(repo_id=cfg.ckpt_repo_id, 
                                           filename=cfg.ckpt_filename, 
                                           cache_dir='../models')

    args = SLConfig.fromfile(cfg.cache_config_file) 
    args.device = cfg.device
    groundingdino_model = build_model(args)

    checkpoint = torch.load(cfg.cache_file, map_location=cfg.device)
    groundingdino_model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    groundingdino_model.eval()
    return groundingdino_model


# SAM
def load_sam():
    sam = sam_model_registry[cfg.model_type](checkpoint=cfg.sam_checkpoint)
    sam.to(device=cfg.device)
    predictor = SamPredictor(sam)
    return predictor


# InstructPix2Pix
def load_instructpix2pix():
    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
    pipe.to(cfg.device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    return pipe
