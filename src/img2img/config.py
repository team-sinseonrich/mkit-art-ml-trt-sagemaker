from dataclasses import dataclass
import groundingdino.datasets.transforms as T


@dataclass
class Config:
    # Env
    device = "cuda"
    
    # GroundingDINO
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filename = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    cache_config_file  = "../models/models--ShilongLiu--GroundingDINO/snapshots/a94c9b567a2a374598f05c584e96798a170c56fb/GroundingDINO_SwinB.cfg.py"
    cache_file = "../models/models--ShilongLiu--GroundingDINO/snapshots/a94c9b567a2a374598f05c584e96798a170c56fb/groundingdino_swinb_cogcoor.pth"
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # SAM
    sam_checkpoint = "../models/sam/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    # InstructPix2Pix
    p2p_model_id = "../models/instruct-pix2pix"

cfg = Config()
