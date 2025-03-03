import base64
import io
import json
import pathlib
from typing import Dict, List, Optional

import PIL.Image
import pydantic
import rich.console
import torch
from torchvision.transforms.functional import resize

from ..datasets.loader import DatasetLoader
from .env import ExpEnv
from .resources import get_recipe, load_cfg_dataset, load_epoch_model


class ImageExplanation(pydantic.BaseModel):
    img_channels: int
    img_px_size: int
    img_patch_size: int
    image: str  # base64-encoded jpg
    explanation: List[List[float]]  # [label][h*w]


class RunImageExplanationResults(pydantic.BaseModel):
    items: Dict[int, ImageExplanation]


console = rich.get_console()


def run_image_explanation(
    env: ExpEnv,
    device: torch.device,
    d_loader: Optional[DatasetLoader],
    into: pathlib.Path,
    limit: Optional[int],
) -> None:
    config = env.config
    m_recipe, m_config = get_recipe(config)
    if d_loader is None:
        d_loader = load_cfg_dataset(config.dataset, env.model_path)

    _epoch_final, m_final = load_epoch_model(env, m_recipe, "final", device=device)

    m_misc = m_recipe.load_misc(env.model_path, m_config)
    num_labels: int = m_config.num_labels
    img_channels: int = m_config.img_channels
    img_px_size: int = m_config.img_px_size
    img_patch_size: int = m_config.img_patch_size

    gen_input = m_recipe.gen_input(m_config, m_misc, device)
    result_buffer: List[ImageExplanation] = []
    for i, (_inputs, _targets, _inputs_raw, _targets_raw) in enumerate(
        d_loader.test_raw(1)
    ):
        if limit is not None and i >= limit:
            break
        Xs, _Zs = gen_input(_inputs, _targets)
        with torch.no_grad():
            logits, attr = m_recipe.fw_final(m_final, Xs)
        # align with original input
        _, _pred_Zs = logits.max(dim=1)
        Zs, pred_Zs = int(_Zs.item()), int(_pred_Zs.item())
        if Zs != pred_Zs:
            continue

        # conv raw tensor -> jpg -> base64
        img = _inputs_raw[0]
        img = resize(img, [img_px_size, img_px_size])
        assert img.shape == (img_channels, img_px_size, img_px_size)
        img = img.cpu().numpy().transpose(1, 2, 0)
        img = (img * 255).astype("uint8")
        img = PIL.Image.fromarray(img)
        assert img.size == (img_px_size, img_px_size)
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="JPEG")
        img = img_buffer.getvalue()
        img = base64.b64encode(img).decode("utf-8")
        # attr tensor -> list
        assert attr.shape == (1, num_labels, (img_px_size // img_patch_size) ** 2)
        attr = attr[0].cpu().numpy().tolist()

        result_item = ImageExplanation(
            img_channels=img_channels,
            img_px_size=img_px_size,
            img_patch_size=img_patch_size,
            image=img,
            explanation=attr,
        )
        result_buffer.append(result_item)
        print(f"    visualized #{i}...")

    env.log(f"saving into: {into}")
    results = RunImageExplanationResults(
        items={i: r for i, r in enumerate(result_buffer)}
    )
    with open(into, "w", encoding="utf-8") as f:
        j = results.model_dump_json()
        r = json.loads(j)
        j = json.dumps(r, indent=None)
        f.write(j + "\n")
        # f.write("{\n")
        # f.write('  "items": {\n')
        # for i, item in results.items.items():
        #     j = item.model_dump_json()
        #     r = json.loads(j)
        #     j = json.dumps(r, indent=None)
        #     trail = "," if i + 1 < len(results.items) else ""
        #     f.write(f'    "{i}": {j}{trail}\n')
        # f.write("  }\n")
        # f.write("}\n")
    return
