import queue
import time
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import pydantic
import torch
import torch.utils.hooks
from torch import Tensor, nn
from torch.nn import functional as F

from ..datasets.loader import DatasetLoader
from ..models.shapley import loss_shapley_new, mask_shapley_new
from ..recipes.types import (
    ModelRecipe,
    ModelRecipe_Measurements_DualTaskSimilarity,
    TAltClassifier,
    TAltExplainer,
    TSurrogate,
)
from ..scripts.env import ExpEnv
from ..scripts.resources import (
    get_epoch_ckpts,
    get_recipe,
    load_cfg_dataset,
    load_epoch_ckpt,
    load_epoch_model,
)


class MeasureDualTaskSimilarityReport(pydantic.BaseModel):
    """When a model is trained to accommodate Classifier & Explainer tasks
    simultaneously, this report measures the cosine similarity between the two
    tasks' input embedding gradients w.r.t. the output loss.

    Requires: surrogate [-1], explainer [ep], `duo_vanilla` family."""

    epochs: List[int]
    cos_sim_avg: List[float]
    cos_sim_std: List[float]


def measure_dual_task_similarity(
    env: ExpEnv, device: torch.device, d_loader: Optional[DatasetLoader]
) -> MeasureDualTaskSimilarityReport:
    env.log("loading models...")
    config = env.config
    m_recipe, m_config = get_recipe(config)
    if m_recipe.measurements.allow_dual_task_similarity is False:
        raise ValueError("unsupported recipe action")
    inspector = m_recipe.measurements.allow_dual_task_similarity

    if d_loader is None:
        env.log("loading dataset...")
        d_config = (
            config.eval_dual_task_similarity.dataset
            if config.eval_dual_task_similarity is not None
            and config.eval_dual_task_similarity.dataset is not None
            else config.dataset
        )
        d_loader = load_cfg_dataset(d_config, env.model_path)
    m_misc = m_recipe.load_misc(env.model_path, m_config)
    n_players = m_recipe.n_players(m_config)
    gen_input = m_recipe.gen_input(m_config, m_misc, device)
    gen_null = m_recipe.gen_null(m_config, m_misc, device)

    _epoch_classifier, m_classifier = load_epoch_model(
        env, m_recipe, "classifier", device=device
    )
    _epoch_surrogate, m_surrogate = load_epoch_model(
        env, m_recipe, "surrogate", device=device
    )
    # surrogate_null
    nil_Xs = gen_null
    nil_mask = torch.ones((1, n_players), dtype=torch.long, device=device)
    m_surrogate.eval()
    with torch.no_grad():
        # <1, n_classes>
        surrogate_null, _ = m_recipe.fw_surrogate(m_surrogate, nil_Xs, nil_mask)

    # we need to go through all epochs of the explainer. if applicable.
    env.log("[[[ running measurement... ]]]")
    all_epochs: List[int] = []
    all_cos_sim_avg: List[float] = []
    all_cos_sim_std: List[float] = []
    for _loading_epoch in get_epoch_ckpts(
        env.model_path, "explainer", config.train_explainer.epochs
    ):
        epoch_explainer, mpt_o_explainer = load_epoch_ckpt(
            env.model_path, "explainer", _loading_epoch, required=True
        )
        m_o_explainer = m_recipe.t_explainer(m_config)
        m_o_explainer.load_state_dict(mpt_o_explainer)
        # and re-interpret it as a model where we can investigate its gradients.
        M_classifier, M_explainer = inspector.conv_alt_models(
            m_config, m_classifier, m_o_explainer
        )
        M_classifier = M_classifier.to(device=device)
        M_explainer = M_explainer.to(device=device)
        inspect_cls, inspect_exp = inspector.grad_modules(M_classifier, M_explainer)
        optim_cls = torch.optim.AdamW(  # type: ignore
            inspect_cls.parameters(), lr=config.train_classifier.lr
        )
        optim_exp = torch.optim.AdamW(  # type: ignore
            inspect_exp.parameters(), lr=config.train_explainer.lr
        )
        if inspect_cls is inspect_exp:
            optim_cls = optim_exp

        ts_begin = time.time()
        cos_sim_avg, cos_sim_std = _explainer_grad_eval(
            env=env,
            device=device,
            n_mask_samples=config.train_explainer.n_mask_samples,
            n_players=n_players,
            surrogate_null=surrogate_null,
            d_items=d_loader.test(
                config.eval_dual_task_similarity.batch_size
                if config.eval_dual_task_similarity
                else config.train_explainer.batch_size
            ),
            m_recipe=m_recipe,
            inspector=inspector,
            M_classifier=M_classifier,
            m_surrogate=m_surrogate,
            M_explainer=M_explainer,
            epoch=epoch_explainer,
            optim_cls=optim_cls,
            optim_exp=optim_exp,
            gen_input=gen_input,
        )
        ts_delta = time.time() - ts_begin

        all_epochs.append(epoch_explainer)
        all_cos_sim_avg.append(cos_sim_avg)
        all_cos_sim_std.append(cos_sim_std)
        info = f"  > epoch {epoch_explainer} done in {ts_delta:.2f}s // "
        info += f"cos_sim: avg {cos_sim_avg:.6f} std {cos_sim_std:.6f}"
        env.log(info)

    return MeasureDualTaskSimilarityReport(
        epochs=all_epochs,
        cos_sim_avg=all_cos_sim_avg,
        cos_sim_std=all_cos_sim_std,
    )


def _explainer_grad_eval(
    env: ExpEnv,
    device: torch.device,
    n_mask_samples: int,
    n_players: int,
    surrogate_null: Tensor,
    d_items: Iterable[Tuple[Any, Any]],
    m_recipe: ModelRecipe[Any, Any, Any, TSurrogate, Any, Any],
    inspector: ModelRecipe_Measurements_DualTaskSimilarity[
        Any, Any, Any, TAltClassifier, TAltExplainer
    ],
    m_surrogate: TSurrogate,
    M_classifier: TAltClassifier,
    M_explainer: TAltExplainer,
    epoch: int,
    optim_cls: torch.optim.Optimizer,  # type: ignore
    optim_exp: torch.optim.Optimizer,  # type: ignore
    gen_input: Callable[[Any, Any], Tuple[Tensor, Tensor]],
) -> Tuple[float, float]:
    all_cos_sim: List[float] = []
    for batch_idx, (_inputs, _targets) in enumerate(d_items):
        # generate everything we need
        Xs, Zs = gen_input(_inputs, _targets)
        batch_size, *_ = Xs.shape
        Xs_mask_1 = torch.ones((batch_size, n_players), dtype=torch.long, device=device)
        Xs_mask_shap_ = mask_shapley_new(batch_size * n_mask_samples, n_players).to(
            device
        )
        Xs_mask_shap = Xs_mask_shap_.reshape((batch_size, n_mask_samples, n_players))
        Xs_EXT = Xs.reshape((batch_size, 1, -1)).repeat(1, n_mask_samples, 1)
        Xs_EXT = Xs_EXT.reshape((batch_size * n_mask_samples, -1))

        # forward on surrogate to get stuff
        m_surrogate.eval()
        with torch.no_grad():
            surrogate_values, _ = m_recipe.fw_surrogate(
                m_surrogate, Xs_EXT, Xs_mask_shap_
            )
            surrogate_grand, _ = m_recipe.fw_surrogate(m_surrogate, Xs, Xs_mask_1)
            surrogate_grand_EXT = surrogate_grand.reshape((batch_size, 1, -1))
            surrogate_grand_EXT = surrogate_grand_EXT.repeat(1, n_mask_samples, 1)
            surrogate_grand_EXT = surrogate_grand_EXT.reshape(
                (batch_size * n_mask_samples, -1)
            )

        # explainer pass 1: classifier gradients
        optim_cls.zero_grad()
        grad_cls, _grad_exp = inspector.grad_modules(M_classifier, M_explainer)
        grad_cls.eval()
        hook = TorchGradientHook(grad_cls)
        explainer_Ys, _ = inspector.fw_alt(
            M_classifier, M_explainer, Xs, Xs_mask_1, surrogate_grand, surrogate_null
        )
        loss_cls = torch.nn.functional.cross_entropy(explainer_Ys, Zs)
        loss_cls.backward()
        grad_in, _grad_out = hook.pull()
        grad_Xs_cls, *_ = grad_in
        hook.remove()

        # explainer pass 2: shapley gradients
        optim_exp.zero_grad()
        _grad_cls, grad_exp = inspector.grad_modules(M_classifier, M_explainer)
        grad_exp.eval()
        hook = TorchGradientHook(grad_exp)
        _, explainer_shap = inspector.fw_alt(
            M_classifier, M_explainer, Xs, Xs_mask_1, surrogate_grand, surrogate_null
        )
        loss_shap = loss_shapley_new(
            batch_size=batch_size,
            n_mask_samples=n_mask_samples,
            n_players=n_players,
            mask=Xs_mask_shap,
            v_0=surrogate_null,
            v_s=surrogate_values,
            v_1=surrogate_grand,
            phi=explainer_shap,
        )
        loss_shap.backward()
        grad_in, _grad_out = hook.pull()
        grad_Xs_exp, *_ = grad_in
        hook.remove()

        grads_cls = grad_Xs_cls.reshape((batch_size, -1))
        grads_exp = grad_Xs_exp.reshape((batch_size, -1))
        cos_sim = F.cosine_similarity(grads_cls, grads_exp)

        all_cos_sim.extend(cos_sim.tolist())
        info = f"  > epoch {epoch} :{batch_idx}:sim // {cos_sim.sum().item() / batch_size:.6f}, fin {len(all_cos_sim)}"
        env.log(info)

    all_cos_sim_t = torch.tensor(all_cos_sim)
    cos_sim_avg = all_cos_sim_t.mean().item()
    cos_sim_std = all_cos_sim_t.std().item()
    return cos_sim_avg, cos_sim_std


class TorchGradientHook:
    _grad_t = Union[Tuple[Tensor, ...], Tensor]

    def __init__(self, target: nn.Module) -> None:
        self._target = target
        self._buffer: queue.Queue[Tuple[List[Tensor], List[Tensor]]] = queue.Queue()
        self._handle: Optional[torch.utils.hooks.RemovableHandle] = (
            self._target.register_full_backward_hook(
                lambda m, i, o: self._hook_fn(m, i, o)
            )
        )
        return

    def _hook_fn(self, m: nn.Module, i: _grad_t, o: _grad_t) -> None:
        il, ol = [], []
        for grad in i:
            if grad is not None:
                il.append(grad)
        for grad in o:
            if grad is not None:
                ol.append(grad)
        self._buffer.put((il, ol))

    def pull(self) -> Tuple[List[Tensor], List[Tensor]]:
        try:
            return self._buffer.get_nowait()
        except queue.Empty:
            return [], []

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
        self._handle = None

    def __del__(self) -> None:
        self.remove()

    pass
