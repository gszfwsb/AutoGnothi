import pathlib
import unittest
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import rich
import rich.table
import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .strings import pattern_replace


def force_save_model(
    id: str,
    path: pathlib.Path,
    save_model: Optional[Type[PreTrainedModel]] = None,
    save_tokenizer: Optional[Type[PreTrainedTokenizerBase]] = None,
) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    if save_model:
        model = cast(PreTrainedModel, save_model.from_pretrained(id))
        model.save_pretrained(path, safe_serialization=False)
    if save_tokenizer:
        tokenizer = save_tokenizer.from_pretrained(
            id, clean_up_tokenization_spaces=True
        )
        tokenizer.save_pretrained(path)
    return


def show_model_fridge(model: nn.Module) -> None:
    console = rich.get_console()
    table = rich.table.Table(
        title=f"Fridge Status on `{model.__class__.__name__}`", title_justify="left"
    )
    table.add_column("Name", justify="left")
    table.add_column("Training", justify="left")
    for name, param in model.named_parameters():
        frozen = "✅" if param.requires_grad else "    "
        table.add_row(name, frozen)
    console.print(table)
    return


def freeze_model_parameters(
    on: nn.Module, *item_names: str, requires_grad: bool = False
) -> None:
    """freeze(model, "layer1", "layer2") || freeze(model, ...)"""

    if len(item_names) == 1 and item_names[0] is ...:
        for param in on.parameters():
            param.requires_grad = requires_grad
    else:
        for name, param in on.named_parameters():
            if any(name.startswith(f"{n}.") for n in item_names):
                param.requires_grad = requires_grad
    return


class New:
    __counter = 0

    def __init__(self):
        self.__class__.__counter += 1

    def __repr__(self):
        return "new()"

    def __hash__(self):
        return self.__class__.__counter

    pass


# type MergeStateDictRules = {
#   [key: string | New]: string | ... | (string | ...)[] | undefined;
# };
MergeStateDictRules = Dict[
    Union[str, New], Union[str, type(Ellipsis), List[Union[str, type(Ellipsis)]], None]
]


def merge_state_dicts(
    *rules_src: Tuple[MergeStateDictRules, Union[nn.Module, Any]],
    into: nn.Module = ...,
) -> None:
    """
    [1]: 'pattern.{a}' -> 'pattern.{b}'  # edit
    [2]: 'pattern.{a}' -> ...  # keep
    [3]: 'pattern.{a}' -> None  # remove
    [4]: New() -> 'pattern.{a}'  # new
    """

    def _duplicate(v: Any) -> Any:
        if isinstance(v, torch.Tensor):
            return v.clone()
        return v

    old_rules_src = [
        (rules, m.state_dict() if isinstance(m, nn.Module) else m)
        for rules, m in rules_src
    ]
    new_sd = into.state_dict()
    result_sd = _merge_items(old_rules_src, new_sd, duplicate_action=_duplicate)
    into.load_state_dict(result_sd)
    return


def _merge_items(
    rules_src: List[Tuple[MergeStateDictRules, Dict[str, Any]]],
    dest: Dict[str, Any],
    duplicate_action: Callable[[Any], Any],
) -> Dict[str, Any]:
    ok = True
    new_rules: Dict[str, List[str]] = {}
    l_edit_repl: List[Callable[[str], Tuple[bool, List[str]]]] = []
    l_rm_repl: List[Callable[[str], Tuple[bool, List[str]]]] = []

    for rules, _ in rules_src:
        edit_rules: Dict[str, List[str]] = {}
        rm_rules: Dict[str, List[str]] = {}
        for k, v in rules.items():
            if isinstance(k, New) and isinstance(v, str):
                new_rules[v] = ["<NEW>"]
            elif isinstance(k, str) and isinstance(v, str):
                edit_rules[k] = [v]
            elif isinstance(k, str) and v is Ellipsis:
                edit_rules[k] = [k]
            elif isinstance(k, str) and isinstance(v, list):
                vvs: List[str] = []
                error = False
                for vv in v:
                    if isinstance(vv, str):
                        vvs.append(vv)
                    elif vv is Ellipsis:
                        vvs.append(k)
                    else:
                        error = True
                if error:
                    raise ValueError(f"invalid rule: {k} -> {v}")
                if vvs:
                    edit_rules[k] = vvs
                else:
                    rm_rules[k] = ["<RM>"]
            elif isinstance(k, str) and v is None:
                rm_rules[k] = ["<RM>"]
            else:
                raise ValueError(f"invalid rule: {k} -> {v}")
        edit_repl = pattern_replace(edit_rules)
        l_edit_repl.append(edit_repl)
        rm_repl = pattern_replace(rm_rules)
        l_rm_repl.append(rm_repl)
    new_repl = pattern_replace(new_rules)

    result: Dict[str, Any] = {}
    for (_, src), edit_repl, rm_repl in zip(rules_src, l_edit_repl, l_rm_repl):
        for k, v in src.items():
            matched, nks = edit_repl(k)
            if matched:
                for _i, nk in enumerate(nks):
                    if nk in result:
                        print(f" [!] duplicate key: {nk}")
                        ok = False
                    # duplicate_action applies to any value other than the first 'rename'
                    dup_v = v if _i == 0 else duplicate_action(v)
                    result[nk] = dup_v
                continue
            matched, nks = rm_repl(k)
            if matched and nks == ["<RM>"]:
                continue
            print(f" [!] no rule matches key from `from_model`: {k}")
            ok = False

    dest = {k: v for k, v in dest.items() if k not in result}
    for k, v in dest.items():
        matched, _flag = new_repl(k)
        if matched and _flag == ["<NEW>"]:
            if k in result:
                print(f" [!] duplicate key: {k}")
                ok = False
            result[k] = v
            continue
        print(f" [!] ignored key from `into_model`: {k}")
        ok = False

    if not ok:
        raise ValueError("merge failed")
    return result


class ObservableModuleMixin:
    def __init__(self):
        self.__om_observing__ = False
        self.__om_observed_features__: Optional[Dict[str, torch.Tensor]] = None

    @classmethod
    def using(cls, m: nn.Module) -> "ObservableModuleMixin":
        if not isinstance(m, ObservableModuleMixin):
            raise ValueError("not an ObservableModuleMixin")
        return m

    def om_retain_observations(self, flag: bool = True) -> None:
        # using an explicit flag and otherwise not keep the tensors means that
        # we can free those computation graphs if we do not need to observe
        self.__om_observing__ = flag
        if not flag:
            self.__om_observed_features__ = None

    def om_record_features(
        self,
        repr_cls: Optional[torch.Tensor] = None,
        repr_srg: Optional[torch.Tensor] = None,
        repr_exp: Optional[torch.Tensor] = None,
        extra: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        features: Dict[str, Optional[torch.Tensor]] = {}
        features["repr_cls"] = repr_cls
        features["repr_srg"] = repr_srg
        features["repr_exp"] = repr_exp
        features.update(extra or {})
        if self.__om_observing__:
            self.__om_observed_features__ = {
                k: v for k, v in features.items() if v is not None
            }

    def om_observe(self) -> Dict[str, torch.Tensor]:
        if self.__om_observed_features__ is None:
            raise ValueError("no features to observe. use `forward()` first")
        return self.__om_observed_features__

    def om_take_observations(self) -> Dict[str, torch.Tensor]:
        features = self.__om_observed_features__
        self.__om_observed_features__ = None
        return features or {}

    pass


class NNModelUtilsTest(unittest.TestCase):
    def test_merge_items(self):
        src_1 = {
            "alpha.default.0": 0,
            "alpha.default.1": 0,
            "alpha.0": 0,
            "alpha.1": 0,
            "beta.2": 0,
            "gamma.3": 0,
        }
        src_2 = {
            "iota.0": 1,
            "kappa.1": 1,
        }
        dest = {
            "gamma.3": 2,
            "theta.4": 2,
        }
        rules_1: MergeStateDictRules = {
            "alpha.default.{_}": ...,
            "alpha.{_}": [..., "epsilon.{_}", "zeta.{_}"],
            "beta.{_}": None,
            "gamma.{_}": None,
            New(): "gamma.{_}",
            New(): "theta.{_}",
        }
        rules_2: MergeStateDictRules = {
            "iota.{_}": ...,
            "kappa.{_}": None,
        }
        actual = _merge_items(
            [(rules_1, src_1), (rules_2, src_2)], dest, duplicate_action=lambda x: x + 5
        )
        expected = {
            "alpha.default.0": 0,
            "alpha.default.1": 0,
            "alpha.0": 0,
            "alpha.1": 0,
            "epsilon.0": 5,
            "epsilon.1": 5,
            "gamma.3": 2,
            "iota.0": 1,
            "theta.4": 2,
            "zeta.0": 5,
            "zeta.1": 5,
        }
        self.assertDictEqual(actual, expected)

    def test_crash(self):
        rule: MergeStateDictRules = {
            "alpha.{_}": "beta.{_}",
        }
        src = {
            "alpha.0": 0,
            "alpha.1": 0,
        }
        dest = {
            "beta.0": 1,
            "beta.1": 1,
            "gamma.0": 1,
        }
        self.assertRaises(
            ValueError, lambda: _merge_items([(rule, src)], dest, lambda x: x)
        )

    pass
