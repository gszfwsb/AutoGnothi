import re
import unittest
from typing import Any, Callable, Dict, List, Tuple, cast


def flatten_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a dictionary to a single level"""
    ret = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for kk, vv in flatten_dict(v).items():
                ret[f"{k}.{kk}"] = vv
        else:
            ret[k] = v
    return ret


def pattern_replace(
    rules: Dict[str, List[str]],
) -> Callable[[str], Tuple[bool, List[str]]]:
    """Replace multiple patterns in `text`"""

    exprs: List[List[Callable[[str], Tuple[bool, str]]]] = []
    for sub, repl in rules.items():
        buffer: List[Callable[[str], Tuple[bool, str]]] = []
        for r in repl:
            buffer.append(pattern_replace_single(sub, r))
        exprs.append(buffer)

    def _replace(text: str) -> Tuple[bool, List[str]]:
        for expr in exprs:
            all_matched = True
            all_text = []
            for e in expr:
                matched, ret = e(text)
                if matched:
                    all_text.append(ret)
                else:
                    all_matched = False
            if all_matched:
                return True, all_text or [text]
        return False, [text]

    return _replace


def pattern_replace_single(sub: str, repl: str) -> Callable[[str], Tuple[bool, str]]:
    """Replace `sub` with `repl` in `text`:
    input [0]: 'format {this} and {that}',
    input [1]: 'into {that} and {this}',
    ret: input: 'format 1 and 2',
    ret: ret [0]: True,
    ret: ret [1]: 'into 2 and 1'
    --- alt:
    ret: input: 'no match',
    ret: ret [0]: False,
    ret: ret [1]: 'no match';"""

    sub_expr = _split_repl(sub)
    repl_expr = _split_repl(repl)
    expr = ""
    components = []
    for mode, buffer in sub_expr:
        if mode:
            expr += r"(.*?)"
            components.append(buffer)
        else:
            expr += re.escape(buffer)
    expr = re.compile(expr)

    def _replace(text: str) -> Tuple[bool, str]:
        nonlocal expr, components, repl_expr
        real_expr = cast(re.Pattern, expr)
        match = real_expr.fullmatch(text)
        if match is None:
            return False, text
        groups = match.groups()
        ret = ""
        for mode, buffer in repl_expr:
            if mode:
                ret += groups[components.index(buffer)]
            else:
                ret += buffer
        return True, ret

    return _replace


def _split_repl(pattern: str) -> List[Tuple[bool, str]]:
    """input: 'format {this} and {that}';
    ret [0]: (False, 'format '),
    ret [1]: (True, 'this'),
    ret [2]: (False, ' and '),
    ret [3]: (True, 'that')"""

    ret: List[Tuple[bool, str]] = []
    mode = False
    buffer = ""

    def _flush() -> None:
        nonlocal buffer
        if buffer:
            ret.append((mode, buffer))
            buffer = ""

    for ch in pattern:
        if ch == "{":
            _flush()
            mode = True
        elif ch == "}":
            _flush()
            mode = False
        else:
            buffer += ch
    _flush()
    return ret


def ranged_modulo_test(pattern: str) -> Callable[[int], bool]:
    """Check if a number matches the modulo expression. Example:

    `<=10:%2==3; <=5:%3==1; <= 20 : %5 == 0`"""

    patt_l = [x.strip() for x in pattern.split(";") if x.strip()]
    rng: List[Tuple[int, int, int]] = []  # upper bound, modulo, remainder
    for patt in patt_l:
        match_1 = re.findall(r"<=\s*(\d+)\s*:\s*%\s*(\d+)\s*==\s*(\d+)", patt)
        match_2 = re.findall(r"_\s*:\s*%\s*(\d+)\s*==\s*(\d+)", patt)
        if match_1:
            bnd, mod, rem = map(int, match_1[0])
        elif match_2:
            bnd, (mod, rem) = 10**9, map(int, match_2[0])
        else:
            raise ValueError(f"invalid pattern: {pattern}")
        rng.append((bnd, mod, rem))
    rng.sort(key=lambda x: x[0])

    # calibrate range
    matches: List[Tuple[int, int, int, int]] = []  # low, upper, mod, rem
    prev_low = 0
    for bnd, mod, rem in rng:
        matches.append((prev_low, bnd, mod, rem))
        prev_low = bnd + 1

    def _test(num: int) -> bool:
        for low, top, mod, rem in matches:
            if num >= low and num <= top and num % mod == rem:
                return True
        return False

    return _test


class StringsUtilsTest(unittest.TestCase):
    def test_pattern_replace_single(self):
        repl = pattern_replace_single(
            "format {this} and {that}", "into {that} and {this}"
        )
        self.assertEqual(repl("format 1 and 2"), (True, "into 2 and 1"))
        self.assertEqual(repl("long format 1 and 2"), (False, "long format 1 and 2"))
        self.assertEqual(repl("no match"), (False, "no match"))

    def test_pattern_replace(self):
        rules = {
            "format {this} and {that}": ["into {that} and {this}"],
            "multi {format}": ["a {format}", "b {format}"],
            "a{b}c": ["a{b}c"],
        }
        repl = pattern_replace(rules)
        self.assertEqual(repl("format 1 and 2"), (True, ["into 2 and 1"]))
        self.assertEqual(repl("multi format"), (True, ["a format", "b format"]))
        self.assertEqual(repl("a1c"), (True, ["a1c"]))
        self.assertEqual(repl("no match"), (False, ["no match"]))

    def test_ranged_modulo_test(self):
        def _test(patt: str, expected: str):
            fn = ranged_modulo_test(patt)
            ss = "".join("*" if fn(i) else "." for i in range(len(expected)))
            self.assertEqual(ss, expected)

        _test("<=10:%2==0; <=5:%3==1; <= 20 : %5 == 0", ".*..*.*.*.*....*....*")
        _test(" <=6:%4==2 ;", "..*...*.......")
        _test("<=5:%2==1; _:%3==0", ".*.*.**..*..*..*..")

    pass
