"""命令行与批处理用的索引解析。"""
import numpy as np


def parse_slice_token(token):
    token = token.strip()
    if not token:
        raise ValueError("Empty index token.")

    if ":" not in token:
        return int(token)

    parts = token.split(":")
    if len(parts) > 3:
        raise ValueError(f"Invalid slice token: {token}")

    def _to_int_or_none(s):
        s = s.strip()
        return None if s == "" else int(s)

    start = _to_int_or_none(parts[0]) if len(parts) >= 1 else None
    stop = _to_int_or_none(parts[1]) if len(parts) >= 2 else None
    step = _to_int_or_none(parts[2]) if len(parts) >= 3 else None

    if step == 0:
        raise ValueError(f"Slice step cannot be 0: {token}")
    return slice(start, stop, step)


def resolve_indices(spec_args, total):
    """解析索引规范，支持并集，如 ['::3,0:10', '25', '-1']。"""
    if not spec_args:
        spec_args = ["0"]

    selected = set()
    all_indices = np.arange(total)

    for raw in spec_args:
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue

            obj = parse_slice_token(token)
            if isinstance(obj, slice):
                selected.update(all_indices[obj].tolist())
            else:
                idx = obj
                if idx < 0:
                    idx += total
                if 0 <= idx < total:
                    selected.add(idx)
                else:
                    print(f"[skip] index {obj} out of range [0, {total - 1}]")

    return sorted(selected)
