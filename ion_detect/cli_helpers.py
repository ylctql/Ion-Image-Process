"""命令行与批处理用的索引解析。"""
import os
from pathlib import Path

import numpy as np

from .frame_io import is_supported_frame_path, list_frame_files


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


def resolve_frame_paths_by_name(names: list[str], files: list[Path]) -> list[tuple[int, Path]]:
    """
    将用户在数据目录下给出的文件名解析为 ``(在 files 中的索引, 路径)``，顺序与 ``names`` 一致。

    匹配规则（依次尝试）：

    1. 与 ``Path.name`` 完全一致；在 Windows 上对扩展名大小写不敏感。
    2. 无扩展名时与 ``Path.stem`` 一致，且目录中仅有一帧满足该 stem；若多帧（如 ``a.npy`` 与 ``a.jpg``）则报错。
    """
    if not names:
        raise ValueError("未提供任何文件名。")
    out: list[tuple[int, Path]] = []
    win = os.name == "nt"

    def name_matches(p: Path, q: str) -> bool:
        if p.name == q:
            return True
        return bool(win and p.name.lower() == q.lower())

    for raw in names:
        name = raw.strip()
        if not name:
            continue
        hit: list[int] = []
        for i, p in enumerate(files):
            if name_matches(p, name):
                hit.append(i)
        if not hit:
            stem_hit = [i for i, p in enumerate(files) if p.stem == name]
            if len(stem_hit) == 1:
                hit = stem_hit
            elif len(stem_hit) > 1:
                amb = [files[i].name for i in stem_hit]
                raise ValueError(
                    f"文件名 {name!r} 对应多帧（无扩展名时歧义），请写全名: {amb}",
                )
        if len(hit) != 1:
            raise FileNotFoundError(
                f"数据目录中未找到帧文件: {name!r}（共 {len(files)} 个候选）",
            )
        i = hit[0]
        out.append((i, files[i]))
    if not out:
        raise ValueError("未提供任何有效文件名。")
    return out


def has_path_component(s: str) -> bool:
    """
    判断用户输入是否包含目录/路径信息（相对或绝对）。

    仅含单个文件名（如 ``foo.npy``）返回 False；含子目录或盘符等则返回 True，
    用于决定是否在 ``--data-dir`` 下拼接路径。
    """
    p = Path(s.strip()).expanduser()
    if p.is_absolute():
        return True
    return len(p.parts) > 1


def resolve_frame_paths_from_args(names: list[str], data_dir: Path) -> list[tuple[int | None, Path]]:
    """
    解析 ``--file`` 列表：若某条含路径分量则按该路径定位文件；否则在 ``data_dir`` 下列表中按名匹配。

    返回 ``(在 data_dir 排序列表中的索引, 路径)``；若文件不在该列表中（例如位于其它目录的绝对路径），
    则索引为 ``None``。
    """
    if not names:
        raise ValueError("未提供任何文件名。")
    data_dir = Path(data_dir).resolve()
    files = list_frame_files(data_dir)
    has_basename_only = any(
        not has_path_component(n) for n in names if n.strip()
    )
    if not files and has_basename_only:
        raise FileNotFoundError(
            f"目录中无可用帧文件: {data_dir}（需要 .npy 或栅格图: .jpg/.jpeg/.png 等）",
        )

    out: list[tuple[int | None, Path]] = []
    for raw in names:
        name = raw.strip()
        if not name:
            continue
        if has_path_component(name):
            p = Path(name).expanduser().resolve()
            if not p.is_file():
                raise FileNotFoundError(f"不是已存在的文件: {p}")
            if not is_supported_frame_path(p):
                raise ValueError(f"不支持的帧文件类型: {p}")
            idx: int | None = None
            if files:
                rp = p.resolve()
                for i, q in enumerate(files):
                    if q.resolve() == rp:
                        idx = i
                        break
            out.append((idx, p))
        else:
            i, path = resolve_frame_paths_by_name([name], files)[0]
            out.append((i, path))
    if not out:
        raise ValueError("未提供任何有效文件名。")
    return out
