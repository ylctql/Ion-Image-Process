"""
离子检测 — 兼容入口，核心实现位于 ``ion_detect`` 包。

推荐使用::

    from ion_detect import detect_ions, visualize, print_summary

或命令行::

    python -m ion_detect [indices] [options]

历史脚本仍可使用 ``from ion_detection import detect_ions``。
"""
from ion_detect import (
    bgsub_binarize,
    bgsub_binarize_u8,
    detect_ions,
    print_summary,
    visualize,
    visualize_bgsub,
    visualize_bgsub_binarized,
)

__all__ = [
    "detect_ions",
    "visualize",
    "visualize_bgsub",
    "visualize_bgsub_binarized",
    "bgsub_binarize",
    "bgsub_binarize_u8",
    "print_summary",
]

if __name__ == "__main__":
    from ion_detect.__main__ import main

    main()
