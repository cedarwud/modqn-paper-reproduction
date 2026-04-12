"""Training script wrapper.

Delegates to the package CLI, which rejects paper-envelope configs and
requires a resolved-run training config.
"""

from modqn_paper_reproduction.cli import train_main


def main() -> int:
    return train_main()


if __name__ == "__main__":
    raise SystemExit(main())
