"""MAPPO-oriented training entry point.

The core benchmark intentionally has no mandatory heavy RL dependency. Install
`.[rl]` to use RLlib's multi-agent PPO implementation. A production MAPPO
experiment can share one policy across homogeneous robots or define one policy
per robot. This script provides a lightweight dependency check and documents
that intended execution path.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--robots", type=int, default=3)
    args = parser.parse_args()

    try:
        import ray  # noqa: F401
        from ray.rllib.algorithms.ppo import PPOConfig  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install RL dependencies with: pip install -e '.[rl]'") from exc

    print(
        "RLlib is available. Configure PPO/MAPPO with a shared policy using "
        f"{args.robots} agents for {args.iterations} training iterations. "
        "The benchmark environment in environment.py exposes the joint dynamics; "
        "adapt it to RLlib MultiAgentEnv/PettingZoo according to the experiment design."
    )


if __name__ == "__main__":
    main()
