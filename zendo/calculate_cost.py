import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

INPUT_COST_PER_MILLION = 2.5
OUTPUT_COST_PER_MILLION = 10.0
TOKENS_PER_MILLION = 1_000_000


@dataclass
class UsageSummary:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    calls_with_usage: int = 0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "UsageSummary":
        data = data or {}
        input_tokens = int(data.get("input_tokens", 0) or 0)
        output_tokens = int(data.get("output_tokens", 0) or 0)
        total_tokens = int(data.get("total_tokens", input_tokens + output_tokens) or 0)
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            calls=int(data.get("calls", 0) or 0),
            calls_with_usage=int(data.get("calls_with_usage", 0) or 0),
        )

    @property
    def input_cost(self) -> float:
        return (self.input_tokens / TOKENS_PER_MILLION) * INPUT_COST_PER_MILLION

    @property
    def output_cost(self) -> float:
        return (self.output_tokens / TOKENS_PER_MILLION) * OUTPUT_COST_PER_MILLION

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost


@dataclass
class GameCost:
    label: str
    source_path: Path
    usage: UsageSummary
    usage_found: bool


def load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def is_history_payload(payload: Any) -> bool:
    return (
        isinstance(payload, list)
        and bool(payload)
        and all(isinstance(item, dict) for item in payload)
        and all({"timestamp", "event", "data"} <= item.keys() for item in payload)
    )


def is_ablation_payload(payload: Any) -> bool:
    return (
        isinstance(payload, list)
        and (
            not payload
            or all(
                isinstance(item, dict)
                and ("history_file" in item or "llm_usage" in item or "experiment_name" in item)
                for item in payload
            )
        )
    )


def extract_usage_from_history_payload(payload: List[Dict[str, Any]]) -> Optional[UsageSummary]:
    for event in reversed(payload):
        if event.get("event") == "llm_usage_summary":
            return UsageSummary.from_dict(event.get("data"))
    return None


def resolve_history_path(reference: str, parent_path: Path) -> Path:
    ref_path = Path(reference)
    candidates = []
    if ref_path.is_absolute():
        candidates.append(ref_path)
    else:
        candidates.append(parent_path / ref_path)
        candidates.append(Path.cwd() / ref_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[0].resolve()


def build_game_label(index: int, item: Dict[str, Any], history_path: Optional[Path]) -> str:
    parts = [f"game_{index + 1}"]
    if item.get("experiment_name"):
        parts.append(str(item["experiment_name"]))
    if item.get("rule_index") is not None:
        parts.append(f"rule_{item['rule_index']}")
    if item.get("run_index") is not None:
        parts.append(f"run_{item['run_index']}")
    if history_path is not None:
        parts.append(history_path.name)
    return " | ".join(parts)


def game_costs_from_history(path: Path, payload: List[Dict[str, Any]]) -> List[GameCost]:
    usage = extract_usage_from_history_payload(payload)
    return [
        GameCost(
            label=path.name,
            source_path=path,
            usage=usage or UsageSummary(),
            usage_found=usage is not None,
        )
    ]


def game_costs_from_ablation(path: Path, payload: List[Dict[str, Any]]) -> List[GameCost]:
    games: List[GameCost] = []
    for index, item in enumerate(payload):
        usage = None
        history_path = None
        if item.get("llm_usage") is not None:
            usage = UsageSummary.from_dict(item["llm_usage"])
        elif item.get("history_file"):
            history_path = resolve_history_path(item["history_file"], path.parent)
            if history_path.exists():
                history_payload = load_json(history_path)
                if is_history_payload(history_payload):
                    usage = extract_usage_from_history_payload(history_payload)

        games.append(
            GameCost(
                label=build_game_label(index, item, history_path),
                source_path=history_path or path,
                usage=usage or UsageSummary(),
                usage_found=usage is not None,
            )
        )
    return games


def game_costs_from_run_result(path: Path, payload: Dict[str, Any]) -> List[GameCost]:
    usage = UsageSummary.from_dict(payload.get("llm_usage"))
    return [
        GameCost(
            label=path.name,
            source_path=path,
            usage=usage,
            usage_found=payload.get("llm_usage") is not None,
        )
    ]


def load_game_costs(path: Path) -> List[GameCost]:
    payload = load_json(path)
    if is_history_payload(payload):
        return game_costs_from_history(path, payload)
    if isinstance(payload, dict):
        return game_costs_from_run_result(path, payload)
    if is_ablation_payload(payload):
        return game_costs_from_ablation(path, payload)
    raise ValueError(f"Unsupported JSON format in {path}")


def aggregate_usage(games: Iterable[GameCost]) -> UsageSummary:
    total = UsageSummary()
    for game in games:
        total.input_tokens += game.usage.input_tokens
        total.output_tokens += game.usage.output_tokens
        total.total_tokens += game.usage.total_tokens
        total.calls += game.usage.calls
        total.calls_with_usage += game.usage.calls_with_usage
    return total


def format_dollars(amount: float) -> str:
    return f"${amount:.6f}"


def print_report(path: Path, games: List[GameCost], summary_only: bool):
    print(f"File: {path}")
    if not summary_only:
        for game in games:
            status = "usage found" if game.usage_found else "usage missing"
            print(
                f"  {game.label}: "
                f"input={game.usage.input_tokens}, "
                f"output={game.usage.output_tokens}, "
                f"total={game.usage.total_tokens}, "
                f"cost={format_dollars(game.usage.total_cost)} "
                f"({status})"
            )

    totals = aggregate_usage(games)
    missing_games = sum(1 for game in games if not game.usage_found)
    print(f"  games={len(games)}")
    print(f"  input_tokens={totals.input_tokens}")
    print(f"  output_tokens={totals.output_tokens}")
    print(f"  total_tokens={totals.total_tokens}")
    print(f"  input_cost={format_dollars(totals.input_cost)}")
    print(f"  output_cost={format_dollars(totals.output_cost)}")
    print(f"  total_cost={format_dollars(totals.total_cost)}")
    print(f"  missing_usage_games={missing_games}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate LLM input/output token costs from game history or ablation results."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more JSON files. Supported inputs: per-game history files, run result JSON, or ablation results JSON.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print aggregate totals for each input file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    all_games: List[GameCost] = []

    for raw_path in args.paths:
        path = Path(raw_path).resolve()
        games = load_game_costs(path)
        print_report(path, games, args.summary_only)
        all_games.extend(games)
        if len(args.paths) > 1:
            print()

    if len(args.paths) > 1:
        print("Overall Totals:")
        overall = aggregate_usage(all_games)
        missing_games = sum(1 for game in all_games if not game.usage_found)
        print(f"  games={len(all_games)}")
        print(f"  input_tokens={overall.input_tokens}")
        print(f"  output_tokens={overall.output_tokens}")
        print(f"  total_tokens={overall.total_tokens}")
        print(f"  input_cost={format_dollars(overall.input_cost)}")
        print(f"  output_cost={format_dollars(overall.output_cost)}")
        print(f"  total_cost={format_dollars(overall.total_cost)}")
        print(f"  missing_usage_games={missing_games}")


if __name__ == "__main__":
    main()
