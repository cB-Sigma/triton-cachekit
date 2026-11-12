"""``stats``: where start-up compile time goes, and what a warm cache would save."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from typing import Any

from ._util import human_duration_us, render_table
from .recording import Recording


@dataclass
class KernelStats:
    kernel: str
    compilations: int = 0
    hits: int = 0
    misses: int = 0
    unknown: int = 0
    miss_us: int = 0
    hit_us: int = 0
    stages_us: dict[str, int] = field(default_factory=dict)


@dataclass
class Stats:
    kernels: list[KernelStats]
    total_us: int
    miss_us: int
    hit_us: int
    hits: int
    misses: int
    unknown: int
    hit_cost_us: float | None  # median cost of a cache hit in this recording
    stages_us: dict[str, int]
    processes: int

    @property
    def removable_us(self) -> float:
        """Compile time a fully warm cache would have removed.

        A hit is not free (Triton still builds the key and reads the entry), so
        when the recording contains hits their median cost is charged for each
        miss; without hits the full miss time is an upper bound.
        """
        if self.hit_cost_us is None:
            return float(self.miss_us)
        return max(0.0, self.miss_us - self.hit_cost_us * self.misses)

    @property
    def removable_share(self) -> float:
        return self.removable_us / self.total_us if self.total_us else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "processes": self.processes,
            "compilations": self.hits + self.misses + self.unknown,
            "hits": self.hits,
            "misses": self.misses,
            "unknown": self.unknown,
            "total_us": self.total_us,
            "miss_us": self.miss_us,
            "hit_us": self.hit_us,
            "median_hit_cost_us": self.hit_cost_us,
            "removable_by_warm_cache_us": round(self.removable_us),
            "removable_share": round(self.removable_share, 4),
            "removable_is_upper_bound": self.hit_cost_us is None,
            "stages_us": self.stages_us,
            "kernels": [
                {
                    "kernel": k.kernel,
                    "compilations": k.compilations,
                    "hits": k.hits,
                    "misses": k.misses,
                    "miss_us": k.miss_us,
                    "hit_us": k.hit_us,
                    "stages_us": k.stages_us,
                }
                for k in self.kernels
            ],
        }


def compute_stats(recordings: list[Recording]) -> Stats:
    table: dict[str, KernelStats] = {}
    hit_costs: list[int] = []
    stages: dict[str, int] = {}
    pids: set[tuple[str, int]] = set()
    for recording in recordings:
        for item in recording.compiles:
            pids.add((recording.path, int(item.raw.get("pid", 0))))
            stat = table.setdefault(item.kernel, KernelStats(kernel=item.kernel))
            stat.compilations += 1
            total = item.total_us or 0
            if item.cache_hit is None:
                stat.unknown += 1
            elif item.cache_hit:
                stat.hits += 1
                stat.hit_us += total
                hit_costs.append(total)
            else:
                stat.misses += 1
                stat.miss_us += total
                for stage, value in item.stage_us.items():
                    stat.stages_us[stage] = stat.stages_us.get(stage, 0) + value
                    stages[stage] = stages.get(stage, 0) + value
    kernels = sorted(table.values(), key=lambda k: (-k.miss_us, k.kernel))
    miss_us = sum(k.miss_us for k in kernels)
    hit_us = sum(k.hit_us for k in kernels)
    return Stats(
        kernels=kernels,
        total_us=miss_us + hit_us,
        miss_us=miss_us,
        hit_us=hit_us,
        hits=sum(k.hits for k in kernels),
        misses=sum(k.misses for k in kernels),
        unknown=sum(k.unknown for k in kernels),
        hit_cost_us=statistics.median(hit_costs) if hit_costs else None,
        stages_us=stages,
        processes=len(pids),
    )


def render_text(stats: Stats, *, top: int = 20) -> str:
    out = []
    count = stats.hits + stats.misses + stats.unknown
    out.append(
        f"{count} compilations in {stats.processes} process(es): "
        f"{stats.misses} cache misses, {stats.hits} hits"
        + (f", {stats.unknown} without outcome" if stats.unknown else "")
    )
    out.append(
        f"Time inside triton.compile: {human_duration_us(stats.total_us)} "
        f"({human_duration_us(stats.miss_us)} on misses, {human_duration_us(stats.hit_us)} on hits)"
    )
    if stats.misses:
        if stats.hit_cost_us is None:
            out.append(
                f"A warm cache would remove at most {human_duration_us(stats.removable_us)} "
                f"({stats.removable_share:.0%} of compile time; upper bound - this recording has no "
                "hits to measure what a hit costs)."
            )
        else:
            out.append(
                f"A warm cache would remove about {human_duration_us(stats.removable_us)} "
                f"({stats.removable_share:.0%} of compile time), charging the median hit cost of "
                f"{human_duration_us(stats.hit_cost_us)} for each of the {stats.misses} misses."
            )
    else:
        out.append("No misses: the cache was already warm for this run.")
    out.append("")
    rows = []
    for kernel in stats.kernels[:top]:
        share = kernel.miss_us / stats.miss_us if stats.miss_us else 0.0
        slowest = (
            max(kernel.stages_us.items(), key=lambda kv: kv[1])[0] if kernel.stages_us else "-"
        )
        rows.append(
            [
                kernel.kernel,
                str(kernel.compilations),
                str(kernel.misses),
                str(kernel.hits),
                human_duration_us(kernel.miss_us),
                f"{share:.0%}",
                slowest,
            ]
        )
    out.append(
        render_table(
            ["KERNEL", "COMPILES", "MISS", "HIT", "MISS TIME", "SHARE", "SLOWEST STAGE"],
            rows,
            right_align=(1, 2, 3, 4, 5),
        )
    )
    if len(stats.kernels) > top:
        out.append(f"... and {len(stats.kernels) - top} more kernels (use --top)")
    if stats.stages_us:
        out.append("")
        out.append("Miss time by stage:")
        total = sum(stats.stages_us.values()) or 1
        out.append(
            render_table(
                ["STAGE", "TIME", "SHARE"],
                [
                    [stage, human_duration_us(value), f"{value / total:.0%}"]
                    for stage, value in sorted(stats.stages_us.items(), key=lambda kv: -kv[1])
                ],
                right_align=(1, 2),
            )
        )
    out.append("")
    out.append(
        "Note: this counts time inside triton.compile only. Time the application spends before "
        "the first compilation (imports, model load) is not in the recording."
    )
    return "\n".join(out) + "\n"


def render_json(stats: Stats) -> str:
    return json.dumps(stats.to_dict(), indent=2)
