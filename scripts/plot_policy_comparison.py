#!/usr/bin/env python3
"""Generate lightweight SVG charts for the selected routing policy.

The challenge has no direct quality labels or latency measurements. These charts
therefore make the proxies explicit:

* quality proxy: continuation burden delta, lower is better;
* latency proxy: cache-aware prefill token work, not wall-clock latency.
"""

import csv
from pathlib import Path


GRID_CANDIDATES = (
    Path("results/current_gated_sim030_summary.csv"),
    Path("results/userref_policy_grid.csv"),
)
CALLS_CANDIDATES = (
    Path("results/current_gated_sim030_lambda02_calls.csv"),
    Path("results/userref_gated_broad_trajectories_lambda05_calls.csv"),
)
OUT_COST = Path("results/cost_quality_frontier.svg")
OUT_LATENCY = Path("results/latency_proxy_comparison.svg")
OUT_COMPARISON_CSV = Path("results/original_vs_gated_cost_quality.csv")
OUT_COMPARISON_SVG = Path("results/original_vs_gated_cost_quality.svg")
OUT_THREE_WAY_CSV = Path("results/three_way_policy_comparison.csv")
OUT_THREE_WAY_SVG = Path("results/three_way_policy_comparison.svg")
OUT_ALL_POLICY_CSV = Path("results/all_policy_comparison.csv")
OUT_ALL_POLICY_SVG = Path("results/all_policy_comparison.svg")
SEGMENT_AGGREGATE = Path("results/cache_segment_router_aggregate.csv")
CLASSIFIER_AGGREGATE = Path("results/classifier_router_aggregate.csv")


COLORS = {
    "logged": "#111827",
    "current_gated_sim030": "#0f766e",
    "userref_gated_sim030": "#0f766e",
    "userref_tight_complex": "#2563eb",
}


def read_grid():
    grid_path = next((path for path in GRID_CANDIDATES if path.exists()), None)
    if grid_path is None:
        raise FileNotFoundError("No policy summary CSV found in results/.")
    with grid_path.open(newline="") as f:
        raw_rows = list(csv.DictReader(f))
    rows = []
    for row in raw_rows:
        if "policy" in row:
            out = {
                "policy": row["policy"],
                "lambda": float(row["lambda"]),
                "logged_cost_usd": float(row["logged_cost_usd"]),
                "routed_cost_usd": float(row["routed_cost_usd"]),
                "saved_usd": float(row["saved_usd"]),
                "cost_delta_pct": float(row["cost_delta_pct"]),
                "burden_delta": float(row["burden_delta"]),
                "logged_match_pct": float(row["logged_match_pct"]),
                "supported_pct": float(row["supported_pct"]),
                "avg_effective_n": float(row["avg_effective_n"]),
            }
        else:
            routed_cost = float(row["routed_replay_cost_usd"])
            logged_cost = float(row["logged_cost_usd"])
            out = {
                "policy": "current_gated_sim030",
                "lambda": float(row["lambda"]),
                "logged_cost_usd": logged_cost,
                "routed_cost_usd": routed_cost,
                "saved_usd": logged_cost - routed_cost,
                "cost_delta_pct": float(row["replay_cost_delta"]) * 100,
                "burden_delta": float(row["xfit_burden_delta"]),
                "logged_match_pct": float(row["logged_match_rate"]) * 100,
                "supported_pct": float(row["supported_rate"]) * 100,
                "avg_effective_n": float(row["avg_effective_n"]),
            }
        rows.append(out)
    return rows


def selected_row(rows):
    for target_policy, target_lambda in (
        ("current_gated_sim030", 0.2),
        ("userref_gated_sim030", 0.05),
    ):
        for row in rows:
            if row["policy"] == target_policy and abs(row["lambda"] - target_lambda) < 1e-9:
                return row
    return min(rows, key=lambda row: row["routed_cost_usd"])


def same_point(left, right):
    return (
        left["policy"] == right["policy"]
        and abs(left["lambda"] - right["lambda"]) < 1e-9
        and abs(left["routed_cost_usd"] - right["routed_cost_usd"]) < 1e-9
    )


def sx(x, xmin, xmax, left, width):
    if xmax == xmin:
        return left + width / 2
    return left + (x - xmin) / (xmax - xmin) * width


def sy(y, ymin, ymax, top, height):
    if ymax == ymin:
        return top + height / 2
    return top + height - (y - ymin) / (ymax - ymin) * height


def svg_text(x, y, text, size=13, fill="#111827", anchor="start", weight="400"):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
        f'font-weight="{weight}">{text}</text>'
    )


def write_cost_quality(rows):
    rows = [r for r in rows if r["policy"] != "userref_score_sim030"]
    selected = selected_row(rows)
    width, height = 980, 620
    left, top, plot_w, plot_h = 100, 80, 690, 410
    costs = [r["routed_cost_usd"] for r in rows] + [rows[0]["logged_cost_usd"]]
    burdens = [r["burden_delta"] for r in rows] + [0.0]
    xmin, xmax = min(costs) - 2, max(costs) + 4
    ymin, ymax = min(burdens) - 0.001, max(burdens) + 0.001

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        svg_text(40, 40, "Cost-quality comparison: cache-aware cost vs continuation burden", 22, weight="700"),
        svg_text(40, 64, "Quality is proxied by burden delta; lower is better. Tokens and cache are estimated from traces.", 13, "#4b5563"),
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#d1d5db"/>',
    ]

    for i in range(6):
        x = xmin + i * (xmax - xmin) / 5
        px = sx(x, xmin, xmax, left, plot_w)
        parts.append(f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{top + plot_h}" stroke="#eef2f7"/>')
        parts.append(svg_text(px, top + plot_h + 28, f"${x:.0f}", 12, "#4b5563", "middle"))
    for i in range(6):
        y = ymin + i * (ymax - ymin) / 5
        py = sy(y, ymin, ymax, top, plot_h)
        parts.append(f'<line x1="{left}" y1="{py:.1f}" x2="{left + plot_w}" y2="{py:.1f}" stroke="#eef2f7"/>')
        parts.append(svg_text(left - 12, py + 4, f"{y:+.3f}", 12, "#4b5563", "end"))

    baseline_x = sx(rows[0]["logged_cost_usd"], xmin, xmax, left, plot_w)
    baseline_y = sy(0.0, ymin, ymax, top, plot_h)
    parts.append(f'<circle cx="{baseline_x:.1f}" cy="{baseline_y:.1f}" r="8" fill="{COLORS["logged"]}"/>')
    parts.append(svg_text(baseline_x + 12, baseline_y - 10, "logged baseline", 13, COLORS["logged"], weight="700"))

    for policy in sorted({r["policy"] for r in rows}):
        prs = sorted((r for r in rows if r["policy"] == policy), key=lambda r: r["routed_cost_usd"])
        color = COLORS.get(policy, "#6b7280")
        points = [
            f'{sx(r["routed_cost_usd"], xmin, xmax, left, plot_w):.1f},{sy(r["burden_delta"], ymin, ymax, top, plot_h):.1f}'
            for r in prs
        ]
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5" opacity="0.8"/>')
        for r in prs:
            px = sx(r["routed_cost_usd"], xmin, xmax, left, plot_w)
            py = sy(r["burden_delta"], ymin, ymax, top, plot_h)
            radius = 7 if same_point(r, selected) else 4
            stroke = "#f59e0b" if radius == 7 else color
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius}" fill="{color}" stroke="{stroke}" stroke-width="2"/>')

    sel_x = sx(selected["routed_cost_usd"], xmin, xmax, left, plot_w)
    sel_y = sy(selected["burden_delta"], ymin, ymax, top, plot_h)
    parts.append(svg_text(sel_x + 12, sel_y - 12, "selected gated policy", 13, "#0f766e", weight="700"))
    parts.append(svg_text(left + plot_w / 2, height - 60, "Estimated cache-aware cost, USD (lower is better)", 14, "#111827", "middle", "700"))
    parts.append(
        f'<text x="28" y="{top + plot_h / 2:.1f}" font-family="Arial, sans-serif" font-size="14" fill="#111827" '
        'text-anchor="middle" font-weight="700" transform="rotate(-90 28 '
        f'{top + plot_h / 2:.1f})">Continuation burden delta (lower is better)</text>'
    )

    lx, ly = 815, 130
    policy_labels = {
        "current_gated_sim030": "gated safe policy",
        "userref_gated_sim030": "gated safe policy",
        "userref_tight_complex": "tight conservative",
    }
    legend = [("logged baseline", COLORS["logged"])]
    for policy in sorted({r["policy"] for r in rows}):
        legend.append((policy_labels.get(policy, policy), COLORS.get(policy, "#6b7280")))
    parts.append(f'<rect x="{lx - 10}" y="{ly - 30}" width="145" height="130" rx="10" fill="#ffffff" stroke="#d1d5db"/>')
    for i, (label, color) in enumerate(legend):
        y = ly + i * 28
        parts.append(f'<circle cx="{lx}" cy="{y}" r="5" fill="{color}"/>')
        parts.append(svg_text(lx + 14, y + 4, label, 12, "#374151"))

    parts.append(svg_text(815, 310, f"Selected cost: ${selected['routed_cost_usd']:.2f}", 13, "#111827", weight="700"))
    parts.append(svg_text(815, 332, f"Savings: {-selected['cost_delta_pct']:.2f}%", 13, "#111827"))
    parts.append(svg_text(815, 354, f"Burden delta: {selected['burden_delta']:+.4f}", 13, "#111827"))
    parts.append(svg_text(815, 376, f"Agreement: {selected['logged_match_pct']:.2f}%", 13, "#111827"))
    parts.append("</svg>")
    OUT_COST.write_text("\n".join(parts))


def latency_proxy_from_calls():
    calls_path = next((path for path in CALLS_CANDIDATES if path.exists()), None)
    if calls_path is None:
        raise FileNotFoundError("No trajectory call CSV found in results/.")
    original = 0.0
    routed = 0.0
    with calls_path.open(newline="") as f:
        for row in csv.DictReader(f):
            original += float(row["original_uncached_tokens_est"]) + 0.2 * float(row["original_cached_tokens_est"])
            routed += float(row["new_uncached_tokens_est"]) + 0.2 * float(row["new_cached_tokens_est"])
    return original, routed


def write_latency_proxy():
    original, routed = latency_proxy_from_calls()
    width, height = 720, 430
    max_v = max(original, routed) * 1.12
    left, top, plot_w, plot_h = 90, 90, 500, 230
    bars = [("logged baseline", original, COLORS["logged"]), ("selected gated", routed, COLORS["userref_gated_sim030"])]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        svg_text(38, 42, "Estimated latency proxy comparison", 22, weight="700"),
        svg_text(38, 66, "Proxy = uncached input tokens + 0.2 × cached prefix tokens. This is not wall-clock latency.", 13, "#4b5563"),
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#d1d5db"/>',
    ]
    for i in range(5):
        v = i * max_v / 4
        y = sy(v, 0, max_v, top, plot_h)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#eef2f7"/>')
        parts.append(svg_text(left - 10, y + 4, f"{v/1e6:.1f}M", 12, "#4b5563", "end"))
    for i, (label, value, color) in enumerate(bars):
        bar_w = 120
        x = left + 110 + i * 210
        y = sy(value, 0, max_v, top, plot_h)
        h = top + plot_h - y
        parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" fill="{color}" rx="6"/>')
        parts.append(svg_text(x + bar_w / 2, top + plot_h + 28, label, 13, "#111827", "middle", "700"))
        parts.append(svg_text(x + bar_w / 2, y - 10, f"{value/1e6:.2f}M", 13, "#111827", "middle"))
    delta = (routed / original - 1) * 100 if original else 0.0
    parts.append(svg_text(38, 375, f"Selected policy changes this proxy by {delta:+.2f}%.", 14, "#111827", weight="700"))
    parts.append(svg_text(38, 398, "Interpretation: expected speed is roughly unchanged; the real benefit is cost reduction.", 13, "#4b5563"))
    parts.append("</svg>")
    OUT_LATENCY.write_text("\n".join(parts))


def write_original_vs_gated(rows):
    selected = selected_row(rows)
    records = [
        {
            "policy": "original_logged",
            "cost_usd": selected["logged_cost_usd"],
            "cost_delta_pct": 0.0,
            "burden_delta": 0.0,
            "supported_pct": 100.0,
            "avg_effective_n": "",
            "note": "Observed logged model choices.",
        },
        {
            "policy": f"gated_router_lambda_{selected['lambda']:.2f}",
            "cost_usd": selected["routed_cost_usd"],
            "cost_delta_pct": selected["cost_delta_pct"],
            "burden_delta": selected["burden_delta"],
            "supported_pct": selected["supported_pct"],
            "avg_effective_n": selected["avg_effective_n"],
            "note": "Trajectory-level gated router; quality is continuation-burden proxy.",
        },
    ]
    with OUT_COMPARISON_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    width, height = 820, 430
    original_cost = selected["logged_cost_usd"]
    gated_cost = selected["routed_cost_usd"]
    saved = selected["saved_usd"]
    cost_reduction = -selected["cost_delta_pct"]
    burden_delta = selected["burden_delta"]
    max_cost = original_cost * 1.12
    left, top, plot_w, plot_h = 90, 100, 470, 230
    bars = [
        ("Original", original_cost, COLORS["logged"]),
        ("Gated router", gated_cost, COLORS["userref_gated_sim030"]),
    ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        svg_text(36, 42, "Original vs gated router: cost-quality comparison", 22, weight="700"),
        svg_text(36, 66, "Cost is cache-aware estimated input cost. Quality is a continuation-burden proxy; lower burden is better.", 13, "#4b5563"),
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#d1d5db"/>',
    ]
    for i in range(5):
        v = i * max_cost / 4
        y = sy(v, 0, max_cost, top, plot_h)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#eef2f7"/>')
        parts.append(svg_text(left - 10, y + 4, f"${v:.0f}", 12, "#4b5563", "end"))
    for i, (label, value, color) in enumerate(bars):
        bar_w = 118
        x = left + 105 + i * 205
        y = sy(value, 0, max_cost, top, plot_h)
        h = top + plot_h - y
        parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" fill="{color}" rx="6"/>')
        parts.append(svg_text(x + bar_w / 2, y - 12, f"${value:.2f}", 14, "#111827", "middle", "700"))
        parts.append(svg_text(x + bar_w / 2, top + plot_h + 28, label, 13, "#111827", "middle", "700"))

    parts.extend(
        [
            svg_text(600, 125, f"Savings: ${saved:.2f}", 16, "#111827", weight="700"),
            svg_text(600, 155, f"Cost reduction: {cost_reduction:.2f}%", 15, "#111827"),
            svg_text(600, 185, f"Burden delta: {burden_delta:+.4f}", 15, "#111827"),
            svg_text(600, 215, f"Supported routes: {selected['supported_pct']:.2f}%", 15, "#111827"),
            svg_text(600, 245, f"Avg effective n: {selected['avg_effective_n']:.1f}", 15, "#111827"),
            svg_text(36, 388, "Interpretation: the gated policy spends less while the proxy quality moves slightly in the favorable direction.", 13, "#374151"),
            "</svg>",
        ]
    )
    OUT_COMPARISON_SVG.write_text("\n".join(parts))


def row_for_lambda(rows, lam):
    candidates = [r for r in rows if abs(r["lambda"] - lam) < 1e-9]
    if not candidates:
        raise ValueError(f"No policy row found for lambda={lam}")
    return candidates[0]


def write_three_way_comparison(rows):
    if SEGMENT_AGGREGATE.exists():
        with SEGMENT_AGGREGATE.open(newline="") as f:
            aggregate = {row["policy"]: row for row in csv.DictReader(f)}
        original = aggregate["original_logged"]
        gated = aggregate["gated_router_lambda_0.20"]
        recommended = aggregate["cache_segment_switch_penalty_2000_tokens"]
        records = [
            {
                "policy": "original_logged",
                "cost_usd": float(original["cost_usd"]),
                "cost_delta_pct": float(original["cost_delta_pct"]),
                "burden_delta": float(original["burden_delta"]),
                "changed_trajectories": int(original["changed_trajectories"]),
                "switch_penalty_usd": float(original["switch_penalty_usd"]),
            },
            {
                "policy": "gated_router_lambda_0.20",
                "cost_usd": float(gated["cost_usd"]),
                "cost_delta_pct": float(gated["cost_delta_pct"]),
                "burden_delta": float(gated["burden_delta"]),
                "changed_trajectories": int(gated["changed_trajectories"]),
                "switch_penalty_usd": float(gated["switch_penalty_usd"]),
            },
            {
                "policy": "cache_segment_router_switch_penalty_2000_tokens",
                "cost_usd": float(recommended["cost_usd"]),
                "cost_delta_pct": float(recommended["cost_delta_pct"]),
                "burden_delta": float(recommended["burden_delta"]),
                "changed_trajectories": int(recommended["changed_trajectories"]),
                "switch_penalty_usd": float(recommended["switch_penalty_usd"]),
            },
        ]
    else:
        gated = row_for_lambda(rows, 0.0)
        recommended = selected_row(rows)
        records = [
            {
                "policy": "original_logged",
                "cost_usd": gated["logged_cost_usd"],
                "cost_delta_pct": 0.0,
                "burden_delta": 0.0,
                "changed_trajectories": 0,
                "switch_penalty_usd": 0.0,
            },
            {
                "policy": "gated_router_lambda_0.00",
                "cost_usd": gated["routed_cost_usd"],
                "cost_delta_pct": gated["cost_delta_pct"],
                "burden_delta": gated["burden_delta"],
                "changed_trajectories": "",
                "switch_penalty_usd": 0.0,
            },
            {
                "policy": f"gated_router_lambda_{recommended['lambda']:.2f}",
                "cost_usd": recommended["routed_cost_usd"],
                "cost_delta_pct": recommended["cost_delta_pct"],
                "burden_delta": recommended["burden_delta"],
                "changed_trajectories": "",
                "switch_penalty_usd": 0.0,
            },
        ]
    with OUT_THREE_WAY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    width, height = 980, 470
    left, top, plot_w, plot_h = 90, 100, 600, 250
    max_cost = max(float(r["cost_usd"]) for r in records) * 1.12
    colors = ["#111827", "#2563eb", "#0f766e"]
    labels = ["Original", "Gated λ=0.20", "Segment + switch penalty"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        svg_text(36, 42, "Three-way policy comparison", 22, weight="700"),
        svg_text(36, 66, "Cost uses cache-aware estimated input tokens. Quality uses continuation-burden delta; lower is better.", 13, "#4b5563"),
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#d1d5db"/>',
    ]
    for i in range(5):
        v = i * max_cost / 4
        y = sy(v, 0, max_cost, top, plot_h)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#eef2f7"/>')
        parts.append(svg_text(left - 10, y + 4, f"${v:.0f}", 12, "#4b5563", "end"))
    for i, (record, label, color) in enumerate(zip(records, labels, colors)):
        value = float(record["cost_usd"])
        bar_w = 115
        x = left + 80 + i * 175
        y = sy(value, 0, max_cost, top, plot_h)
        h = top + plot_h - y
        parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" fill="{color}" rx="6"/>')
        parts.append(svg_text(x + bar_w / 2, y - 12, f"${value:.2f}", 14, "#111827", "middle", "700"))
        parts.append(svg_text(x + bar_w / 2, top + plot_h + 26, label, 12, "#111827", "middle", "700"))

    recommended = records[2]
    saved_vs_original = -float(recommended["cost_delta_pct"])
    parts.extend(
        [
            svg_text(730, 112, "Concrete recommendation", 15, "#111827", weight="700"),
            svg_text(730, 140, "Cache-aware segment router", 14, "#111827"),
            svg_text(730, 166, "Use gated λ=0.20 as proposal", 14, "#111827"),
            svg_text(730, 192, "Keep/switch per call segment", 14, "#111827"),
            svg_text(730, 218, "Charge 2k-token switch penalty", 14, "#111827"),
            svg_text(730, 258, f"Savings vs original: {saved_vs_original:.2f}%", 14, "#111827", weight="700"),
            svg_text(730, 284, f"Burden delta: {float(recommended['burden_delta']):+.4f}", 14, "#111827"),
            svg_text(730, 310, f"Switch penalty: ${float(recommended['switch_penalty_usd']):.4f}", 14, "#111827"),
            svg_text(36, 425, "Interpretation: the segment router keeps the gated quality guard, then avoids model switches unless cache-aware savings survive the switch penalty.", 13, "#374151"),
            "</svg>",
        ]
    )
    OUT_THREE_WAY_SVG.write_text("\n".join(parts))


def read_aggregate(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_all_policy_comparison():
    if not SEGMENT_AGGREGATE.exists() or not CLASSIFIER_AGGREGATE.exists():
        return
    segment_rows = {row["policy"]: row for row in read_aggregate(SEGMENT_AGGREGATE)}
    classifier_rows = {row["policy"]: row for row in read_aggregate(CLASSIFIER_AGGREGATE)}
    records = [
        segment_rows["original_logged"],
        segment_rows["gated_router_lambda_0.20"],
        segment_rows["cache_segment_switch_penalty_2000_tokens"],
        classifier_rows["classifier_cache_segment_switch_penalty_2000_tokens"],
    ]
    display_names = {
        "original_logged": "Original logged",
        "gated_router_lambda_0.20": "Gated router λ=0.20",
        "cache_segment_switch_penalty_2000_tokens": "Cache segment router",
        "classifier_cache_segment_switch_penalty_2000_tokens": "Classifier + cache segment",
    }
    normalized = []
    for row in records:
        normalized.append(
            {
                "policy": row["policy"],
                "display_name": display_names[row["policy"]],
                "cost_usd": float(row["cost_usd"]),
                "cost_delta_pct": float(row["cost_delta_pct"]),
                "burden_delta": float(row["burden_delta"]),
                "changed_trajectories": row["changed_trajectories"],
                "switch_penalty_usd": float(row["switch_penalty_usd"]),
                "classifier_label_match_pct": row.get("classifier_label_match_pct", ""),
            }
        )
    with OUT_ALL_POLICY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(normalized[0]))
        writer.writeheader()
        writer.writerows(normalized)

    width, height = 1060, 500
    left, top, plot_w, plot_h = 90, 100, 690, 260
    max_cost = max(row["cost_usd"] for row in normalized) * 1.12
    colors = ["#111827", "#2563eb", "#0f766e", "#ea580c"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        svg_text(36, 42, "All-policy comparison", 22, weight="700"),
        svg_text(36, 66, "Classifier is cross-fitted and used only as proposal; cache segment optimizer remains the final guard.", 13, "#4b5563"),
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#d1d5db"/>',
    ]
    for i in range(5):
        v = i * max_cost / 4
        y = sy(v, 0, max_cost, top, plot_h)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#eef2f7"/>')
        parts.append(svg_text(left - 10, y + 4, f"${v:.0f}", 12, "#4b5563", "end"))
    for i, (row, color) in enumerate(zip(normalized, colors)):
        value = row["cost_usd"]
        bar_w = 105
        x = left + 60 + i * 155
        y = sy(value, 0, max_cost, top, plot_h)
        h = top + plot_h - y
        parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" fill="{color}" rx="6"/>')
        parts.append(svg_text(x + bar_w / 2, y - 12, f"${value:.2f}", 14, "#111827", "middle", "700"))
        parts.append(svg_text(x + bar_w / 2, top + plot_h + 24, row["display_name"], 11, "#111827", "middle", "700"))

    best = min(normalized, key=lambda row: row["cost_usd"])
    parts.extend(
        [
            svg_text(815, 110, "Best offline cost", 15, "#111827", weight="700"),
            svg_text(815, 138, best["display_name"], 14, "#111827"),
            svg_text(815, 166, f"Cost: ${best['cost_usd']:.2f}", 14, "#111827"),
            svg_text(815, 194, f"Cost delta: {best['cost_delta_pct']:.2f}%", 14, "#111827"),
            svg_text(815, 222, f"Burden delta: {best['burden_delta']:+.4f}", 14, "#111827"),
            svg_text(815, 250, f"Switch penalty: ${best['switch_penalty_usd']:.4f}", 14, "#111827"),
            svg_text(36, 450, "Interpretation: the classifier improves estimated cost only when the cache-aware segment guard is retained; the raw classifier route is not the recommended policy.", 13, "#374151"),
            "</svg>",
        ]
    )
    OUT_ALL_POLICY_SVG.write_text("\n".join(parts))


def main():
    rows = read_grid()
    write_cost_quality(rows)
    write_latency_proxy()
    write_original_vs_gated(rows)
    write_three_way_comparison(rows)
    write_all_policy_comparison()
    print(f"wrote {OUT_COST}")
    print(f"wrote {OUT_LATENCY}")
    print(f"wrote {OUT_COMPARISON_CSV}")
    print(f"wrote {OUT_COMPARISON_SVG}")
    print(f"wrote {OUT_THREE_WAY_CSV}")
    print(f"wrote {OUT_THREE_WAY_SVG}")
    if OUT_ALL_POLICY_CSV.exists():
        print(f"wrote {OUT_ALL_POLICY_CSV}")
    if OUT_ALL_POLICY_SVG.exists():
        print(f"wrote {OUT_ALL_POLICY_SVG}")


if __name__ == "__main__":
    main()
