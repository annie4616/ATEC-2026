"""Compile the whole grasp project into one summary: method generations per depth
(v4 noise-free -> rgbl perception-noise -> real full-noise) plus the CAN-focused
inference-time heuristic ablation. Writes grasp_rl/GRASP_SUMMARY.md.

All numbers are re-parsed from the round reports and eval logs on disk, so the script
can be re-run at any time to refresh the summary.
"""

import re
from pathlib import Path

RW = Path(__file__).resolve().parents[2]
D = RW / "grasp_rl" / "detached"
OUT = RW / "grasp_rl" / "GRASP_SUMMARY.md"
DEPTHS = ["0.2", "0.21", "0.22", "0.23", "0.24", "0.25", "0.3"]
HVARS = ["none", "prior", "ema", "open", "reopen", "shield", "combo"]


def rate(path: Path) -> str:
    try:
        m = re.search(r"grasp successes \((\d+\.?\d*)%\)", path.read_text())
        return m.group(1) if m else "—"
    except OSError:
        return "—"


def report_after(report: Path, depth: str) -> str:
    """Success % of the [after] line in a depth section of a FINAL_REPORT_* file."""
    try:
        txt = report.read_text()
    except OSError:
        return "—"
    sec = re.search(rf"--- depth {re.escape(depth)}\n(.*?)(?=--- depth |\Z)", txt, re.S)
    if not sec:
        return "—"
    m = re.search(r"\[after\].*?\((\d+\.?\d*)%\)", sec.group(1))
    return m.group(1) if m else "—"


def main():
    lines = ["# B2+Piper 파지 프로젝트 종합 결과", ""]

    lines += [
        "## 방법 세대별 성공률 (깊이별, 혼합 물체 분포 검증)",
        "",
        "| crouch 깊이 | v4 무노이즈 | rgbl 인식노이즈 승자 | real 전체노이즈 승자 |",
        "|---|---|---|---|",
    ]
    v4_new = D / "FINAL_REPORT_NEWDEPTHS.txt"
    v4_old = D / "FINAL_REPORT_V4.txt"
    for d in DEPTHS:
        v4 = report_after(v4_new if d in ("0.21", "0.23", "0.25") else v4_old, d)
        lines.append(
            f"| {d} | {v4}% | {report_after(D / 'FINAL_REPORT_RGBL.txt', d)}% "
            f"| {report_after(D / 'FINAL_REPORT_REAL.txt', d)}% |"
        )
    lines += ["", "(각 열은 해당 라운드의 자기 노이즈 조건에서 측정 — 열끼리 절대비교가 아니라",
              "각 배포 가정에서의 도달 성능임. rgbl/real은 체크포인트 사다리 선택 승자.)", ""]

    lines += [
        "## CAN(⌀66×115) heuristic 애블레이션 — real 방법 기준, 32env×400step",
        "",
        "| 깊이 | " + " | ".join(HVARS) + " | 최적 |",
        "|---|" + "---|" * (len(HVARS) + 1),
    ]
    best_re = {}
    for gpu_log in (D / "heur5.log", D / "heur4.log"):
        try:
            for m in re.finditer(r"HCAN_BEST method=real depth=([\d.]+) var=(\S+) rate=([\d.]+)%", gpu_log.read_text()):
                best_re[m.group(1)] = f"{m.group(2)} ({m.group(3)}%)"
        except OSError:
            pass
    for d in DEPTHS:
        row = [rate(D / f"hcan_real_{d}_{v}.log") for v in HVARS]
        lines.append(f"| {d} | " + " | ".join(f"{r}%" if r != "—" else "—" for r in row)
                     + f" | {best_re.get(d, '—')} |")

    lines += [
        "",
        "## CAN — 방법별 기준선/개선치",
        "",
        "| 깊이 | v4 none | v4 shield | rgbl none | rgbl+승자변형 | real none | real 최적 |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in DEPTHS:
        rk_best = "—"
        for v in HVARS[1:]:
            r = rate(D / f"hcan_rgbl_{d}_{v}.log")
            if r != "—":
                rk_best = f"{r}% ({v})"
        best = best_re.get(d, "—")
        lines.append(
            f"| {d} | {rate(D / f'hcan_v4_{d}_none.log')}% | {rate(D / f'hcan_v4_{d}_shield.log')}% "
            f"| {rate(D / f'hcan_rgbl_{d}_none.log')}% | {rk_best} "
            f"| {rate(D / f'hcan_real_{d}_none.log')}% | {best} |"
        )

    lines += ["", "## 산출물 위치",
              "- 각 라운드 리포트: `grasp_rl/detached/FINAL_REPORT_{V4,NEWDEPTHS,RGBL,REAL}.txt`",
              "- CAN heuristic 영상: `grasp_rl/videos/heur_d<깊이>_can.mp4` (real 방법 최적 변형)",
              "- 라운드별 영상: `videos/d*`(무노이즈), `videos/rgbl_d*`, `videos/real_d*`", ""]

    OUT.write_text("\n".join(lines))
    print(f"SUMMARY_WRITTEN {OUT}")


if __name__ == "__main__":
    main()
