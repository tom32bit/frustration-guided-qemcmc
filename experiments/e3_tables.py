"""Markdown tables for the E3 reports, generated from results/e3_tier{A,B}_summary.json (no transcription by hand).

Usage: e3_tables.py A|B
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
NS = list(range(7, 13))


def ci(x):
    return f"[{x[0]:+.2f}, {x[1]:+.2f}]"


def ep_row(name, e, at="mean_r_at_n12"):
    if not e:
        return f"| {name} | — | — | — |"
    return (f"| {name} | {e['slope']:+.3f} {ci(e['slope_ci95'])} | {e[at]:+.2f} {ci(e[at + '_ci95'])} | "
            f"{'**pass**' if e['passes'] else 'fail'} |")


def tier_a():
    S = json.load(open(os.path.join(RES, "e3_tierA_summary.json")))
    M = S["modalities"]
    out = [f"Instances: {S['instances']} (errors recorded: {S['errors']})", ""]
    out += ["### Primary endpoint A (β̃ = 1, partition space)", "",
            "| Modality | Instances | Slope of mean r per added node [95 % CI] | Mean r at n = 12 [95 % CI] | Endpoint A |",
            "|---|---|---|---|---|"]
    for m in ("FC", "SC"):
        e = M[m]["PRIMARY_endpoint_A_beta1"]
        out.append(f"| {m} | {M[m]['instances']} | {e['slope']:+.3f} {ci(e['slope_ci95'])} | "
                   f"{e['mean_r_at_n12']:+.2f} {ci(e['mean_r_at_n12_ci95'])} | {'**PASS**' if e['passes'] else 'FAIL'} |")
    out += ["", "### Mean r and win share by n (β̃ = 1)", "",
            "| | " + " | ".join(f"n = {n}" for n in NS) + " |", "|---|" + "---|" * len(NS)]
    for m in ("FC", "SC"):
        e = M[m]["PRIMARY_endpoint_A_beta1"]
        d = M[m]["descriptive"]
        out.append(f"| {m}: mean r | " + " | ".join(f"{e['mean_r_by_n'][str(n)]:+.2f}" for n in NS) + " |")
        out.append(f"| {m}: share r > 0 | " + " | ".join(f"{100 * d['win_share_by_n'][str(n)]:.0f} %" for n in NS) + " |")
        out.append(f"| {m}: instances | " + " | ".join(str(e['instances_by_n'][str(n)]) for n in NS) + " |")
    out += ["", "### Secondary endpoints (same statistics, same instances)", "",
            "| Modality | Quantity | Slope per node [95 % CI] | Value at n = 12 [95 % CI] | Both CIs > 0 |", "|---|---|---|---|---|"]
    for m in ("FC", "SC"):
        for k, name in (("secondary_full_space_beta1", "Labelled-state (full-space) gaps"),
                        ("secondary_asymptotic_variance_beta1", "log₂ asymptotic-variance ratio (best classical / quantum)"),
                        ("secondary_endpoint_A_beta0.5", "r at β̃ = 0.5"),
                        ("secondary_endpoint_A_beta2.0", "r at β̃ = 2"),
                        ("secondary_qaoa_grid_beta1", "r of QAOA-MC, grid average"),
                        ):
            out.append(ep_row(f"{m} | {name}", M[m][k]))
        out.append(ep_row(f"{m} | r of QAOA-MC, angle 0 (out of sample)", M[m]["descriptive"]["qaoa_angle0_endpoint_beta1"]))
    out += ["", "### G1b minimum-exponent rule over n = 7–12 (secondary)", "",
            "| Modality | k quantum | Best classical (k) | Δk [95 % CI] |", "|---|---|---|---|"]
    for m in ("FC", "SC"):
        g = M[m]["secondary_G1b_min_exponent_rule_7to12"]
        out.append(f"| {m} | {g['k']['quantum']:.3f} | {g['best_classical']} ({g['k_best_classical']:.3f}) | "
                   f"{g['delta_k']:+.3f} {ci(g['ci95'])} |")
    out += ["", "Per-chain exponents k (log₂ gap lost per added node):", "",
            "| Modality | " + " | ".join(sorted(M['FC']['secondary_G1b_min_exponent_rule_7to12']['k'])) + " |",
            "|---|" + "---|" * len(M['FC']['secondary_G1b_min_exponent_rule_7to12']['k'])]
    for m in ("FC", "SC"):
        k = M[m]["secondary_G1b_min_exponent_rule_7to12"]["k"]
        out.append(f"| {m} | " + " | ".join(f"{k[c]:.3f}" for c in sorted(k)) + " |")
    out += ["", "### Routing test (prospective, fresh instances, all Tier A rows)", "",
            "| Threshold | Routed: wins / total | Unrouted: wins / total | One-sided Fisher p |", "|---|---|---|---|"]
    for thr in ("0.05", "0.1"):
        t = S[f"routing_test_F>={thr}"]
        out.append(f"| F ≥ {thr} | {t['routed_wins']} / {t['routed_wins'] + t['routed_losses']} "
                   f"({100 * t['win_share_routed']:.0f} %) | {t['unrouted_wins']} / {t['unrouted_wins'] + t['unrouted_losses']} "
                   f"({100 * t['win_share_unrouted']:.0f} %) | {t['fisher_one_sided_p']:.2g} |")
    out += ["", "Win share by frustration bin:", "", "| F bin | Instances | Share r > 0 |", "|---|---|---|"]
    for b, v in S["win_share_by_F_bin"].items():
        share = "—" if v["win_share"] is None else f"{100 * v['win_share']:.0f} %"
        out.append(f"| {b} | {v['n']} | {share} |")
    out += ["", "### Which classical chain is the oracle (β̃ = 1)", ""]
    for m in ("FC", "SC"):
        oc = M[m]["descriptive"]["oracle_chain_counts_by_n"]
        chains = sorted({c for v in oc.values() for c in v})
        out += [f"{m}:", "", "| Chain | " + " | ".join(f"n = {n}" for n in NS) + " |", "|---|" + "---|" * len(NS)]
        for c in chains:
            out.append(f"| {c} | " + " | ".join(str(oc[str(n)].get(c, 0)) for n in NS) + " |")
        out.append("")
    out += ["### Mean log₂ partition-space gap by chain (β̃ = 1)", ""]
    for m in ("FC", "SC"):
        g = M[m]["descriptive"]["mean_log2_gap_by_n"]
        out += [f"{m}:", "", "| Chain | " + " | ".join(f"n = {n}" for n in NS) + " |", "|---|" + "---|" * len(NS)]
        for c in ("quantum", "qaoa_a0", "qaoa", "local", "label_swap", "uniform", "sw", "wolff", "heat_bath"):
            out.append(f"| {c} | " + " | ".join(f"{g[c][str(n)]:.2f}" for n in NS) + " |")
        out.append("")
    out += ["### By ROI kind (mean r, β̃ = 1)", "", "| Modality | Kind | " + " | ".join(f"n = {n}" for n in NS) + " |",
            "|---|---|" + "---|" * len(NS)]
    for m in ("FC", "SC"):
        for k, v in M[m]["descriptive"]["mean_r_by_kind_and_n"].items():
            out.append(f"| {m} | {k} | " + " | ".join(f"{v[str(n)]:+.2f}" for n in NS) + " |")
    out += ["", "### QAOA-MC angles on fresh instances (β̃ = 1)", "",
            "| Modality | Quantity | " + " | ".join(f"n = {n}" for n in NS) + " |", "|---|---|" + "---|" * len(NS)]
    for m in ("FC", "SC"):
        d = M[m]["descriptive"]
        out.append(f"| {m} | best angle index | " + " | ".join(str(d['qaoa_best_angle_by_n'][str(n)]) for n in NS) + " |")
        out.append(f"| {m} | rank of angle 0 (of 18) | " + " | ".join(str(d['qaoa_angle0_rank_by_n'][str(n)]) for n in NS) + " |")
        out.append(f"| {m} | mean r, best angle | " + " | ".join(f"{d['qaoa_best_angle_mean_r_by_n'][str(n)]:+.2f}" for n in NS) + " |")
        out.append(f"| {m} | mean r, angle 0 | " + " | ".join(
            f"{d['qaoa_angle0_endpoint_beta1']['mean_r_by_n'][str(n)]:+.2f}" for n in NS) + " |")
    out += ["", "### PT-2 check at n = 7–8 (six ladders, oracle-tuned per instance)", "",
            "| Modality | Rows | Share where PT-2 beats the endpoint oracle | Mean log₂(PT-2 / oracle) |", "|---|---|---|---|"]
    for m in ("FC", "SC"):
        p = M[m]["PT2_check_n7to8"]
        out.append(f"| {m} | {p['rows']} | {100 * p['share_pt2_beats_oracle']:.0f} % | {p['mean_log2_pt2_over_oracle']:+.2f} |")
    out += ["", "### Canonical ROIs (β̃ = 1)", "", "| Dataset | ROI | n | F | r |", "|---|---|---|---|---|"]
    for c in S["canonical_beta1"]:
        out.append(f"| {c['ds']} | {c['roi']} | {c['n']} | {c['F']:.3f} | {c['r']:+.2f} |")
    return "\n".join(out)


def tier_b():
    S = json.load(open(os.path.join(RES, "e3_tierB_summary.json")))
    M = S["modalities"]
    out = [f"Instances: {S['instances']}", "", "### Primary endpoint B (per-step accounting)", "",
           "| Modality | Slope of mean ρ per added node [95 % CI] | Mean ρ at n = 14 [95 % CI] | Endpoint B |", "|---|---|---|---|"]
    for m in ("FC", "SC"):
        e = M[m]["PRIMARY_endpoint_B"]
        if not e:
            out.append(f"| {m} | — | — | — |")
            continue
        out.append(f"| {m} | {e['slope']:+.3f} {ci(e['slope_ci95'])} | {e['mean_rho_at_n14']:+.2f} {ci(e['mean_rho_at_n14_ci95'])} | "
                   f"{'**PASS**' if e['passes'] else 'FAIL'} |")
    out += ["", "### Mean ρ by n", "", "| Modality | Accounting | n = 12 | n = 14 | n = 16 |", "|---|---|---|---|---|"]
    for m in ("FC", "SC"):
        for k, name in (("PRIMARY_endpoint_B", "per step (primary)"), ("sensitivity_classical_nT_steps", "classical gets n·T steps")):
            e = M[m][k]
            if e:
                out.append(f"| {m} | {name} | " + " | ".join(
                    f"{e['mean_rho_by_n'][str(n)]:+.2f} ({e['instances_by_n'][str(n)]})" if str(n) in e["mean_rho_by_n"] else "—"
                    for n in (12, 14, 16)) + " |")
    out += ["", "### Best classical chain at T (counts)", "", "| Modality | " + " | ".join(
        sorted(M["FC"]["best_classical_counts"])) + " |", "|---|" + "---|" * len(M["FC"]["best_classical_counts"])]
    for m in ("FC", "SC"):
        b = M[m]["best_classical_counts"]
        out.append(f"| {m} | " + " | ".join(str(b[c]) for c in sorted(b)) + " |")
    v = S.get("validation_n12_simulated_vs_exact")
    if v:
        out += ["", "### Validation at n = 12 against the exact asymptotic variances (same instances)", "",
                "| Pairs | Spearman | Mean simulated ρ | Mean exact log₂ variance ratio |", "|---|---|---|---|",
                f"| {v['pairs']} | {v['spearman']:+.2f} | {v['mean_simulated_rho']:+.2f} | {v['mean_exact_log2_s2_ratio']:+.2f} |"]
    a = S.get("amendment13_routing_test")
    if a:
        pw, rs = a["predicted_win_class"], a["rest"]
        fmt = lambda c: (f"{c['wins']} / {c['wins'] + c['losses']} | {c['mean_rho']:+.2f}" if c["mean_rho"] is not None
                         else f"{c['wins']} / {c['wins'] + c['losses']} | —")
        out += ["", "### Amendment 13: sealed two-factor routing predictions (n = 14, 16; fresh instances)", "",
                f"Predictions written {a['predictions_written_at']}, with {a['rows_existing_when_written']} outcome rows in existence.", "",
                "| Class | ρ > 0 / instances | Mean ρ |", "|---|---|---|",
                f"| Predicted win (F ≥ 0.10 and βD ≥ 1) | {fmt(pw)} |", f"| Rest | {fmt(rs)} |", "",
                f"One-sided Fisher exact p = {a['fisher_one_sided_p']:.3g}; classification accuracy {100 * a['accuracy']:.0f} %."]
    return "\n".join(out)


if __name__ == "__main__":
    print(tier_a() if sys.argv[1:] == ["A"] else tier_b())
