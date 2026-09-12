"""LaTeX MSE tables for the Monte Carlo study.

Helpers used by MC_simulations.ipynb:

merge_grf_dml_gente_with_ganite
    Combines the GRF/DML/GenTE results table with the GANITE/GenTE table on
    (effect, n_samples, n_features).  GenTE is taken from the first table; the
    GenTE column of the GANITE table is a second, independent run of the same
    estimator and is dropped.

mse_table_to_latex
    Renders the merged table: one block per effect type, rows indexed by sample
    size, columns grouped by method and covariate dimension, with the mean MSE
    above the standard deviation in parentheses.  The final columns report the
    percent reduction in MSE of GenTE relative to each competitor, averaged
    over the covariate dimensions in that row.

gente_sep_joint_to_latex
    Renders the separate-versus-joint GenTE comparison table
    (``df_table_ganite_gen``): same layout, two method groups only.
"""

import pandas as pd

KEYS = ["effect", "n_samples", "n_features"]

EFFECT_ORDER = ["linear", "non-linear", "interaction", "non-linear interaction"]

EFFECT_LABELS = {
    "linear": "Linear",
    "non-linear": "Non-linear",
    "interaction": "Interaction",
    "non-linear interaction": "Non-linear interaction",
}

# (column stem in the data frame, header label in the table)
METHODS = [
    ("GRF", "GRF"),
    ("DML", "DML-XGB"),
    ("GANITE", "GANITE"),
    ("GenTE", "GenTE"),
]

GAIN_LABELS = {"GRF": "vs GRF", "DML": "vs DML", "GANITE": "vs GANITE"}


CAPTION = (
    "Mean squared error (MSE) of the generalised random forest (GRF), double "
    "machine learning based on XGBoost (DML-XGB), GANITE, and GenTE for "
    "different effect types, sample sizes $S$, and covariate dimensions $p$. "
    "Each cell reports the mean MSE with standard deviation in parentheses "
    "based on {n_mc} Monte Carlo experiments. The last three columns report "
    "the average percent gain of GenTE relative to (i) GRF, (ii) DML-XGB, and "
    "(iii) GANITE, averaged over $p\\in\\{{{p_set}\\}}$ in each row."
)


def _normalise_ganite(df):
    """Accept 'GanITE'/'Ganite'/'GANITE' as the same stem."""
    ren = {c: c.replace("GanITE", "GANITE").replace("Ganite", "GANITE")
           for c in df.columns if c.lower().startswith("ganite")}
    return df.rename(columns=ren)


def merge_grf_dml_gente_with_ganite(df_grf_dml_gente, df_ganite_gen):
    """Merge the two results tables on (effect, n_samples, n_features)."""
    df_ganite_gen = _normalise_ganite(df_ganite_gen)

    left_cols = KEYS + [
        c for c in df_grf_dml_gente.columns
        if c.startswith(("GRF MSE", "DML MSE", "GenTE MSE"))
    ]
    right_cols = KEYS + [
        c for c in df_ganite_gen.columns if c.startswith("GANITE MSE")
    ]
    if len(right_cols) == len(KEYS):
        raise KeyError(
            "no GANITE MSE columns found; got "
            f"{list(df_ganite_gen.columns)}"
        )

    merged = df_grf_dml_gente[left_cols].merge(
        df_ganite_gen[right_cols], on=KEYS, how="outer"
    )
    merged["effect"] = pd.Categorical(
        merged["effect"], categories=EFFECT_ORDER, ordered=True
    )
    return merged.sort_values(KEYS).reset_index(drop=True)


def _fmt(value, nd=3):
    return "---" if pd.isna(value) else f"{value:.{nd}f}"


def _percent_gain(block, stem, p_values):
    """Mean over p of the percent reduction in MSE of GenTE relative to stem."""
    gains = []
    for p in p_values:
        row = block[block["n_features"] == p]
        if row.empty:
            continue
        gen = row["GenTE MSE"].iloc[0]
        comp = row[f"{stem} MSE"].iloc[0]
        if pd.isna(gen) or pd.isna(comp) or comp == 0:
            continue
        gains.append(100.0 * (1.0 - gen / comp))
    if not gains:
        return "---"
    return f"{sum(gains) / len(gains):.1f}\\%"


def mse_table_to_latex(
    df,
    outpath=None,
    caption=None,
    label="tab_mse",
    n_mc=100,
    p_values=None,
    n_values=None,
    effect_order=None,
):
    """Render the merged results table as a LaTeX table.

    Parameters
    ----------
    df : merged data frame from ``merge_grf_dml_gente_with_ganite``.
    outpath : if given, the table is also written to this path.
    caption, label : table caption and cross-reference label.
    n_mc : number of Monte Carlo replications, reported in the caption.
    p_values, n_values : column and row orderings; inferred from ``df`` if None.
    effect_order : effect blocks to render, in order; defaults to those present.

    Returns
    -------
    The LaTeX source as a string.
    """
    if p_values is None:
        p_values = sorted(df["n_features"].dropna().unique())
    if n_values is None:
        n_values = sorted(df["n_samples"].dropna().unique())
    if effect_order is None:
        present = set(df["effect"].dropna().unique())
        effect_order = [e for e in EFFECT_ORDER if e in present]
    if caption is None:
        caption = CAPTION.format(
            n_mc=n_mc, p_set=",".join(str(int(p)) for p in p_values)
        )

    comparators = [stem for stem, _ in METHODS if stem != "GenTE"]
    n_cols = 1 + len(METHODS) * len(p_values) + len(comparators)

    out = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l" + "r" * (n_cols - 1) + "}",
        "\\toprule",
    ]

    # Method group headers, then the per-p sub-headers.
    groups = [f"\\multicolumn{{{len(p_values)}}}{{c}}{{{lab}}}" for _, lab in METHODS]
    groups.append(f"\\multicolumn{{{len(comparators)}}}{{c}}{{Avg.\\ \\% gain (GenTE)}}")
    out.append("& " + " & ".join(groups) + " \\\\")

    rules, start = [], 2
    for width in [len(p_values)] * len(METHODS) + [len(comparators)]:
        rules.append(f"\\cmidrule(lr){{{start}-{start + width - 1}}}")
        start += width
    out.append("".join(rules))

    sub = ["Sample size"]
    for _ in METHODS:
        sub.append("& " + " & ".join(f"$p{{=}}{int(p)}$" for p in p_values))
    sub.append("& " + " & ".join(GAIN_LABELS[stem] for stem in comparators) + " \\\\")
    out.extend(sub)
    out.append("\\midrule")

    for i, effect in enumerate(effect_order):
        block = df[df["effect"] == effect]
        out.append(f"\\multicolumn{{{n_cols}}}{{l}}{{{EFFECT_LABELS[effect]}}} \\\\")
        out.append("\\midrule")

        for n in n_values:
            rows = block[block["n_samples"] == n]
            if rows.empty:
                continue

            means, stds = [], []
            for stem, _ in METHODS:
                for p in p_values:
                    cell = rows[rows["n_features"] == p]
                    if cell.empty:
                        means.append("---")
                        stds.append("")
                        continue
                    means.append(_fmt(cell[f"{stem} MSE"].iloc[0]))
                    stds.append(f"({_fmt(cell[f'{stem} MSE std'].iloc[0])})")

            gains = [_percent_gain(rows, stem, p_values) for stem in comparators]

            out.append(
                f"{int(n)} & " + " & ".join(means + gains) + " \\\\"
            )
            out.append(
                "  & " + " & ".join(stds + [""] * len(comparators)) + " \\\\"
            )

        if i < len(effect_order) - 1:
            out.append("\\addlinespace")
            out.append("\\midrule")

    out.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        f"\\label{{{label}}}",
        "\\end{table}",
    ])

    tex = "\n".join(out) + "\n"

    if outpath is not None:
        with open(outpath, "w") as fh:
            fh.write(tex)

    return tex


SEP_JOINT_METHODS = [
    ("GenTE Separate", "GenTE (separate)"),
    ("GenTE Joint Estimation", "GenTE (joint)"),
]

SEP_JOINT_CAPTION = (
    "Mean squared error of GenTE with separate treated/control quantile "
    "functions versus joint estimation, by effect type, sample size $n$, "
    "and covariate dimension $p$. Each cell is the mean MSE (standard "
    "deviation in parentheses) over {n_mc} Monte Carlo replications."
)


SEP_JOINT_METHODS = (
    ("GenTE Separate",         "GenTE (separate)"),
    ("GenTE Joint Estimation", "GenTE (joint, grid)"),
    ("GenTE Joint MC",         "GenTE (joint, MC)"),
)


def gente_sep_joint_to_latex(
    df,
    outpath=None,
    caption=None,
    label="tab:gente_sep_joint",
    n_mc=100,
    p_values=None,
    n_values=None,
    effect_order=None,
    methods=None,
):
    """Render the separate-versus-joint GenTE MSE table as LaTeX.

    Parameters
    ----------
    df : data frame with one mean column and one ``std`` column per method
        stem in ``methods``.
    outpath : if given, the table is also written to this path.
    caption, label : table caption and cross-reference label.
    n_mc : number of Monte Carlo replications, reported in the caption.
    p_values, n_values : column and row orderings; inferred from ``df`` if None.
    effect_order : effect blocks to render, in order; defaults to those present.
    methods : sequence of ``(column stem, display label)`` pairs, one block of
        ``len(p_values)`` columns each. Defaults to ``SEP_JOINT_METHODS``.
        Stems absent from ``df`` are dropped.

    Returns
    -------
    The LaTeX source as a string.
    """
    if methods is None:
        methods = SEP_JOINT_METHODS
    # drop any method whose columns are not in the frame
    methods = [(stem, lab) for stem, lab in methods
               if f"{stem} MSE" in df.columns]
    if not methods:
        raise ValueError("none of the requested method stems are in df")

    if p_values is None:
        p_values = sorted(df["n_features"].dropna().unique())
    if n_values is None:
        n_values = sorted(df["n_samples"].dropna().unique())
    if effect_order is None:
        present = set(df["effect"].dropna().unique())
        effect_order = [e for e in EFFECT_ORDER if e in present]
    if caption is None:
        caption = SEP_JOINT_CAPTION.format(n_mc=n_mc)

    n_cols = 1 + len(methods) * len(p_values)

    out = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l" + "r" * (n_cols - 1) + "}",
        "\\toprule",
    ]

    groups = [
        f"\\multicolumn{{{len(p_values)}}}{{c}}{{{lab}}}"
        for _, lab in methods
    ]
    out.append("& " + " & ".join(groups) + " \\\\")

    rules, start = [], 2
    for _ in methods:
        rules.append(f"\\cmidrule(lr){{{start}-{start + len(p_values) - 1}}}")
        start += len(p_values)
    out.append("".join(rules))

    out.append(
        "Sample size & "
        + " & ".join([f"$p{{=}}{int(p)}$" for p in p_values] * len(methods))
        + " \\\\"
    )
    out.append("\\midrule")

    for i, effect in enumerate(effect_order):
        block = df[df["effect"] == effect]
        out.append(f"\\multicolumn{{{n_cols}}}{{l}}{{{EFFECT_LABELS[effect]}}} \\\\")
        out.append("\\midrule")

        for n in n_values:
            rows = block[block["n_samples"] == n]
            if rows.empty:
                continue

            means, stds = [], []
            for stem, _ in methods:
                for p in p_values:
                    cell = rows[rows["n_features"] == p]
                    if cell.empty:
                        means.append("---")
                        stds.append("")
                        continue
                    means.append(_fmt(cell[f"{stem} MSE"].iloc[0]))
                    stds.append(f"({_fmt(cell[f'{stem} MSE std'].iloc[0])})")

            out.append(f"{int(n)} & " + " & ".join(means) + " \\\\")
            out.append("  & " + " & ".join(stds) + " \\\\")

        if i < len(effect_order) - 1:
            out.append("\\addlinespace")
            out.append("\\midrule")

    out.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
    ])

    tex = "\n".join(out) + "\n"

    if outpath is not None:
        with open(outpath, "w") as fh:
            fh.write(tex)

    return tex
    


DESIGN_ORDER = ('gaussian', 'heavy', 'skew', 'mixture')
DESIGN_LABELS = {
    'gaussian': 'Gaussian errors',
    'heavy': 'Heavy-tailed errors ($t_3$)',
    'skew': 'Skewed treated arm',
    'mixture': 'Mixture errors',
}

ROBUST_METHODS = (
    ("GenTE Joint Estimation", "GenTE (quantile, grid)"),
    ("GenTE Joint MC",         "GenTE (quantile, MC)"),
    ("GenTE Direct",           "GenTE (direct, squared loss)"),
)


def gente_quantile_direct_to_latex(
    df,
    outpath=None,
    caption=None,
    label="tab:gente_quantile_direct",
    n_mc=100,
    p_values=None,
    n_values=None,
    effect_order=None,
    methods=None,
    group_col="design",
    group_order=None,
    group_labels=None,
    baseline_stem="GenTE Direct",
    mark_baseline_wins=True,
):
    """Render a GenTE MSE table as LaTeX.

    ``group_col`` selects the column defining the row blocks: "effect" for the
    effect-form tables, "design" for the error-distribution table.

    When ``mark_baseline_wins`` is True, each ``baseline_stem`` cell is wrapped
    in ``\\bt{}`` whenever it attains a lower MSE than every other method at the
    same (group, S, p).  With ``baseline_stem="GenTE Direct"`` this marks the
    cells in which the direct specification outperforms the quantile-indexed
    one under both quadratures, matching the convention of the other tables, in
    which dark blue marks cells where the specification reported in the main
    text does not lead.
    """
    if methods is None:
        methods = ROBUST_METHODS
    methods = [(stem, lab) for stem, lab in methods
               if f"{stem} MSE" in df.columns]
    if not methods:
        raise ValueError("none of the requested method stems are in df")

    other_stems = [s for s, _ in methods if s != baseline_stem]
    if mark_baseline_wins:
        if not any(s == baseline_stem for s, _ in methods):
            raise ValueError(f"baseline_stem '{baseline_stem}' is not among methods")
        if not other_stems:
            raise ValueError("nothing to compare the baseline against")

    if group_order is None:
        group_order = DESIGN_ORDER if group_col == "design" else EFFECT_ORDER
    if group_labels is None:
        group_labels = DESIGN_LABELS if group_col == "design" else EFFECT_LABELS

    if p_values is None:
        p_values = sorted(df["n_features"].dropna().unique())
    if n_values is None:
        n_values = sorted(df["n_samples"].dropna().unique())
    if effect_order is None:
        present = set(df[group_col].dropna().unique())
        effect_order = [e for e in group_order if e in present]
    if caption is None:
        caption = SEP_JOINT_CAPTION.format(n_mc=n_mc)

    n_cols = 1 + len(methods) * len(p_values)

    out = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l" + "r" * (n_cols - 1) + "}",
        "\\toprule",
    ]

    groups = [
        f"\\multicolumn{{{len(p_values)}}}{{c}}{{{lab}}}"
        for _, lab in methods
    ]
    out.append("& " + " & ".join(groups) + " \\\\")

    rules, start = [], 2
    for _ in methods:
        rules.append(f"\\cmidrule(lr){{{start}-{start + len(p_values) - 1}}}")
        start += len(p_values)
    out.append("".join(rules))

    out.append(
        "Sample size & "
        + " & ".join([f"$p{{=}}{int(p)}$" for p in p_values] * len(methods))
        + " \\\\"
    )
    out.append("\\midrule")

    for i, grp in enumerate(effect_order):
        block = df[df[group_col] == grp]
        out.append(f"\\multicolumn{{{n_cols}}}{{l}}{{{group_labels[grp]}}} \\\\")
        out.append("\\midrule")

        for n in n_values:
            rows = block[block["n_samples"] == n]
            if rows.empty:
                continue

            # at each p, does the baseline beat every other method?
            base_wins = {}
            if mark_baseline_wins:
                for p in p_values:
                    cell = rows[rows["n_features"] == p]
                    if cell.empty:
                        base_wins[p] = False
                        continue
                    b = float(cell[f"{baseline_stem} MSE"].iloc[0])
                    base_wins[p] = all(
                        b < float(cell[f"{s} MSE"].iloc[0]) for s in other_stems
                    )

            means, stds = [], []
            for stem, _ in methods:
                for p in p_values:
                    cell = rows[rows["n_features"] == p]
                    if cell.empty:
                        means.append("---")
                        stds.append("")
                        continue
                    txt = _fmt(float(cell[f"{stem} MSE"].iloc[0]))
                    if (mark_baseline_wins and stem == baseline_stem
                            and base_wins.get(p, False)):
                        txt = f"\\bt{{{txt}}}"
                    means.append(txt)
                    stds.append(f"({_fmt(cell[f'{stem} MSE std'].iloc[0])})")

            out.append(f"{int(n)} & " + " & ".join(means) + " \\\\")
            out.append("  & " + " & ".join(stds) + " \\\\")

        if i < len(effect_order) - 1:
            out.append("\\addlinespace")
            out.append("\\midrule")

    out.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
    ])

    tex = "\n".join(out) + "\n"

    if outpath is not None:
        with open(outpath, "w") as fh:
            fh.write(tex)

    return tex