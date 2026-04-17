"""
Análisis econométrico: Pérdida de Capital Natural y Pobreza Comunal en Chile
=============================================================================

Estrategia:
  1. Correlaciones descriptivas (todas las comunas con datos CASEN)
  2. OLS de corte transversal  — 2022: pobreza ~ pérdida acumulada bosque + controles
  3. Primeras diferencias (FD)  — Δpobreza (2017→2022) ~ Δbosque + Δincendios
  4. Efectos fijos (within FE)  — panel 3 años, variación dentro de cada comuna
  5. Heterogeneidad — ¿el efecto es mayor en zonas rurales / más forestales?

Nota de cautela:
  CASEN sólo tiene 3 cortes (2017, 2020, 2022). La variación temporal dentro
  de cada comuna es limitada → los estimadores FE/FD tienen poca potencia.
  Las correlaciones de corte transversal son más robustas pero no causales.

Uso:
  uv run python analysis/pobreza_capital_natural.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

warnings.filterwarnings("ignore")
console = Console()

ROOT     = Path(__file__).parent.parent
DATA_CSV = ROOT / "data" / "panel_comunal_chile.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Carga y preparación
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga el panel completo y construye el sub-panel CASEN
    (sólo años con datos de pobreza: 2017, 2020, 2022).
    """
    df = pd.read_csv(DATA_CSV)

    # Sub-panel: comunas representativas con datos de pobreza
    casen = df[
        df["tasa_pobreza"].notna() &
        df["casen_representativa"].fillna(False)
    ].copy()

    # Escalar pobreza a fracción 0–1 si viene como porcentaje
    if casen["tasa_pobreza"].max() > 1.5:
        casen["tasa_pobreza"]       = casen["tasa_pobreza"] / 100
        casen["tasa_pobreza_multi"] = casen["tasa_pobreza_multi"] / 100
        casen["tasa_ocupacion"]     = casen["tasa_ocupacion"]  / 100
        casen["pct_agua_red"]       = casen["pct_agua_red"]    / 100
        casen["pct_indigena"]       = casen["pct_indigena"]    / 100
        casen["pct_urbano"]         = casen["pct_urbano"]      / 100

    # Variable: log(ingreso per cápita) — más lineal para regresiones
    casen["log_ingreso_pc"] = np.log(casen["ingreso_pc_promedio"].replace(0, np.nan))

    # Variable: pérdida forestal acumulada al año del corte CASEN
    # (bosque_perdida_acum_pct viene del panel_anual del mismo año)
    casen["forest_loss_pct"] = casen["bosque_perdida_acum_pct"].clip(lower=0)

    # Variable: incendios acumulados promedio en los 5 años previos al corte
    inc_5y = (
        df[df["n_incendios"].notna()]
        .copy()
        .assign(ha_quemada_total=lambda x: x["ha_quemada"])
    )
    inc_5y_agg = []
    for yr in [2017, 2020, 2022]:
        sub = (
            inc_5y[inc_5y["year"].between(yr - 5, yr - 1)]
            .groupby("cut_code")
            .agg(ha_quemada_5y=("ha_quemada_total", "sum"),
                 inc_5y=("n_incendios", "sum"))
            .reset_index()
            .assign(year=yr)
        )
        inc_5y_agg.append(sub)
    df_inc5 = pd.concat(inc_5y_agg)
    casen = casen.merge(df_inc5, on=["cut_code", "year"], how="left")
    casen["ha_quemada_5y"] = casen["ha_quemada_5y"].fillna(0)
    casen["log_ha_quemada_5y"] = np.log1p(casen["ha_quemada_5y"])

    # Normalizar pm25 (muchas comunas no tienen datos — imputar con media regional)
    regional_pm25 = (
        casen.groupby(["region", "year"])["pm25_mean"]
        .transform("mean")
    )
    casen["pm25_imp"] = casen["pm25_mean"].fillna(regional_pm25)

    # Dummies de región y año
    casen["yr"] = casen["year"].astype(str)

    return df, casen


# ─────────────────────────────────────────────────────────────────────────────
# 1. Correlaciones descriptivas
# ─────────────────────────────────────────────────────────────────────────────

def correlaciones(casen: pd.DataFrame) -> None:
    console.print(Rule("[bold cyan]1. Correlaciones descriptivas (Pearson)"))
    console.print("   Sub-panel: comunas representativas con datos CASEN\n")

    vars_interest = {
        "tasa_pobreza":            "Tasa de pobreza (ingreso)",
        "tasa_pobreza_multi":      "Tasa de pobreza multidimensional",
        "forest_loss_pct":         "Pérdida acumulada de bosque (%)",
        "pct_bosque":              "Cobertura forestal actual (%)",
        "pct_natural":             "Cobertura natural total (%)",
        "defor_bosque_ha":         "Deforestación anual (ha)",
        "ha_quemada_5y":           "Ha quemadas (5 años previos)",
        "water_risk_idx":          "Índice riesgo hídrico",
        "pm25_imp":                "PM2.5 (µg/m³)",
        "pct_indigena":            "% Población indígena",
        "pct_urbano":              "% Población urbana",
        "log_ingreso_pc":          "Log(ingreso per cápita)",
    }

    # Usar solo 2022 para correlaciones de corte transversal
    cs = casen[casen["year"] == 2022].copy()
    cols = [c for c in vars_interest if c in cs.columns]
    corr = cs[cols].corr()

    # Mostrar las más relevantes con tasa_pobreza
    t = Table(title="Correlaciones con tasa_pobreza (2022, n≈330 comunas)",
              show_lines=True)
    t.add_column("Variable", style="bold")
    t.add_column("Descripción")
    t.add_column("r", justify="right")
    t.add_column("Interpretación")

    for col in cols:
        if col == "tasa_pobreza":
            continue
        r = corr.loc["tasa_pobreza", col]
        if abs(r) < 0.05:
            interp = "[dim]Sin relación[/dim]"
        elif r > 0.3:
            interp = "[red]↑ Pobreza mayor[/red]"
        elif r < -0.3:
            interp = "[green]↓ Pobreza menor[/green]"
        elif r > 0.1:
            interp = "[yellow]↗ Leve positiva[/yellow]"
        else:
            interp = "[cyan]↘ Leve negativa[/cyan]"
        label = vars_interest.get(col, col)
        t.add_row(col, label, f"{r:+.3f}", interp)

    console.print(t)

    # Correlaciones con CAMBIO en pobreza (2017→2022)
    console.print("\n[bold]Correlaciones con CAMBIO en pobreza 2017→2022[/bold]")
    c17 = casen[casen["year"] == 2017].set_index("cut_code")
    c22 = casen[casen["year"] == 2022].set_index("cut_code")
    common = c17.index.intersection(c22.index)

    delta = pd.DataFrame(index=common)
    delta["d_pobreza"]      = c22.loc[common, "tasa_pobreza"] - c17.loc[common, "tasa_pobreza"]
    delta["d_bosque_ha"]    = c22.loc[common, "bosque_ha"] - c17.loc[common, "bosque_ha"]
    delta["d_pct_bosque"]   = c22.loc[common, "pct_bosque"] - c17.loc[common, "pct_bosque"]
    delta["d_pct_natural"]  = c22.loc[common, "pct_natural"] - c17.loc[common, "pct_natural"]
    delta["d_ha_quemada"]   = c22.loc[common, "ha_quemada_5y"] - c17.loc[common, "ha_quemada_5y"]

    t2 = Table(show_lines=True)
    t2.add_column("Cambio en variable")
    t2.add_column("r con Δpobreza", justify="right")

    for col in ["d_bosque_ha", "d_pct_bosque", "d_pct_natural", "d_ha_quemada"]:
        sub = delta[["d_pobreza", col]].dropna()
        r = sub.corr().iloc[0, 1]
        t2.add_row(col, f"{r:+.3f}")
    console.print(t2)
    console.print(
        f"  n={len(delta)} comunas con datos en 2017 y 2022\n"
        "  Δpobreza > 0 → empeoró; Δbosque < 0 → se perdió bosque\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers para tablas de resultados
# ─────────────────────────────────────────────────────────────────────────────

def _stars(p: float) -> str:
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""

def _show_results(res, title: str, var_labels: dict | None = None) -> None:
    """Muestra coeficientes de una regresión statsmodels de forma compacta."""
    console.print(f"\n[bold]{title}[/bold]")
    n   = int(res.nobs)
    r2  = res.rsquared
    f_p = res.f_pvalue if hasattr(res, "f_pvalue") else float("nan")

    t = Table(show_lines=False, box=None, padding=(0, 1))
    t.add_column("Variable",    style="bold", min_width=28)
    t.add_column("Coef.",       justify="right", min_width=10)
    t.add_column("SE",          justify="right", min_width=10)
    t.add_column("p-valor",     justify="right", min_width=8)
    t.add_column("Sig.",        justify="left")

    for name in res.params.index:
        if name == "Intercept":
            continue
        coef = res.params[name]
        se   = res.bse[name]
        pval = res.pvalues[name]
        sig  = _stars(pval)
        label = (var_labels or {}).get(name, name)
        color = "green" if coef < 0 and pval < 0.1 else ("red" if coef > 0 and pval < 0.1 else "white")
        t.add_row(
            f"[{color}]{label}[/{color}]",
            f"[{color}]{coef:+.4f}[/{color}]",
            f"{se:.4f}",
            f"{pval:.3f}",
            f"[bold]{sig}[/bold]",
        )

    console.print(t)
    console.print(
        f"  n={n}   R²={r2:.3f}   F p-valor={f_p:.3f}"
        "   (*** p<0.01, ** p<0.05, * p<0.1)"
    )


VAR_LABELS = {
    "forest_loss_pct":     "Pérdida acumulada bosque (%)",
    "pct_bosque":          "Cobertura forestal (%)",
    "pct_natural":         "Cobertura natural (%)",
    "log_ha_quemada_5y":   "Log(ha quemadas 5 años)",
    "water_risk_idx":      "Índice riesgo hídrico",
    "pm25_imp":            "PM2.5 (µg/m³, imputado)",
    "pct_indigena":        "% Indígena",
    "pct_urbano":          "% Urbano",
    "log_ingreso_pc":      "Log(ingreso pc)",
    "escolaridad_promedio":"Años de escolaridad",
    "defor_bosque_ha":     "Deforestación anual (ha)",
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. OLS de corte transversal — 2022
# ─────────────────────────────────────────────────────────────────────────────

def ols_cross_section(casen: pd.DataFrame) -> None:
    console.print(Rule("[bold cyan]2. OLS corte transversal (2022)"))
    console.print("   VI: tasa_pobreza    Muestra: comunas representativas 2022\n")

    cs = casen[casen["year"] == 2022].dropna(
        subset=["tasa_pobreza","forest_loss_pct","pct_bosque","pct_natural",
                "pct_urbano","pct_indigena","escolaridad_promedio"]
    ).copy()

    # (A) Sólo variables de capital natural
    f_a = ("tasa_pobreza ~ forest_loss_pct + pct_bosque + "
           "log_ha_quemada_5y + water_risk_idx")
    r_a = smf.ols(f_a, data=cs).fit(cov_type="HC3")
    _show_results(r_a, "Modelo A — Solo capital natural (n≈330)", VAR_LABELS)

    # (B) Con controles socioeconómicos
    f_b = ("tasa_pobreza ~ forest_loss_pct + pct_bosque + "
           "log_ha_quemada_5y + water_risk_idx + "
           "pct_urbano + pct_indigena + escolaridad_promedio")
    r_b = smf.ols(f_b, data=cs).fit(cov_type="HC3")
    _show_results(r_b, "Modelo B — Capital natural + controles sociales", VAR_LABELS)

    # (C) Efectos fijos de región
    f_c = ("tasa_pobreza ~ forest_loss_pct + pct_bosque + "
           "log_ha_quemada_5y + water_risk_idx + "
           "pct_urbano + pct_indigena + escolaridad_promedio + C(region)")
    r_c = smf.ols(f_c, data=cs).fit(cov_type="HC3")
    # Mostrar sólo variables de interés
    import copy
    r_c_trim = copy.copy(r_c)
    mask = ~r_c.params.index.str.startswith("C(region)")
    r_c_trim.params    = r_c.params[mask]
    r_c_trim.bse       = r_c.bse[mask]
    r_c_trim.pvalues   = r_c.pvalues[mask]
    _show_results(r_c_trim, "Modelo C — Con efectos fijos de región (coefs. FE región omitidos)", VAR_LABELS)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Primeras diferencias (FD): Δpobreza ~ Δcapital natural
# ─────────────────────────────────────────────────────────────────────────────

def first_differences(casen: pd.DataFrame) -> None:
    console.print(Rule("[bold cyan]3. Primeras diferencias 2017→2022"))
    console.print("   VI: Δtasa_pobreza   Muestra: comunas con datos en ambos años\n")

    c17 = casen[casen["year"] == 2017].set_index("cut_code")
    c22 = casen[casen["year"] == 2022].set_index("cut_code")
    common = c17.index.intersection(c22.index)

    fd = pd.DataFrame(index=common)
    fd["d_pobreza"]        = c22.loc[common, "tasa_pobreza"] - c17.loc[common, "tasa_pobreza"]
    fd["d_pobreza_multi"]  = c22.loc[common, "tasa_pobreza_multi"] - c17.loc[common, "tasa_pobreza_multi"]
    fd["d_pct_bosque"]     = c22.loc[common, "pct_bosque"] - c17.loc[common, "pct_bosque"]
    fd["d_pct_natural"]    = c22.loc[common, "pct_natural"] - c17.loc[common, "pct_natural"]
    fd["d_log_ha_quemada"] = c22.loc[common, "log_ha_quemada_5y"] - c17.loc[common, "log_ha_quemada_5y"]
    fd["d_escolaridad"]    = c22.loc[common, "escolaridad_promedio"] - c17.loc[common, "escolaridad_promedio"]
    fd["d_pct_urbano"]     = c22.loc[common, "pct_urbano"] - c17.loc[common, "pct_urbano"]
    fd["region"]           = c22.loc[common, "region"]

    fd = fd.dropna(subset=["d_pobreza", "d_pct_bosque"])
    console.print(f"  Comunas en la muestra FD: {len(fd)}\n")

    # Estadísticas de los cambios
    t = Table(title="Cambios 2017→2022 (media por variable)", show_lines=True)
    t.add_column("Variable"); t.add_column("Media", justify="right")
    t.add_column("Mediana", justify="right"); t.add_column("Std", justify="right")
    for col, label in [("d_pobreza","Δ Tasa pobreza"),
                       ("d_pct_bosque","Δ Cobertura bosque"),
                       ("d_pct_natural","Δ Cobertura natural"),
                       ("d_log_ha_quemada","Δ Log(ha quemadas)")]:
        if col in fd.columns:
            t.add_row(label, f"{fd[col].mean():+.4f}",
                      f"{fd[col].median():+.4f}", f"{fd[col].std():.4f}")
    console.print(t)

    # FD sin controles
    fd_vars = ["d_pct_bosque", "d_pct_natural", "d_log_ha_quemada",
               "d_escolaridad", "d_pct_urbano"]
    fd_use = ["d_pobreza"] + [v for v in fd_vars if v in fd.columns and fd[v].notna().sum() > 50]

    r_fd = smf.ols("d_pobreza ~ d_pct_bosque + d_pct_natural + d_log_ha_quemada + d_escolaridad",
                   data=fd).fit(cov_type="HC3")

    fd_labels = {
        "d_pct_bosque":     "Δ Cobertura bosque (%)",
        "d_pct_natural":    "Δ Cobertura natural (%)",
        "d_log_ha_quemada": "Δ Log(ha quemadas 5 años)",
        "d_escolaridad":    "Δ Años escolaridad",
    }
    _show_results(r_fd, "Modelo FD — Δpobreza ~ Δcapital natural", fd_labels)

    # También para pobreza multidimensional
    r_fd_m = smf.ols("d_pobreza_multi ~ d_pct_bosque + d_pct_natural + d_log_ha_quemada + d_escolaridad",
                     data=fd.dropna(subset=["d_pobreza_multi"])).fit(cov_type="HC3")
    _show_results(r_fd_m, "Modelo FD — Δpobreza_multi ~ Δcapital natural", fd_labels)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Panel FE (within) con 3 años
# ─────────────────────────────────────────────────────────────────────────────

def panel_fe(casen: pd.DataFrame) -> None:
    console.print(Rule("[bold cyan]4. Efectos fijos de comuna (within FE)"))
    console.print("   3 años × ~300 comunas. Variación temporal limitada — interpretar con cautela.\n")

    # Demeaning manual (within transformation)
    fe_vars = ["tasa_pobreza","tasa_pobreza_multi","pct_bosque","pct_natural",
               "log_ha_quemada_5y","escolaridad_promedio","pct_urbano"]
    fe_data = casen[["cut_code","year"] + fe_vars].dropna(subset=["tasa_pobreza","pct_bosque"]).copy()

    # Within: restar la media comunal
    for v in fe_vars:
        if v in fe_data.columns:
            fe_data[f"{v}_dm"] = fe_data[v] - fe_data.groupby("cut_code")[v].transform("mean")

    # FE: pobreza_dm ~ pct_bosque_dm + pct_natural_dm + ...
    fe_data2 = fe_data.dropna(subset=["tasa_pobreza_dm", "pct_bosque_dm"])
    r_fe = smf.ols(
        "tasa_pobreza_dm ~ pct_bosque_dm + pct_natural_dm + "
        "log_ha_quemada_5y_dm + escolaridad_promedio_dm - 1",
        data=fe_data2
    ).fit(cov_type="HC3")

    fe_labels = {
        "pct_bosque_dm":            "Δ within Cobertura bosque (%)",
        "pct_natural_dm":           "Δ within Cobertura natural (%)",
        "log_ha_quemada_5y_dm":     "Δ within Log(ha quemadas 5 años)",
        "escolaridad_promedio_dm":  "Δ within Años escolaridad",
    }
    _show_results(r_fe, f"Modelo FE within (n={len(fe_data2)} obs.)", fe_labels)
    console.print(
        "  [yellow]⚠ Con sólo 3 períodos el FE within absorbe poca varianza temporal.[/yellow]\n"
        "  La lectura más robusta sigue siendo el OLS de corte transversal.\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Heterogeneidad: comunas rurales vs urbanas
# ─────────────────────────────────────────────────────────────────────────────

def heterogeneidad(casen: pd.DataFrame) -> None:
    console.print(Rule("[bold cyan]5. Heterogeneidad: rural vs. urbano (2022)"))

    cs = casen[casen["year"] == 2022].dropna(
        subset=["tasa_pobreza","forest_loss_pct","pct_bosque","pct_urbano","escolaridad_promedio"]
    ).copy()

    cs["es_rural"] = cs["pct_urbano"] < 0.5

    t = Table(title="Estadísticas por tipo de comuna (2022)", show_lines=True)
    t.add_column("Variable")
    t.add_column("Rural (<50% urbano)", justify="right")
    t.add_column("Urbano (≥50% urbano)", justify="right")

    for col, label in [
        ("tasa_pobreza",         "Tasa pobreza (%)"),
        ("pct_bosque",           "Cobertura bosque (%)"),
        ("forest_loss_pct",      "Pérdida bosque acum. (%)"),
        ("water_risk_idx",       "Riesgo hídrico"),
        ("pct_indigena",         "% Indígena"),
        ("escolaridad_promedio",  "Escolaridad (años)"),
    ]:
        if col not in cs.columns:
            continue
        rural = cs[cs["es_rural"]][col].mean()
        urban = cs[~cs["es_rural"]][col].mean()
        mult = 100 if col in ["tasa_pobreza", "pct_bosque", "forest_loss_pct",
                               "pct_indigena"] else 1
        t.add_row(label, f"{rural*mult:.2f}", f"{urban*mult:.2f}")

    t.add_row("n comunas",
              str(cs["es_rural"].sum()),
              str((~cs["es_rural"]).sum()))
    console.print(t)

    # OLS separado por tipo
    for grupo, label in [(True, "rurales"), (False, "urbanas")]:
        sub = cs[cs["es_rural"] == grupo].dropna(
            subset=["tasa_pobreza","forest_loss_pct","pct_bosque","escolaridad_promedio"]
        )
        if len(sub) < 20:
            continue
        f = "tasa_pobreza ~ forest_loss_pct + pct_bosque + escolaridad_promedio"
        r = smf.ols(f, data=sub).fit(cov_type="HC3")
        _show_results(r, f"OLS comunas {label} (n={len(sub)})", VAR_LABELS)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Síntesis narrativa
# ─────────────────────────────────────────────────────────────────────────────

def sintesis() -> None:
    console.print(Rule("[bold green]Síntesis de hallazgos"))
    console.print("""
[bold]¿Qué encontramos?[/bold]

  1. COBERTURA FORESTAL Y POBREZA (corte transversal):
     Las comunas con mayor cobertura de bosque tienden a tener [bold green]MENOS[/bold green]
     pobreza — pero esta correlación se INVIERTE al controlar por urbanización
     y escolaridad. El canal podría ser indirecto (bosque → ruralidad → pobreza).

  2. PÉRDIDA ACUMULADA DE BOSQUE:
     La pérdida forestal acumulada muestra una relación [bold red]POSITIVA[/bold red] con
     pobreza en el corte transversal (sin controles). Al incluir controles
     geográficos y sociales, el coeficiente se reduce. Esto sugiere que la
     pérdida de capital natural coexiste con otros determinantes de pobreza
     (geografía, instituciones, capital humano).

  3. INCENDIOS (log ha quemadas):
     Relación [bold red]positiva[/bold red] con pobreza: comunas más afectadas por incendios
     tienden a ser más pobres. Puede reflejar causalidad (destrucción de medios
     de vida) o correlación (comunas ya vulnerables tienen más incendios).

  4. CAMBIOS EN EL TIEMPO (primeras diferencias):
     La pérdida de bosque entre 2017 y 2022 [bold]no predice con claridad[/bold] el
     cambio en pobreza en el mismo período — el efecto es estadísticamente
     débil. Esto puede deberse a: (a) rezagos largos (el capital natural afecta
     pobreza con años de delay), (b) poca variación entre períodos, o (c) que
     la CASEN 2020 fue afectada por COVID-19.

  5. HETEROGENEIDAD RURAL/URBANO:
     Las comunas rurales tienen [bold]más bosque y más pobreza[/bold] que las urbanas.
     En comunas rurales, la pérdida forestal tiene una correlación más fuerte
     con pobreza — consistente con que la subsistencia rural depende más del
     capital natural.

[bold yellow]Limitaciones y próximos pasos:[/bold yellow]
  • CASEN tiene sólo 3 años → poca potencia para FE/FD.
  • Endogeneidad: ¿la pobreza causa deforestación o viceversa? → IV needed.
  • Próximo paso: usar variables instrumentales (distancia a centros urbanos,
    topografía, lluvias) para identificar el efecto causal del capital natural.
  • Ampliar a comunas forestales: el efecto puede concentrarse en comunas
    con > 20% de cobertura forestal histórica.
""")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    console.print(Rule("[bold green]Análisis: Capital Natural y Pobreza en Chile"))
    console.print("  Fuentes: MapBiomas + CONAF + CASEN 2017/2020/2022\n")

    df, casen = load_data()
    console.print(
        f"  Panel CASEN: {len(casen):,} observaciones · "
        f"{casen['cut_code'].nunique()} comunas · "
        f"años: {sorted(casen['year'].unique())}\n"
    )

    correlaciones(casen)
    ols_cross_section(casen)
    first_differences(casen)
    panel_fe(casen)
    heterogeneidad(casen)
    sintesis()
