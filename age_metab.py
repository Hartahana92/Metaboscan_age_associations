import io
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

st.set_page_config(page_title="Метаболит vs Возраст", layout="wide")
st.title("Зависимость концентрации метаболита от возраста и агрегированные тренды")

# ---------- загрузка данных ----------
uploaded = st.file_uploader("Загрузите файл CSV или XLSX", type=["csv", "xlsx"])
if uploaded is None:
    st.info("Ожидаю файл с данными. Пример: колонки Age, Sex, C2, C3, ...")
    st.stop()

with st.expander("Параметры импорта"):
    filetype = "xlsx" if uploaded.name.lower().endswith("xlsx") else "csv"
    sep = st.text_input("Разделитель (CSV)", value=",")
    sheet = st.text_input("Имя листа (XLSX, пусто=первый)", value="")
    decimal = st.text_input("Десятичный разделитель", value=".")
    thousands = st.text_input("Разделитель тысяч (опц.)", value="")

try:
    if filetype == "csv":
        df = pd.read_csv(uploaded, sep=sep or ",", decimal=decimal or ".", thousands=thousands or None)
    else:
        xls = pd.ExcelFile(uploaded)
        sheet_to_use = sheet if (sheet and sheet in xls.sheet_names) else xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet_to_use)
except Exception as e:
    st.error(f"Ошибка чтения файла: {e}")
    st.stop()

# ---------- выбор колонок ----------
candidate_age = None
for name in df.columns:
    if str(name).strip().lower() in {"age", "возраст"}:
        candidate_age = name
        break

age_col = st.selectbox("Колонка возраста", list(df.columns),
                       index=(list(df.columns).index(candidate_age) if candidate_age in df.columns else 0))

met_cols = st.multiselect("Метаболиты", [c for c in df.columns if c != age_col],
                          default=[c for c in df.columns if c != age_col][:1])
if not met_cols:
    st.warning("Выберите хотя бы один метаболит.")
    st.stop()

sex_candidates = [c for c in df.columns if str(c).strip().lower() in {"sex", "gender", "пол"}]
sex_col = st.selectbox("Колонка пола", ["<нет>"] + list(df.columns),
                       index=(1 + list(df.columns).index(sex_candidates[0])) if sex_candidates else 0)
if sex_col == "<нет>":
    sex_col = None

# ---------- настройки ----------
with st.expander("Настройки анализа"):
    age_min = float(np.nanmin(pd.to_numeric(df[age_col], errors="coerce")))
    age_max = float(np.nanmax(pd.to_numeric(df[age_col], errors="coerce")))
    sel_range = st.slider("Фильтр по возрасту",
                          float(np.floor(age_min)), float(np.ceil(age_max)),
                          (float(np.floor(age_min)), float(np.ceil(age_max))))
    log_y = st.checkbox("Логарифмировать концентрации (log10)", value=False)
    show_ci = st.checkbox("Показывать 95% доверительный интервал", value=False)
    local_sd_window = st.slider("Окно (лет) для локальной SD вокруг линии", 2, 20, 10)
    bin_years = st.slider("Ширина окна (лет) для агрегирования", 1, 10, 3)
    show_sd_band = st.checkbox("Показывать SD-полосы на агрегированном графике", value=True)

# ---------- подготовка ----------
def _clean_numeric(s): 
    return pd.to_numeric(s, errors="coerce")

df_work = df.copy()
df_work[age_col] = _clean_numeric(df_work[age_col])
df_work = df_work[(df_work[age_col] >= sel_range[0]) & (df_work[age_col] <= sel_range[1])]

# ---------- функции ----------
def linreg_stats(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3: 
        return np.nan, np.nan, np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1] if len(x) >= 2 else np.nan
    r2 = r**2 if np.isfinite(r) else np.nan
    pvalue = np.nan
    if SCIPY_OK and len(x) >= 3:
        r_sc, p_sc = stats.pearsonr(x, y)
        r, pvalue = r_sc, p_sc
        r2 = r**2
    return slope, intercept, r, r2, pvalue

def ci_band(x, y, slope, intercept):
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    if n < 3: 
        return None, None
    y_hat = slope * x + intercept
    resid = y - y_hat
    s_err = np.sqrt(np.sum(resid**2) / (n - 2))
    x_mean = np.mean(x)
    denom = np.sum((x - x_mean)**2)
    if denom == 0:
        return None, None
    t = 1.96
    se_line = lambda xv: s_err * np.sqrt(1/n + (xv - x_mean)**2 / denom)
    y_low = y_hat - t * np.array([se_line(xv) for xv in x])
    y_high = y_hat + t * np.array([se_line(xv) for xv in x])
    return y_low, y_high

def local_sd_along_line(x, y, x_line, window_years):
    df_temp = pd.DataFrame({"x": x, "y": y}).sort_values("x")
    sd_values = []
    half = window_years / 2
    for xv in x_line:
        y_local = df_temp.loc[(df_temp["x"] >= xv - half) & (df_temp["x"] <= xv + half), "y"]
        sd_values.append(y_local.std() if len(y_local) > 3 else np.nan)
    return np.array(sd_values)

def mean_sd_by_age(gr, var, win_years):
    m = gr.groupby("Age_round")[var].mean()
    s = gr.groupby("Age_round")[var].std()
    if len(m) > 1:
        win = max(1, int(win_years))
        m = m.rolling(window=win, center=True, min_periods=1).mean()
        s = s.rolling(window=win, center=True, min_periods=1).mean()
    return m, s

# ---------- цикл: для каждого метаболита рисуем ДВА графика ----------
for met in met_cols:
    st.markdown("---")
    st.subheader(f"Метаболит: {met}")

    # ДАННЫЕ для текущего метаболита
    y = _clean_numeric(df_work[met])
    x = df_work[age_col].astype(float)
    if log_y:
        y = np.log10(y)

    # === ГРАФИК 1: scatter + регрессия ===
    slope, intercept, r, r2, pvalue = linreg_stats(x.values, y.values)

    fig, ax = plt.subplots(figsize=(5, 2))
    ax.scatter(x, y, alpha=0.6, label="наблюдения")

    if np.isfinite(slope) and np.isfinite(intercept):
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="red", linewidth=2, label="линейная регрессия")

        sd_values = local_sd_along_line(x.values, y.values, x_line, local_sd_window)
        if np.isfinite(sd_values).any():
            ax.fill_between(x_line, y_line - sd_values, y_line + sd_values,
                            color="red", alpha=0.25, label=f"±1 SD (локальная, окно={local_sd_window})")

        if show_ci:
            y_low, y_high = ci_band(x.values, y.values, slope, intercept)
            if y_low is not None:
                order = np.argsort(x.values)
                x_sorted = x.values[order]
                y_low_i = np.interp(x_line, x_sorted, y_low[order])
                y_high_i = np.interp(x_line, x_sorted, y_high[order])
                ax.fill_between(x_line, y_low_i, y_high_i, alpha=0.18, color='red', label="95% ДИ")

    ax.set_xlabel("Возраст")
    ax.set_ylabel(f"{'log10(' + met + ')' if log_y else met}")
    ax.grid(True, alpha=0.25)
    title = f"R²={r2:.3f}" if np.isfinite(r2) else "R²: n/a"
    if np.isfinite(r):
        title += f" | r={r:.3f}"
    if np.isfinite(pvalue):
        title += f" | p={pvalue:.3g}"
    ax.set_title(f"{met} vs {age_col} — {title}")
    ax.legend(loc="best")
    st.pyplot(fig)

    # === ГРАФИК 2: агрегированные средние по возрасту для ЭТОГО метаболита ===
    df_agg = df_work[[age_col, met] + ([sex_col] if sex_col else [])].dropna()
    if df_agg.empty:
        st.info("Недостаточно данных для агрегированного графика.")
        continue

    if log_y:
        df_agg[met] = np.log10(df_agg[met])
    df_agg["Age_round"] = df_agg[age_col].round(0).astype(int)

    mean_all, sd_all = mean_sd_by_age(df_agg, met, bin_years)

    fig2, ax2 = plt.subplots(figsize=(5, 2))
    ax2.plot(mean_all.index, mean_all.values, linewidth=2, color="black", label="Среднее (все)")
    if show_sd_band and sd_all.notna().any():
        ax2.fill_between(mean_all.index, (mean_all - sd_all).values, (mean_all + sd_all).values,
                         alpha=0.15, color="black", label="±1 SD (все)")

    if sex_col:
        palette = {"M": "blue", "F": "red", "м": "blue", "ж": "red",
                   "М": "blue", "Ж": "red", "Male": "blue", "Female": "red"}
        for val in pd.Series(df_agg[sex_col].astype(str).unique()):
            grp = df_agg[df_agg[sex_col].astype(str) == val]
            if grp.empty:
                continue
            mean_s, sd_s = mean_sd_by_age(grp, met, bin_years)
            color = palette.get(str(val), None)
            ax2.plot(mean_s.index, mean_s.values, linewidth=2, color=color, label=f"Среднее ({val})")
            if show_sd_band and sd_s.notna().any():
                ax2.fill_between(mean_s.index, (mean_s - sd_s).values, (mean_s + sd_s).values,
                                 alpha=0.18, color=color)

    ax2.set_xlabel("Возраст, лет")
    ax2.set_ylabel(f"{'log10(' + met + ')' if log_y else met}")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")
    st.pyplot(fig2)
