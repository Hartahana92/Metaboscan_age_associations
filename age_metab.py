import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    from scipy import stats
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

st.set_page_config(page_title="Метаболит vs Возраст", layout="wide")

st.title("Зависимость концентрации метаболита от возраста")
st.caption("Загрузите CSV/XLSX → выберите колонку возраста и метаболиты → получите графики и R²")

# ---------- загрузка данных ----------
uploaded = st.file_uploader("Загрузите файл CSV или XLSX", type=["csv", "xlsx"])
if uploaded is None:
    st.info("Ожидаю файл с данными. Пример: колонки Age, C2, C3, ...")
    st.stop()

# параметры чтения
with st.expander("Параметры импорта"):
    filetype = "xlsx" if uploaded.name.lower().endswith("xlsx") else "csv"
    sep = st.text_input("Разделитель (для CSV)", value=",")
    sheet = st.text_input("Имя листа (для XLSX; оставьте пустым для первого)", value="")
    decimal = st.text_input("Десятичный разделитель", value=".")
    thousands = st.text_input("Разделитель тысяч (при наличии)", value="")

# читаем
try:
    if filetype == "csv":
        df = pd.read_csv(uploaded, sep=sep or ",", decimal=decimal or ".", thousands=thousands or None)
    else:
        xls = pd.ExcelFile(uploaded)
        sheet_to_use = sheet if sheet in xls.sheet_names else (xls.sheet_names[0] if sheet == "" else xls.sheet_names[0])
        df = pd.read_excel(xls, sheet_name=sheet_to_use)
except Exception as e:
    st.error(f"Не удалось прочитать файл: {e}")
    st.stop()


# ---------- выбор колонок ----------
# предложим колонку возраста по типичным именам
candidate_age = None
for name in df.columns:
    if str(name).strip().lower() in {"age", "Возраст"}:
        candidate_age = name
        break
age_options = list(df.columns)
age_index = age_options.index(candidate_age) if candidate_age in age_options else 0
age_col = st.selectbox("Колонка возраста", options=list(df.columns), index=(list(df.columns).index(candidate_age) if candidate_age in df.columns else 0))
met_cols = st.multiselect(
    "Колонки метаболитов (1+)",
    options=[c for c in df.columns if c != age_col],
    default=[c for c in df.columns if c != age_col][:1]
)

if not met_cols:
    st.warning("Выберите хотя бы один метаболит.")
    st.stop()

sex_candidates = [c for c in df.columns if str(c).strip().lower() in {"sex", "gender", "Пол"}]
sex_col = st.selectbox("Колонка пола (опционально)", options=["<нет>"] + list(df.columns),
                       index=(1 + list(df.columns).index(sex_candidates[0])) if sex_candidates else 0)

if sex_col == "<нет>":
    sex_col = None

# ---------- настройки анализа ----------
with st.expander("Настройки анализа и отображения"):
    age_min = float(np.nanmin(pd.to_numeric(df[age_col], errors="coerce")))
    age_max = float(np.nanmax(pd.to_numeric(df[age_col], errors="coerce")))
    sel_range = st.slider(
        "Фильтр по возрасту",
        min_value=float(np.floor(age_min)),
        max_value=float(np.ceil(age_max)),
        value=(float(np.floor(age_min)), float(np.ceil(age_max)))
    )
    log_y = st.checkbox("Логарифмировать концентрацию (log10(y))", value=False)
    show_ci = st.checkbox("Показывать 95% ДИ для линии (по упрощённой формуле)", value=False)
    # локальная SD-полоса вокруг регрессии (по окну лет)
    local_sd_window = st.slider("Окно (лет) для локальной SD вокруг регрессии", 2, 20, 10)

    show_agg = st.checkbox("Показать агрегированный график (средние по возрасту)", value=False)
    agg_met = st.selectbox("Метаболит для агрегированного графика", options=met_cols) if show_agg else None
    bin_years = st.slider("Ширина окна (годы) для усреднения/сглаживания", 1, 10, 3) if show_agg else None
    show_sd_band = st.checkbox("Показывать SD-полосы на агрегированном графике", value=True) if show_agg else None

# ---------- подготовка данных ----------
def _clean_numeric(s):
    return pd.to_numeric(s, errors="coerce")

df_work = df.copy()
df_work[age_col] = _clean_numeric(df_work[age_col])
df_work = df_work[(df_work[age_col] >= sel_range[0]) & (df_work[age_col] <= sel_range[1])]

# ---------- функции статистики ----------
def linreg_stats(x, y):
    """
    Возвращает:
      slope, intercept, r (Pearson), r2, pvalue
    Без SciPy: r и pvalue будут None (кроме r2 через corrcoef).
    """
    # очистка NaN
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # линейная регрессия по МНК
    slope, intercept = np.polyfit(x, y, 1)

    # r2 как квадрат коэффициента корреляции Пирсона
    if len(x) >= 2:
        r = np.corrcoef(x, y)[0, 1]
        r2 = r**2
    else:
        r = np.nan
        r2 = np.nan

    if SCIPY_OK and len(x) >= 3:
        r_sc, p_sc = stats.pearsonr(x, y)
        r = r_sc
        pvalue = p_sc
    else:
        pvalue = np.nan

    return slope, intercept, r, r2, pvalue

def ci_band(x, y, slope, intercept):
    """ Приблизительная 95% доверительная полоса для прогноза (упрощённо). """
    x = np.asarray(x); y = np.asarray(y)
    n = len(x)
    if n < 3: 
        return None, None
    y_hat = slope * x + intercept
    resid = y - y_hat
    s_err = np.sqrt(np.sum(resid**2) / (n - 2))
    x_mean = np.mean(x)
    t = 1.96  # ~95%
    if denom == 0:
        return None, None
    
    se_line = lambda xv: s_err * np.sqrt(1/n + (xv - x_mean)**2 / np.sum((x - x_mean)**2))
    y_low = y_hat - t * np.array([se_line(xv) for xv in x])
    y_high = y_hat + t * np.array([se_line(xv) for xv in x])
    return y_low, y_high

def local_sd_along_line(x, y, x_line, window_years):
    """
    Локальная SD (по окну ±window_years/2) относительно оси X (возраста).
    Возвращает массив sd_values по точкам x_line (NaN, где мало наблюдений).
    """
    df_temp = pd.DataFrame({"x": x, "y": y}).sort_values("x")
    sd_values = []
    half = window_years / 2.0
    for xv in x_line:
        mask = (df_temp["x"] >= xv - half) & (df_temp["x"] <= xv + half)
        y_local = df_temp.loc[mask, "y"]
        sd_values.append(y_local.std() if len(y_local) > 3 else np.nan)
    return np.array(sd_values)
    
# ---------- вывод ----------
st.subheader("Результаты")
cols = st.columns(min(2, len(met_cols)))

for i, met in enumerate(met_cols):
    with cols[i % len(cols)]:
        y = _clean_numeric(df_work[met])
        x = df_work[age_col].astype(float)

        if log_y:
            y = np.log10(y)

        slope, intercept, r, r2, pvalue = linreg_stats(x.values, y.values)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(x, y, alpha=0.6, label="наблюдения")

        if np.isfinite(slope) and np.isfinite(intercept):
            x_line = np.linspace(np.nanmin(x), np.nanmax(x), 200)
            y_line = slope * x_line + intercept

            # линия регрессии (красная)
            ax.plot(x_line, y_line, linewidth=2, color='red', label="линейная регрессия")

            # локальная SD-полоса вокруг линии
            sd_values = local_sd_along_line(x.values, y.values, x_line, window_years=local_sd_window)
            if np.isfinite(sd_values).any():
                ax.fill_between(
                    x_line,
                    y_line - sd_values,
                    y_line + sd_values,
                    color='red', alpha=0.25, label=f'±1 SD (локальная, окно={local_sd_window} лет)'
                )

            # 95% доверительная полоса (опция)
            if show_ci:
                y_low, y_high = ci_band(x.values, y.values, slope, intercept)
                if y_low is not None:
                    # интерполируем для x_line
                    order = np.argsort(x.values)
                    x_sorted = x.values[order]
                    y_low_i = np.interp(x_line, x_sorted, y_low[order])
                    y_high_i = np.interp(x_line, x_sorted, y_high[order])
                    ax.fill_between(x_line, y_low_i, y_high_i, alpha=0.18, color='red', label="95% ДИ")

        ax.set_xlabel("Возраст")
        ax.set_ylabel(f"{'log10(' + met + ')' if log_y else met}")
        ax.grid(True, alpha=0.25)

        title_parts = [f"{met} vs {age_col}", f"R² = {r2:.3f}" if np.isfinite(r2) else "R²: n/a"]
        if np.isfinite(slope) and np.isfinite(intercept):
            title_parts.append(f"y = {slope:.3f}·x + {intercept:.3f}")
        if np.isfinite(r):
            title_parts.append(f"r = {r:.3f}")
        if np.isfinite(pvalue):
            title_parts.append(f"p = {pvalue:.3g}")
        ax.set_title(" | ".join(title_parts))
        ax.legend(loc="best")
        st.pyplot(fig)


# ---------- агрегированный график средних по возрасту ----------
if show_agg:
    st.markdown("---")
    st.subheader("Агрегированный график средних по возрасту")

    df_agg = df_work[[age_col, agg_met] + ([sex_col] if sex_col else [])].copy()
    df_agg[age_col] = pd.to_numeric(df_agg[age_col], errors="coerce")
    df_agg[agg_met] = pd.to_numeric(df_agg[agg_met], errors="coerce")
    df_agg = df_agg.dropna(subset=[age_col, agg_met])

    if log_y:
        df_agg[agg_met] = np.log10(df_agg[agg_met])

    # округлим возраст до целых и применим скользящее окно по годам
    df_agg["Age_round"] = df_agg[age_col].round(0).astype(int)

    def mean_sd_by_age(gr):
        m = gr.groupby("Age_round")[agg_met].mean()
        s = gr.groupby("Age_round")[agg_met].std()
        if len(m) > 1:
            win = max(1, int(bin_years))
            m = m.rolling(window=win, center=True, min_periods=1).mean()
            s = s.rolling(window=win, center=True, min_periods=1).mean()
        return m, s

    mean_all, sd_all = mean_sd_by_age(df_agg)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(mean_all.index, mean_all.values, linewidth=2, color="black", label="Среднее (все)")
    if show_sd_band and sd_all.notna().any():
        ax2.fill_between(mean_all.index,
                         (mean_all - sd_all).values,
                         (mean_all + sd_all).values,
                         alpha=0.15, color="black", label="±1 SD (все)")

    # по полу (если есть колонка)
    if sex_col:
        # палитра и метки
        palette = {"M": "blue", "F": "red", "м": "blue", "ж": "red",
                   "М": "blue", "Ж": "red", "Male": "blue", "Female": "red"}
        for val in pd.Series(df_agg[sex_col].astype(str).unique()):
            group = df_agg[df_agg[sex_col].astype(str) == val]
            if group.empty:
                continue
            mean_sex, sd_sex = mean_sd_by_age(group)
            color = palette.get(str(val), None)
            ax2.plot(mean_sex.index, mean_sex.values,
                     linewidth=2, label=f"Среднее ({val})", color=color)
            if show_sd_band and sd_sex.notna().any():
                ax2.fill_between(mean_sex.index,
                                 (mean_sex - sd_sex).values,
                                 (mean_sex + sd_sex).values,
                                 alpha=0.18, color=color)

    ax2.set_xlabel("Возраст, лет")
    ax2.set_ylabel(f"{'log10(' + agg_met + ')' if log_y else agg_met}")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")
    st.pyplot(fig2)

# ---------- сводная таблица метрик ----------
summary = []
for met in met_cols:
    y = _clean_numeric(df_work[met])
    x = df_work[age_col].astype(float)
    if log_y:
        y = np.log10(y)
    slope, intercept, r, r2, pvalue = linreg_stats(x.values, y.values)
    n_obs = int((np.isfinite(x) & np.isfinite(y)).sum())
    summary.append({
        "Metabolite": met,
        "N": n_obs,
        "slope": slope, "intercept": intercept,
        "Pearson r": r, "R^2": r2, "p-value": pvalue,
        "log10(y)": log_y
    })

st.markdown("---")
st.subheader("Сводка по метрикам")
st.dataframe(pd.DataFrame(summary))
