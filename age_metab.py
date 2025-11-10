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

age_col = st.selectbox("Колонка возраста", options=list(df.columns), index=(list(df.columns).index(candidate_age) if candidate_age in df.columns else 0))
met_cols = st.multiselect(
    "Колонки метаболитов (1+)",
    options=[c for c in df.columns if c != age_col],
    default=[c for c in df.columns if c != age_col][:1]
)

if not met_cols:
    st.warning("Выберите хотя бы один метаболит.")
    st.stop()

# ---------- настройки анализа ----------
with st.expander("Настройки анализа и отображения"):
    age_min, age_max = float(np.nanmin(pd.to_numeric(df[age_col], errors="coerce"))), float(np.nanmax(pd.to_numeric(df[age_col], errors="coerce")))
    sel_range = st.slider("Фильтр по возрасту", min_value=float(np.floor(age_min)), max_value=float(np.ceil(age_max)),
                          value=(float(np.floor(age_min)), float(np.ceil(age_max))))
    log_y = st.checkbox("Логарифмировать концентрацию (log10(y))", value=False)
    show_ci = st.checkbox("Показывать 95% ДИ для линии (по простой аппроксимации)", value=False)

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
    se_line = lambda xv: s_err * np.sqrt(1/n + (xv - x_mean)**2 / np.sum((x - x_mean)**2))
    y_low = y_hat - t * np.array([se_line(xv) for xv in x])
    y_high = y_hat + t * np.array([se_line(xv) for xv in x])
    return y_low, y_high

# ---------- вывод ----------
st.subheader("Результаты")

cols = st.columns(min(2, len(met_cols)))
for i, met in enumerate(met_cols):
    with cols[i % len(cols)]:
        y = _clean_numeric(df_work[met])
        x = df_work[age_col].astype(float)

        # логарифмирование
        if log_y:
            y = np.log10(y)

        # расчёт статистики
        slope, intercept, r, r2, pvalue = linreg_stats(x.values, y.values)

        # график
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(x, y, alpha=0.6, label="наблюдения")

        if np.isfinite(slope) and np.isfinite(intercept):
            x_line = np.linspace(x.min(), x.max(), 200)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, linewidth=2, color='red', label="линейная регрессия")
            y_pred = slope * x + intercept
            residuals = y - y_pred
            sd = np.nanstd(residuals)
            ax.fill_between(
                x_line,
                y_line - sd,
                y_line + sd,
                color='red',
                alpha=0.2,
                label='±1 SD'
            )
            
            if show_ci:
                y_low, y_high = ci_band(x.values, y.values, slope, intercept)
                if y_low is not None:
                    # интерполируем на x_line для плавности
                    y_low_i = np.interp(x_line, np.sort(x.values), y_low[np.argsort(x.values)])
                    y_high_i = np.interp(x_line, np.sort(x.values), y_high[np.argsort(x.values)])
                    ax.fill_between(x_line, y_low_i, y_high_i, alpha=0.2, label="95% ДИ")

        ax.set_xlabel("Возраст")
        ax.set_ylabel(f"{'log10(' + met + ')' if log_y else met}")
        ax.grid(True, alpha=0.25)

        # заголовок с метриками
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

# сводная таблица метрик
summary = []
for met in met_cols:
    y = _clean_numeric(df_work[met])
    x = df_work[age_col].astype(float)
    if log_y:
        y = np.log10(y)
    slope, intercept, r, r2, pvalue = linreg_stats(x.values, y.values)
    summary.append({
        "Metabolite": met,
        "N": int((np.isfinite(x) & np.isfinite(y)).sum()),
        "slope": slope, "intercept": intercept,
        "Pearson r": r, "R^2": r2, "p-value": pvalue,
        "log10(y)": log_y
    })

st.markdown("---")
st.subheader("Сводка по метрикам")
st.dataframe(pd.DataFrame(summary))
