"""Streamlit app para analizar datos de temperatura y humedad replicando el notebook base."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# -----------------------------------------------------------------------------
# Configuración global
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Reportes de Sensores", layout="wide")
sns.set_theme(style="whitegrid")

DEFAULT_TEMP_SHEET_HINT = "tem"
DEFAULT_HUM_SHEET_HINT = "hum"
DEFAULT_INTERVAL_MINUTES = 1
PERCENTIL_ALTO = 0.99
PERCENTIL_BAJO = 0.01
DEFAULT_DECIMALES = 1
PLOT_SMOOTH_WINDOW = 5
PLOT_DIFF_QUANTILE = 0.995


class PruebaTipo:
    MAX = "max"
    MIN = "min"


NOMBRES_GRUPOS = {
    "almacen": "Almacén",
    "stage": "Stage",
    "maquila": "Maquila",
}


# -----------------------------------------------------------------------------
# Utilidades generales
# -----------------------------------------------------------------------------
def safe_stop() -> None:
    """Detiene la ejecución cuando falta información en Streamlit."""
    try:
        st.stop()
    except Exception:
        pass


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def _hampel(series: pd.Series, window: int = 7, n_sigmas: float = 3.0) -> pd.Series:
    datos = to_numeric(series)
    if datos.empty:
        return datos
    mediana = datos.rolling(window, center=True, min_periods=1).median()
    mad = 1.4826 * (datos - mediana).abs().rolling(window, center=True, min_periods=1).median()
    umbral = n_sigmas * mad.replace(0, np.nan)
    return datos.mask((datos - mediana).abs() > umbral)

def fmt_num(valor, decimales: int) -> str:
    if valor is None:
        return ""
    if isinstance(valor, (float, np.floating)) and not np.isfinite(valor):
        return ""
    try:
        return f"{float(valor):.{decimales}f}"
    except (TypeError, ValueError):
        return str(valor)

def _series_for_plot(series: pd.Series, smooth_window: int = PLOT_SMOOTH_WINDOW, diff_quantile: float = PLOT_DIFF_QUANTILE) -> pd.Series:
    datos = to_numeric(series).replace([np.inf, -np.inf], np.nan)
    #if datos.empty:
    return datos
    #filtrado = _hampel(datos)
    #suavizado = filtrado.rolling(smooth_window, center=True, min_periods=1).mean()
    #diff = suavizado.diff().abs()
    #if diff.dropna().empty:
    #    return suavizado
    #threshold = diff.quantile(diff_quantile)
    #if not np.isfinite(threshold) or threshold <= 0:
    #    med = diff.median()
    #    if np.isfinite(med) and med > 0:
    #        threshold = med * 5
    #    else:
    #        return suavizado
    #mask = diff > threshold
    #mask_prev = mask.shift(1, fill_value=False)
    #suavizado = suavizado.mask(mask | mask_prev)
    #return suavizado


def read_excel_sheet(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)


def list_sheets(file_bytes: bytes) -> list[str]:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    return list(xls.sheet_names)


def parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if "Timestamp" in data.columns:
        data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")
        return data

    fecha_col = next((c for c in data.columns if str(c).lower().startswith("fecha")), None)
    hora_col = next((c for c in data.columns if str(c).lower().startswith("hora")), None)
    if fecha_col is None:
        raise ValueError("La hoja no contiene columnas de fecha/Timestamp")

    fecha_texto = data[fecha_col].astype(str).str.strip()
    if hora_col is not None:
        hora_texto = data[hora_col].astype(str).str.strip()
        timestamp = pd.to_datetime(fecha_texto + " " + hora_texto, errors="coerce", dayfirst=True)
    else:
        timestamp = pd.to_datetime(fecha_texto, errors="coerce", dayfirst=True)

    data["Timestamp"] = timestamp
    return data


def limpiar_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = parse_timestamp(raw_df)
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    df = df.drop_duplicates(subset=["Timestamp"], keep="first").reset_index(drop=True)
    return df


def detectar_sensores(df: pd.DataFrame, excluir: set[str]) -> list[str]:
    sensores = []
    for col in df.columns:
        if col in excluir:
            continue
        serie = to_numeric(df[col])
        if serie.notna().sum() > 0:
            sensores.append(col)
    return sensores


def completar_nulos(df: pd.DataFrame, sensores: Iterable[str]) -> pd.DataFrame:
    data = df.copy()
    for sensor in sensores:
        serie = to_numeric(data[sensor])
        data[sensor] = serie.where(~serie.isna(), np.nan)
    return data


def id_solo(valor: str) -> str:
    coincidencia = re.search(r"(\d+)", str(valor))
    return coincidencia.group(1) if coincidencia else str(valor).strip()


def load_map_file(uploaded) -> pd.DataFrame | None:
    if uploaded is None:
        return None
    try:
        uploaded.seek(0)
    except Exception:
        pass
    try:
        if uploaded.name.lower().endswith(".csv"):
            return pd.read_csv(uploaded)
        return pd.read_excel(uploaded)
    except Exception:
        try:
            uploaded.seek(0)
        except Exception:
            pass
        try:
            return pd.read_csv(uploaded, sep=";")
        except Exception:
            return None


def dividir_grupos(
    sensores: Iterable[str],
    stage: Iterable[str],
    maquila: Iterable[str],
) -> dict[str, list[str]]:
    sensores_unicos = list(dict.fromkeys(sensores))
    stage_set = {s for s in stage if s in sensores_unicos}
    maquila_set = {s for s in maquila if s in sensores_unicos and s not in stage_set}
    almacen = [s for s in sensores_unicos if s not in stage_set and s not in maquila_set]
    stage_list = [s for s in sensores_unicos if s in stage_set]
    maquila_list = [s for s in sensores_unicos if s in maquila_set]
    return {
        "almacen": almacen,
        "stage": stage_list,
        "maquila": maquila_list,
    }


def construir_mapa_niveles(
    df_map: pd.DataFrame | None,
    sensores: list[str],
    niveles_manual: dict[str, str],
    alturas_manual: dict[str, str],
) -> tuple[dict[str, int | None], dict[int, float]]:
    level_override: dict[str, int | None] = {}
    level_meters: dict[int, float] = {}

    if df_map is not None and not df_map.empty:
        columnas = {str(col).strip().lower(): col for col in df_map.columns}
        col_sensor = columnas.get("sensor_id") or columnas.get("sensor") or df_map.columns[0]
        col_level = columnas.get("level") or columnas.get("nivel")
        if col_level is None and df_map.shape[1] > 1:
            col_level = df_map.columns[1]
        col_height = columnas.get("height") or columnas.get("altura")

        if col_level is not None:
            for _, row in df_map.iterrows():
                sensor = str(row.get(col_sensor, "")).strip()
                if not sensor:
                    continue
                try:
                    nivel = int(row[col_level])
                except Exception:
                    continue
                level_override[sensor] = nivel
                if col_height is not None:
                    try:
                        altura = float(row[col_height])
                        level_meters[nivel] = altura
                    except Exception:
                        continue

    for sensor in sensores:
        level_override.setdefault(sensor, None)

    for sensor, nivel_txt in niveles_manual.items():
        nivel_txt = str(nivel_txt or "").strip()
        if not nivel_txt:
            continue
        try:
            level_override[sensor] = int(nivel_txt)
        except ValueError:
            continue

    for nivel_txt, altura_txt in alturas_manual.items():
        nivel_txt = str(nivel_txt or "").strip()
        altura_txt = str(altura_txt or "").strip()
        if not nivel_txt or not altura_txt:
            continue
        try:
            nivel = int(nivel_txt)
            altura = float(altura_txt)
        except ValueError:
            continue
        level_meters[nivel] = altura

    return level_override, level_meters


def nivel_de(sensor_id: str, level_override: dict[str, int | None]) -> int | None:
    if sensor_id in level_override:
        nivel = level_override[sensor_id]
        return int(nivel) if nivel is not None else None
    sensor_simple = id_solo(sensor_id)
    for clave, valor in level_override.items():
        if id_solo(clave) == sensor_simple:
            return int(valor) if valor is not None else None
    return None


def ubicacion_text(sensor_id: str, level_override: dict[str, int | None], level_meters: dict[int, float]) -> str:
    nivel = nivel_de(sensor_id, level_override)
    if nivel is None:
        return "Nivel desconocido"
    altura = level_meters.get(nivel)
    if altura is None:
        return f"Nivel {nivel}"
    return f"Nivel {nivel} -> {altura:.2f} m a N.P.T."


MESES_ES_CORTO = [
    "ene",
    "feb",
    "mar",
    "abr",
    "may",
    "jun",
    "jul",
    "ago",
    "sep",
    "oct",
    "nov",
    "dic",
]

MESES_ABREV_A_NUMERO = {mes: idx + 1 for idx, mes in enumerate(MESES_ES_CORTO)}
FORMATO_FECHA_ES_REGEX = re.compile(
    r"^\s*(\d{1,2})\s*-\s*([a-záéíóúñ]{3,})\s*-\s*(\d{4})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?\s*$",
    flags=re.IGNORECASE,
)


def _timestamp_es(
    dt: pd.Timestamp | datetime | None,
    *,
    include_time: bool,
    include_seconds: bool = True,
) -> str:
    if dt is None or pd.isna(dt):
        return ""
    ts = pd.to_datetime(dt)
    fecha = f"{ts.day:02d} - {MESES_ES_CORTO[ts.month - 1]} - {ts.year}"
    if not include_time:
        return fecha
    formato_hora = "%H:%M:%S" if include_seconds else "%H:%M"
    return f"{fecha} {ts.strftime(formato_hora)}"


def format_datetime_es(dt: pd.Timestamp | None, include_seconds: bool = True) -> str:
    return _timestamp_es(dt, include_time=True, include_seconds=include_seconds)


def format_date_es(dt: pd.Timestamp | None) -> str:
    return _timestamp_es(dt, include_time=False)


def parse_datetime_es(texto: str) -> pd.Timestamp | None:
    if texto is None:
        return None
    match = FORMATO_FECHA_ES_REGEX.match(texto.strip())
    if not match:
        return None
    dia = int(match.group(1))
    mes_texto = match.group(2).lower()
    mes = MESES_ABREV_A_NUMERO.get(mes_texto)
    if mes is None:
        return None
    anio = int(match.group(3))
    hora = int(match.group(4)) if match.group(4) is not None else 0
    minuto = int(match.group(5)) if match.group(5) is not None else 0
    segundo = int(match.group(6)) if match.group(6) is not None else 0
    try:
        return pd.Timestamp(year=anio, month=mes, day=dia, hour=hora, minute=minuto, second=segundo)
    except ValueError:
        return None


def matplotlib_date_formatter(include_seconds: bool = False) -> mdates.Formatter:
    def _formatter(value, _):
        dt = mdates.num2date(value)
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return _timestamp_es(pd.Timestamp(dt), include_time=True, include_seconds=include_seconds)

    return FuncFormatter(_formatter)


def titulo_con_contexto(titulo_base: str, contexto: str | None) -> str:
    contexto_limpio = (contexto or "").strip()
    if not contexto_limpio:
        return titulo_base
    return f"{titulo_base} — {contexto_limpio}"


def contexto_con_division(contexto_base: str, division: str | None) -> str:
    partes = [s.strip() for s in (contexto_base or "", division or "") if s and s.strip()]
    return " · ".join(partes)


def agregar_lineas_umbral(ax: plt.Axes, lineas: list[tuple[float, str]] | None) -> None:
    if not lineas:
        return
    colores = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd"]
    for idx, (valor, etiqueta) in enumerate(lineas):
        if valor is None or not np.isfinite(valor):
            continue
        color = colores[idx % len(colores)]
        etiqueta_linea = etiqueta or f"Umbral {idx + 1}"
        ax.axhline(valor, color=color, linestyle="--", linewidth=1.2, label=etiqueta_linea)


def lineas_umbral_temperatura(valores: Iterable[float]) -> list[tuple[float, str]]:
    valores_validos = [
        float(valor) for valor in valores if valor is not None and np.isfinite(valor)
    ]
    if not valores_validos:
        return []
    valores_unicos = sorted(dict.fromkeys(valores_validos))
    lineas = []
    for idx, valor in enumerate(valores_unicos, start=1):
        etiqueta = "Umbral temperatura" if len(valores_unicos) == 1 else f"Umbral temperatura {idx}"
        lineas.append((valor, etiqueta))
    return lineas


def lineas_umbral_humedad(valor: float | None) -> list[tuple[float, str]]:
    if valor is None or not np.isfinite(valor):
        return []
    return [(float(valor), "Umbral humedad máxima")]


def resaltar_periodo_prueba(
    ax: plt.Axes,
    ts: pd.Series,
    t_ini: pd.Timestamp,
    t_fin: pd.Timestamp,
    *,
    color_span: str,
    legend_kwargs: dict[str, Any] | None = None,
    color_estabilizacion: str = "#b0bec5",
    alpha_prueba: float = 0.14,
    alpha_estabilizacion: float = 0.05,
) -> None:
    if t_ini is None or t_fin is None:
        return
    ts_series = pd.to_datetime(ts, errors="coerce")
    if ts_series.dropna().empty:
        return
    t_min = ts_series.min()
    t_max = ts_series.max()
    if pd.isna(t_min) or pd.isna(t_max):
        return

    ax.axvspan(t_ini, t_fin, color=color_span, alpha=alpha_prueba)
    estabilizacion = None
    if t_fin < t_max:
        estabilizacion = ax.axvspan(t_fin, t_max, color=color_estabilizacion, alpha=alpha_estabilizacion)

    ylim = ax.get_ylim()
    if t_fin > t_ini:
        y_text = ylim[1] - (ylim[1] - ylim[0]) * 0.06
        x_prueba = t_ini + (t_fin - t_ini) / 2
        ax.text(
            x_prueba,
            y_text,
            "Periodo de Prueba",
            color=color_span,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="top",
            alpha=0.9,
        )
        if estabilizacion is not None:
            dur_est = t_max - t_fin
            if pd.notna(dur_est) and dur_est > pd.Timedelta(0):
                x_est = t_fin + dur_est / 2
                ax.text(
                    x_est,
                    y_text,
                    "Estabilización",
                    color="#4a4a4a",
                    fontsize=9,
                    fontweight="bold",
                    ha="center",
                    va="top",
                    alpha=0.85,
                )

    handles, labels = ax.get_legend_handles_labels()
    # No añadir parches extra a la leyenda para mantener la misma escala visual.
    if handles:
        legend_kwargs = legend_kwargs or {}
        ax.legend(handles, labels, **legend_kwargs)


def filtrar_rango(
    df: pd.DataFrame,
    fecha_ini,
    fecha_fin,
) -> tuple[pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None]:
    if df.empty:
        return df.copy(), None, None

    def _to_timestamp(valor, fallback):
        if valor is None or valor == "":
            return fallback
        if isinstance(valor, (pd.Timestamp, datetime)):
            return pd.to_datetime(valor, errors="coerce")
        if isinstance(valor, tuple) and len(valor) == 2:
            return pd.to_datetime(valor[0], errors="coerce")
        if isinstance(valor, str):
            texto = valor.strip()
            if not texto:
                return fallback
            ts_custom = parse_datetime_es(texto)
            if ts_custom is not None:
                return ts_custom
            formatos = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%d",
                "%d-%m-%Y %H:%M:%S",
                "%d-%m-%Y %H:%M",
                "%d-%m-%Y",
            ]
            for fmt in formatos:
                try:
                    return pd.to_datetime(texto, format=fmt, errors="raise")
                except Exception:
                    continue
            ts = pd.to_datetime(texto, errors="coerce", dayfirst=True)
            if pd.isna(ts):
                ts = pd.to_datetime(texto, errors="coerce")
            return ts
        return pd.to_datetime(valor, errors="coerce")

    ti = _to_timestamp(fecha_ini, df["Timestamp"].min())
    tf = _to_timestamp(fecha_fin, df["Timestamp"].max())

    if pd.isna(ti) or pd.isna(tf):
        return df.copy(), df["Timestamp"].min(), df["Timestamp"].max()
    if ti > tf:
        ti, tf = tf, ti

    mask = (df["Timestamp"] >= ti) & (df["Timestamp"] <= tf)
    filtrado = df.loc[mask].copy()
    if filtrado.empty:
        return filtrado, ti, tf
    return filtrado, filtrado["Timestamp"].min(), filtrado["Timestamp"].max()


def _interp_at(ts: pd.Series, serie: pd.Series, t_ref: pd.Timestamp) -> float:
    x = pd.to_datetime(ts).values.astype("datetime64[ns]").astype("int64")
    y = pd.to_numeric(serie, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return np.nan
    t_ref_i64 = np.datetime64(t_ref).astype("datetime64[ns]").astype("int64")
    if t_ref_i64 <= x.min():
        return float(y[0])
    if t_ref_i64 >= x.max():
        return float(y[-1])
    j = np.searchsorted(x, t_ref_i64)
    x0, x1 = x[j - 1], x[j]
    y0, y1 = y[j - 1], y[j]
    if x1 == x0:
        return float(y0)
    frac = (t_ref_i64 - x0) / (x1 - x0)
    return float(y0 + frac * (y1 - y0))


def _first_cross_time(series: pd.Series, ts: pd.Series, umbral: float, tipo: str, mask: pd.Series | None = None) -> pd.Timestamp | None:
    y = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    t = pd.to_datetime(ts).to_numpy()
    if mask is not None:
        mask_np = mask.to_numpy()
        y = y[mask_np]
        t = t[mask_np]
    if len(t) < 2:
        return None
    if tipo == PruebaTipo.MAX:
        a = y[:-1] <= umbral
        b = y[1:] > umbral
    else:
        a = y[:-1] >= umbral
        b = y[1:] < umbral
    idx = np.where(a & b)[0]
    if idx.size == 0:
        return None
    i = int(idx[0])
    y0, y1 = y[i], y[i + 1]
    t0, t1 = pd.Timestamp(t[i]), pd.Timestamp(t[i + 1])
    if np.isfinite(y0) and np.isfinite(y1) and (y1 != y0):
        frac = (umbral - y0) / (y1 - y0)
        frac = float(min(max(frac, 0.0), 1.0))
        return t0 + (t1 - t0) * frac
    return t0


def _tiempo_fuera(series: pd.Series, ts: pd.Series, umbral: float, tipo: str, mask: pd.Series | None = None) -> float:
    y = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    t = pd.to_datetime(ts).to_numpy()
    if mask is not None:
        mask_np = mask.to_numpy()
        y = y[mask_np]
        t = t[mask_np]
    if len(t) < 2:
        return 0.0
    fuera = (y > umbral) if tipo == PruebaTipo.MAX else (y < umbral)
    dt_sec = np.diff(t.astype("datetime64[ns]")).astype("timedelta64[s]").astype(float)
    if dt_sec.size == 0:
        return 0.0
    fuera_left = fuera[:-1]
    total_seconds = dt_sec[fuera_left].sum()
    return float(total_seconds) / 3600.0


def _first_cross_down_after(series: pd.Series, ts: pd.Series, umbral: float, t_ref: pd.Timestamp) -> pd.Timestamp | None:
    y = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    t = pd.to_datetime(ts).to_numpy()
    n = len(t)
    if n < 2:
        return None
    idx0 = int(np.searchsorted(t, np.datetime64(t_ref), side="left"))
    i_start = max(0, idx0 - 1)
    for i in range(i_start, n - 1):
        y0, y1 = y[i], y[i + 1]
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue
        t0, t1 = pd.Timestamp(t[i]), pd.Timestamp(t[i + 1])
        if (y0 > umbral) and (y1 <= umbral):
            if y1 != y0:
                frac = (umbral - y0) / (y1 - y0)
                frac = float(min(max(frac, 0.0), 1.0))
                t_cross = t0 + (t1 - t0) * frac
            else:
                t_cross = t0
            if t_cross >= t_ref:
                return t_cross
    return None


def fmt_hm(hours: float | None) -> str:
    if hours is None:
        return "—"
    sign = "-" if hours < 0 else ""
    total_min = int(round(abs(hours) * 60))
    h, m = divmod(total_min, 60)
    return f"{sign}{h:d}h {m:02d}m"


# -----------------------------------------------------------------------------
# Cálculos de métricas y eventos
# -----------------------------------------------------------------------------
@dataclass
class SensorStats:
    stats: pd.DataFrame
    resumen: dict


def calcular_estadisticas(df: pd.DataFrame, sensores: list[str]) -> SensorStats:
    registros: list[dict] = []
    for sensor in sensores:
        serie = to_numeric(df[sensor]).dropna()
        if serie.empty:
            continue
        max_val = float(serie.max())
        min_val = float(serie.min())
        mean_val = float(serie.mean())
        registros.append(
            {
                "Sensor": sensor,
                "Máximo": max_val,
                "Frecuencia Máx": int((serie == max_val).sum()),
                "Mínimo": min_val,
                "Frecuencia Mín": int((serie == min_val).sum()),
                "Promedio": mean_val,
                "Timestamp Máx": df.loc[serie.idxmax(), "Timestamp"],
                "Timestamp Mín": df.loc[serie.idxmin(), "Timestamp"],
            }
        )

    stats_df = pd.DataFrame(registros)
    if stats_df.empty:
        return SensorStats(stats_df, {})

    resumen: dict = {}
    idx_max = stats_df["Máximo"].idxmax()
    idx_min = stats_df["Mínimo"].idxmin()
    idx_avg_max = stats_df["Promedio"].idxmax()
    idx_avg_min = stats_df["Promedio"].idxmin()

    resumen["max"] = stats_df.loc[idx_max].to_dict()
    resumen["min"] = stats_df.loc[idx_min].to_dict()
    resumen["avg_max"] = stats_df.loc[idx_avg_max].to_dict()
    resumen["avg_min"] = stats_df.loc[idx_avg_min].to_dict()

    return SensorStats(stats_df, resumen)


def construir_eventos_extremos(df: pd.DataFrame, stats: SensorStats) -> pd.DataFrame:
    if df.empty or not stats.resumen:
        return pd.DataFrame(columns=["Sensor", "Fecha", "Hora", "Valor", "Tipo"])

    eventos: list[dict] = []
    valores = {
        "Máximo General": stats.resumen["max"]["Máximo"],
        "Mínimo General": stats.resumen["min"]["Mínimo"],
    }

    fecha_col = next((c for c in df.columns if str(c).lower().startswith("fecha")), None)
    hora_col = next((c for c in df.columns if str(c).lower().startswith("hora")), None)

    for sensor in stats.stats["Sensor"]:
        serie = to_numeric(df[sensor])
        for tipo, valor in valores.items():
            mask = np.isclose(serie, valor, equal_nan=False)
            if not mask.any():
                continue
            for idx in serie[mask].index:
                ts = df.loc[idx, "Timestamp"]
                fecha = df.loc[idx, fecha_col] if fecha_col else ts.date()
                hora = df.loc[idx, hora_col] if hora_col else ts.time().strftime("%H:%M:%S")
                eventos.append(
                    {
                        "Sensor": sensor,
                        "Fecha": pd.to_datetime(fecha).date() if not pd.isna(fecha) else ts.date(),
                        "Hora": str(hora),
                        "Valor": float(serie.loc[idx]),
                        "Tipo": tipo,
                    }
                )

    if not eventos:
        return pd.DataFrame(columns=["Sensor", "Fecha", "Hora", "Valor", "Tipo"])

    return pd.DataFrame(eventos).sort_values(["Sensor", "Tipo", "Fecha", "Hora"]).reset_index(drop=True)


def agrupar_eventos(eventos: pd.DataFrame, intervalo_minutos: int) -> pd.DataFrame:
    if eventos.empty:
        return pd.DataFrame(columns=["Sensor", "Fecha", "Hora (Rango)", "Valor Registrado", "Frecuencia", "Tipo"])

    tmp = eventos.copy()
    tmp["Fecha"] = pd.to_datetime(tmp["Fecha"], errors="coerce").dt.date
    tmp["Timestamp"] = pd.to_datetime(tmp["Fecha"].astype(str) + " " + tmp["Hora"], errors="coerce")
    tmp = tmp.dropna(subset=["Timestamp"]).sort_values(["Sensor", "Tipo", "Timestamp"])

    registros: list[dict] = []
    delta = pd.Timedelta(minutes=intervalo_minutos)
    for (sensor, tipo), bloque in tmp.groupby(["Sensor", "Tipo"]):
        timestamps = bloque["Timestamp"].to_list()
        inicio = 0
        for i in range(1, len(timestamps)):
            if timestamps[i] - timestamps[i - 1] > delta:
                sub = bloque.iloc[inicio:i]
                registros.append(_compactar_rango(sensor, tipo, sub))
                inicio = i
        sub_final = bloque.iloc[inicio:]
        registros.append(_compactar_rango(sensor, tipo, sub_final))

    return pd.DataFrame(registros).sort_values(["Sensor", "Tipo", "Fecha"]).reset_index(drop=True)


def _compactar_rango(sensor: str, tipo: str, bloque: pd.DataFrame) -> dict:
    if bloque.empty:
        return {
            "Sensor": sensor,
            "Fecha": None,
            "Hora (Rango)": "",
            "Valor Registrado": np.nan,
            "Frecuencia": 0,
            "Tipo": tipo,
        }
    ts_ini = bloque["Timestamp"].iloc[0]
    ts_fin = bloque["Timestamp"].iloc[-1]
    hora_ini = ts_ini.strftime("%H:%M:%S")
    hora_fin = ts_fin.strftime("%H:%M:%S")
    rango = hora_ini if hora_ini == hora_fin else f"{hora_ini}-{hora_fin}"
    return {
        "Sensor": sensor,
        "Fecha": ts_ini.date(),
        "Hora (Rango)": rango,
        "Valor Registrado": float(bloque["Valor"].iloc[0]),
        "Frecuencia": int(bloque.shape[0]),
        "Tipo": tipo,
    }


@dataclass
class PromedioAnalisis:
    promedio: pd.Series
    p99: float
    p01: float
    max_val: float
    min_val: float
    ts_max: pd.Timestamp
    ts_min: pd.Timestamp
    intervalos_altos: list[tuple[pd.Timestamp, pd.Timestamp]]
    intervalos_bajos: list[tuple[pd.Timestamp, pd.Timestamp]]


def analizar_promedio(df: pd.DataFrame, sensores: list[str]) -> PromedioAnalisis | None:
    if df.empty:
        return None
    valores = df[sensores].apply(to_numeric)
    promedio = valores.mean(axis=1, skipna=True)
    if promedio.dropna().empty:
        return None

    p99 = float(promedio.quantile(PERCENTIL_ALTO))
    p01 = float(promedio.quantile(PERCENTIL_BAJO))

    flat = valores.stack(dropna=False)
    if flat.dropna().empty:
        return None

    max_val = float(flat.max())
    min_val = float(flat.min())
    idx_max = flat.idxmax()
    idx_min = flat.idxmin()

    def _ts_from_idx(idx) -> pd.Timestamp:
        if idx is None or (isinstance(idx, float) and pd.isna(idx)):
            return pd.NaT
        if isinstance(idx, tuple):
            fila = idx[0]
        else:
            fila = idx
        if fila in df.index:
            return pd.to_datetime(df.loc[fila, "Timestamp"], errors="coerce")
        try:
            pos = int(fila)
        except Exception:
            return pd.NaT
        if 0 <= pos < len(df):
            return pd.to_datetime(df.iloc[pos]["Timestamp"], errors="coerce")
        return pd.NaT

    ts_max = _ts_from_idx(idx_max)
    ts_min = _ts_from_idx(idx_min)

    intervalos_altos = _intervalos_contiguos(df["Timestamp"], promedio >= p99)
    intervalos_bajos = _intervalos_contiguos(df["Timestamp"], promedio <= p01)

    return PromedioAnalisis(
        promedio=promedio,
        p99=p99,
        p01=p01,
        max_val=max_val,
        min_val=min_val,
        ts_max=ts_max,
        ts_min=ts_min,
        intervalos_altos=intervalos_altos,
        intervalos_bajos=intervalos_bajos,
    )


def _intervalos_contiguos(ts: pd.Series, mascara: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    valid = pd.to_datetime(ts[mascara], errors="coerce").dropna().sort_values()
    if valid.empty:
        return []
    delta = pd.Timedelta(minutes=1)
    rangos = []
    inicio = valid.iloc[0]
    anterior = inicio
    for actual in valid.iloc[1:]:
        if actual - anterior <= delta:
            anterior = actual
            continue
        rangos.append((inicio, anterior))
        inicio = actual
        anterior = actual
    rangos.append((inicio, anterior))
    return rangos


# -----------------------------------------------------------------------------
# Mapeo de sensores y niveles
# -----------------------------------------------------------------------------
def _bloque_metricas_map(
    titulo: str,
    valor: float,
    sensores: list[str],
    unidad: str,
    level_override: dict[str, int | None],
    level_meters: dict[int, float],
    decimales: int,
) -> list[str]:
    if not sensores or not np.isfinite(valor):
        return []
    lineas = [f"{titulo}: {fmt_num(valor, decimales)} {unidad}"]
    sensor_principal = sensores[0]
    lineas.append(f"Sensor: {sensor_principal}")
    lineas.append(f"Ubicación: {ubicacion_text(sensor_principal, level_override, level_meters)}")
    empatados = sensores[1:]
    if empatados:
        lineas.append("Empatados:")
        for sid in empatados:
            lineas.append(f"  - Sensor: {sid} | {ubicacion_text(sid, level_override, level_meters)}")
    return lineas


def construir_bloques_mapeo(
    df_temp: pd.DataFrame,
    df_hum: pd.DataFrame,
    sensores_temp: list[str],
    sensores_hum: list[str],
    level_override: dict[str, int | None],
    level_meters: dict[int, float],
    decimales: int,
) -> str:
    bloques: list[str] = []

    def _info_metricas(df: pd.DataFrame, sensores: list[str], incluir_mkt: bool) -> dict[str, tuple[float, list[str]]]:
        data = df.copy()
        data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")
        data = data.dropna(subset=["Timestamp"]).sort_values("Timestamp")
        if data.empty or not sensores:
            return {}
        valores = data[sensores].apply(to_numeric)

        maxima_col = valores.max(axis=0, skipna=True)
        max_val = float(maxima_col.max()) if maxima_col.notna().any() else np.nan
        sensores_max = maxima_col.index[maxima_col == max_val].tolist() if np.isfinite(max_val) else []

        minima_col = valores.min(axis=0, skipna=True)
        min_val = float(minima_col.min()) if minima_col.notna().any() else np.nan
        sensores_min = minima_col.index[minima_col == min_val].tolist() if np.isfinite(min_val) else []

        medias = valores.mean(axis=0, skipna=True)
        prom_max_val, sensores_prom_max, prom_min_val, sensores_prom_min = _empates_por_promedio(medias, decimales)

        mkt_max_val = mkt_min_val = np.nan
        sensores_mkt_max: list[str] = []
        sensores_mkt_min: list[str] = []
        if incluir_mkt:
            mkt_vals = {
                sensor: mkt_celsius(valores[sensor])
                for sensor in sensores
                if sensor in valores.columns and not valores[sensor].dropna().empty
            }
            if mkt_vals:
                serie = pd.Series(mkt_vals)
                mkt_max_val = float(serie.max())
                mkt_min_val = float(serie.min())
                serie_round = serie.round(decimales)
                sensores_mkt_max = serie_round.index[serie_round == serie_round.max()].tolist()
                sensores_mkt_min = serie_round.index[serie_round == serie_round.min()].tolist()

        return {
            "max": (max_val, sensores_max),
            "min": (min_val, sensores_min),
            "avg_max": (prom_max_val, sensores_prom_max),
            "avg_min": (prom_min_val, sensores_prom_min),
            "mkt_max": (mkt_max_val, sensores_mkt_max),
            "mkt_min": (mkt_min_val, sensores_mkt_min),
        }

    temp_info = _info_metricas(df_temp, sensores_temp, incluir_mkt=True) if sensores_temp else {}
    hum_info = _info_metricas(df_hum, sensores_hum, incluir_mkt=False) if sensores_hum else {}

    if temp_info:
        lineas = ["=== TEMPERATURA ==="]
        lineas += _bloque_metricas_map(
            "Temperatura Máxima",
            temp_info.get("max", (np.nan, []))[0],
            temp_info.get("max", (np.nan, []))[1],
            "°C",
            level_override,
            level_meters,
            decimales,
        )
        lineas.append("")
        lineas += _bloque_metricas_map(
            "Temperatura Mínima",
            temp_info.get("min", (np.nan, []))[0],
            temp_info.get("min", (np.nan, []))[1],
            "°C",
            level_override,
            level_meters,
            decimales,
        )
        lineas.append("")
        lineas += _bloque_metricas_map(
            "Temperatura Promedio Máx.",
            temp_info.get("avg_max", (np.nan, []))[0],
            temp_info.get("avg_max", (np.nan, []))[1],
            "°C",
            level_override,
            level_meters,
            decimales,
        )
        lineas.append("")
        lineas += _bloque_metricas_map(
            "Temperatura Promedio Mín.",
            temp_info.get("avg_min", (np.nan, []))[0],
            temp_info.get("avg_min", (np.nan, []))[1],
            "°C",
            level_override,
            level_meters,
            decimales,
        )
        lineas.append("")
        lineas += _bloque_metricas_map(
            "MKT Máx.",
            temp_info.get("mkt_max", (np.nan, []))[0],
            temp_info.get("mkt_max", (np.nan, []))[1],
            "°C",
            level_override,
            level_meters,
            decimales,
        )
        lineas.append("")
        lineas += _bloque_metricas_map(
            "MKT Mín.",
            temp_info.get("mkt_min", (np.nan, []))[0],
            temp_info.get("mkt_min", (np.nan, []))[1],
            "°C",
            level_override,
            level_meters,
            decimales,
        )
        bloques.append("\n".join([texto for texto in lineas if texto.strip()]))

    if hum_info:
        lineas = ["=== HUMEDAD ==="]
        lineas += _bloque_metricas_map(
            "Humedad Máxima",
            hum_info.get("max", (np.nan, []))[0],
            hum_info.get("max", (np.nan, []))[1],
            "%HR",
            level_override,
            level_meters,
            decimales,
        )
        lineas.append("")
        lineas += _bloque_metricas_map(
            "Humedad Mínima",
            hum_info.get("min", (np.nan, []))[0],
            hum_info.get("min", (np.nan, []))[1],
            "%HR",
            level_override,
            level_meters,
            decimales,
        )
        lineas.append("")
        lineas += _bloque_metricas_map(
            "Humedad Máx. Promedio",
            hum_info.get("avg_max", (np.nan, []))[0],
            hum_info.get("avg_max", (np.nan, []))[1],
            "%HR",
            level_override,
            level_meters,
            decimales,
        )
        lineas.append("")
        lineas += _bloque_metricas_map(
            "Humedad Mín. Promedio",
            hum_info.get("avg_min", (np.nan, []))[0],
            hum_info.get("avg_min", (np.nan, []))[1],
            "%HR",
            level_override,
            level_meters,
            decimales,
        )
        bloques.append("\n".join([texto for texto in lineas if texto.strip()]))

    if temp_info or hum_info:
        lineas = ["=== PUNTOS CRÍTICOS ==="]
        if temp_info.get("max") and temp_info["max"][1]:
            sensor = temp_info["max"][1][0]
            lineas.append("Punto crítico caliente")
            lineas.append(f"Sensor: {sensor}")
            lineas.append(f"Ubicación: {ubicacion_text(sensor, level_override, level_meters)}")
            lineas.append("")
        if temp_info.get("min") and temp_info["min"][1]:
            sensor = temp_info["min"][1][0]
            lineas.append("Punto crítico frío")
            lineas.append(f"Sensor: {sensor}")
            lineas.append(f"Ubicación: {ubicacion_text(sensor, level_override, level_meters)}")
            lineas.append("")
        if hum_info.get("max") and hum_info["max"][1]:
            sensor = hum_info["max"][1][0]
            lineas.append("Punto crítico humedad")
            lineas.append(f"Sensor: {sensor}")
            lineas.append(f"Ubicación: {ubicacion_text(sensor, level_override, level_meters)}")
        bloques.append("\n".join([texto for texto in lineas if texto.strip()]))

    return "\n\n".join([bloque for bloque in bloques if bloque]) if bloques else ""


# -----------------------------------------------------------------------------
# Gráficas (Matplotlib) y reportes de texto
# -----------------------------------------------------------------------------
def mkt_celsius(vals_c, Ea_kJ: float = 83.144) -> float:
    serie = pd.to_numeric(pd.Series(vals_c), errors="coerce")
    vals = serie.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    R_kJ = 0.008314462618
    Ea_over_R = Ea_kJ / R_kJ
    TK = vals + 273.15
    arr = np.exp(-Ea_over_R / TK)
    m = arr.mean()
    if not np.isfinite(m) or m <= 0:
        return np.nan
    TK_mkt = Ea_over_R / (-np.log(m))
    return TK_mkt - 273.15

def _string_list(valores: list[str]) -> str:
    return repr(valores) if valores else "[]"


def _tabla_texto(titulo: str, filas: list[dict], columnas: list[str]) -> str:
    if not filas:
        return f"{titulo}: (sin datos)"
    widths = [len(col) for col in columnas]
    for fila in filas:
        for i, col in enumerate(columnas):
            widths[i] = max(widths[i], len(str(fila.get(col, ""))))
    header = "  ".join(f"{col:<{widths[i]}}" for i, col in enumerate(columnas))
    separador = "  ".join("-" * widths[i] for i in range(len(columnas)))
    lines = [titulo, header, separador]
    for fila in filas:
        lines.append("  ".join(f"{str(fila.get(col, '')):<{widths[i]}}" for i, col in enumerate(columnas)))
    return "\n".join(lines)


def _compactar_eventos_valor(
    df: pd.DataFrame,
    sensores: list[str],
    objetivo: float,
    columna_valor: str,
    tolerancia: float = 1e-3,
    intervalo_minutos: int = DEFAULT_INTERVAL_MINUTES,
    decimales: int = DEFAULT_DECIMALES,
) -> list[dict]:
    if df.empty or not sensores or not np.isfinite(objetivo):
        return []
    registros: list[dict] = []
    data = df.sort_values("Timestamp")
    for _, row in data.iterrows():
        ts = row["Timestamp"]
        sensores_match = []
        for sensor in sensores:
            val = pd.to_numeric(row.get(sensor), errors="coerce")
            if pd.notna(val) and abs(val - objetivo) <= tolerancia:
                sensores_match.append(sensor)
        if sensores_match:
            registros.append(
                {
                    "Timestamp": ts,
                    columna_valor: float(pd.to_numeric(row[sensores_match[0]], errors="coerce")),
                    "sensores": tuple(sorted(set(sensores_match))),
                }
            )
    if not registros:
        return []
    eventos = pd.DataFrame(registros).sort_values("Timestamp")
    grupos = (
        eventos["Timestamp"].diff().fillna(pd.Timedelta(0))
        > pd.Timedelta(minutes=intervalo_minutos)
    ).cumsum()
    filas: list[dict] = []
    for _, bloque in eventos.groupby(grupos):
        ts_ini = bloque["Timestamp"].iloc[0]
        ts_fin = bloque["Timestamp"].iloc[-1]
        sensores_bloque = sorted({sid for grupo in bloque["sensores"] for sid in grupo})
        filas.append(
            {
                "fecha_inicio": format_date_es(ts_ini),
                "hora_inicio": ts_ini.strftime("%H:%M:%S"),
                "fecha_fin": format_date_es(ts_fin),
                "hora_fin": ts_fin.strftime("%H:%M:%S"),
                columna_valor: fmt_num(bloque[columna_valor].iloc[0], decimales),
                "n_registros": bloque.shape[0],
                "n_sensores": len(sensores_bloque),
                "sensores": ", ".join(sensores_bloque),
            }
        )
    return filas


def _empates_por_promedio(series_medias: pd.Series, decimales: int = DEFAULT_DECIMALES) -> tuple[float, list[str], float, list[str]]:
    if series_medias.dropna().empty:
        return np.nan, [], np.nan, []
    medias_r = series_medias.round(decimales)
    prom_max = float(medias_r.max())
    prom_min = float(medias_r.min())
    sensores_prom_max = sorted(medias_r.index[medias_r == prom_max].tolist())
    sensores_prom_min = sorted(medias_r.index[medias_r == prom_min].tolist())
    return prom_max, sensores_prom_max, prom_min, sensores_prom_min


def _resumen_generico(
    df: pd.DataFrame,
    sensores: list[str],
    unidad_label: str,
    incluir_mkt: bool = False,
    intervalo_minutos: int = DEFAULT_INTERVAL_MINUTES,
    decimales: int = DEFAULT_DECIMALES,
) -> tuple[str, list[dict], list[dict]]:
    if df.empty or not sensores:
        return "Sin datos", [], []

    datos = df.copy()
    datos["Timestamp"] = pd.to_datetime(datos["Timestamp"], errors="coerce")
    datos = datos.dropna(subset=["Timestamp"]).sort_values("Timestamp")

    ts = datos["Timestamp"]
    fecha_inicio = ts.min()
    fecha_final = ts.max()
    tiempo_total_horas = (
        (fecha_final - fecha_inicio).total_seconds() / 3600.0 if pd.notna(fecha_inicio) and pd.notna(fecha_final) else np.nan
    )

    valores = datos[sensores].apply(to_numeric)
    flat = valores.stack(dropna=False)
    promedio_general = float(flat.mean()) if not flat.dropna().empty else np.nan
    max_global = float(flat.max()) if not flat.dropna().empty else np.nan
    min_global = float(flat.min()) if not flat.dropna().empty else np.nan
    reps_max = int(np.isclose(valores, max_global, equal_nan=False).sum().sum()) if np.isfinite(max_global) else 0
    reps_min = int(np.isclose(valores, min_global, equal_nan=False).sum().sum()) if np.isfinite(min_global) else 0

    dif_max_instantanea = np.nan
    if not valores.dropna(how="all").empty:
        row_max = valores.max(axis=1, skipna=True)
        row_min = valores.min(axis=1, skipna=True)
        dif = (row_max - row_min).dropna()
        if not dif.empty:
            dif_max_instantanea = float(dif.max())

    medias = valores.mean(axis=0, skipna=True).dropna()
    if medias.empty:
        prom_max = prom_min = np.nan
        sensores_prom_max = []
        sensores_prom_min = []
    else:
        r = medias.round(decimales)
        prom_max = float(r.max())
        prom_min = float(r.min())
        sensores_prom_max = sorted(r.index[r == prom_max].tolist())
        sensores_prom_min = sorted(r.index[r == prom_min].tolist())

    resumen_lineas = ["=== Resumen ===", "Parametro                     : Resultado", "-------------------------------------------:"]

    def linea(nombre: str, valor) -> None:
        resumen_lineas.append(f"{nombre:<31}: {valor}")

    linea("fecha_hora_inicio", format_datetime_es(fecha_inicio))
    linea("fecha_hora_final", format_datetime_es(fecha_final))
    linea("tiempo_total_horas.", fmt_num(tiempo_total_horas, decimales) if np.isfinite(tiempo_total_horas) else "")
    linea("cantidad de sensores utilizados.", str(len(sensores)))
    linea(f"{unidad_label}_promedio.", fmt_num(promedio_general, decimales))
    linea(f"{unidad_label}_maxima.", fmt_num(max_global, decimales))
    linea(f"repeticiones_{unidad_label}_maxima.", str(reps_max))
    linea(f"{unidad_label}_minima.", fmt_num(min_global, decimales))
    linea(f"repeticiones_{unidad_label}_minima.", str(reps_min))
    linea("diferencia_maxima_instantanea.", fmt_num(dif_max_instantanea, decimales))
    linea(f"{unidad_label}_promedio_maxima.", fmt_num(prom_max, decimales))
    linea("sensores_promedio_maximo.", _string_list(sensores_prom_max))
    linea(f"{unidad_label}_promedio_minima.", fmt_num(prom_min, decimales))
    linea("sensores_promedio_minimo.", _string_list(sensores_prom_min))

    if incluir_mkt:
        mkt_vals = {sensor: mkt_celsius(valores[sensor]) for sensor in sensores if not valores[sensor].dropna().empty}
        if mkt_vals:
            mkt_series = pd.Series(mkt_vals)
            mkt_max = float(mkt_series.max())
            mkt_min = float(mkt_series.min())
            serie_round = mkt_series.round(decimales)
            sensores_mkt_max = sorted(serie_round.index[serie_round == serie_round.max()].tolist())
            sensores_mkt_min = sorted(serie_round.index[serie_round == serie_round.min()].tolist())
        else:
            mkt_max = mkt_min = np.nan
            sensores_mkt_max = sensores_mkt_min = []
        linea("mkt_maximo.", fmt_num(mkt_max, decimales))
        linea("sensores_mkt_maximo.", _string_list(sensores_mkt_max))
        linea("mkt_minimo.", fmt_num(mkt_min, decimales))
        linea("sensores_mkt_minimo.", _string_list(sensores_mkt_min))

    columna = "temperatura" if incluir_mkt else "humedad"
    tabla_max = _compactar_eventos_valor(
        datos,
        sensores,
        max_global,
        columna,
        intervalo_minutos=intervalo_minutos,
        decimales=decimales,
    )
    tabla_min = _compactar_eventos_valor(
        datos,
        sensores,
        min_global,
        columna,
        intervalo_minutos=intervalo_minutos,
        decimales=decimales,
    )

    return "\n".join(resumen_lineas), tabla_max, tabla_min


def grafico_boxplot(
    df: pd.DataFrame,
    sensores: list[str],
    titulo: str,
    unidad: str,
    lineas_umbral: list[tuple[float, str]] | None = None,
) -> plt.Figure:
    columnas = [sensor for sensor in sensores if sensor in df.columns]
    datos = df[columnas].apply(to_numeric, result_type="broadcast") if columnas else pd.DataFrame()
    datos = datos.dropna(axis=1, how="all").dropna(how="all")

    fig, ax = plt.subplots(figsize=(12, 6))
    if datos.empty:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Sin datos suficientes para boxplot",
            ha="center",
            va="center",
            fontsize=12,
        )
    else:
        sns.boxplot(data=datos, ax=ax)
        ax.set_ylabel(unidad)
        ax.tick_params(axis="x", rotation=90)
        agregar_lineas_umbral(ax, lineas_umbral)
        if lineas_umbral:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc="upper right")

    ax.set_title(titulo)
    fig.tight_layout()
    return fig


def grafico_tendencias(
    df: pd.DataFrame,
    sensores: list[str],
    titulo: str,
    unidad: str,
    mostrar_promedio: bool = True,
    lineas_umbral: list[tuple[float, str]] | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 6))
    ts = df["Timestamp"]
    for sensor in sensores:
        serie_plot = _series_for_plot(df[sensor])
        ax.plot(ts, serie_plot, linewidth=0.6, label=sensor)

    if mostrar_promedio:
        promedio = df[sensores].apply(to_numeric).mean(axis=1, skipna=True)
        promedio_plot = _series_for_plot(promedio)
        ax.plot(ts, promedio_plot, color="black", linestyle="--", linewidth=2, label="Promedio General")

    agregar_lineas_umbral(ax, lineas_umbral)
    ax.set_title(titulo)
    ax.set_xlabel("Periodo de tiempo")
    ax.set_ylabel(unidad)
    ax.xaxis.set_major_formatter(matplotlib_date_formatter(include_seconds=False))
    ax.tick_params(axis="x", rotation=90)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    return fig


def grafico_destacados(
    df: pd.DataFrame,
    stats: SensorStats,
    analisis: PromedioAnalisis | None,
    titulo: str,
    unidad: str,
    lineas_umbral: list[tuple[float, str]] | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 6))
    ts = df["Timestamp"]
    sensores = [s for s in stats.stats["Sensor"].tolist() if s in df.columns]

    for sensor in sensores:
        serie_plot = _series_for_plot(df[sensor])
        ax.plot(ts, serie_plot, color="#C0C0C0", linewidth=0.6, alpha=0.6)

    colores = {
        "max": "darkred",
        "min": "royalblue",
        "avg_max": "magenta",
        "avg_min": "teal",
    }
    for clave, color in colores.items():
        info = stats.resumen.get(clave)
        if not info:
            continue
        sensor = info.get("Sensor")
        if sensor not in df.columns:
            continue
        serie_dest = _series_for_plot(df[sensor])
        ax.plot(ts, serie_dest, color=color, linewidth=1.6, label=f"{clave.upper()}: {sensor}")

    if analisis:
        promedio_plot = _series_for_plot(analisis.promedio)
        ax.plot(ts, promedio_plot, color="black", linestyle="--", linewidth=2, label="Promedio General")
        ax.axhline(analisis.max_val, color="red", linestyle=":", linewidth=1.4, label=f"Máximo {analisis.max_val:.1f}{unidad}")
        ax.axhline(analisis.min_val, color="blue", linestyle=":", linewidth=1.4, label=f"Mínimo {analisis.min_val:.1f}{unidad}")

    agregar_lineas_umbral(ax, lineas_umbral)
    ax.set_title(titulo)
    ax.set_xlabel("Periodo de tiempo")
    ax.set_ylabel(unidad)
    ax.xaxis.set_major_formatter(matplotlib_date_formatter(include_seconds=False))
    ax.tick_params(axis="x", rotation=90)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    return fig


def grafico_promedio_intervalos(
    df: pd.DataFrame,
    analisis: PromedioAnalisis,
    titulo: str,
    unidad: str,
    lineas_umbral: list[tuple[float, str]] | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 6))
    ts = df["Timestamp"]
    promedio = analisis.promedio

    promedio_plot = _series_for_plot(promedio)
    ax.plot(ts, promedio_plot, color="black", linewidth=1.5, label="Promedio General")
    if pd.notna(analisis.ts_max):
        idx_max = (ts - analisis.ts_max).abs().idxmin()
        ax.plot(analisis.ts_max, promedio.loc[idx_max], "ro", label=f"Máximo {analisis.max_val:.1f}{unidad}")
    if pd.notna(analisis.ts_min):
        idx_min = (ts - analisis.ts_min).abs().idxmin()
        ax.plot(analisis.ts_min, promedio.loc[idx_min], "bo", label=f"Mínimo {analisis.min_val:.1f}{unidad}")

    for i, (ini, fin) in enumerate(analisis.intervalos_altos):
        ax.axvspan(ini, fin, color="red", alpha=0.18, label="Intervalo promedio alto" if i == 0 else "")
    for i, (ini, fin) in enumerate(analisis.intervalos_bajos):
        ax.axvspan(ini, fin, color="blue", alpha=0.15, label="Intervalo promedio bajo" if i == 0 else "")

    agregar_lineas_umbral(ax, lineas_umbral)
    ax.set_title(titulo)
    ax.set_xlabel("Periodo de tiempo")
    ax.set_ylabel(unidad)
    ax.xaxis.set_major_formatter(matplotlib_date_formatter(include_seconds=False))
    ax.tick_params(axis="x", rotation=90)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    return fig


# -----------------------------------------------------------------------------
# Render de reportes
# -----------------------------------------------------------------------------
@dataclass
class AnalisisPrueba:
    df_contexto: pd.DataFrame
    df_rango: pd.DataFrame
    mask_rango: pd.Series
    sensores: list[str]
    umbral: float
    tipo: str
    t_ini_clip: pd.Timestamp
    t_fin_clip: pd.Timestamp
    cruces: dict[str, pd.Timestamp]
    tiempos_fuera: dict[str, float]
    sensor_primero: str | None
    t_primera: pd.Timestamp | None
    cruces_down_after: dict[str, pd.Timestamp]
    retardos_horas: dict[str, float]
    sensor_baja_primero: str | None
    sensor_baja_ultimo: str | None
    t_baja_primero: pd.Timestamp | None
    t_baja_ultimo: pd.Timestamp | None


def analizar_prueba(
    df_contexto: pd.DataFrame,
    sensores: list[str],
    t_ini_clip: pd.Timestamp,
    t_fin_clip: pd.Timestamp,
    umbral: float,
    tipo: str,
) -> AnalisisPrueba | None:
    if t_ini_clip is None or t_fin_clip is None:
        return None

    data = df_contexto.copy()
    data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")
    data = data.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    if data.empty:
        return None

    mask_rango = (data["Timestamp"] >= t_ini_clip) & (data["Timestamp"] <= t_fin_clip)
    df_rango = data.loc[mask_rango].copy()
    if df_rango.empty:
        return None

    cruces: dict[str, pd.Timestamp] = {}
    tiempos_fuera: dict[str, float] = {}
    for sensor in sensores:
        if sensor not in data.columns:
            continue
        serie = data[sensor]
        t_cross = _first_cross_time(serie, data["Timestamp"], umbral, tipo, mask_rango)
        if t_cross is not None:
            cruces[sensor] = t_cross
        tiempos_fuera[sensor] = _tiempo_fuera(serie, data["Timestamp"], umbral, tipo, mask_rango)

    sensor_primero = min(cruces, key=cruces.get) if cruces else None
    t_primera = cruces.get(sensor_primero) if sensor_primero else None

    t_ref = t_fin_clip
    cruces_down_after: dict[str, pd.Timestamp] = {}
    retardos_horas: dict[str, float] = {}
    for sensor in sensores:
        if sensor not in data.columns:
            continue
        t_cross = _first_cross_down_after(data[sensor], data["Timestamp"], umbral, t_ref)
        if t_cross is not None:
            cruces_down_after[sensor] = t_cross
            retardos_horas[sensor] = (t_cross - t_ref).total_seconds() / 3600.0

    sensor_baja_primero = min(cruces_down_after, key=cruces_down_after.get) if cruces_down_after else None
    sensor_baja_ultimo = max(cruces_down_after, key=cruces_down_after.get) if cruces_down_after else None
    t_baja_primero = cruces_down_after.get(sensor_baja_primero) if sensor_baja_primero else None
    t_baja_ultimo = cruces_down_after.get(sensor_baja_ultimo) if sensor_baja_ultimo else None

    return AnalisisPrueba(
        df_contexto=data,
        df_rango=df_rango,
        mask_rango=mask_rango,
        sensores=[s for s in sensores if s in data.columns],
        umbral=umbral,
        tipo=tipo,
        t_ini_clip=t_ini_clip,
        t_fin_clip=t_fin_clip,
        cruces=cruces,
        tiempos_fuera=tiempos_fuera,
        sensor_primero=sensor_primero,
        t_primera=t_primera,
        cruces_down_after=cruces_down_after,
        retardos_horas=retardos_horas,
        sensor_baja_primero=sensor_baja_primero,
        sensor_baja_ultimo=sensor_baja_ultimo,
        t_baja_primero=t_baja_primero,
        t_baja_ultimo=t_baja_ultimo,
    )


def render_bloque(
    nombre: str,
    df: pd.DataFrame,
    sensores: list[str],
    unidad: str,
    intervalo_minutos: int,
    decimales: int,
    titulo_contexto: str = "",
    lineas_umbral: list[tuple[float, str]] | None = None,
    aplicar_umbrales: bool = True,
) -> None:
    if df.empty or not sensores:
        st.info(f"Sin datos de {nombre.lower()} para mostrar.")
        return

    stats = calcular_estadisticas(df, sensores)
    es_temperatura = nombre.lower().startswith("temperatura")
    etiqueta = "temperatura" if es_temperatura else "humedad"
    resumen_texto, tabla_max, tabla_min = _resumen_generico(
        df,
        sensores,
        etiqueta,
        incluir_mkt=es_temperatura,
        intervalo_minutos=intervalo_minutos,
        decimales=decimales,
    )

    st.text(resumen_texto)

    titulo_max = f"Tabla Repetición Máxima ({nombre} compactada)"
    titulo_min = f"Tabla Repetición Mínima ({nombre} compactada)"
    columnas = [
        "fecha_inicio",
        "hora_inicio",
        "fecha_fin",
        "hora_fin",
        etiqueta,
        "n_registros",
        "n_sensores",
        "sensores",
    ]
    st.text(_tabla_texto(titulo_max, tabla_max, columnas))
    st.text(_tabla_texto(titulo_min, tabla_min, columnas))

    titulo_box = titulo_con_contexto(f"Distribución de {nombre.lower()} por sensor", titulo_contexto)
    fig_box = grafico_boxplot(df, sensores, titulo_box, unidad, lineas_umbral=None)
    st.pyplot(fig_box)
    plt.close(fig_box)

    titulo_tend = titulo_con_contexto(f"Tendencia de {nombre.lower()}", titulo_contexto)
    fig_trend = grafico_tendencias(
        df,
        sensores,
        titulo_tend,
        unidad,
        lineas_umbral=lineas_umbral if aplicar_umbrales else None,
    )
    st.pyplot(fig_trend)
    plt.close(fig_trend)

    analisis = analizar_promedio(df, sensores)
    if analisis and not stats.stats.empty:
        titulo_dest = titulo_con_contexto(f"Sensores destacados ({nombre.lower()})", titulo_contexto)
        fig_dest = grafico_destacados(
            df,
            stats,
            analisis,
            titulo_dest,
            unidad,
            lineas_umbral=None,
        )
        st.pyplot(fig_dest)
        plt.close(fig_dest)
        titulo_prom = titulo_con_contexto(f"Promedio general de {nombre.lower()}", titulo_contexto)
        fig_avg = grafico_promedio_intervalos(
            df,
            analisis,
            titulo_prom,
            unidad,
            lineas_umbral=None,
        )
        st.pyplot(fig_avg)
        plt.close(fig_avg)
        st.caption(
            " | ".join(
                [
                    f"Percentil 99: {fmt_num(analisis.p99, decimales)}{unidad}",
                    f"Percentil 1: {fmt_num(analisis.p01, decimales)}{unidad}",
                    f"Máximo global: {fmt_num(analisis.max_val, decimales)}{unidad}",
                    f"Mínimo global: {fmt_num(analisis.min_val, decimales)}{unidad}",
                ]
            )
        )


def render_seccion(
    titulo: str,
    df_temp: pd.DataFrame,
    df_hum: pd.DataFrame,
    sensores_temp: list[str],
    sensores_hum: list[str],
    unidad_temp: str,
    unidad_hum: str,
    periodo_temp: tuple[pd.Timestamp | None, pd.Timestamp | None],
    periodo_hum: tuple[pd.Timestamp | None, pd.Timestamp | None],
    intervalo_minutos: int,
    level_override: dict[str, int | None],
    level_meters: dict[int, float],
    decimales: int,
    titulo_contexto: str = "",
    umbrales_temp: list[float] | None = None,
    umbral_hum_max: float | None = None,
    grupos_temp: dict[str, list[str]] | None = None,
    grupos_hum: dict[str, list[str]] | None = None,
    mostrar_humedad: bool = True,
) -> None:
    st.header(titulo)

    st.markdown("### Temperatura")
    if periodo_temp[0] is not None and periodo_temp[1] is not None:
        st.caption(f"Periodo: {format_datetime_es(periodo_temp[0])} → {format_datetime_es(periodo_temp[1])}")
    contexto_general = contexto_con_division(titulo_contexto, "General")
    lineas_temp = lineas_umbral_temperatura(umbrales_temp or [])
    render_bloque(
        "Temperatura",
        df_temp,
        sensores_temp,
        unidad_temp,
        intervalo_minutos,
        decimales,
        contexto_general,
        lineas_temp,
        aplicar_umbrales=True,
    )

    if grupos_temp:
        st.markdown("#### Temperatura por área")
        tabs_temp = st.tabs([NOMBRES_GRUPOS.get(g, g.title()) for g in grupos_temp])
        for tab, (grupo_nombre, sensores_grupo) in zip(tabs_temp, grupos_temp.items()):
            with tab:
                division_label = NOMBRES_GRUPOS.get(grupo_nombre, grupo_nombre.title())
                contexto_grupo = contexto_con_division(titulo_contexto, division_label)
                render_bloque(
                    f"Temperatura ({division_label})",
                    df_temp,
                    sensores_grupo,
                    unidad_temp,
                    intervalo_minutos,
                    decimales,
                    contexto_grupo,
                    lineas_temp,
                    aplicar_umbrales=False,
                )

    if mostrar_humedad and sensores_hum:
        st.markdown("### Humedad")
        if periodo_hum[0] is not None and periodo_hum[1] is not None:
            st.caption(f"Periodo: {format_datetime_es(periodo_hum[0])} → {format_datetime_es(periodo_hum[1])}")
        lineas_hum = lineas_umbral_humedad(umbral_hum_max)
        render_bloque(
            "Humedad",
            df_hum,
            sensores_hum,
            unidad_hum,
            intervalo_minutos,
            decimales,
            contexto_general,
            lineas_hum,
            aplicar_umbrales=bool(lineas_hum),
        )

        if grupos_hum:
            st.markdown("#### Humedad por área")
            tabs_hum = st.tabs([NOMBRES_GRUPOS.get(g, g.title()) for g in grupos_hum])
            for tab, (grupo_nombre, sensores_grupo) in zip(tabs_hum, grupos_hum.items()):
                with tab:
                    division_label = NOMBRES_GRUPOS.get(grupo_nombre, grupo_nombre.title())
                    contexto_grupo = contexto_con_division(titulo_contexto, division_label)
                    render_bloque(
                        f"Humedad ({division_label})",
                        df_hum,
                        sensores_grupo,
                        unidad_hum,
                        intervalo_minutos,
                        decimales,
                        contexto_grupo,
                        lineas_hum,
                        aplicar_umbrales=False,
                    )

    bloques_mapeo = construir_bloques_mapeo(
        df_temp,
        df_hum,
        sensores_temp,
        sensores_hum,
        level_override or {},
        level_meters or {},
        decimales,
    )
    if bloques_mapeo:
        st.markdown("### Mapeo de sensores y puntos críticos")
        st.text(bloques_mapeo)


def plot_prueba_sensor_principal(analisis: AnalisisPrueba, titulo_contexto: str = "") -> plt.Figure | None:
    if not analisis.sensor_primero or analisis.t_primera is None:
        return None
    sensor = analisis.sensor_primero
    data = analisis.df_contexto.sort_values("Timestamp")
    ts = data["Timestamp"]
    y = to_numeric(data[sensor])
    y_plot = _series_for_plot(data[sensor])
    y_ini = _interp_at(ts, y, analisis.t_ini_clip)
    y_fin = _interp_at(ts, y, analisis.t_fin_clip)

    fig, ax = plt.subplots(figsize=(14.5, 6.5))
    ax.plot(ts, y_plot, label=sensor, linewidth=1.2)
    mask_rango = analisis.mask_rango.reindex(data.index, fill_value=False)
    ax.plot(ts[mask_rango], y_plot[mask_rango], linewidth=2.6, alpha=0.95)

    ax.axhline(analisis.umbral, linestyle=":", linewidth=1.6)
    ax.text(ts.iloc[0], analisis.umbral, f"  Umbral = {fmt_num(analisis.umbral, DEFAULT_DECIMALES)}", va="bottom")
    ax.axvline(analisis.t_ini_clip, linestyle="--", linewidth=1.2)
    ax.axvline(analisis.t_fin_clip, linestyle="--", linewidth=1.2)

    ax.scatter([analisis.t_ini_clip], [y_ini], s=50, zorder=5)
    ax.annotate(
        f"Inicio\n{format_datetime_es(analisis.t_ini_clip, include_seconds=False)}\n{fmt_num(y_ini, DEFAULT_DECIMALES)}",
        xy=(analisis.t_ini_clip, y_ini),
        xytext=(15, 15),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->"),
    )

    ax.scatter([analisis.t_fin_clip], [y_fin], s=50, zorder=5)
    ax.annotate(
        f"Fin\n{format_datetime_es(analisis.t_fin_clip, include_seconds=False)}\n{fmt_num(y_fin, DEFAULT_DECIMALES)}",
        xy=(analisis.t_fin_clip, y_fin),
        xytext=(15, -35),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->"),
    )

    ax.scatter([analisis.t_primera], [_interp_at(ts, y, analisis.t_primera)], s=50, zorder=6)
    ax.annotate(
        f"Primer cruce\n{format_datetime_es(analisis.t_primera, include_seconds=False)}",
        xy=(analisis.t_primera, _interp_at(ts, y, analisis.t_primera)),
        xytext=(15, 15),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->"),
    )

    ax.set_title(titulo_con_contexto("Sensor que cruza primero el umbral", titulo_contexto))
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Temperatura (°C)")
    ax.xaxis.set_major_formatter(matplotlib_date_formatter(include_seconds=False))
    ax.tick_params(axis="x", rotation=90)
    resaltar_periodo_prueba(
        ax,
        ts,
        analisis.t_ini_clip,
        analisis.t_fin_clip,
        color_span="tab:orange",
        legend_kwargs={"loc": "upper left"},
    )
    fig.tight_layout()
    return fig


def plot_prueba_rango(analisis: AnalisisPrueba, titulo_contexto: str = "") -> plt.Figure | None:
    sensores_zoom = [s for s, _, horas in sorted(
        [(s, analisis.cruces[s], analisis.tiempos_fuera.get(s, 0.0)) for s in analisis.cruces],
        key=lambda x: x[1]
    ) if analisis.tiempos_fuera.get(s, 0.0) > 0]
    if not sensores_zoom:
        return None

    data = analisis.df_contexto.sort_values("Timestamp")
    ts = data["Timestamp"]
    pad = pd.Timedelta(minutes=5)
    t0 = max(analisis.t_ini_clip - pad, ts.min())
    t1 = min(analisis.t_fin_clip + pad, ts.max())
    mask_zoom = (ts >= t0) & (ts <= t1)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    for sensor in sensores_zoom:
        serie_plot = _series_for_plot(data[sensor])
        ax.plot(ts[mask_zoom], serie_plot[mask_zoom], linewidth=1.6, label=sensor)
        tcr = analisis.cruces.get(sensor)
        if tcr and t0 <= tcr <= t1:
            ax.axvline(tcr, linestyle="-.", linewidth=1.0, alpha=0.85)

    ax.axhline(analisis.umbral, linestyle=":", linewidth=1.4)
   
    ax.axvline(analisis.t_ini_clip, linestyle="--", linewidth=1.1)
    ax.axvline(analisis.t_fin_clip, linestyle="--", linewidth=1.1)

    dur_min = max(1, int(round((t1 - t0).total_seconds() / 60)))
    if dur_min <= 60:
        step = 5
    elif dur_min <= 180:
        step = 10
    elif dur_min <= 360:
        step = 15
    else:
        step = 30
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=step))
    ax.xaxis.set_major_formatter(matplotlib_date_formatter(include_seconds=False))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=max(1, step // 2)))
    ax.grid(True, which="major", axis="x", alpha=0.25)
    ax.set_xlim(t0, t1)
    ax.set_title(titulo_con_contexto("Sensores fuera del umbral dentro del rango", titulo_contexto))
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Temperatura (°C)")
    plt.xticks(rotation=90, ha="right")
    resaltar_periodo_prueba(
        ax,
        ts[mask_zoom],
        analisis.t_ini_clip,
        analisis.t_fin_clip,
        color_span="tab:orange",
        legend_kwargs={
            "bbox_to_anchor": (1.02, 1),
            "loc": "upper left",
            "borderaxespad": 0.0,
            "title": "Orden de cruce",
        },
    )
    fig.tight_layout(rect=[0, 0, 0.86, 1])
    return fig


def plot_prueba_descenso(analisis: AnalisisPrueba, titulo_contexto: str = "") -> plt.Figure | None:
    sensores = []
    if analisis.sensor_baja_primero:
        sensores.append(analisis.sensor_baja_primero)
    if analisis.sensor_baja_ultimo and analisis.sensor_baja_ultimo != analisis.sensor_baja_primero:
        sensores.append(analisis.sensor_baja_ultimo)
    if not sensores:
        return None

    data = analisis.df_contexto.sort_values("Timestamp")
    ts = data["Timestamp"]

    fig, ax = plt.subplots(figsize=(14.5, 6.5))
    for sensor in sensores:
        serie_plot = _series_for_plot(data[sensor])
        ax.plot(ts, serie_plot, linewidth=1.6, label=sensor)

    ax.axhline(analisis.umbral, linestyle=":", linewidth=1.6)
    ax.text(ts.min(), analisis.umbral, f"  Umbral = {fmt_num(analisis.umbral, DEFAULT_DECIMALES)}", va="bottom")
    ax.axvline(analisis.t_ini_clip, linestyle="--", linewidth=1.2)
    ax.axvline(analisis.t_fin_clip, linestyle="--", linewidth=1.2)

    if analisis.t_baja_primero is not None and analisis.sensor_baja_primero:
        y_near = data.loc[(ts - analisis.t_baja_primero).abs().idxmin(), analisis.sensor_baja_primero]
        ax.scatter([analisis.t_baja_primero], [y_near], s=35, zorder=5)
        ax.axvline(analisis.t_baja_primero, linestyle="-.", linewidth=1.1)
        ax.annotate(
            f"Baja 1º: {analisis.sensor_baja_primero}\n{format_datetime_es(analisis.t_baja_primero, include_seconds=False)}\nΔ={fmt_hm(analisis.retardos_horas.get(analisis.sensor_baja_primero))}",
            xy=(analisis.t_baja_primero, y_near),
            xytext=(15, 15),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->"),
        )

    if analisis.t_baja_ultimo is not None and analisis.sensor_baja_ultimo:
        y_near = data.loc[(ts - analisis.t_baja_ultimo).abs().idxmin(), analisis.sensor_baja_ultimo]
        ax.scatter([analisis.t_baja_ultimo], [y_near], s=35, zorder=5)
        ax.axvline(analisis.t_baja_ultimo, linestyle="-.", linewidth=1.1)
        ax.annotate(
            f"Baja último: {analisis.sensor_baja_ultimo}\n{format_datetime_es(analisis.t_baja_ultimo, include_seconds=False)}\nΔ={fmt_hm(analisis.retardos_horas.get(analisis.sensor_baja_ultimo))}",
            xy=(analisis.t_baja_ultimo, y_near),
            xytext=(15, -35),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->"),
        )

    ax.set_xlim(ts.min(), ts.max())
    ax.set_title(titulo_con_contexto("Sensores que regresan dentro del límite", titulo_contexto))
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Temperatura (°C)")
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(matplotlib_date_formatter(include_seconds=False))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0, 30]))
    ax.grid(True, which="major", axis="x", alpha=0.2)
    plt.xticks(rotation=90, ha="right")
    resaltar_periodo_prueba(
        ax,
        ts,
        analisis.t_ini_clip,
        analisis.t_fin_clip,
        color_span="tab:orange",
        legend_kwargs={"bbox_to_anchor": (1.02, 1), "loc": "upper left", "borderaxespad": 0.0},
    )
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    return fig


def _grafico_prueba_full(
    df: pd.DataFrame,
    sensores: list[str],
    t_ini: pd.Timestamp,
    t_fin: pd.Timestamp,
    unidad: str,
    titulo: str,
    color_span: str,
    decimales: int = DEFAULT_DECIMALES,
    lineas_umbral: list[tuple[float, str]] | None = None,
) -> plt.Figure | None:
    if df.empty or not sensores or t_ini is None or t_fin is None:
        return None

    data = df.copy()
    data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")
    data = data.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    if data.empty:
        return None

    ts = data["Timestamp"]
    fig, ax = plt.subplots(figsize=(14.5, 6.5))
    for sensor in sensores:
        if sensor not in data.columns:
            continue
        serie_plot = _series_for_plot(data[sensor])
        ax.plot(ts, serie_plot, linewidth=0.8, alpha=0.9, label=sensor)

    valores = data[sensores].apply(to_numeric)
    promedio = valores.mean(axis=1, skipna=True)
    if not promedio.dropna().empty:
        promedio_plot = _series_for_plot(promedio)
        ax.plot(ts, promedio_plot, color="black", linestyle="--", linewidth=2.0, label="Promedio General")
    else:
        promedio_plot = promedio

    def _anotar(punto: pd.Timestamp, etiqueta: str) -> None:
        if promedio_plot.dropna().empty:
            return
        y_val = _interp_at(ts, promedio_plot, punto)
        if not np.isfinite(y_val):
            return
        ax.scatter([punto], [y_val], s=55, zorder=5, color=color_span)
        lado = "right" if etiqueta == "Fin" else "left"
        dx = 12 if lado == "right" else -12
        ax.annotate(
            f"{etiqueta}\n{format_datetime_es(punto, include_seconds=False)}\n{fmt_num(y_val, decimales)}{unidad}",
            xy=(punto, y_val),
            xytext=(dx, 18),
            textcoords="offset points",
            ha="left" if lado == "right" else "right",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
            arrowprops=dict(arrowstyle="->", lw=1),
        )

    _anotar(t_ini, "Inicio")
    _anotar(t_fin, "Fin")

    agregar_lineas_umbral(ax, lineas_umbral)
    ax.set_title(titulo)
    ax.set_xlabel("Periodo de tiempo")
    ax.set_ylabel(unidad)
    ax.xaxis.set_major_formatter(matplotlib_date_formatter(include_seconds=False))
    ax.tick_params(axis="x", rotation=90)
    ax.set_xlim(ts.min(), ts.max())
    resaltar_periodo_prueba(
        ax,
        ts,
        t_ini,
        t_fin,
        color_span=color_span,
        legend_kwargs={"bbox_to_anchor": (1.02, 1), "loc": "upper left", "borderaxespad": 0.},
    )
    ax.grid(True, axis="x", alpha=0.15)
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    return fig


def grafico_temp_prueba_full(
    df_temp: pd.DataFrame,
    sensores_temp: list[str],
    t_ini: pd.Timestamp,
    t_fin: pd.Timestamp,
    decimales: int = DEFAULT_DECIMALES,
    titulo_contexto: str = "",
    lineas_umbral: list[tuple[float, str]] | None = None,
) -> plt.Figure | None:
    titulo = titulo_con_contexto("Tendencia de temperatura — Ventana de prueba", titulo_contexto)
    return _grafico_prueba_full(
        df_temp,
        sensores_temp,
        t_ini,
        t_fin,
        "°C",
        titulo,
        color_span="tab:orange",
        decimales=decimales,
        lineas_umbral=lineas_umbral,
    )


def grafico_hum_prueba_full(
    df_hum: pd.DataFrame,
    sensores_hum: list[str],
    t_ini: pd.Timestamp,
    t_fin: pd.Timestamp,
    decimales: int = DEFAULT_DECIMALES,
    titulo_contexto: str = "",
    lineas_umbral: list[tuple[float, str]] | None = None,
) -> plt.Figure | None:
    titulo = titulo_con_contexto("Tendencia de humedad — Ventana de prueba", titulo_contexto)
    return _grafico_prueba_full(
        df_hum,
        sensores_hum,
        t_ini,
        t_fin,
        "%HR",
        titulo,
        color_span="tab:blue",
        decimales=decimales,
        lineas_umbral=lineas_umbral,
    )


def render_prueba_umbrales(
    analisis_general: AnalisisPrueba | None,
    decimales: int,
    contexto_general: str = "",
    analisis_por_division: dict[str, AnalisisPrueba] | None = None,
    contextos_division: dict[str, str] | None = None,
) -> None:
    bloques: list[tuple[str, str, AnalisisPrueba]] = []
    if analisis_general is not None:
        bloques.append(("General", contexto_general, analisis_general))
    if analisis_por_division:
        for clave, analisis_div in analisis_por_division.items():
            if analisis_div is None:
                continue
            etiqueta = NOMBRES_GRUPOS.get(clave, clave.title())
            contexto_div = (contextos_division or {}).get(clave)
            if not contexto_div:
                contexto_div = contexto_con_division(contexto_general, etiqueta)
            bloques.append((etiqueta, contexto_div, analisis_div))

    if not bloques:
        st.info("Selecciona un rango válido y configura el umbral para ver el análisis de prueba.")
        return

    st.markdown("### Análisis de umbral (Prueba)")
    for idx, (etiqueta, contexto, analisis_actual) in enumerate(bloques):
        if idx > 0:
            st.markdown(f"#### División: {etiqueta}")
        _render_prueba_umbrales_detalle(analisis_actual, decimales, contexto, etiqueta)


def _render_prueba_umbrales_detalle(
    analisis: AnalisisPrueba,
    decimales: int,
    contexto: str,
    division_label: str,
) -> None:
    fmt = lambda valor: fmt_num(valor, decimales)

    st.markdown(f"**División analizada:** {division_label}")
    st.caption(
        " | ".join(
            [
                f"Umbral = {fmt(analisis.umbral)} °C",
                "Condición: mayor que" if analisis.tipo == PruebaTipo.MAX else "Condición: menor que",
                f"Periodo: {format_datetime_es(analisis.t_ini_clip, include_seconds=False)} → {format_datetime_es(analisis.t_fin_clip, include_seconds=False)}",
            ]
        )
    )

    fig_first = plot_prueba_sensor_principal(analisis, contexto)
    if fig_first is not None:
        st.pyplot(fig_first)
        plt.close(fig_first)
    else:
        st.info("Ningún sensor cruza el umbral en el periodo seleccionado.")

    fig_range = plot_prueba_rango(analisis, contexto)
    if fig_range is not None:
        st.pyplot(fig_range)
        plt.close(fig_range)

    fig_descenso = plot_prueba_descenso(analisis, contexto)
    if fig_descenso is not None:
        st.pyplot(fig_descenso)
        plt.close(fig_descenso)

    filas = []
    for sensor, horas in analisis.tiempos_fuera.items():
        t_cruce = analisis.cruces.get(sensor)
        if t_cruce is None or not (np.isfinite(horas) and horas > 0):
            continue
        filas.append(
            {
                "Sensor": sensor,
                "Primer cruce": t_cruce,
                "Tiempo fuera (h)": horas,
                "Tiempo fuera": fmt_hm(horas),
            }
        )
    tabla_fuera = pd.DataFrame(filas)
    if not tabla_fuera.empty:
        tabla_fuera = tabla_fuera.sort_values("Primer cruce")
        primero = tabla_fuera.iloc[0]
        ultimo = tabla_fuera.iloc[-1]
        t_inicio = analisis.t_ini_clip
        t_fin = analisis.t_fin_clip
        filas_salida = [
            {
                "Sensor": fila["Sensor"],
                "Salida": format_datetime_es(pd.to_datetime(fila["Primer cruce"], errors="coerce"), include_seconds=False),
                "Δ fuera": fmt_hm(fila["Tiempo fuera (h)"]),
                "Δ inicio": fmt_hm((pd.to_datetime(fila["Primer cruce"], errors="coerce") - t_inicio).total_seconds() / 3600.0) if pd.notna(t_inicio) else "",
            }
            for _, fila in tabla_fuera.iterrows()
        ]
        resumen_fuera = [
            f"Periodos de análisis: {format_datetime_es(t_inicio, include_seconds=False)} → {format_datetime_es(t_fin, include_seconds=False)}",
            f"Umbral evaluado: {fmt(analisis.umbral)} °C",
            f"Primero en salir: {primero['Sensor']} ({format_datetime_es(pd.to_datetime(primero['Primer cruce'], errors='coerce'), include_seconds=False)})",
            f"Último en salir: {ultimo['Sensor']} ({format_datetime_es(pd.to_datetime(ultimo['Primer cruce'], errors='coerce'), include_seconds=False)})",
            "",
            _tabla_texto(
                "Tiempo fuera del límite por sensor",
                filas_salida,
                ["Sensor", "Salida", "Δ fuera", "Δ inicio"],
            ),
        ]
        st.text("\n".join(resumen_fuera))

    if analisis.cruces_down_after:
        orden = [
            (s, analisis.cruces_down_after[s], analisis.retardos_horas.get(s, 0.0))
            for s in analisis.cruces_down_after
            if analisis.retardos_horas.get(s, 0.0) > 0
        ]
        orden.sort(key=lambda x: x[1])
        if orden:
            s_first, t_first, d_first = orden[0]
            s_last, t_last, d_last = orden[-1]
            resumen = [
                f"Periodos de análisis: {format_datetime_es(analisis.t_ini_clip, include_seconds=False)} → {format_datetime_es(analisis.t_fin_clip, include_seconds=False)}",
                f"Umbral evaluado: {fmt(analisis.umbral)} °C",
                f"Primero en regresar: {s_first} ({format_datetime_es(t_first, include_seconds=False)})",
                f"Último en regresar: {s_last} ({format_datetime_es(t_last, include_seconds=False)})",
                "",
                _tabla_texto(
                    "Orden de regreso (más temprano → más tarde)",
                    [
                        {
                            "Sensor": s,
                            "Regreso": format_datetime_es(t_cross, include_seconds=False),
                            "Δ regreso": fmt_hm(delay),
                        }
                        for s, t_cross, delay in orden
                    ],
                    ["Sensor", "Regreso", "Δ regreso"],
                ),
            ]
            st.text("\n".join(resumen))
        else:
            st.text(
                "Umbral: "
                f"{fmt(analisis.umbral)} °C | Inicio {format_datetime_es(analisis.t_ini_clip, include_seconds=False)}\n"
                "Periodo de prueba sin reingresos registrados."
            )

# -----------------------------------------------------------------------------
# UI principal
# -----------------------------------------------------------------------------
def main() -> None:
    st.title("📈 Reportes de sensores (Monitoreo / Prueba)")
    st.write("Carga un archivo Excel con hojas de temperatura y humedad para obtener el mismo análisis del notebook.")

    incluir_humedad = True
    contexto_general = ""
    contextos_division: dict[str, str] = {}

    with st.sidebar:
        st.header("Carga de datos")
        excel_file = st.file_uploader("Excel de sensores", type=["xlsx", "xlsm", "xls"])
        if excel_file is None:
            st.info("Sube un Excel para comenzar.")
            safe_stop()
            return
        file_bytes = excel_file.getvalue()

        sheets = list_sheets(file_bytes)
        if not sheets:
            st.error("No se pudieron leer las hojas del Excel.")
            safe_stop()
            return

        def default_index(keyword: str) -> int:
            for i, sheet in enumerate(sheets):
                if keyword in sheet.lower():
                    return i
            return 0
        incluir_humedad = st.checkbox("Incluir análisis de humedad", value=True, key="toggle_incluir_humedad")
        sheet_temp = st.selectbox("Hoja de temperatura", sheets, index=default_index(DEFAULT_TEMP_SHEET_HINT))
        sheet_hum = None
        if incluir_humedad:
            sheet_hum = st.selectbox("Hoja de humedad", sheets, index=default_index(DEFAULT_HUM_SHEET_HINT))

    try:
        df_temp_raw = read_excel_sheet(file_bytes, sheet_temp)
        if incluir_humedad and sheet_hum is not None:
            df_hum_raw = read_excel_sheet(file_bytes, sheet_hum)
        else:
            df_hum_raw = pd.DataFrame(columns=["Timestamp"])
    except Exception as exc:
        st.error(f"No fue posible leer las hojas seleccionadas: {exc}")
        safe_stop()
        return

    df_temp = limpiar_dataframe(df_temp_raw)
    df_hum = limpiar_dataframe(df_hum_raw) if incluir_humedad and not df_hum_raw.empty else df_hum_raw.copy()

    sensores_temp = detectar_sensores(df_temp, {"Timestamp", "Fecha", "Hora"})
    sensores_hum = detectar_sensores(df_hum, {"Timestamp", "Fecha", "Hora"}) if incluir_humedad else []

    if not sensores_temp:
        st.error("No se detectaron columnas de sensores de temperatura.")
        safe_stop()
        return
    if incluir_humedad and not sensores_hum:
        st.warning("No se detectaron sensores de humedad. El análisis continuará solo con temperatura.")
        incluir_humedad = False
        df_hum = pd.DataFrame(columns=["Timestamp"])
        sensores_hum = []

    sel_temp = list(sensores_temp)
    sel_hum = list(sensores_hum)
    df_map_override: pd.DataFrame | None = None
    niveles_manual: dict[str, str] = {}
    alturas_manual: dict[str, str] = {}
    grupos_temp_config: dict[str, list[str]] = {"stage": [], "maquila": []}
    grupos_hum_config: dict[str, list[str]] = {"stage": [], "maquila": []}
    habilitar_grupos = False
    contexto_titulos = ""
    umbral_hum_max: float | None = None

    with st.sidebar:
        st.header("Panel de configuraciones")

        mostrar_config_sensores = st.checkbox("Mostrar configuración de sensores", value=False, key="toggle_config_sensores")
        if mostrar_config_sensores:
            st.subheader("Sensores incluidos")
            sel_temp = st.multiselect(
                "Sensores de temperatura",
                sensores_temp,
                default=sel_temp,
                key="sel_temp_sidebar",
            )
            if incluir_humedad and sensores_hum:
                sel_hum = st.multiselect(
                    "Sensores de humedad",
                    sensores_hum,
                    default=sel_hum,
                    key="sel_hum_sidebar",
                )
        else:
            st.caption("Usando todos los sensores disponibles (activa la casilla para personalizar).")

        st.subheader("División por área")
        habilitar_grupos = st.checkbox("Dividir sensores en Almacén / Stage / Maquila", value=False, key="toggle_grupos")
        if habilitar_grupos:
            st.caption("Selecciona los sensores para Stage y Maquila; el resto se asignará automáticamente a Almacén.")
            grupos_temp_config["stage"] = st.multiselect(
                "Temperatura · Stage",
                sel_temp,
                default=[],
                key="grupo_stage_temp",
            )
            grupos_temp_config["maquila"] = st.multiselect(
                "Temperatura · Maquila",
                [s for s in sel_temp if s not in grupos_temp_config["stage"]],
                default=[],
                key="grupo_maquila_temp",
            )
            if incluir_humedad and sel_hum:
                grupos_hum_config["stage"] = st.multiselect(
                    "Humedad · Stage",
                    sel_hum,
                    default=[],
                    key="grupo_stage_hum",
                )
                grupos_hum_config["maquila"] = st.multiselect(
                    "Humedad · Maquila",
                    [s for s in sel_hum if s not in grupos_hum_config["stage"]],
                    default=[],
                    key="grupo_maquila_hum",
                )

        st.subheader("Asignación de niveles")
        modo_niveles = st.radio(
            "Modo de niveles",
            ("Ocultar", "Automático (archivo)", "Manual"),
            index=0,
            key="modo_niveles",
        )
        if modo_niveles == "Automático (archivo)":
            map_file = st.file_uploader("Archivo CSV/XLSX de niveles", type=["csv", "xlsx"], key="map_file")
            df_map_override = load_map_file(map_file)
            st.caption("Columnas sugeridas: sensor_id, level, height")
        elif modo_niveles == "Manual":
            sensores_para_niveles = list(dict.fromkeys(sel_temp + sel_hum))
            if not sensores_para_niveles:
                st.info("Selecciona sensores para ingresar niveles manualmente.")
            else:
                st.caption("Define el nivel para cada sensor y las alturas por nivel (las demás se dejan vacías).")
                for sensor in sensores_para_niveles:
                    niveles_manual[sensor] = st.text_input(
                        f"Nivel sensor {sensor}",
                        value="",
                        key=f"nivel_{sensor}",
                    )
                niveles_unicos = sorted(
                    {int(valor) for valor in niveles_manual.values() if str(valor).strip().isdigit()}
                )
                niveles_referencia = niveles_unicos or [1, 2, 3]
                for nivel in niveles_referencia:
                    alturas_manual[str(nivel)] = st.text_input(
                        f"Altura Nivel {nivel} (m)",
                        value="",
                        key=f"altura_{nivel}",
                    )

        st.subheader("Contexto para gráficas")
        opciones_contexto = [
            "Sin contexto adicional",
            "apertura de puertas",
            "falla generalizada",
            "Personalizar…",
        ]
        opcion_contexto = st.selectbox("Texto extra en títulos", opciones_contexto)
        if opcion_contexto == "Personalizar…":
            contexto_titulos = st.text_input("Escribe el contexto", value="", key="contexto_personalizado")
        else:
            contexto_titulos = "" if opcion_contexto == "Sin contexto adicional" else opcion_contexto
        contexto_general = contexto_con_division(contexto_titulos, "General")

        st.subheader("Filtro para 'Prueba'")
        min_candidates = [df_temp["Timestamp"].min(), df_hum["Timestamp"].min()]
        max_candidates = [df_temp["Timestamp"].max(), df_hum["Timestamp"].max()]
        valid_min = [ts for ts in min_candidates if pd.notna(ts)]
        valid_max = [ts for ts in max_candidates if pd.notna(ts)]
        if valid_min and valid_max:
            min_ts = min(valid_min)
            max_ts = max(valid_max)
            default_ini = format_datetime_es(min_ts, include_seconds=False)
            default_fin = format_datetime_es(max_ts, include_seconds=False)
            fecha_ini = st.text_input("Fecha inicio (DD - mes - YYYY HH:MM)", value=default_ini)
            fecha_fin = st.text_input("Fecha fin (DD - mes - YYYY HH:MM)", value=default_fin)
        else:
            fecha_ini = st.text_input("Fecha inicio (DD - mes - YYYY HH:MM)")
            fecha_fin = st.text_input("Fecha fin (DD - mes - YYYY HH:MM)")
        intervalo_minutos = st.number_input(
            "Intervalo para rachas (min)",
            min_value=1,
            max_value=120,
            value=DEFAULT_INTERVAL_MINUTES,
            step=1,
        )
        umbral_temp = st.number_input("Umbral temperatura (°C)", value=20.0, step=0.5)
        habilitar_umbral_temp_extra = st.checkbox("Agregar segundo umbral de temperatura", value=False)
        umbral_temp_extra = None
        if habilitar_umbral_temp_extra:
            umbral_temp_extra = st.number_input(
                "Segundo umbral de temperatura (°C)",
                value=float(umbral_temp + 5.0),
                step=0.5,
            )
        if incluir_humedad and sensores_hum:
            habilitar_umbral_hum = st.checkbox("Agregar umbral máximo de humedad", value=False)
            if habilitar_umbral_hum:
                umbral_hum_max = st.number_input(
                    "Umbral máximo de humedad (%HR)",
                    min_value=0.0,
                    max_value=100.0,
                    value=80.0,
                    step=1.0,
                )
        tipo_limite_label = st.selectbox(
            "Tipo de límite",
            ("Mayor que (sobre límite)", "Menor que (bajo límite)"),
            index=0,
        )
        tipo_limite = PruebaTipo.MAX if tipo_limite_label.startswith("Mayor") else PruebaTipo.MIN
        decimales_report = int(
            st.number_input(
                "Decimales a mostrar",
                min_value=0,
                max_value=4,
                value=DEFAULT_DECIMALES,
                step=1,
            )
        )

    if not sel_temp:
        sel_temp = sensores_temp
    if incluir_humedad:
        if not sel_hum:
            sel_hum = sensores_hum
    else:
        sel_hum = []

    grupos_temp = dividir_grupos(sel_temp, grupos_temp_config["stage"], grupos_temp_config["maquila"]) if habilitar_grupos else None
    grupos_hum = (
        dividir_grupos(sel_hum, grupos_hum_config["stage"], grupos_hum_config["maquila"])
        if habilitar_grupos and incluir_humedad and sel_hum
        else None
    )

    umbrales_temp_monitoreo: list[float] = []
    if umbral_temp is not None and np.isfinite(umbral_temp):
        umbrales_temp_monitoreo.append(float(umbral_temp))
    if umbral_temp_extra is not None and np.isfinite(umbral_temp_extra):
        valor_extra = float(umbral_temp_extra)
        if valor_extra not in umbrales_temp_monitoreo:
            umbrales_temp_monitoreo.append(valor_extra)
    umbrales_temp_monitoreo.sort()
    lineas_temp_general = lineas_umbral_temperatura(umbrales_temp_monitoreo)
    lineas_hum_general = lineas_umbral_humedad(umbral_hum_max) if incluir_humedad and sel_hum else []
    contextos_division = (
        {
            clave: contexto_con_division(
                contexto_titulos,
                NOMBRES_GRUPOS.get(clave, clave.title()),
            )
            for clave in (grupos_temp or {})
        }
    )

    sensores_mapeo = list(dict.fromkeys(sel_temp + sel_hum))
    level_override, level_meters = construir_mapa_niveles(
        df_map_override,
        sensores_mapeo,
        niveles_manual,
        alturas_manual,
    )

    df_temp = completar_nulos(df_temp, sel_temp)
    df_hum = completar_nulos(df_hum, sel_hum)

    tab_monitoreo, tab_prueba = st.tabs(["Monitoreo", "Prueba"])

    with tab_monitoreo:
        periodo_temp = (df_temp["Timestamp"].min(), df_temp["Timestamp"].max())
        periodo_hum = (
            (df_hum["Timestamp"].min(), df_hum["Timestamp"].max()) if sel_hum else (None, None)
        )
        render_seccion(
            "Monitoreo",
            df_temp,
            df_hum,
            sel_temp,
            sel_hum,
            "°C",
            "%HR",
            periodo_temp,
            periodo_hum,
            intervalo_minutos,
            level_override,
            level_meters,
            decimales_report,
            contexto_titulos,
            umbrales_temp=umbrales_temp_monitoreo,
            umbral_hum_max=umbral_hum_max,
            grupos_temp=grupos_temp,
            grupos_hum=grupos_hum,
            mostrar_humedad=bool(sel_hum),
        )

    with tab_prueba:
        df_temp_prueba, ti_temp, tf_temp = filtrar_rango(df_temp, fecha_ini, fecha_fin)
        if sel_hum:
            df_hum_prueba, ti_hum, tf_hum = filtrar_rango(df_hum, fecha_ini, fecha_fin)
        else:
            df_hum_prueba = df_hum.copy()
            ti_hum = tf_hum = None
        render_seccion(
            "Prueba",
            df_temp_prueba,
            df_hum_prueba,
            sel_temp,
            sel_hum,
            "°C",
            "%HR",
            (ti_temp, tf_temp),
            (ti_hum, tf_hum),
            intervalo_minutos,
            level_override,
            level_meters,
            decimales_report,
            contexto_titulos,
            umbrales_temp=umbrales_temp_monitoreo,
            umbral_hum_max=umbral_hum_max,
            grupos_temp=grupos_temp,
            grupos_hum=grupos_hum,
            mostrar_humedad=bool(sel_hum),
        )

        st.markdown("### Tendencia de temperatura con ventana de prueba")
        if (ti_temp is not None) and (tf_temp is not None) and sel_temp:
            fig_temp_full = grafico_temp_prueba_full(
                df_temp,
                sel_temp,
                ti_temp,
                tf_temp,
                decimales_report,
                titulo_contexto=contexto_general,
                lineas_umbral=lineas_temp_general,
            )
            if fig_temp_full is not None:
                st.pyplot(fig_temp_full)
                plt.close(fig_temp_full)

        st.markdown("### Tendencia de humedad con ventana de prueba")
        if (ti_hum is not None) and (tf_hum is not None) and sel_hum:
            fig_hum_full = grafico_hum_prueba_full(
                df_hum,
                sel_hum,
                ti_hum,
                tf_hum,
                decimales_report,
                titulo_contexto=contexto_general,
                lineas_umbral=lineas_hum_general,
            )
            if fig_hum_full is not None:
                st.pyplot(fig_hum_full)
                plt.close(fig_hum_full)

        analisis_prueba = None
        analisis_divisiones: dict[str, AnalisisPrueba] = {}
        if ti_temp is not None and tf_temp is not None:
            analisis_prueba = analizar_prueba(df_temp, sel_temp, ti_temp, tf_temp, umbral_temp, tipo_limite)
            if grupos_temp:
                for clave, sensores_grupo in grupos_temp.items():
                    if not sensores_grupo:
                        continue
                    analisis_div = analizar_prueba(df_temp, sensores_grupo, ti_temp, tf_temp, umbral_temp, tipo_limite)
                    if analisis_div is not None:
                        analisis_divisiones[clave] = analisis_div
        render_prueba_umbrales(
            analisis_prueba,
            decimales_report,
            contexto_general,
            analisis_divisiones if analisis_divisiones else None,
            contextos_division,
        )


if __name__ == "__main__":
    main()
