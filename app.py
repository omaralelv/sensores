"""Streamlit app para analizar datos de temperatura y humedad replicando el notebook base."""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_INTERVAL_MINUTES = 1
PERCENTIL_ALTO = 0.99
PERCENTIL_BAJO = 0.01

class PruebaTipo:
    MAX = "max"
    MIN = "min"

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

def id_solo(valor: str) -> str:
    import re

    m = re.search(r"(\d+)", str(valor))
    return m.group(1) if m else str(valor).strip()

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
        if serie.notna().any():
            data[sensor] = serie.fillna(serie.mean())
        else:
            data[sensor] = serie
    return data

MESES_ES = [
    "ene",
    "feb",
    "mar",
    "abr",
    "may",
    "jun",
    "jul",
    "ago",
    "sept",
    "oct",
    "nov",
    "dic",
]

def format_datetime_es(dt: pd.Timestamp | None) -> str:
    if dt is None or pd.isna(dt):
        return ""
    return f"{dt.day} - {MESES_ES[dt.month - 1]} - {dt.year} {dt.strftime('%H:%M:%S')}"

def format_date_es(dt: pd.Timestamp | None) -> str:
    if dt is None or pd.isna(dt):
        return ""
    return f"{dt.day} - {MESES_ES[dt.month - 1]} - {dt.year}"

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

def construir_mapa_niveles(
    df_map: pd.DataFrame | None,
    sensores: list[str],
    niveles_manual: dict[str, str],
    alturas_manual: dict[str, str],
) -> tuple[dict[str, int | None], dict[int, float]]:
    level_override: dict[str, int | None] = {}
    level_meters: dict[int, float] = {}

    if df_map is not None and not df_map.empty:
        columnas = {str(c).strip().lower(): c for c in df_map.columns}
        col_sensor = columnas.get("sensor_id") or columnas.get("sensor") or df_map.columns[0]
        col_level = columnas.get("level") or columnas.get("nivel") or (
            df_map.columns[1] if df_map.shape[1] > 1 else None
        )
        col_height = columnas.get("height") or columnas.get("altura")
        if col_level is not None:
            for _, row in df_map.iterrows():
                sensor = str(row[col_sensor]).strip()
                try:
                    level = int(row[col_level])
                except Exception:
                    continue
                if sensor:
                    level_override[sensor] = level
                if col_height is not None:
                    try:
                        altura = float(row[col_height])
                        level_meters[level] = altura
                    except Exception:
                        pass

    for sensor in sensores:
        level_override.setdefault(sensor, None)

    for sensor, nivel_txt in niveles_manual.items():
        if nivel_txt.strip():
            try:
                level_override[sensor] = int(nivel_txt)
            except ValueError:
                pass

    for nivel_txt, altura_txt in alturas_manual.items():
        nivel_txt = nivel_txt.strip()
        altura_txt = altura_txt.strip()
        if not nivel_txt or not altura_txt:
            continue
        try:
            level = int(nivel_txt)
            altura = float(altura_txt)
        except ValueError:
            continue
        level_meters[level] = altura

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
            return pd.to_datetime(valor.strip(), errors="coerce", dayfirst=True)
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
    ts_max = (
        df.loc[idx_max[0], "Timestamp"]
        if isinstance(idx_max, tuple) and idx_max[0] in df.index
        else pd.NaT
    )
    ts_min = (
        df.loc[idx_min[0], "Timestamp"]
        if isinstance(idx_min, tuple) and idx_min[0] in df.index
        else pd.NaT
    )

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

def _empates_por_promedio(series_medias: pd.Series, decimales: int = 1):
    """
    Recibe la serie de medias por sensor (una columna por sensor).
    Devuelve:
        prom_max, sensores_prom_max, prom_min, sensores_prom_min
    usando el valor redondeado a 'decimales' para definir empates.
    """
    if series_medias.dropna().empty:
        return np.nan, [], np.nan, []

    medias_r = series_medias.round(decimales)

    prom_max = float(medias_r.max())
    prom_min = float(medias_r.min())

    sensores_prom_max = sorted(medias_r.index[medias_r == prom_max].tolist())
    sensores_prom_min = sorted(medias_r.index[medias_r == prom_min].tolist())

    return prom_max, sensores_prom_max, prom_min, sensores_prom_min

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
) -> list[dict]:
    if df.empty or not sensores or not np.isfinite(objetivo):
        return []
    registros: list[dict] = []
    data = df.sort_values("Timestamp")
    for _, row in data.iterrows():
        ts = row["Timestamp"]
        for sensor in sensores:
            val = pd.to_numeric(row.get(sensor), errors="coerce")
            if pd.notna(val) and abs(val - objetivo) <= tolerancia:
                registros.append({"Timestamp": ts, columna_valor: float(val)})
                break
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
        filas.append(
            {
                "fecha_inicio": format_date_es(ts_ini),
                "hora_inicio": ts_ini.strftime("%H:%M:%S"),
                "fecha_fin": format_date_es(ts_fin),
                "hora_fin": ts_fin.strftime("%H:%M:%S"),
                columna_valor: f"{bloque[columna_valor].iloc[0]:.1f}",
                "n_registros": bloque.shape[0],
            }
        )
    return filas

def _resumen_generico(
    df: pd.DataFrame,
    sensores: list[str],
    unidad_label: str,
    incluir_mkt: bool = False,
    intervalo_minutos: int = DEFAULT_INTERVAL_MINUTES,
) -> tuple[str, list[dict], list[dict]]:
    """
    Devuelve:
      - texto del bloque '=== Resumen ==='
      - tabla de rachas en el valor máximo global
      - tabla de rachas en el valor mínimo global

    Siempre incluye los EMPATES por PROMEDIO (max/min). Si 'incluir_mkt' es True,
    también incluye los empates de MKT (max/min).
    """
    if df.empty or not sensores:
        return "Sin datos", [], []

    datos = df.copy()
    datos["Timestamp"] = pd.to_datetime(datos["Timestamp"], errors="coerce")
    datos = datos.dropna(subset=["Timestamp"]).sort_values("Timestamp")

    ts = datos["Timestamp"]
    fecha_inicio = ts.min()
    fecha_final  = ts.max()
    tiempo_total_horas = (
        (fecha_final - fecha_inicio).total_seconds() / 3600.0
        if pd.notna(fecha_inicio) and pd.notna(fecha_final) else np.nan
    )

    # -------- básicos
    valores = datos[sensores].apply(to_numeric)
    flat = valores.stack(dropna=False)

    promedio_general = float(flat.mean()) if not flat.dropna().empty else np.nan
    max_global = float(flat.max()) if not flat.dropna().empty else np.nan
    min_global = float(flat.min()) if not flat.dropna().empty else np.nan

    reps_max = int(np.isclose(valores, max_global, equal_nan=False).sum().sum()) if np.isfinite(max_global) else 0
    reps_min = int(np.isclose(valores, min_global, equal_nan=False).sum().sum()) if np.isfinite(min_global) else 0

    # Diferencia instantánea máxima entre sensores por timestamp
    dif_max_instantanea = np.nan
    if not valores.dropna(how="all").empty:
        dif = (valores.max(axis=1, skipna=True) - valores.min(axis=1, skipna=True)).dropna()
        if not dif.empty:
            dif_max_instantanea = float(dif.max())

    # -------- promedios por sensor + EMPATES (SIEMPRE)
    medias = valores.mean(axis=0, skipna=True).dropna()
    if medias.empty:
        prom_max = prom_min = np.nan
        sensores_prom_max = []
        sensores_prom_min = []
    else:
        # utiliza atol pequeño o redondeo a 1 decimal para evitar diferencias de ruido
        prom_max = float(medias.max())
        prom_min = float(medias.min())
        # opción 1 (atol):
        #sensores_prom_max = sorted(medias.index[np.isclose(medias, prom_max, atol=1e-3)].tolist())
        #sensores_prom_min = sorted(medias.index[np.isclose(medias, prom_min, atol=1e-3)].tolist())
        # Si prefieres exactamente como se imprime a 1 decimal, descomenta:
        r = medias.round(1)
        prom_max = float(r.max()); prom_min = float(r.min())
        sensores_prom_max = sorted(r.index[r == prom_max].tolist())
        sensores_prom_min = sorted(r.index[r == prom_min].tolist())

    # -------- armado del texto
    resumen_lineas = [
        "=== Resumen ===",
        "Parametro                     : Resultado",
        "-------------------------------------------:",
    ]
    def linea(nombre: str, valor) -> None:
        resumen_lineas.append(f"{nombre:<31}: {valor}")

    linea("fecha_hora_inicio", format_datetime_es(fecha_inicio))
    linea("fecha_hora_final",  format_datetime_es(fecha_final))
    linea("tiempo_total_horas.", f"{tiempo_total_horas:.1f}" if np.isfinite(tiempo_total_horas) else "")
    linea("cantidad de sensores utilizados.", str(len(sensores)))

    linea(f"{unidad_label}_promedio.", f"{promedio_general:.1f}" if np.isfinite(promedio_general) else "")
    linea(f"{unidad_label}_maxima.",   f"{max_global:.1f}"     if np.isfinite(max_global)     else "")
    linea(f"repeticiones_{unidad_label}_maxima.", str(reps_max))
    linea(f"{unidad_label}_minima.",   f"{min_global:.1f}"     if np.isfinite(min_global)     else "")
    linea(f"repeticiones_{unidad_label}_minima.", str(reps_min))
    linea("diferencia_maxima_instantanea.", f"{dif_max_instantanea:.1f}" if np.isfinite(dif_max_instantanea) else "")

    # ---> PROMEDIO (estos dos SIEMPRE deben salir)
    linea(f"{unidad_label}_promedio_maxima.", f"{prom_max:.1f}" if np.isfinite(prom_max) else "")
    linea("sensores_promedio_maximo.", _string_list(sensores_prom_max))
    linea(f"{unidad_label}_promedio_minima.", f"{prom_min:.1f}" if np.isfinite(prom_min) else "")
    linea("sensores_promedio_minimo.", _string_list(sensores_prom_min))

    # ---> MKT (opcional) con empates
    if incluir_mkt:
        mkt_vals = {
            s: mkt_celsius(valores[s]) for s in sensores
            if not valores[s].dropna().empty
        }
        if mkt_vals:
            mkt_series = pd.Series(mkt_vals).round(1)  # empates coherentes con salida a 1 decimal
            mkt_max = float(mkt_series.max())
            mkt_min = float(mkt_series.min())
            sensores_mkt_max = sorted(mkt_series.index[mkt_series == mkt_max].tolist())
            sensores_mkt_min = sorted(mkt_series.index[mkt_series == mkt_min].tolist())
        else:
            mkt_max = mkt_min = np.nan
            sensores_mkt_max = sensores_mkt_min = []

        linea("mkt_maximo.", f"{mkt_max:.1f}" if np.isfinite(mkt_max) else "")
        linea("sensores_mkt_maximo.", _string_list(sensores_mkt_max))
        linea("mkt_minimo.", f"{mkt_min:.1f}" if np.isfinite(mkt_min) else "")
        linea("sensores_mkt_minimo.", _string_list(sensores_mkt_min))

    # -------- tablas por rachas (usando máximos/mínimos globales)
    columna_valor = "temperatura" if incluir_mkt else "humedad"
    tabla_max = _compactar_eventos_valor(datos, sensores, max_global, columna_valor, intervalo_minutos=intervalo_minutos)
    tabla_min = _compactar_eventos_valor(datos, sensores, min_global, columna_valor, intervalo_minutos=intervalo_minutos)

    return "\n".join(resumen_lineas), tabla_max, tabla_min

import re

def _quitar_texto_umbral(ax):
    for t in list(ax.texts):
        if re.search(r"\bUmbral\b", t.get_text(), flags=re.IGNORECASE):
            t.remove()

def _smart_annot(
    ax,
    x, y,
    text,
    side="right",           # "left" o "right"
    dx_points=12,           # desplazamiento horizontal en puntos
    dy_points=16,           # desplazamiento vertical en puntos
    fontsize=9,
):
    # posición del texto respecto al punto
    if side == "left":
        xoff = -abs(dx_points)
        ha = "right"
    else:
        xoff = abs(dx_points)
        ha = "left"

    ax.annotate(
        text,
        xy=(x, y),
        xytext=(xoff, dy_points),
        textcoords="offset points",
        ha=ha, va="bottom",
        fontsize=fontsize,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8),
        arrowprops=dict(arrowstyle="->", lw=1),
    )

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

def render_bloque(nombre: str, df: pd.DataFrame, sensores: list[str], unidad: str, intervalo_minutos: int) -> None:
    if df.empty or not sensores:
        st.info(f"Sin datos de {nombre.lower()} para mostrar.")
        return

    es_temperatura = nombre.lower().startswith("temperatura")
    etiqueta = "temperatura" if es_temperatura else "humedad"
    resumen_texto, tabla_max, tabla_min = _resumen_generico(
        df,
        sensores,
        etiqueta,
        incluir_mkt=es_temperatura,
        intervalo_minutos=intervalo_minutos,
    )

    st.text(resumen_texto)

    titulo_max = f"Tabla Repetición Máxima ({nombre} compactada)"
    titulo_min = f"Tabla Repetición Mínima ({nombre} compactada)"
    columnas = ["fecha_inicio", "hora_inicio", "fecha_fin", "hora_fin", etiqueta, "n_registros"]
    st.text(_tabla_texto(titulo_max, tabla_max, columnas))
    st.text(_tabla_texto(titulo_min, tabla_min, columnas))

    fig_box = grafico_boxplot(df, sensores, f"Distribución de {nombre.lower()} por sensor", unidad)
    st.pyplot(fig_box)
    plt.close(fig_box)

    fig_trend = grafico_tendencias(df, sensores, f"Tendencia de {nombre.lower()} (sensores seleccionados)", unidad)
    st.pyplot(fig_trend)
    plt.close(fig_trend)

    analisis = analizar_promedio(df, sensores)
    stats = calcular_estadisticas(df, sensores)
    if analisis and not stats.stats.empty:
        fig_dest = grafico_destacados(df, stats, analisis, f"Sensores destacados ({nombre.lower()})", unidad)
        st.pyplot(fig_dest)
        plt.close(fig_dest)
     
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
) -> None:
    st.header(titulo)

    st.markdown("### Temperatura")
    if periodo_temp[0] is not None and periodo_temp[1] is not None:
        st.caption(f"Periodo: {periodo_temp[0].strftime(TIME_FORMAT)} → {periodo_temp[1].strftime(TIME_FORMAT)}")
    render_bloque("Temperatura", df_temp, sensores_temp, unidad_temp, intervalo_minutos)

    st.markdown("### Humedad") 
    if periodo_hum[0] is not None and periodo_hum[1] is not None:
        st.caption(f"Periodo: {periodo_hum[0].strftime(TIME_FORMAT)} → {periodo_hum[1].strftime(TIME_FORMAT)}")
    render_bloque("Humedad", df_hum, sensores_hum, unidad_hum, intervalo_minutos)

    
    bloques_mapeo = construir_bloques_mapeo(df_temp, df_hum, sensores_temp, sensores_hum, level_override, level_meters)
    st.markdown("### Mapeo de sensores y puntos críticos")
    st.text(bloques_mapeo)

def plot_prueba_sensor_principal(analisis: AnalisisPrueba) -> plt.Figure | None:
    if not analisis.sensor_primero or analisis.t_primera is None:
        return None

    sensor = analisis.sensor_primero
    data = analisis.df_contexto.sort_values("Timestamp")
    ts = data["Timestamp"]
    y = to_numeric(data[sensor])

    # Valores en los límites del rango
    y_ini = _interp_at(ts, y, analisis.t_ini_clip)
    y_fin = _interp_at(ts, y, analisis.t_fin_clip)
    y_cruce = _interp_at(ts, y, analisis.t_primera)

    fig, ax = plt.subplots(figsize=(14.5, 6.5))

    # Serie completa y resaltado del rango
    ax.plot(ts, y, label=sensor, linewidth=1.2)
    mask_rango = analisis.mask_rango.reindex(data.index, fill_value=False)
    ax.plot(ts[mask_rango], y[mask_rango], linewidth=2.6, alpha=0.95)

    # Umbral y ventana
    ax.axhline(analisis.umbral, linestyle=":", linewidth=1.6)
    ax.axvline(analisis.t_ini_clip, linestyle="--", linewidth=1.2)
    ax.axvline(analisis.t_fin_clip, linestyle="--", linewidth=1.2)
    ax.axvspan(analisis.t_ini_clip, analisis.t_fin_clip, alpha=0.12)

    # --- Etiquetas con helper (para que no se encimen) ---
    # INICIO (caja a la izquierda)
    ax.scatter([analisis.t_ini_clip], [y_ini], s=55, zorder=5, color="tab:orange")
    _smart_annot(
        ax,
        analisis.t_ini_clip, y_ini,
        f"Inicio\n{format_datetime_es(analisis.t_ini_clip)}\n{y_ini:.2f}",
        side="left",
        dx_points=14, dy_points=20, fontsize=9,
    )
    
    # FIN (caja a la derecha)
    ax.scatter([analisis.t_fin_clip], [y_fin], s=50, zorder=5, color="tab:orange")
    _smart_annot(
        ax,
        analisis.t_fin_clip, y_fin,
        f"Fin\n{format_datetime_es(analisis.t_fin_clip)}\n{y_fin:.2f}",
        side="right",
        dx_points=14, dy_points=20, fontsize=9,
    )

    # PRIMER CRUCE (caja a la derecha, un poco más alta)
    ax.scatter([analisis.t_primera], [y_cruce], s=55, zorder=6, color="tab:green")
    _smart_annot(
        ax,
        analisis.t_primera, y_cruce,
        f"Primer cruce\n{analisis.t_primera:%Y-%m-%d %H:%M}",
        side="right",
        dx_points=14, dy_points=26, fontsize=9,
    )

    # Estética
    ax.set_title("Sensor que cruza primero el umbral")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Temperatura (°C)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
    ax.tick_params(axis="x", rotation=90)
    ax.legend(loc="upper left")

    _quitar_texto_umbral(ax)

    fig.tight_layout()
    return fig

def plot_prueba_rango(analisis: AnalisisPrueba) -> plt.Figure | None:
    # Sensores que realmente estuvieron fuera del umbral dentro del rango
    sensores_zoom = [
        s for s, _, horas in sorted(
            [(s, analisis.cruces[s], analisis.tiempos_fuera.get(s, 0.0)) for s in analisis.cruces],
            key=lambda x: x[1]
        ) if analisis.tiempos_fuera.get(s, 0.0) > 0
    ]
    if not sensores_zoom:
        return None

    data = analisis.df_contexto.sort_values("Timestamp")
    ts = data["Timestamp"]
    pad = pd.Timedelta(minutes=5)
    t0 = max(analisis.t_ini_clip - pad, ts.min())
    t1 = min(analisis.t_fin_clip + pad, ts.max())
    mask_zoom = (ts >= t0) & (ts <= t1)

    # ⬅️ Hacemos el canvas igual de grande que los otros
    fig, ax = plt.subplots(figsize=(14.5, 6.5), dpi=100)

    for sensor in sensores_zoom:
        ax.plot(ts[mask_zoom], to_numeric(data.loc[mask_zoom, sensor]), linewidth=1.6, label=sensor)
        tcr = analisis.cruces.get(sensor)
        if tcr and t0 <= tcr <= t1:
            ax.axvline(tcr, linestyle="-.", linewidth=1.0, alpha=0.85)

    ax.axhline(analisis.umbral, linestyle=":", linewidth=1.6)
    ax.text(ts.iloc[0], analisis.umbral, f"  Umbral = {analisis.umbral:.2f}", va="bottom")
    ax.axvline(analisis.t_ini_clip, linestyle="--", linewidth=1.2)
    ax.axvline(analisis.t_fin_clip, linestyle="--", linewidth=1.2)
    ax.axvspan(analisis.t_ini_clip, analisis.t_fin_clip, alpha=0.12)

    # Formato de ejes y grid (igual al resto)
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
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=max(1, step // 2)))
    ax.grid(True, which="major", axis="x", alpha=0.25)
    ax.set_xlim(t0, t1)

    ax.set_title("Sensores fuera del umbral dentro del rango")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Temperatura (°C)")
    plt.xticks(rotation=90, ha="right")

    # Leyenda fuera del área de ploteo, igual que los otros
    ax.legend(title="Orden de cruce", bbox_to_anchor=(1.02, 1), loc="upper left")
    _quitar_texto_umbral(ax)
    fig.tight_layout(rect=[0, 0, 0.86, 1])
    return fig

def plot_prueba_descenso(analisis: AnalisisPrueba) -> plt.Figure | None:
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
        ax.plot(ts, to_numeric(data[sensor]), linewidth=1.6, label=sensor)

    ax.axhline(analisis.umbral, linestyle=":", linewidth=1.6)
    ax.text(ts.min(), analisis.umbral, f"  Umbral = {analisis.umbral:.2f}", va="bottom")
    ax.axvline(analisis.t_ini_clip, linestyle="--", linewidth=1.2)
    ax.axvline(analisis.t_fin_clip, linestyle="--", linewidth=1.2)
    ax.axvspan(analisis.t_ini_clip, analisis.t_fin_clip, alpha=0.12)

    if analisis.t_baja_primero is not None and analisis.sensor_baja_primero:
        y_near = data.loc[(ts - analisis.t_baja_primero).abs().idxmin(), analisis.sensor_baja_primero]
        ax.scatter([analisis.t_baja_primero], [y_near], s=35, zorder=5)
        ax.axvline(analisis.t_baja_primero, linestyle="-.", linewidth=1.1)
        ax.annotate(
            f"Baja 1º: {analisis.sensor_baja_primero}\n{format_datetime_es(analisis.t_baja_primero)}\nΔ={fmt_hm(analisis.retardos_horas.get(analisis.sensor_baja_primero))}",
            xy=(analisis.t_baja_primero, y_near),
            xytext=(15, 15),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->"),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9)
        )

    if analisis.t_baja_ultimo is not None and analisis.sensor_baja_ultimo:
        y_near = data.loc[(ts - analisis.t_baja_ultimo).abs().idxmin(), analisis.sensor_baja_ultimo]
        ax.scatter([analisis.t_baja_ultimo], [y_near], s=35, zorder=5)
        ax.axvline(analisis.t_baja_ultimo, linestyle="-.", linewidth=1.1)
        ax.annotate(
            f"Baja último: {analisis.sensor_baja_ultimo}\n{format_datetime_es(analisis.t_baja_ultimo)}\nΔ={fmt_hm(analisis.retardos_horas.get(analisis.sensor_baja_ultimo))}",
            xy=(analisis.t_baja_ultimo, y_near),
            xytext=(15, -35),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->"),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9)
        )

    ax.set_xlim(ts.min(), ts.max())
    ax.set_title("Sensores que regresan dentro del límite")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Temperatura (°C)")
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0, 30]))
    ax.grid(True, which="major", axis="x", alpha=0.2)
    plt.xticks(rotation=90, ha="right")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    _quitar_texto_umbral(ax)
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    return fig

# === NUEVO: humedad en PRUEBA con TODO el histórico y sombreado de ventana ===
def grafico_humedad_prueba_full(
    df_hum: pd.DataFrame,
    sensores_hum: list[str],
    t_ini: pd.Timestamp,
    t_fin: pd.Timestamp,
    titulo: str = "Tendencia de humedad (sensores seleccionados) — Ventana de prueba",
) -> plt.Figure | None:
    if df_hum.empty or not sensores_hum or t_ini is None or t_fin is None:
        return None

    data = df_hum.sort_values("Timestamp").copy()
    ts = data["Timestamp"]

    # Serie promedio (para los marcadores inicio/fin)
    valores = data[sensores_hum].apply(to_numeric)
    promedio = valores.mean(axis=1, skipna=True)

    fig, ax = plt.subplots(figsize=(14.5, 6.5), dpi=100)

    for s in sensores_hum:
        ax.plot(ts, to_numeric(data[s]), linewidth=0.6, alpha=0.9, label=s)

    # Promedio general
    ax.plot(ts, promedio, linestyle="--", linewidth=2.0, color="black", label="Promedio General")

    # Sombreado solo en la ventana de prueba
    ax.axvspan(t_ini, t_fin, color="tab:blue", alpha=0.12)

    # Marcadores/etiquetas en inicio y fin (sobre el promedio)
    y_ini = _interp_at(ts, promedio, t_ini)
    y_fin = _interp_at(ts, promedio, t_fin)

    ax.scatter([t_ini], [y_ini], s=55, zorder=5, color="tab:orange")
    _smart_annot(
        ax, t_ini, y_ini,
        f"Inicio\n{format_datetime_es(t_ini)}\n{y_ini:.2f}",
        side="left", dx_points=12, dy_points=16, fontsize=9,
    )

    ax.scatter([t_fin], [y_fin], s=55, zorder=5, color="tab:orange")
    _smart_annot(
        ax, t_fin, y_fin,
        f"Fin\n{format_datetime_es(t_fin)}\n{y_fin:.2f}",
        side="right", dx_points=12, dy_points=16, fontsize=9,
    )

    # Ejes/leyenda
    ax.set_title(titulo)
    ax.set_xlabel("Periodo de tiempo")
    ax.set_ylabel("%HR")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
    ax.tick_params(axis="x", rotation=90)
    ax.set_xlim(ts.min(), ts.max())  
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    return fig
# === FIN NUEVO ===

def render_prueba_umbrales(analisis: AnalisisPrueba) -> None:
    if analisis is None:
        st.info("Selecciona un rango válido y configura el umbral para ver el análisis de prueba.")
        return

    st.markdown("### Análisis de umbral (Prueba)")
    st.caption(
        " | ".join(
            [
                f"Umbral = {analisis.umbral:.2f} °C",
                "Condición: mayor que" if analisis.tipo == PruebaTipo.MAX else "Condición: menor que",
                f"Periodo: {format_datetime_es(analisis.t_ini_clip)} → {format_datetime_es(analisis.t_fin_clip)}",
            ]
        )
    )

    fig_first = plot_prueba_sensor_principal(analisis)
    if fig_first is not None:
        st.pyplot(fig_first)
        plt.close(fig_first)
    else:
        st.info("Ningún sensor cruza el umbral en el periodo seleccionado.")

    fig_range = plot_prueba_rango(analisis)
    if fig_range is not None:
        st.pyplot(fig_range, use_container_width=True)  # ⬅️ ocupa todo el ancho
        plt.close(fig_range)

    fig_descenso = plot_prueba_descenso(analisis)
    if fig_descenso is not None:
        st.pyplot(fig_descenso)
        plt.close(fig_descenso)

    filas = []
    for sensor, horas in analisis.tiempos_fuera.items():
        t_cruce = analisis.cruces.get(sensor)   # <- primer cruce
        # Muestra solo si tiene primer cruce y el tiempo fuera es positivo/finito
        if t_cruce is None or not (np.isfinite(horas) and horas > 0):
            continue
        filas.append(
            {
                "Sensor": sensor,
                "Primer cruce": t_cruce,
                "Tiempo fuera (h)": round(horas, 3),
                "Tiempo fuera": fmt_hm(horas),
            }
        )

    tabla_fuera = pd.DataFrame(filas)
    if not tabla_fuera.empty:
        tabla_texto = []
        for _, fila in tabla_fuera.iterrows():
            tabla_texto.append(
                {
                    "Sensor": fila["Sensor"],
                    "Primer cruce": format_datetime_es(pd.to_datetime(fila["Primer cruce"], errors="coerce")),
                    "Tiempo (h)": fila["Tiempo fuera (h)"],
                    "Tiempo": fila["Tiempo fuera"],
                }
            )
        st.text(
            _tabla_texto(
                "Tiempo fuera del límite por sensor",
                tabla_texto,
                ["Sensor", "Primer cruce", "Tiempo (h)", "Tiempo"],
            )
        )

    if analisis.cruces_down_after:
        orden = [
            (s, analisis.cruces_down_after[s], analisis.retardos_horas.get(s, 0.0))
            for s in analisis.cruces_down_after
            if analisis.retardos_horas.get(s, 0.0) > 0
        ]
        orden.sort(key=lambda x: x[1])
        lines = [f"Umbral: {analisis.umbral:.1f} °C | Referencia: después de {analisis.t_fin_clip:%Y-%m-%d %H:%M}"]
        if orden:
            s_first, t_first, d_first = orden[0]
            s_last, t_last, d_last = orden[-1]
            lines.append(
                f"Primero en regresar: {s_first} a las {t_first:%Y-%m-%d %H:%M} (Δ={fmt_hm(d_first)})"
            )
            lines.append(
                f"Último en regresar: {s_last} a las {t_last:%Y-%m-%d %H:%M} (Δ={fmt_hm(d_last)})"
            )
            lines.append("\nOrden de regreso (más temprano → más tarde):")
            for s, t_cross, delay in orden:
                lines.append(f"  {t_cross:%Y-%m-%d %H:%M} — {s}: Δ={fmt_hm(delay)}")
        else:
            lines.append("Ningún sensor regresó con retardo > 0 tras el límite.")
        st.text("\n".join(lines))

# -----------------------------------------------------------------------------
# Mapeo (bloques por niveles)
# -----------------------------------------------------------------------------
def _bloque_metricas_map(
    titulo: str,
    valor: float,
    sensores: list[str],
    unidad: str,
    level_override: dict[str, int | None],
    level_meters: dict[int, float],
) -> list[str]:
    if not sensores or not np.isfinite(valor):
        return []
    lineas = [f"{titulo}: {valor:.1f} {unidad}"]
    sensor_principal = sensores[0]
    lineas.append(f"Sensor: {sensor_principal}")
    lineas.append(f"Ubicación: {ubicacion_text(sensor_principal, level_override, level_meters)}")
    empatados = sensores[1:]
    if empatados:
        lineas.append("Empatados:")
        for sid in empatados:
            lineas.append(
                f"  - Sensor: {sid} | {ubicacion_text(sid, level_override, level_meters)}"
            )
    return lineas

def construir_bloques_mapeo(
    df_temp: pd.DataFrame,
    df_hum: pd.DataFrame,
    sensores_temp: list[str],
    sensores_hum: list[str],
    level_override: dict[str, int | None],
    level_meters: dict[int, float],
) -> str:
    bloques: list[str] = []

    def _info_metricas(df: pd.DataFrame, sensores: list[str], unidad: str, include_mkt: bool):
        data = df.copy()
        data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")
        data = data.dropna(subset=["Timestamp"]).sort_values("Timestamp")
        valores = data[sensores].apply(to_numeric)
        maxima_col = valores.max(axis=0, skipna=True)
        max_val = float(maxima_col.max()) if maxima_col.notna().any() else np.nan
        sensores_max = maxima_col.index[maxima_col == max_val].tolist() if np.isfinite(max_val) else []
        minima_col = valores.min(axis=0, skipna=True)
        min_val = float(minima_col.min()) if minima_col.notna().any() else np.nan
        sensores_min = minima_col.index[minima_col == min_val].tolist() if np.isfinite(min_val) else []
        medias = valores.mean(axis=0, skipna=True)
        prom_max_val = float(medias.max()) if medias.notna().any() else np.nan
        prom_min_val = float(medias.min()) if medias.notna().any() else np.nan
        # Empates por promedio (redondeado a 1 decimal)
        prom_max_val, sensores_prom_max, prom_min_val, sensores_prom_min = _empates_por_promedio(medias, decimales=1)

        mkt_max_val = mkt_min_val = np.nan
        sensores_mkt_max: list[str] = []
        sensores_mkt_min: list[str] = []
        if include_mkt:
            mkt_vals = {sensor: mkt_celsius(valores[sensor]) for sensor in sensores if not valores[sensor].dropna().empty}
            if mkt_vals:
                serie = pd.Series(mkt_vals)
                mkt_max_val = float(serie.max())
                mkt_min_val = float(serie.min())
                sensores_mkt_max = serie.index[np.isclose(serie, mkt_max_val, atol=1e-3)].tolist()
                sensores_mkt_min = serie.index[np.isclose(serie, mkt_min_val, atol=1e-3)].tolist()
        return {
            "max": (max_val, sensores_max),
            "min": (min_val, sensores_min),
            "avg_max": (prom_max_val, sensores_prom_max),
            "avg_min": (prom_min_val, sensores_prom_min),
            "mkt_max": (mkt_max_val, sensores_mkt_max),
            "mkt_min": (mkt_min_val, sensores_mkt_min),
        }

    temp_info = _info_metricas(df_temp, sensores_temp, "°C", include_mkt=True) if sensores_temp else {}
    hum_info = _info_metricas(df_hum, sensores_hum, "%HR", include_mkt=False) if sensores_hum else {}

    if temp_info:
        lineas = ["=== TEMPERATURA ==="]
        lineas += _bloque_metricas_map("Temperatura Máxima", temp_info["max"][0], temp_info["max"][1], "°C", level_override, level_meters)
        lineas.append("")
        lineas += _bloque_metricas_map("Temperatura Mínima", temp_info["min"][0], temp_info["min"][1], "°C", level_override, level_meters)
        lineas.append("")
        lineas += _bloque_metricas_map("Temperatura Promedio Máx.", temp_info["avg_max"][0], temp_info["avg_max"][1], "°C", level_override, level_meters)
        lineas.append("")
        lineas += _bloque_metricas_map("Temperatura Promedio Mín.", temp_info["avg_min"][0], temp_info["avg_min"][1], "°C", level_override, level_meters)
        lineas.append("")
        lineas += _bloque_metricas_map("MKT Máx.", temp_info["mkt_max"][0], temp_info["mkt_max"][1], "°C", level_override, level_meters)
        lineas.append("")
        lineas += _bloque_metricas_map("MKT Mín.", temp_info["mkt_min"][0], temp_info["mkt_min"][1], "°C", level_override, level_meters)
        bloques.append("\n".join(l for l in lineas if l.strip()))

    if hum_info:
        lineas = ["=== HUMEDAD ==="]
        lineas += _bloque_metricas_map("Humedad Máxima", hum_info["max"][0], hum_info["max"][1], "%HR", level_override, level_meters)
        lineas.append("")
        lineas += _bloque_metricas_map("Humedad Mínima", hum_info["min"][0], hum_info["min"][1], "%HR", level_override, level_meters)
        lineas.append("")
        lineas += _bloque_metricas_map("Humedad Máx. Promedio", hum_info["avg_max"][0], hum_info["avg_max"][1], "%HR", level_override, level_meters)
        lineas.append("")
        lineas += _bloque_metricas_map("Humedad Mín. Promedio", hum_info["avg_min"][0], hum_info["avg_min"][1], "%HR", level_override, level_meters)
        bloques.append("\n".join(l for l in lineas if l.strip()))

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
        bloques.append("\n".join(l for l in lineas if l.strip()))

    return "\n\n".join([b for b in bloques if b]) if bloques else ""

def grafico_boxplot(df: pd.DataFrame, sensores: list[str], titulo: str, unidad: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df[sensores], ax=ax)
    ax.set_title(titulo)
    ax.set_ylabel(unidad)
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    return fig

def grafico_tendencias(df: pd.DataFrame, sensores: list[str], titulo: str, unidad: str, mostrar_promedio: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 6))
    ts = df["Timestamp"]
    for sensor in sensores:
        ax.plot(ts, to_numeric(df[sensor]), linewidth=0.6, label=sensor)

    if mostrar_promedio:
        promedio = df[sensores].apply(to_numeric).mean(axis=1, skipna=True)
        ax.plot(ts, promedio, color="black", linestyle="--", linewidth=2, label="Promedio General")

    ax.set_title(titulo)
    ax.set_xlabel("Periodo de tiempo")
    ax.set_ylabel(unidad)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
    ax.tick_params(axis="x", rotation=90)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    return fig

def grafico_destacados(df: pd.DataFrame, stats: SensorStats, analisis: PromedioAnalisis | None, titulo: str, unidad: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 6))
    ts = df["Timestamp"]
    sensores = [s for s in stats.stats["Sensor"].tolist() if s in df.columns]

    for sensor in sensores:
        ax.plot(ts, to_numeric(df[sensor]), color="#C0C0C0", linewidth=0.6, alpha=0.6)

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
        ax.plot(ts, to_numeric(df[sensor]), color=color, linewidth=1.6, label=f"{clave.upper()}: {sensor}")

    if analisis:
        ax.plot(ts, analisis.promedio, color="black", linestyle="--", linewidth=2, label="Promedio General")
        ax.axhline(analisis.max_val, color="red", linestyle=":", linewidth=1.4, label=f"Máximo {analisis.max_val:.1f}{unidad}")
        ax.axhline(analisis.min_val, color="blue", linestyle=":", linewidth=1.4, label=f"Mínimo {analisis.min_val:.1f}{unidad}")

    ax.set_title(titulo)
    ax.set_xlabel("Periodo de tiempo")
    ax.set_ylabel(unidad)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
    ax.tick_params(axis="x", rotation=90)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    return fig

def grafico_promedio_intervalos(df: pd.DataFrame, analisis: PromedioAnalisis, titulo: str, unidad: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 6))
    ts = df["Timestamp"]
    promedio = analisis.promedio

    ax.plot(ts, promedio, color="black", linewidth=1.5, label="Promedio General")
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

    ax.set_title(titulo)
    ax.set_xlabel("Periodo de tiempo")
    ax.set_ylabel(unidad)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
    ax.tick_params(axis="x", rotation=90)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    return fig

# -----------------------------------------------------------------------------
# UI principal
# -----------------------------------------------------------------------------
def main() -> None:
    st.title("📈 Reportes de sensores (Normal / Prueba)")
    st.write("Carga un archivo Excel con hojas de temperatura y humedad para obtener el mismo análisis del notebook.")

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

        sheet_temp = st.selectbox("Hoja de temperatura", sheets, index=default_index(DEFAULT_TEMP_SHEET_HINT))
        sheet_hum = st.selectbox("Hoja de humedad", sheets, index=default_index(DEFAULT_HUM_SHEET_HINT))

    try:
        df_temp_raw = read_excel_sheet(file_bytes, sheet_temp)
        df_hum_raw = read_excel_sheet(file_bytes, sheet_hum)
    except Exception as exc:
        st.error(f"No fue posible leer las hojas seleccionadas: {exc}")
        safe_stop()
        return

    df_temp = limpiar_dataframe(df_temp_raw)
    df_hum = limpiar_dataframe(df_hum_raw)

    sensores_temp = detectar_sensores(df_temp, {"Timestamp", "Fecha", "Hora"})
    sensores_hum = detectar_sensores(df_hum, {"Timestamp", "Fecha", "Hora"})

    if not sensores_temp or not sensores_hum:
        st.error("No se detectaron columnas de sensores en una de las hojas.")
        safe_stop()
        return

    with st.sidebar:
        st.header("Configuración de sensores")
        sel_temp = st.multiselect("Sensores de temperatura", sensores_temp, default=sensores_temp)
        sel_hum = st.multiselect("Sensores de humedad", sensores_hum, default=sensores_hum)

        st.header("Archivo de niveles (opcional)")
        map_file = st.file_uploader("CSV/XLSX", type=["csv", "xlsx"], key="map_file")
        df_map_override = load_map_file(map_file)
        #Ejemplo
        st.caption("Encabezados requeridos (primera fila):")
        st.code("sensor_id,level,height", language="text")
        st.code("54853789,1,2.40", language="text")

        st.header("Niveles por sensor (opcional)")
        niveles_manual: dict[str, str] = {}
        for sensor in sel_temp + sel_hum:
            niveles_manual[sensor] = st.text_input(f"Nivel sensor {sensor}", value="", key=f"nivel_{sensor}")

        st.header("Altura por nivel (opcional)")
        alturas_manual: dict[str, str] = {}
        niveles_unicos = sorted({int(val) for val in niveles_manual.values() if val.strip().isdigit()}) or [1, 2, 3]
        for nivel in niveles_unicos:
            alturas_manual[str(nivel)] = st.text_input(
                f"Altura Nivel {nivel} (m)", value="", key=f"altura_{nivel}"
            )

        st.header("Filtro para 'Prueba'")
        min_candidates = [df_temp["Timestamp"].min(), df_hum["Timestamp"].min()]
        max_candidates = [df_temp["Timestamp"].max(), df_hum["Timestamp"].max()]
        valid_min = [ts for ts in min_candidates if pd.notna(ts)]
        valid_max = [ts for ts in max_candidates if pd.notna(ts)]
        if valid_min and valid_max:
            min_ts = min(valid_min)
            max_ts = max(valid_max)
            default_ini = min_ts.strftime("%Y-%m-%d %H:%M")
            default_fin = max_ts.strftime("%Y-%m-%d %H:%M")
            fecha_ini = st.text_input("Fecha inicio (YYYY-MM-DD HH:MM)", value=default_ini)
            fecha_fin = st.text_input("Fecha fin (YYYY-MM-DD HH:MM)", value=default_fin)
        else:
            fecha_ini = st.text_input("Fecha inicio (YYYY-MM-DD HH:MM)")
            fecha_fin = st.text_input("Fecha fin (YYYY-MM-DD HH:MM)")
        intervalo_minutos = st.number_input("Intervalo para rachas (min)", min_value=1, max_value=120, value=DEFAULT_INTERVAL_MINUTES, step=1)
        umbral_temp = st.number_input("Umbral temperatura (°C)", value=20.0, step=0.5)
        tipo_limite_label = st.selectbox(
            "Tipo de límite",
            ("Mayor que (sobre límite)", "Menor que (bajo límite)"),
            index=0,
        )
        tipo_limite = PruebaTipo.MAX if tipo_limite_label.startswith("Mayor") else PruebaTipo.MIN

    if not sel_temp:
        sel_temp = sensores_temp
    if not sel_hum:
        sel_hum = sensores_hum

    df_temp = completar_nulos(df_temp, sel_temp)
    df_hum = completar_nulos(df_hum, sel_hum)

    level_override: dict[str, int | None] = {}
    level_meters: dict[int, float] = {}

    tab_normal, tab_prueba = st.tabs(["Normal", "Prueba"])

    with tab_normal:
        periodo_temp = (df_temp["Timestamp"].min(), df_temp["Timestamp"].max())
        periodo_hum = (df_hum["Timestamp"].min(), df_hum["Timestamp"].max())
        level_override, level_meters = construir_mapa_niveles(
            df_map_override,
            list(dict.fromkeys(sel_temp + sel_hum)),
            niveles_manual,
            alturas_manual,
        )
        render_seccion(
            "Normal",
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
        )

    with tab_prueba:
        df_temp_prueba, ti_temp, tf_temp = filtrar_rango(df_temp, fecha_ini, fecha_fin)
        df_hum_prueba, ti_hum, tf_hum = filtrar_rango(df_hum, fecha_ini, fecha_fin)
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
        )

        # === NUEVO: humedad de TODO el histórico con sombreado de ventana ===
        st.markdown("### Tendencia de humedad con ventana")
        if (ti_hum is not None) and (tf_hum is not None) and sel_hum:
            fig_hum_full = grafico_humedad_prueba_full(df_hum, sel_hum, ti_hum, tf_hum)
            if fig_hum_full is not None:
                st.pyplot(fig_hum_full)
                plt.close(fig_hum_full)
        # === FIN NUEVO ===

        analisis_prueba = None
        if ti_temp is not None and tf_temp is not None:
            analisis_prueba = analizar_prueba(df_temp, sel_temp, ti_temp, tf_temp, umbral_temp, tipo_limite)
        render_prueba_umbrales(analisis_prueba)

if __name__ == "__main__":
    main()
