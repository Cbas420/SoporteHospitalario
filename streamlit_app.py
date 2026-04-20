"""
Dashboard Streamlit del Sistema Inteligente de Soporte Hospitalario.
5 vistas segun SDD §05: Inicio, Diagnosticos, Modelo IA, Alertas, Calidad de Datos.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from config.settings import API_BASE_URL, DASHBOARD_REFRESH_SECONDS, MODELS_DIR

st.set_page_config(
    page_title="Hospital AI Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_json(endpoint: str, fallback):
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=5)
        response.raise_for_status()
        return response.json(), None
    except Exception as exc:
        return fallback, str(exc)


def patch_json(endpoint: str, payload: dict):
    try:
        response = requests.patch(f"{API_BASE_URL}{endpoint}", json=payload, timeout=5)
        response.raise_for_status()
        return response.json(), None
    except Exception as exc:
        return None, str(exc)


def severity_badge(severity: str) -> str:
    colors = {
        "critical": "🔴",
        "high": "🟠",
        "warning": "🟡",
        "medium": "🟡",
        "low": "🟢",
        "info": "🔵",
    }
    return colors.get(severity, "⚪") + f" {severity.upper()}"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🏥 Hospital AI")
page = st.sidebar.radio(
    "Navegacion",
    ["🏠 Inicio", "🩻 Diagnosticos", "🤖 Modelo IA", "🚨 Alertas", "📊 Calidad de Datos"],
)

refresh_seconds = st.sidebar.slider(
    "Refresco (segundos)",
    min_value=5,
    max_value=120,
    value=DASHBOARD_REFRESH_SECONDS,
)
st_autorefresh(interval=refresh_seconds * 1000, key="hospital-dashboard-refresh")

api_health, _ = fetch_json("/health", {})
api_status = "✅ API en linea" if api_health.get("status") == "healthy" else "❌ API no disponible"
model_status = "✅ Modelo cargado" if api_health.get("model_loaded") else "⚠️ Modelo no cargado"
st.sidebar.caption(api_status)
st.sidebar.caption(model_status)


# ===========================================================================
# PAGINA 1: INICIO
# ===========================================================================
if page == "🏠 Inicio":
    st.title("Sistema Inteligente de Soporte Hospitalario")
    st.caption("Dashboard en tiempo real para gestion de pacientes e inferencia de radiografias.")

    stats, stats_err = fetch_json("/stats", {})
    if stats_err:
        st.error(f"No se pudo cargar /stats: {stats_err}")

    # KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🧑‍⚕️ Pacientes", stats.get("patients_total", 0))
    k2.metric("🩻 Predicciones", stats.get("predictions_total", 0))
    k3.metric("🚨 Alertas activas", stats.get("alerts_total", 0))
    k4.metric("⚠️ Requieren revision", stats.get("low_confidence_total", 0))

    st.divider()

    col_chart, col_pred = st.columns([1.3, 1])

    with col_chart:
        st.subheader("Distribucion de pacientes por diagnostico")
        by_diag = stats.get("patients_by_diagnosis", {})
        if by_diag:
            df_diag = pd.DataFrame(
                {"Diagnostico": list(by_diag.keys()), "Pacientes": list(by_diag.values())}
            )
            st.bar_chart(df_diag.set_index("Diagnostico"))
        else:
            st.info("No hay pacientes sincronizados aun.")

        st.subheader("Predicciones por clase (modelo)")
        by_pred = stats.get("predictions_by_class", {})
        if by_pred:
            df_pred = pd.DataFrame(
                {"Clase": list(by_pred.keys()), "Predicciones": list(by_pred.values())}
            )
            st.bar_chart(df_pred.set_index("Clase"))
        else:
            st.info("Aun no hay predicciones registradas.")

    with col_pred:
        st.subheader("Predicciones recientes")
        recent, recent_err = fetch_json("/predictions/recent?limit=10", [])
        if recent_err:
            st.warning(f"Error: {recent_err}")
        elif recent:
            df_recent = pd.DataFrame(recent)[
                ["patient_id", "prediction", "confidence", "created_at"]
            ].copy()
            df_recent["confidence"] = df_recent["confidence"].map("{:.1%}".format)
            st.dataframe(df_recent, use_container_width=True, hide_index=True)
        else:
            st.info("Sin predicciones recientes.")

        st.subheader("Alertas recientes")
        alerts, alerts_err = fetch_json("/alerts?limit=5", [])
        if alerts_err:
            st.warning(f"Error: {alerts_err}")
        elif alerts:
            for alert in alerts[:5]:
                badge = severity_badge(alert.get("severity", ""))
                st.markdown(f"**{badge}** — {alert.get('message', '')[:80]}...")
        else:
            st.success("No hay alertas activas.")


# ===========================================================================
# PAGINA 2: DIAGNOSTICOS
# ===========================================================================
elif page == "🩻 Diagnosticos":
    st.title("🩻 Diagnosticos y Predicciones")

    stats, _ = fetch_json("/stats", {})
    predictions, pred_err = fetch_json("/predictions/recent?limit=200", [])
    patients, pat_err = fetch_json("/patients?limit=200", [])

    # Distribucion de predicciones
    col_pie, col_metrics = st.columns(2)

    with col_pie:
        st.subheader("Distribucion de predicciones del modelo")
        by_pred = stats.get("predictions_by_class", {})
        if by_pred:
            try:
                import plotly.express as px
                fig = px.pie(
                    names=list(by_pred.keys()),
                    values=list(by_pred.values()),
                    color=list(by_pred.keys()),
                    color_discrete_map={
                        "COVID19": "#e74c3c",
                        "Normal": "#2ecc71",
                        "Pneumonia": "#f39c12",
                    },
                    title="Distribucion de clases predichas",
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                df = pd.DataFrame(
                    {"Clase": list(by_pred.keys()), "Predicciones": list(by_pred.values())}
                )
                st.bar_chart(df.set_index("Clase"))
        else:
            st.info("No hay predicciones aun.")

    with col_metrics:
        st.subheader("Distribucion de pacientes (ground truth)")
        by_diag = stats.get("patients_by_diagnosis", {})
        if by_diag:
            try:
                import plotly.express as px
                fig2 = px.pie(
                    names=list(by_diag.keys()),
                    values=list(by_diag.values()),
                    color=list(by_diag.keys()),
                    color_discrete_map={
                        "COVID19": "#e74c3c",
                        "Normal": "#2ecc71",
                        "Pneumonia": "#f39c12",
                    },
                    title="Distribucion de diagnosticos (CSV)",
                )
                st.plotly_chart(fig2, use_container_width=True)
            except ImportError:
                df2 = pd.DataFrame(
                    {"Clase": list(by_diag.keys()), "Pacientes": list(by_diag.values())}
                )
                st.bar_chart(df2.set_index("Clase"))
        else:
            st.info("No hay datos de pacientes.")

    st.divider()

    # Tabla de predicciones con filtros
    st.subheader("Tabla de predicciones")
    if pred_err:
        st.error(f"Error: {pred_err}")
    elif predictions:
        df = pd.DataFrame(predictions)

        filter_col, conf_col = st.columns([1, 1])
        with filter_col:
            clases_disponibles = ["Todas"] + sorted(df["prediction"].unique().tolist())
            clase_filtro = st.selectbox("Filtrar por clase", clases_disponibles)
        with conf_col:
            conf_min = st.slider("Confianza minima", 0.0, 1.0, 0.0, 0.05)

        df_filtered = df.copy()
        if clase_filtro != "Todas":
            df_filtered = df_filtered[df_filtered["prediction"] == clase_filtro]
        df_filtered = df_filtered[df_filtered["confidence"] >= conf_min]

        df_display = df_filtered[
            ["patient_id", "prediction", "confidence", "requires_review", "source", "created_at"]
        ].copy()
        df_display["confidence"] = df_display["confidence"].map("{:.1%}".format)

        st.dataframe(df_display, use_container_width=True, hide_index=True)
        st.caption(f"Mostrando {len(df_display)} de {len(df)} predicciones")
    else:
        st.info("Sin predicciones registradas.")

    st.divider()

    # Tabla de pacientes
    st.subheader("Pacientes registrados")
    if pat_err:
        st.error(f"Error: {pat_err}")
    elif patients:
        df_pat = pd.DataFrame(patients)
        st.dataframe(df_pat, use_container_width=True, hide_index=True)
    else:
        st.info("No hay pacientes en el repositorio.")


# ===========================================================================
# PAGINA 3: MODELO IA
# ===========================================================================
elif page == "🤖 Modelo IA":
    st.title("🤖 Modelo de Inteligencia Artificial")

    model_info, _ = fetch_json("/model/info", {})
    evaluation, eval_err = fetch_json("/evaluation", {})

    # Informacion del modelo
    info_col, _ = st.columns([2, 1])
    with info_col:
        st.subheader("Informacion del modelo")
        if model_info:
            c1, c2, c3 = st.columns(3)
            c1.metric("Nombre", model_info.get("model_name", "N/A"))
            c2.metric("Version", model_info.get("version", "N/A"))
            c3.metric("Estado", model_info.get("status", "N/A"))
            st.caption(f"Clases: {', '.join(model_info.get('classes', []))}")
            st.caption(f"Ruta: {model_info.get('model_path', 'N/A')}")

    st.divider()

    if eval_err:
        st.warning(f"Evaluacion no disponible: {eval_err}")
        st.info("Ejecuta el entrenamiento para generar el informe de evaluacion.")
    else:
        metrics = evaluation.get("evaluation_metrics", {})
        global_metrics = metrics.get("global", {})

        # Metricas globales
        st.subheader("Metricas globales del modelo")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{global_metrics.get('accuracy', 0):.4f}")
        m2.metric("Precision Macro", f"{global_metrics.get('precision_macro', 0):.4f}")
        m3.metric("Recall Macro", f"{global_metrics.get('recall_macro', 0):.4f}")
        m4.metric("F1 Macro", f"{global_metrics.get('f1_macro', 0):.4f}")

        # Metricas por clase
        per_class = metrics.get("per_class", {})
        if per_class:
            st.subheader("Metricas por clase")
            rows = []
            for cls, cls_metrics in per_class.items():
                rows.append({
                    "Clase": cls,
                    "Precision": f"{cls_metrics.get('precision', 0):.4f}",
                    "Recall": f"{cls_metrics.get('recall', 0):.4f}",
                    "F1": f"{cls_metrics.get('f1', 0):.4f}",
                    "Especificidad": f"{cls_metrics.get('specificity', 0):.4f}",
                    "Soporte": cls_metrics.get("support", 0),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()

        # Imagenes generadas por el entrenamiento
        img_col1, img_col2 = st.columns(2)
        confusion_path = MODELS_DIR / "confusion_matrix.png"
        history_path = MODELS_DIR / "training_history.png"

        with img_col1:
            st.subheader("Matriz de confusion")
            if confusion_path.exists():
                st.image(str(confusion_path), use_container_width=True)
            else:
                # Intentar generar desde datos de evaluacion
                confusion_data = metrics.get("confusion_matrix")
                if confusion_data:
                    try:
                        import plotly.figure_factory as ff
                        import numpy as np
                        classes = ["COVID19", "Normal", "Pneumonia"]
                        fig = ff.create_annotated_heatmap(
                            z=confusion_data,
                            x=classes,
                            y=classes,
                            colorscale="Blues",
                            showscale=True,
                        )
                        fig.update_layout(
                            title="Matriz de Confusion",
                            xaxis_title="Prediccion",
                            yaxis_title="Real",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        st.info("Matriz de confusion no disponible aun.")
                else:
                    st.info("Ejecuta el entrenamiento para generar la matriz de confusion.")

        with img_col2:
            st.subheader("Curvas de entrenamiento")
            if history_path.exists():
                st.image(str(history_path), use_container_width=True)
            else:
                history_json = MODELS_DIR / "training_history.json"
                if history_json.exists():
                    try:
                        import plotly.graph_objects as go
                        with history_json.open("r") as f:
                            history = json.load(f)
                        fig = go.Figure()
                        if "accuracy" in history:
                            fig.add_trace(go.Scatter(y=history["accuracy"], name="Train Acc"))
                        if "val_accuracy" in history:
                            fig.add_trace(go.Scatter(y=history["val_accuracy"], name="Val Acc"))
                        fig.update_layout(title="Accuracy por Epoch", xaxis_title="Epoch")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        st.info("Historial de entrenamiento no disponible.")
                else:
                    st.info("Ejecuta el entrenamiento para ver las curvas.")

        # Analisis clinico
        clinical = evaluation.get("clinical_analysis", {})
        if clinical.get("risk_assessment"):
            st.divider()
            st.subheader("Evaluacion de riesgo clinico")
            for warning in clinical["risk_assessment"]:
                st.warning(warning)


# ===========================================================================
# PAGINA 4: ALERTAS
# ===========================================================================
elif page == "🚨 Alertas":
    st.title("🚨 Alertas Clinicas")

    alerts, alerts_err = fetch_json("/alerts?limit=200", [])

    if alerts_err:
        st.error(f"No se pudieron cargar las alertas: {alerts_err}")
    elif not alerts:
        st.success("✅ No hay alertas activas en el sistema.")
    else:
        df_alerts = pd.DataFrame(alerts)

        # Contadores por severidad
        severities = df_alerts["severity"].value_counts().to_dict()
        cols = st.columns(len(severities) or 1)
        for idx, (sev, count) in enumerate(severities.items()):
            badge = severity_badge(sev)
            cols[idx].metric(badge, count)

        st.divider()

        # Filtros
        f1, f2, f3 = st.columns(3)
        with f1:
            sev_options = ["Todas"] + sorted(df_alerts["severity"].unique().tolist())
            sev_filter = st.selectbox("Severidad", sev_options)
        with f2:
            type_options = ["Todos"] + sorted(df_alerts["alert_type"].unique().tolist())
            type_filter = st.selectbox("Tipo de alerta", type_options)
        with f3:
            resolved_filter = st.selectbox("Estado", ["Todas", "Pendientes", "Resueltas"])

        df_filtered = df_alerts.copy()
        if sev_filter != "Todas":
            df_filtered = df_filtered[df_filtered["severity"] == sev_filter]
        if type_filter != "Todos":
            df_filtered = df_filtered[df_filtered["alert_type"] == type_filter]
        if resolved_filter == "Pendientes":
            df_filtered = df_filtered[df_filtered["resolved"] == False]
        elif resolved_filter == "Resueltas":
            df_filtered = df_filtered[df_filtered["resolved"] == True]

        st.subheader(f"Alertas ({len(df_filtered)} resultados)")

        # Tabla de alertas con detalles
        for _, row in df_filtered.head(50).iterrows():
            badge = severity_badge(row.get("severity", ""))
            resolved_icon = "✅" if row.get("resolved") else "🔔"
            with st.expander(
                f"{resolved_icon} {badge} | {row.get('alert_type', '')} | Paciente: {row.get('patient_id', 'N/A')}",
                expanded=row.get("severity") in ("critical", "high") and not row.get("resolved"),
            ):
                st.write(f"**Mensaje:** {row.get('message', '')}")
                st.write(f"**Fuente:** {row.get('source', 'N/A')}")
                st.write(f"**Creada:** {row.get('created_at', 'N/A')}")
                if row.get("image_path"):
                    st.caption(f"Imagen: {row.get('image_path')}")


# ===========================================================================
# PAGINA 5: CALIDAD DE DATOS
# ===========================================================================
elif page == "📊 Calidad de Datos":
    st.title("📊 Calidad de Datos y Pipeline")

    pipeline_report, pipe_err = fetch_json("/pipeline/report", {})

    if pipe_err:
        st.warning(f"No se pudo cargar el informe del pipeline: {pipe_err}")
        st.info("Ejecuta el pipeline de datos para generar el informe.")
    else:
        # Metricas del pipeline de pacientes
        patients_info = pipeline_report.get("patients", {})
        dataset_info = pipeline_report.get("dataset", {})
        storage_info = pipeline_report.get("storage", {})

        st.subheader("Resumen del pipeline")
        status_icon = "✅" if pipeline_report.get("status") == "ready" else "⚠️"
        st.markdown(f"**Estado:** {status_icon} {pipeline_report.get('status', 'N/A').upper()}")

        col_pat, col_ds = st.columns(2)

        with col_pat:
            st.subheader("Datos de pacientes (CSV)")
            if patients_info:
                p1, p2, p3 = st.columns(3)
                p1.metric("Registros entrada", patients_info.get("input_rows", 0))
                p2.metric("Registros limpios", patients_info.get("records_after_cleaning", 0))
                p3.metric("Duplicados eliminados", patients_info.get("duplicates_removed", 0))

                q1, q2 = st.columns(2)
                q1.metric("Diagnosticos invalidos", patients_info.get("invalid_diagnoses_removed", 0))
                q2.metric("Imagenes faltantes", patients_info.get("missing_images_removed", 0))

                # Indicador de calidad
                total_in = patients_info.get("input_rows", 0)
                total_out = patients_info.get("records_after_cleaning", 0)
                if total_in > 0:
                    quality_pct = total_out / total_in * 100
                    color = "normal" if quality_pct > 80 else "inverse"
                    st.metric("Tasa de calidad", f"{quality_pct:.1f}%")
            else:
                st.info("Sin datos de pacientes disponibles.")

        with col_ds:
            st.subheader("Dataset de imagenes")
            if dataset_info:
                total_images = dataset_info.get("total_images", 0)
                st.metric("Total imagenes", total_images)

                classes_dist = dataset_info.get("classes", {})
                if classes_dist:
                    df_cls = pd.DataFrame(
                        {"Clase": list(classes_dist.keys()), "Imagenes": list(classes_dist.values())}
                    )
                    st.bar_chart(df_cls.set_index("Clase"))
            else:
                st.info("Sin informacion del dataset.")

        st.divider()

        # Estado del almacenamiento
        st.subheader("Estado del almacenamiento")
        obj_storage = storage_info.get("object_storage", {})

        s1, s2 = st.columns(2)
        with s1:
            st.markdown("**Base de datos**")
            db_backend = storage_info.get("database", "N/A")
            st.info(f"Backend: {db_backend}")

        with s2:
            st.markdown("**Almacenamiento de objetos (MinIO)**")
            if obj_storage.get("reachable"):
                bucket_exists = obj_storage.get("bucket_exists", False)
                st.success(f"✅ MinIO accesible — Bucket: {'✅' if bucket_exists else '❌'}")
            else:
                err = obj_storage.get("error", "No disponible")
                st.warning(f"⚠️ MinIO no disponible: {err}")
                st.caption("MinIO esta disponible solo con Docker Compose.")
