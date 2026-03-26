import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import find_peaks
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.units import cm
import matplotlib.pyplot as plt

st.set_page_config(page_title="NMR Report Generator", layout="wide")

# -------------------------------------------------
# Session state
# -------------------------------------------------
if "nmr_results_ready" not in st.session_state:
    st.session_state.nmr_results_ready = False
if "nmr_review_df" not in st.session_state:
    st.session_state.nmr_review_df = pd.DataFrame()
if "nmr_parsed" not in st.session_state:
    st.session_state.nmr_parsed = {}
if "nmr_warnings_df" not in st.session_state:
    st.session_state.nmr_warnings_df = pd.DataFrame()
if "nmr_integrals_df" not in st.session_state:
    st.session_state.nmr_integrals_df = pd.DataFrame()
if "nmr_peaks_df" not in st.session_state:
    st.session_state.nmr_peaks_df = pd.DataFrame()
if "nmr_current_dataset" not in st.session_state:
    st.session_state.nmr_current_dataset = None


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def normalize_name(name: str) -> str:
    stem = Path(name).stem.strip()
    stem = re.sub(r"\s+", " ", stem)
    return stem


def read_text_lines(uploaded_file):
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    raw = uploaded_file.read()
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = str(raw)
    return text.splitlines()


def parse_nmr_file(uploaded_file):
    """
    Generic parser for exported NMR text/asc/csv files.
    Looks for a numeric 2-column table: ppm, intensity.
    """
    lines = read_text_lines(uploaded_file)
    candidates = []

    for delimiter in [",", ";", "\t", None]:
        xs = []
        ys = []
        for line in lines:
            if delimiter is None:
                parts = line.strip().replace(",", ".").split()
            else:
                parts = [p.strip().strip('"') for p in line.split(delimiter)]
            if len(parts) < 2:
                continue
            nums = []
            for p in parts[:4]:
                try:
                    nums.append(float(p.replace(",", ".")))
                except Exception:
                    nums.append(None)
            found = False
            for i in range(len(nums) - 1):
                if nums[i] is not None and nums[i + 1] is not None:
                    x = nums[i]
                    y = nums[i + 1]
                    if -5 <= x <= 20:
                        xs.append(x)
                        ys.append(y)
                        found = True
                        break
            if not found:
                continue
        if len(xs) >= 10:
            candidates.append((delimiter, np.array(xs, dtype=float), np.array(ys, dtype=float)))

    if not candidates:
        raise ValueError("Could not find a numeric ppm/intensity table in the file.")

    delimiter, x, y = max(candidates, key=lambda t: len(t[1]))
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique_x, idx = np.unique(x, return_index=True)
    x = unique_x
    y = y[idx]

    if len(x) < 10:
        raise ValueError("Parsed too few usable spectral points.")

    return x, y, delimiter


def build_review_table(uploaded_files):
    rows = []
    parsed = {}
    warnings = []

    for f in uploaded_files:
        dataset = normalize_name(f.name)
        try:
            ppm, intensity, delimiter = parse_nmr_file(f)
            parsed[dataset] = {
                "ppm": ppm,
                "intensity": intensity,
                "source_file": f.name,
                "delimiter": delimiter,
            }
            rows.append({
                "Dataset": dataset,
                "Points": len(ppm),
                "Min ppm": float(np.min(ppm)),
                "Max ppm": float(np.max(ppm)),
            })
        except Exception as e:
            warnings.append({
                "Dataset": dataset,
                "Type": "Parsing error",
                "Message": str(e),
            })

    return pd.DataFrame(rows), parsed, pd.DataFrame(warnings)


def subset_region(ppm, intensity, lo, hi):
    left = min(lo, hi)
    right = max(lo, hi)
    mask = (ppm >= left) & (ppm <= right)
    return ppm[mask], intensity[mask]


def integrate_region(ppm, intensity, start_ppm, end_ppm):
    x, y = subset_region(ppm, intensity, start_ppm, end_ppm)
    if len(x) < 2:
        return np.nan
    return float(abs(np.trapezoid(y, x)))


def peak_pick(ppm, intensity, prominence_factor=0.03, min_distance_pts=8, lo=None, hi=None):
    x = ppm
    y = intensity
    if lo is not None and hi is not None:
        x, y = subset_region(ppm, intensity, lo, hi)
    if len(x) < 5:
        return pd.DataFrame(columns=["Peak ppm", "Height"])

    y_range = float(np.max(y) - np.min(y))
    prominence = max(y_range * prominence_factor, 1e-12)
    peaks, props = find_peaks(y, prominence=prominence, distance=min_distance_pts)
    out = pd.DataFrame({
        "Peak ppm": x[peaks],
        "Height": y[peaks],
    })
    out = out.sort_values(by="Peak ppm", ascending=False).reset_index(drop=True)
    return out


def build_plotly_figure(ppm, intensity, integrations_df=None, peaks_df=None, display_range=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ppm, y=intensity, mode="lines", name="Spectrum"))

    if integrations_df is not None and not integrations_df.empty:
        for _, row in integrations_df.iterrows():
            start_ppm = float(row["Start ppm"])
            end_ppm = float(row["End ppm"])
            label = str(row["Label"])
            left = min(start_ppm, end_ppm)
            right = max(start_ppm, end_ppm)
            fig.add_vrect(
                x0=left,
                x1=right,
                fillcolor="lightblue",
                opacity=0.18,
                line_width=0,
                annotation_text=label,
                annotation_position="top left",
            )

    if peaks_df is not None and not peaks_df.empty:
        fig.add_trace(
            go.Scatter(
                x=peaks_df["Peak ppm"],
                y=peaks_df["Height"],
                mode="markers+text",
                name="Picked peaks",
                text=[f"{v:.3f}" for v in peaks_df["Peak ppm"]],
                textposition="top center",
            )
        )

    fig.update_layout(
        title="NMR Spectrum",
        xaxis_title="Chemical shift (ppm)",
        yaxis_title="Intensity",
        hovermode="x unified",
        legend_title="Traces",
    )

    if display_range is not None:
        fig.update_xaxes(range=[display_range[1], display_range[0]])
    else:
        fig.update_xaxes(autorange="reversed")

    return fig


def spectrum_png_bytes(dataset_name, ppm, intensity, integrations_df=None, peaks_df=None, display_range=None):
    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    ax.plot(ppm, intensity, lw=1.2)
    ax.set_xlabel("Chemical shift (ppm)")
    ax.set_ylabel("Intensity")
    ax.set_title(dataset_name)
    ax.grid(True, alpha=0.2)
    ax.invert_xaxis()

    if display_range is not None:
        lo, hi = display_range
        ax.set_xlim(max(lo, hi), min(lo, hi))

    if integrations_df is not None and not integrations_df.empty:
        for _, row in integrations_df.iterrows():
            start_ppm = float(row["Start ppm"])
            end_ppm = float(row["End ppm"])
            label = str(row["Label"])
            left = min(start_ppm, end_ppm)
            right = max(start_ppm, end_ppm)
            mask = (ppm >= left) & (ppm <= right)
            if np.any(mask):
                ax.fill_between(ppm[mask], intensity[mask], alpha=0.18)
                xm = (left + right) / 2
                ym = float(np.max(intensity[mask]))
                ax.text(xm, ym, label, fontsize=8, ha="center", va="bottom")

    if peaks_df is not None and not peaks_df.empty:
        ax.scatter(peaks_df["Peak ppm"], peaks_df["Height"], s=12)
        for _, row in peaks_df.iterrows():
            ax.text(float(row["Peak ppm"]), float(row["Height"]), f"{row['Peak ppm']:.3f}", fontsize=7)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def build_pdf_report(dataset_name, ppm, intensity, peaks_df, integrals_df, display_range=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), leftMargin=1.0 * cm, rightMargin=1.0 * cm,
                            topMargin=1.0 * cm, bottomMargin=1.0 * cm)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"NMR Report — {dataset_name}", styles["Title"]))
    story.append(Spacer(1, 0.3 * cm))

    img_buf = spectrum_png_bytes(dataset_name, ppm, intensity, integrals_df, peaks_df, display_range)
    story.append(Image(img_buf, width=25 * cm, height=9 * cm))
    story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph("Picked Peaks", styles["Heading2"]))
    if peaks_df is None or peaks_df.empty:
        story.append(Paragraph("No peaks picked.", styles["BodyText"]))
    else:
        peak_table_data = [["Peak ppm", "Height"]] + [
            [f"{float(r['Peak ppm']):.4f}", f"{float(r['Height']):.4f}"] for _, r in peaks_df.iterrows()
        ]
        t = Table(peak_table_data)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]))
        story.append(t)

    story.append(Spacer(1, 0.45 * cm))
    story.append(Paragraph("Integrals", styles["Heading2"]))
    if integrals_df is None or integrals_df.empty:
        story.append(Paragraph("No integration regions defined.", styles["BodyText"]))
    else:
        cols = ["Label", "Start ppm", "End ppm", "Raw integral", "Normalized integral"]
        integral_table_data = [cols]
        for _, r in integrals_df.iterrows():
            raw_val = r["Raw integral"]
            norm_val = r.get("Normalized integral", np.nan)
            integral_table_data.append([
                str(r["Label"]),
                f"{float(r['Start ppm']):.4f}",
                f"{float(r['End ppm']):.4f}",
                "" if pd.isna(raw_val) else f"{float(raw_val):.6f}",
                "" if pd.isna(norm_val) else f"{float(norm_val):.6f}",
            ])
        t2 = Table(integral_table_data)
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]))
        story.append(t2)

    doc.build(story)
    buffer.seek(0)
    return buffer


# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("NMR Report Generator")
st.caption(
    "Upload one or more exported NMR spectra, inspect them interactively, pick peaks, define integration regions, and export one PDF report per spectrum."
)

left, right = st.columns([1, 1.8], gap="large")

with left:
    st.subheader("Inputs")

    uploaded_files = st.file_uploader(
        "1. Drop NMR export files",
        type=["asc", "txt", "dat", "csv"],
        accept_multiple_files=True,
        key="nmr_uploaded_files",
    )

    preview = st.button("Preview parsing", type="secondary", width="stretch")
    run_analysis = st.button("Load spectra", type="primary", width="stretch")

with right:
    st.subheader("Review and results")

    if preview or (uploaded_files and st.session_state.nmr_review_df.empty):
        if not uploaded_files:
            st.warning("Upload NMR files first.")
        else:
            review_df, parsed, warnings_df = build_review_table(uploaded_files)
            st.session_state.nmr_review_df = review_df
            st.session_state.nmr_parsed = parsed
            st.session_state.nmr_warnings_df = warnings_df
            if not review_df.empty and st.session_state.nmr_current_dataset is None:
                st.session_state.nmr_current_dataset = review_df.iloc[0]["Dataset"]

    if run_analysis:
        if not uploaded_files:
            st.error("Please upload the NMR files.")
        else:
            review_df, parsed, warnings_df = build_review_table(uploaded_files)
            st.session_state.nmr_review_df = review_df
            st.session_state.nmr_parsed = parsed
            st.session_state.nmr_warnings_df = warnings_df
            st.session_state.nmr_results_ready = True
            if not review_df.empty:
                st.session_state.nmr_current_dataset = review_df.iloc[0]["Dataset"]

    if st.session_state.nmr_results_ready:
        review_df = st.session_state.nmr_review_df
        parsed = st.session_state.nmr_parsed
        warnings_df = st.session_state.nmr_warnings_df

        tab1, tab2, tab3, tab4 = st.tabs([
            "Summary",
            "Spectrum",
            "Integrals + Peaks",
            "Warnings",
        ])

        with tab1:
            st.subheader("Parsed spectra")
            st.dataframe(review_df, width="stretch")

        with tab2:
            st.subheader("Interactive spectrum")
            if not parsed:
                st.info("No spectra available.")
            else:
                dataset_options = sorted(parsed.keys())
                selected_dataset = st.selectbox(
                    "Select spectrum",
                    options=dataset_options,
                    index=dataset_options.index(st.session_state.nmr_current_dataset) if st.session_state.nmr_current_dataset in dataset_options else 0,
                    key="nmr_dataset_selector",
                )
                st.session_state.nmr_current_dataset = selected_dataset

                d = parsed[selected_dataset]
                ppm = d["ppm"]
                intensity = d["intensity"]

                col_a, col_b = st.columns(2)
                with col_a:
                    x_min = st.number_input("Display min ppm", value=float(np.min(ppm)), step=0.1, key="nmr_xmin")
                with col_b:
                    x_max = st.number_input("Display max ppm", value=float(np.max(ppm)), step=0.1, key="nmr_xmax")

                display_range = (x_min, x_max)

                integrals_editor = st.data_editor(
                    pd.DataFrame(
                        [{"Label": "Region 1", "Start ppm": 7.30, "End ppm": 7.10}]
                    ),
                    width="stretch",
                    num_rows="dynamic",
                    key=f"nmr_integrals_editor_{selected_dataset}",
                    column_config={
                        "Label": st.column_config.TextColumn("Label"),
                        "Start ppm": st.column_config.NumberColumn("Start ppm", step=0.01),
                        "End ppm": st.column_config.NumberColumn("End ppm", step=0.01),
                    },
                )

                normalize_choice = st.selectbox(
                    "Normalize integrals to region",
                    options=["None"] + integrals_editor["Label"].dropna().astype(str).tolist(),
                    key=f"nmr_norm_region_{selected_dataset}",
                )
                normalize_target = None
                if normalize_choice != "None":
                    normalize_target = st.number_input(
                        "Normalized value for selected region",
                        min_value=0.0001,
                        value=1.0,
                        step=0.1,
                        key=f"nmr_norm_target_{selected_dataset}",
                    )

                prominence_factor = st.slider(
                    "Peak-picking prominence factor",
                    min_value=0.001,
                    max_value=0.2,
                    value=0.03,
                    step=0.001,
                    key=f"nmr_prom_{selected_dataset}",
                )
                min_distance_pts = st.slider(
                    "Peak-picking minimum point distance",
                    min_value=1,
                    max_value=100,
                    value=8,
                    step=1,
                    key=f"nmr_dist_{selected_dataset}",
                )
                peak_lo, peak_hi = st.slider(
                    "Peak-picking ppm range",
                    min_value=float(np.min(ppm)),
                    max_value=float(np.max(ppm)),
                    value=(float(np.min(ppm)), float(np.max(ppm))),
                    step=0.01,
                    key=f"nmr_peak_range_{selected_dataset}",
                )

                peaks_df = peak_pick(
                    ppm,
                    intensity,
                    prominence_factor=prominence_factor,
                    min_distance_pts=min_distance_pts,
                    lo=peak_lo,
                    hi=peak_hi,
                )

                integrals_rows = []
                for _, row in integrals_editor.iterrows():
                    label = row.get("Label")
                    start_ppm = row.get("Start ppm")
                    end_ppm = row.get("End ppm")
                    if pd.isna(start_ppm) or pd.isna(end_ppm):
                        continue
                    raw_integral = integrate_region(ppm, intensity, float(start_ppm), float(end_ppm))
                    integrals_rows.append({
                        "Label": label,
                        "Start ppm": float(start_ppm),
                        "End ppm": float(end_ppm),
                        "Raw integral": raw_integral,
                    })
                integrals_df = pd.DataFrame(integrals_rows)

                if not integrals_df.empty:
                    if normalize_choice != "None":
                        ref_rows = integrals_df[integrals_df["Label"].astype(str) == normalize_choice]
                        if not ref_rows.empty and pd.notna(ref_rows.iloc[0]["Raw integral"]):
                            ref_val = float(ref_rows.iloc[0]["Raw integral"])
                            if abs(ref_val) > 1e-12:
                                scale = float(normalize_target) / ref_val
                                integrals_df["Normalized integral"] = integrals_df["Raw integral"] * scale
                            else:
                                integrals_df["Normalized integral"] = np.nan
                        else:
                            integrals_df["Normalized integral"] = np.nan
                    else:
                        integrals_df["Normalized integral"] = np.nan

                st.session_state.nmr_integrals_df = integrals_df
                st.session_state.nmr_peaks_df = peaks_df

                fig = build_plotly_figure(
                    ppm,
                    intensity,
                    integrations_df=integrals_df,
                    peaks_df=peaks_df,
                    display_range=display_range,
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("Peaks, integrals, and PDF export")
            dataset_name = st.session_state.nmr_current_dataset
            if not dataset_name or dataset_name not in parsed:
                st.info("Load and select a spectrum first.")
            else:
                ppm = parsed[dataset_name]["ppm"]
                intensity = parsed[dataset_name]["intensity"]
                peaks_df = st.session_state.nmr_peaks_df
                integrals_df = st.session_state.nmr_integrals_df
                display_range = (st.session_state.get("nmr_xmin", float(np.min(ppm))), st.session_state.get("nmr_xmax", float(np.max(ppm))))

                st.subheader("Picked peaks")
                st.dataframe(peaks_df, width="stretch")
                st.download_button(
                    "Download peaks CSV",
                    data=peaks_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{dataset_name}_peaks.csv",
                    mime="text/csv",
                    width="stretch",
                )

                st.subheader("Integrals")
                st.dataframe(integrals_df, width="stretch")
                st.download_button(
                    "Download integrals CSV",
                    data=integrals_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{dataset_name}_integrals.csv",
                    mime="text/csv",
                    width="stretch",
                )

                pdf_buf = build_pdf_report(
                    dataset_name=dataset_name,
                    ppm=ppm,
                    intensity=intensity,
                    peaks_df=peaks_df,
                    integrals_df=integrals_df,
                    display_range=display_range,
                )
                st.download_button(
                    "Download PDF report",
                    data=pdf_buf,
                    file_name=f"{dataset_name}_nmr_report.pdf",
                    mime="application/pdf",
                    width="stretch",
                )

        with tab4:
            st.subheader("Warnings")
            if warnings_df.empty:
                st.success("No warnings.")
            else:
                st.dataframe(warnings_df, width="stretch")
    else:
        st.info("Upload files, preview parsing, and load spectra to start.")
