import json
import os
import re
import zipfile
from io import BytesIO
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


st.set_page_config(
    page_title="Traffic Violation Detection",
    page_icon="TV",
    layout="wide",
)


def _find_model_path() -> str | None:
    env_path = os.getenv("TRAFFIC_MODEL_PATH")
    candidates = [
        env_path,
        "best_traffic_model.pt",
        "traffic_model.pt",
        os.path.join("runs", "detect", "train", "weights", "best.pt"),
        os.path.join("runs", "detect", "train-3", "weights", "best.pt"),
        os.path.join("runs", "detect", "train-2", "weights", "best.pt"),
        os.path.join("runs", "detect", "train-1", "weights", "best.pt"),
        os.path.join("master_traffic_violation_dataset", "best_traffic_model.pt"),
        os.path.join("models", "best_traffic_model.pt"),
        os.path.join("weights", "best_traffic_model.pt"),
    ]

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def _save_uploaded_model(uploaded_model) -> str:
    file_name = uploaded_model.name
    target_path = os.path.join(os.getcwd(), file_name)
    with open(target_path, "wb") as target_file:
        target_file.write(uploaded_model.getbuffer())

    if file_name.lower().endswith(".zip"):
        with zipfile.ZipFile(target_path, "r") as archive:
            pt_members = [name for name in archive.namelist() if name.lower().endswith(".pt")]
            if not pt_members:
                names = set(archive.namelist())
                is_torch_bundle = any(name.endswith("data.pkl") for name in names)
                if is_torch_bundle:
                    # Torch checkpoints are often zip-backed files with entries like
                    # data.pkl, .data/, byteorder, and version. In this case, the zip
                    # file itself is the model payload, so store it as .pt.
                    final_path = os.path.join(os.getcwd(), "best_traffic_model.pt")
                    with open(target_path, "rb") as src_file, open(final_path, "wb") as dst_file:
                        dst_file.write(src_file.read())
                    return final_path
                raise ValueError("Zip file does not contain .pt weights or a Torch checkpoint bundle.")
            chosen_member = pt_members[0]
            extracted_path = archive.extract(chosen_member, os.getcwd())
            final_path = os.path.join(os.getcwd(), "best_traffic_model.pt")
            os.replace(extracted_path, final_path)
            return final_path

    return target_path


@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


@st.cache_resource(show_spinner=False)
def load_reader():
    try:
        import easyocr
    except ModuleNotFoundError:
        return None

    return easyocr.Reader(["en"], gpu=False)


@st.cache_resource(show_spinner=False)
def load_rapidocr():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ModuleNotFoundError:
        return None

    return RapidOCR()


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def clean_plate_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _prepare_ocr_variants(plate_bgr: np.ndarray) -> list[np.ndarray]:
    grayscale = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    enlarged = cv2.resize(grayscale, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    equalized = cv2.equalizeHist(enlarged)
    blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
    sharpened = cv2.addWeighted(equalized, 1.6, blurred, -0.6, 0)

    _, threshold = cv2.threshold(
        sharpened,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    inverted_threshold = cv2.bitwise_not(threshold)

    return [
        cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB),
        equalized,
        sharpened,
        threshold,
        inverted_threshold,
    ]


def extract_plate_number(plate_bgr: np.ndarray) -> tuple[str, float]:
    reader = load_reader()
    best_text = ""
    best_conf = 0.0

    if reader is not None:
        for variant in _prepare_ocr_variants(plate_bgr):
            try:
                results = reader.readtext(
                    variant,
                    detail=1,
                    paragraph=False,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    rotation_info=[0, 180],
                    decoder="beamsearch",
                )
            except TypeError:
                results = reader.readtext(variant, detail=1, paragraph=False)

            for _, text, confidence in results:
                normalized = clean_plate_text(text)
                if len(normalized) < 4:
                    continue
                if confidence > best_conf or (confidence == best_conf and len(normalized) > len(best_text)):
                    best_text = normalized
                    best_conf = float(confidence)

            if best_conf >= 0.55 and best_text:
                break

    if best_text:
        return best_text, best_conf

    rapidocr = load_rapidocr()
    if rapidocr is None:
        return "", 0.0

    for variant in _prepare_ocr_variants(plate_bgr):
        result = rapidocr(variant)
        if not result:
            continue

        texts, _ = result
        for item in texts or []:
            if len(item) < 3:
                continue
            _, text, confidence = item
            normalized = clean_plate_text(str(text))
            if len(normalized) < 4:
                continue
            if confidence > best_conf or (confidence == best_conf and len(normalized) > len(best_text)):
                best_text = normalized
                best_conf = float(confidence)

        if best_text:
            break

    return best_text, best_conf


def extract_plate_number_with_fallback(image_bgr: np.ndarray, plate_crop: np.ndarray | None) -> tuple[str, float]:
    if plate_crop is not None and plate_crop.size:
        plate_number, plate_confidence = extract_plate_number(plate_crop)
        if plate_number:
            return plate_number, plate_confidence

    return extract_plate_number(image_bgr)


def crop_with_padding(image_bgr: np.ndarray, xyxy: np.ndarray, padding: int = 8) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = xyxy.astype(int)
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return image_bgr[y1:y2, x1:x2].copy()


def crop_plate_region(image_bgr: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = xyxy.astype(float)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    padding = int(max(12, round(max(width, height) * 0.18)))
    return crop_with_padding(image_bgr, xyxy, padding=padding)


def annotate_detections(result, image_bgr: np.ndarray) -> np.ndarray:
    annotated = result.plot()
    return annotated if annotated is not None else image_bgr


def render_summary_cards(result: dict):
    c1, c2, c3 = st.columns(3)
    c1.metric("Violation Count", result["violation_count"])
    c2.metric("Vehicle Number", result["plate_number"] or "Not detected")
    c3.metric("Detections", len(result["detections"]))


def build_report(upload_name: str, result: dict) -> dict:
    return {
        "file_name": upload_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "violation_type": result["violation_type"],
        "violation_items": result["violation_items"],
        "violation_count": result["violation_count"],
        "violation_flag": result["has_violation"],
        "vehicle_number": result["plate_number"] or "Not detected",
        "detections": result["detections"],
    }


def _pretty_violation_label(label: str) -> str:
    return {
        "WithoutHelmet": "No Helmet",
        "TripleRiding": "Triple Riding",
        "No OCR Number": "No OCR Number",
    }.get(label, label)


def _build_violation_items(detections: list[dict], plate_number: str) -> list[str]:
    violation_items: list[str] = []

    if any(item["class_name"] == "WithoutHelmet" for item in detections):
        violation_items.append("WithoutHelmet")

    if any(item["class_name"] == "TripleRiding" for item in detections):
        violation_items.append("TripleRiding")

    if not plate_number:
        violation_items.append("No OCR Number")

    return violation_items


def build_pdf_report(report: dict) -> bytes:
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("AI-Based Traffic Violation Detection Report", styles["Title"]))
    story.append(Spacer(1, 0.2 * inch))

    summary_rows = [
        ["File Name", report.get("file_name", "")],
        ["Generated At", report.get("generated_at", "")],
        ["Violation Type", report.get("violation_type", "")],
        ["Violation Count", str(report.get("violation_count", 0))],
        ["Vehicle Number", report.get("vehicle_number", "Not detected")],
        ["Status", "Violation detected" if report.get("violation_flag") else "No violation detected"],
    ]

    summary_table = Table(summary_rows, colWidths=[1.7 * inch, 4.7 * inch])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#4b5563")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#111827")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#1f2937"), colors.HexColor("#111827")]),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 0.2 * inch))

    violation_items = report.get("violation_items", [])
    story.append(Paragraph("Violation List", styles["Heading2"]))
    if violation_items:
        for item in violation_items:
            story.append(Paragraph(f"- {_pretty_violation_label(item)}", styles["BodyText"]))
    else:
        story.append(Paragraph("No violations detected.", styles["BodyText"]))

    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Detected Objects", styles["Heading2"]))
    detections = report.get("detections", [])
    if detections:
        detection_rows = [["Class", "Confidence", "Box"]]
        for item in detections:
            detection_rows.append([
                item.get("class_name", ""),
                f"{item.get('confidence', 0):.4f}",
                ", ".join(str(round(value, 1)) for value in item.get("xyxy", [])),
            ])
        detection_table = Table(detection_rows, colWidths=[1.7 * inch, 1.0 * inch, 3.7 * inch])
        detection_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#374151")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#4b5563")),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(detection_table)
    else:
        story.append(Paragraph("No detected objects.", styles["BodyText"]))

    document.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def main():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; }
        .hero {
            padding: 1.4rem 1.6rem;
            border-radius: 1.1rem;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #0b1120 100%);
            color: white;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 18px 50px rgba(15, 23, 42, 0.25);
        }
        .hero h1 { margin-bottom: 0.35rem; }
        .hero p { margin: 0; opacity: 0.88; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <h1>AI-Based Traffic Violation Detection</h1>
            <p>Upload a vehicle image, run YOLOv8 detection, extract the plate number with OCR, and download a violation report.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_path = _find_model_path()
    with st.sidebar:
        st.header("Pipeline Settings")
        st.write("Model path")
        model_path_input = st.text_input(
            "Path to YOLO .pt file",
            value=model_path or "best_traffic_model.pt",
            help="Use the exact local path to your trained YOLO weights.",
        )
        if model_path_input and os.path.exists(model_path_input):
            model_path = model_path_input
        st.code(model_path or "best_traffic_model.pt not found")
        uploaded_model = st.file_uploader("Upload model weights (.pt or .zip)", type=["pt", "zip"])
        if uploaded_model is not None:
            try:
                model_path = _save_uploaded_model(uploaded_model)
                st.success(f"Model saved to {model_path}")
            except Exception as model_error:
                st.error(f"Model upload failed: {model_error}")
                model_path = None
        conf_threshold = st.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.05)
        st.caption("Place best_traffic_model.pt in the workspace root, or set TRAFFIC_MODEL_PATH / paste the path above.")
        st.caption("If you trained locally with Ultralytics, the app also checks runs/detect/*/weights/best.pt.")
        if load_reader() is None:
            st.warning("EasyOCR not installed. OCR is disabled until you install easyocr.")

    uploaded_file = st.file_uploader(
        "Upload vehicle image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is None:
        st.info("Upload an image to start detection.")
        st.stop()

    input_image = Image.open(uploaded_file)
    image_bgr = pil_to_bgr(input_image)

    left, right = st.columns([1, 1.15])
    with left:
        st.subheader("Uploaded Image")
        st.image(input_image, use_container_width=True)

    if model_path is None:
        st.warning("No model found yet. Upload the .pt file from the sidebar to enable detection.")
        annotated_bgr = image_bgr.copy()
        detections = []
        plate_crop = None
        plate_number = ""
        plate_confidence = 0.0
        violation_items = ["Demo mode"]
        violation_type = "Demo mode"
        violation_count = 0
        has_violation = False
        pipeline_result = {
            "annotated_bgr": annotated_bgr,
            "detections": detections,
            "plate_crop": plate_crop,
            "plate_number": plate_number,
            "plate_confidence": plate_confidence,
            "violation_type": violation_type,
            "violation_items": violation_items,
            "violation_count": violation_count,
            "has_violation": has_violation,
        }
    else:
        model = load_model(model_path)
        result = model.predict(image_bgr, conf=conf_threshold, verbose=False)[0]
        annotated_bgr = annotate_detections(result, image_bgr)

        detections = []
        plate_candidates = []
        class_names = model.names

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                cls_name = class_names.get(cls_id, str(cls_id))
                conf = float(box.conf.item())
                xyxy = box.xyxy[0].cpu().numpy()
                detections.append(
                    {
                        "class_name": cls_name,
                        "confidence": conf,
                        "xyxy": [float(v) for v in xyxy],
                    }
                )
                if str(cls_name).strip().lower() == "plate":
                    plate_candidates.append((conf, xyxy))

        plate_number = ""
        plate_confidence = 0.0
        plate_crop = None
        if plate_candidates:
            plate_candidates.sort(key=lambda item: item[0], reverse=True)
            plate_confidence, best_xyxy = plate_candidates[0]
            plate_crop = crop_plate_region(image_bgr, best_xyxy)
            plate_number, plate_confidence = extract_plate_number_with_fallback(image_bgr, plate_crop)

        violation_items = _build_violation_items(detections, plate_number)
        has_violation = bool(violation_items)

        if violation_items:
            violation_type = ", ".join(dict.fromkeys(violation_items))
        else:
            violation_type = "No major violation detected"

        violation_count = len(violation_items)

        pipeline_result = {
            "annotated_bgr": annotated_bgr,
            "detections": detections,
            "plate_crop": plate_crop,
            "plate_number": plate_number,
            "plate_confidence": plate_confidence,
            "violation_type": violation_type,
            "violation_items": violation_items,
            "violation_count": violation_count,
            "has_violation": has_violation,
        }

    with right:
        st.subheader("Detected Result")
        st.image(bgr_to_rgb(annotated_bgr), use_container_width=True)

    st.divider()
    render_summary_cards(pipeline_result)

    detection_rows = [
        {
            "Class": item["class_name"],
            "Confidence": round(item["confidence"], 4),
            "Box": [round(value, 1) for value in item["xyxy"]],
        }
        for item in detections
    ]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Violation Details")
        st.write(f"**Status:** {'Violation detected' if has_violation else 'No violation detected'}")
        st.write(f"**Type:** {violation_type}")
        st.write(f"**Count:** {pipeline_result['violation_count']}")
        st.write(f"**OCR Number:** {plate_number or 'Not detected'}")
        st.write(f"**OCR Confidence:** {plate_confidence:.2f}")

        if pipeline_result["violation_items"]:
            st.write("**Violation List:**")
            for item in pipeline_result["violation_items"]:
                st.write(f"- {_pretty_violation_label(item)}")

    with col2:
        st.subheader("Detected Objects")
        st.dataframe(detection_rows, use_container_width=True, hide_index=True)

    if plate_crop is not None:
        st.subheader("Cropped Plate")
        st.image(bgr_to_rgb(plate_crop), width=360)

    report = build_report(uploaded_file.name, pipeline_result)
    report_bytes = build_pdf_report(report)

    st.download_button(
        "Download violation report (PDF)",
        data=report_bytes,
        file_name=f"violation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
    )

    with st.expander("Raw report preview", expanded=False):
        st.json(report)


if __name__ == "__main__":
    main()