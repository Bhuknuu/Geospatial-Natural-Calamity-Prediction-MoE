"""
Nabh-Drishti: Geospatial Disaster Calamity Prediction & Monitoring System
========================================================================
A Minimalist Dark-Themed Desktop Application featuring a 6-Stage Visual Pipeline
powered by Foundation Vision Transformer, Spectral Preprocessing,
OMaR / Clef Multi-Hazard Calamity Routing, Urgency Scoring, and Fast AI Briefings.

Supports:
  - Live NASA GIBS WMS real-time satellite streaming across India (Current UTC Date/Time)
  - Curated Indian Calamity Presets (Cyclone, Flood, Wildfire, Heatwave, Landslide)
  - Custom User Image Ingestion (Upload arbitrary satellite/aerial JPG, PNG, TIF, WEBP)
  - High-speed Local Qwen2.5 AI situational reasoning & IMD/NDRF emergency response directives
"""

import os
import sys
import time
import argparse
import threading
from typing import Dict, Any, Optional
from PIL import Image
import torch

# Ensure workspace root and library paths
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, WORKSPACE_DIR)

LIB_PATH = "/home/bhuknu/Documents/Library_Books/SystemDesign/.lib/usr/lib"
TK_LIB_PATH = "/home/bhuknu/Documents/Library_Books/SystemDesign/.lib/usr/lib/tk8.6"
TCL_LIB_PATH = "/usr/lib/tcl8.6"

if os.path.exists(LIB_PATH) and LIB_PATH not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = f"{LIB_PATH}:{os.environ.get('LD_LIBRARY_PATH', '')}"
if os.path.exists(TK_LIB_PATH):
    os.environ["TK_LIBRARY"] = TK_LIB_PATH
if os.path.exists(TCL_LIB_PATH):
    os.environ["TCL_LIBRARY"] = TCL_LIB_PATH

from src.data_feed import DataFeedManager, HISTORICAL_PRESETS, REGIONAL_BBOXES
from src.preprocessor import SpectralPreprocessor, FROZEN_CALAMITIES
from src.prithvi_model import PrithviCalamityEngine
from src.vlm_narrator import VLMNarrator

# UI Color Palette: Minimalist Dark Theme
BG_DARK = "#0d0d0d"
BG_PANEL = "#171717"
BG_CARD = "#212121"
BORDER_COLOR = "#383838"
TEXT_WHITE = "#FFFFFF"
TEXT_GRAY = "#A0A0A0"
TEXT_DIM = "#666666"
ACCENT_GREEN = "#2ECC71"
ACCENT_ORANGE = "#F39C12"
ACCENT_RED = "#E74C3C"
ACCENT_CYAN = "#00FFFF"


class CalamityPipelineRunner:
    """Core execution pipeline shared between GUI and CLI."""

    def __init__(self, models_dir: str = "models"):
        self.data_manager = DataFeedManager(base_dir=WORKSPACE_DIR)
        self.preprocessor = SpectralPreprocessor()
        checkpoint = os.path.join(WORKSPACE_DIR, models_dir, "Prithvi_100M.pt")
        self.prithvi_engine = PrithviCalamityEngine(checkpoint_path=checkpoint)
        self.vlm_narrator = VLMNarrator()

    def run_full_pipeline(
        self,
        feed_mode: str = "preset",
        preset_id: str = "cyclone_biparjoy",
        region_name: str = "All India Synoptic Composite",
        custom_image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Executes all 6 stages of the calamity prediction pipeline fresh from scratch."""
        t0 = time.time()

        # Stage 1: Data Ingestion
        if feed_mode == "custom" and custom_image_path:
            raw_img, metadata = self.data_manager.load_custom_image(custom_image_path)
        elif feed_mode == "live":
            raw_img, metadata = self.data_manager.fetch_live_stream(region_name=region_name)
        else:
            raw_img, metadata = self.data_manager.load_historical(preset_id=preset_id)

        # Stage 2 & 3: Preprocessing & OMaR / Clef Routing
        preprocessed = self.preprocessor.process_scene(raw_img, metadata)

        # Stage 4 & 5: Foundation ViT Inference & Urgency Scoring
        inference_res = self.prithvi_engine.run_inference(preprocessed)

        # Stage 6: Fast AI Situational Briefing
        vlm_res = self.vlm_narrator.narrate(inference_res, preprocessed)

        total_latency = (time.time() - t0) * 1000.0

        return {
            "stage1_raw": {"image": raw_img, "metadata": metadata},
            "stage2_prep": preprocessed,
            "stage3_route": {
                "target": preprocessed["routing_target"],
                "confidence": preprocessed["routing_confidence"],
                "calamity_probabilities": preprocessed.get("calamity_probabilities", {}),
                "top_calamity_name": preprocessed.get("top_calamity_name", ""),
                "metrics": preprocessed["spectral_metrics"],
                "routing_card_img": preprocessed.get("routing_card_img")
            },
            "stage4_inference": inference_res,
            "stage5_calamity": {
                "urgency": inference_res["urgency"],
                "urgency_label": inference_res["urgency_label"],
                "urgency_color": inference_res["urgency_color"],
                "severity_score": inference_res["severity_score"],
                "affected_area_pct": inference_res["affected_area_pct"],
                "comparison_image": inference_res["comparison_image"],
                "event_name": inference_res.get("event_name", ""),
                "target_calamity": inference_res.get("target_calamity", ""),
            },
            "stage6_vlm": vlm_res,
            "total_latency_ms": total_latency,
            "device_name": inference_res.get("device_name", "PyTorch ViT")
        }


def launch_gui():
    """Launches the streamlined Tkinter 6-Stage Desktop Application."""
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    from PIL import ImageTk

    root = tk.Tk()
    root.title("Nabh-Drishti: Multi-Hazard Calamity Prediction Platform")
    root.geometry("1260x860")
    root.configure(bg=BG_DARK)
    root.minsize(1080, 720)

    pipeline = CalamityPipelineRunner()

    current_data = {
        "pipeline_result": None,
        "current_stage": 1,
        "is_streaming": False,
        "stream_thread": None,
        "tk_images": {},
        "briefing_card_img": None
    }

    # Style definitions
    style = ttk.Style()
    style.theme_use("clam")
    style.configure(".", background=BG_DARK, foreground=TEXT_WHITE, font=("Helvetica", 10))
    style.configure("TCombobox", fieldbackground=BG_PANEL, background=BORDER_COLOR, foreground=TEXT_WHITE, arrowcolor=TEXT_WHITE)
    style.map("TCombobox", fieldbackground=[("readonly", BG_PANEL)], selectbackground=[("readonly", BG_PANEL)], selectforeground=[("readonly", TEXT_WHITE)])

    # Top Header Banner
    header_frame = tk.Frame(root, bg=BG_PANEL, highlightbackground=BORDER_COLOR, highlightthickness=1)
    header_frame.pack(fill=tk.X, padx=12, pady=(10, 4))

    top_title_row = tk.Frame(header_frame, bg=BG_PANEL)
    top_title_row.pack(fill=tk.X, padx=16, pady=(8, 2))

    lbl_title = tk.Label(
        top_title_row,
        text="NABH-DRISHTI: MULTI-HAZARD CALAMITY PREDICTION PLATFORM",
        font=("Helvetica", 13, "bold"),
        bg=BG_PANEL,
        fg=TEXT_WHITE
    )
    lbl_title.pack(side=tk.LEFT)

    lbl_status_top = tk.Label(
        top_title_row,
        text="SATELLITE INTELLIGENCE PIPELINE | ACTIVE",
        font=("Helvetica", 9, "bold"),
        bg=BG_PANEL,
        fg=ACCENT_CYAN
    )
    lbl_status_top.pack(side=tk.RIGHT)

    lbl_sub = tk.Label(
        header_frame,
        text="Multi-Hazard Geospatial Observation, Spectral Preprocessing & Autonomous Calamity Evaluation",
        font=("Helvetica", 9),
        bg=BG_PANEL,
        fg=TEXT_GRAY
    )
    lbl_sub.pack(anchor="w", padx=16, pady=(0, 8))

    # Control Bar Frame
    control_frame = tk.Frame(root, bg=BG_PANEL, highlightbackground=BORDER_COLOR, highlightthickness=1)
    control_frame.pack(fill=tk.X, padx=12, pady=4)

    # Preset selection
    tk.Label(control_frame, text="PRESET:", font=("Helvetica", 9, "bold"), bg=BG_PANEL, fg=TEXT_WHITE).grid(row=0, column=0, padx=(10, 4), pady=8, sticky="w")
    preset_names = [f"{p['name']} ({p['region']})" for p in HISTORICAL_PRESETS]
    preset_keys = [p["id"] for p in HISTORICAL_PRESETS]
    combo_preset = ttk.Combobox(control_frame, values=preset_names, width=32, state="readonly")
    combo_preset.current(0)
    combo_preset.grid(row=0, column=1, padx=4, pady=8, sticky="w")

    # Live region selection
    tk.Label(control_frame, text="LIVE SECTOR:", font=("Helvetica", 9, "bold"), bg=BG_PANEL, fg=TEXT_WHITE).grid(row=0, column=2, padx=(10, 4), pady=8, sticky="w")
    region_names = list(REGIONAL_BBOXES.keys())
    combo_region = ttk.Combobox(control_frame, values=region_names, width=24, state="readonly")
    combo_region.current(0)
    combo_region.grid(row=0, column=3, padx=4, pady=8, sticky="w")

    # Action Buttons
    btn_frame = tk.Frame(control_frame, bg=BG_PANEL)
    btn_frame.grid(row=0, column=4, padx=8, pady=8, sticky="e")

    def make_btn(parent, text, cmd, bg=BG_CARD, fg=TEXT_WHITE, **kwargs):
        b = tk.Button(
            parent,
            text=text,
            command=cmd,
            bg=bg,
            fg=fg,
            activebackground="#333333",
            activeforeground=TEXT_WHITE,
            font=("Helvetica", 9, "bold"),
            relief=tk.FLAT,
            bd=1,
            padx=8,
            pady=4,
            highlightbackground=BORDER_COLOR,
            highlightthickness=1
        )
        return b

    # Stage Navigation Tabs
    stage_nav_frame = tk.Frame(root, bg=BG_DARK)
    stage_nav_frame.pack(fill=tk.X, padx=12, pady=(6, 2))

    stage_buttons = []
    stages_info = [
        ("STAGE 1: RAW INGESTION", 1),
        ("STAGE 2: SPECTRAL PREPROCESSING", 2),
        ("STAGE 3: OMaR / CLEF ROUTING", 3),
        ("STAGE 4: GEOSPATIAL ENCODER", 4),
        ("STAGE 5: CALAMITY HUD", 5),
        ("STAGE 6: SITUATIONAL BRIEFING", 6)
    ]

    def set_stage(stage_num: int):
        current_data["current_stage"] = stage_num
        for idx, (btn, s_num) in enumerate(stage_buttons):
            if s_num == stage_num:
                btn.configure(bg="#ffffff", fg="#000000")
            else:
                btn.configure(bg=BG_CARD, fg=TEXT_GRAY)
        render_current_stage()

    for label_text, s_num in stages_info:
        btn = make_btn(stage_nav_frame, label_text, lambda s=s_num: set_stage(s))
        btn.pack(side=tk.LEFT, padx=3)
        stage_buttons.append((btn, s_num))

    # Main Body: Full-Width High-Resolution Canvas
    body_frame = tk.Frame(root, bg=BG_DARK)
    body_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

    canvas_container = tk.Frame(body_frame, bg=BG_PANEL, highlightbackground=BORDER_COLOR, highlightthickness=1)
    canvas_container.pack(fill=tk.BOTH, expand=True)

    canvas_header = tk.Frame(canvas_container, bg=BG_PANEL)
    canvas_header.pack(fill=tk.X, padx=12, pady=6)

    lbl_canvas_title = tk.Label(canvas_header, text="STAGE VISUALIZATION", font=("Helvetica", 11, "bold"), bg=BG_PANEL, fg=TEXT_WHITE)
    lbl_canvas_title.pack(side=tk.LEFT)

    lbl_canvas_meta = tk.Label(canvas_header, text="READY", font=("Helvetica", 9), bg=BG_PANEL, fg=TEXT_GRAY)
    lbl_canvas_meta.pack(side=tk.RIGHT)

    canvas = tk.Canvas(canvas_container, bg="#050505", highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    # Bottom Telemetry & Status Bar
    status_bar = tk.Frame(root, bg=BG_PANEL, height=30, highlightbackground=BORDER_COLOR, highlightthickness=1)
    status_bar.pack(fill=tk.X, padx=12, pady=(0, 6))

    lbl_status_left = tk.Label(status_bar, text="SYSTEM READY | OLLAMA QWEN2.5 ACTIVE", font=("Helvetica", 9), bg=BG_PANEL, fg=TEXT_GRAY)
    lbl_status_left.pack(side=tk.LEFT, padx=12)

    lbl_status_center = tk.Label(status_bar, text="HAZARD: STANDBY", font=("Helvetica", 9, "bold"), bg=BG_PANEL, fg=TEXT_WHITE)
    lbl_status_center.pack(side=tk.LEFT, expand=True)

    lbl_status_right = tk.Label(status_bar, text="STATUS: NORMAL", font=("Helvetica", 9, "bold"), bg=BG_PANEL, fg=ACCENT_GREEN)
    lbl_status_right.pack(side=tk.RIGHT, padx=12)

    def render_current_stage():
        res = current_data["pipeline_result"]
        if not res:
            canvas.delete("all")
            cw = canvas.winfo_width() or 800
            ch = canvas.winfo_height() or 500
            canvas.create_text(
                cw / 2, ch / 2,
                text="NO DATA LOADED\nClick [ LOAD PRESET ], [ FETCH LIVE ], or [ CUSTOM IMAGE ] to begin analysis.",
                fill=TEXT_GRAY,
                font=("Helvetica", 12),
                justify=tk.CENTER
            )
            return

        stage = current_data["current_stage"]
        canvas.delete("all")
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 100:
            cw, ch = 960, 560

        if stage == 1:
            lbl_canvas_title.config(text="STAGE 1: RAW SENSOR INGESTION")
            meta = res["stage1_raw"]["metadata"]
            lbl_canvas_meta.config(text=f"SOURCE: {meta.get('source', 'NASA GIBS')} | DATE: {meta.get('date', 'Live')}")
            img = res["stage1_raw"]["image"]
            max_w, max_h = cw - 40, ch - 40
            scale = min(max_w / img.width, max_h / img.height, 1.8)
            disp_w, disp_h = int(img.width * scale), int(img.height * scale)
            disp_img = img.resize((disp_w, disp_h), Image.Resampling.BILINEAR)
            tk_img = ImageTk.PhotoImage(disp_img)
            current_data["tk_images"]["stage1"] = tk_img
            canvas.create_image(cw / 2, ch / 2, image=tk_img)

        elif stage == 2:
            lbl_canvas_title.config(text="STAGE 2: SPECTRAL PREPROCESSING & GEOPHYSICAL ANOMALY EXTRACTION")
            lbl_canvas_meta.config(text="FALSE-COLOR (NIR-R-G) + ANOMALY MAP + SENSOR GRID + 6-BAND SPECTRAL PROFILES")
            prep_card = res["stage2_prep"].get("preprocessing_card_img")
            if prep_card:
                aspect = prep_card.width / prep_card.height
                target_w = min(cw - 30, int(prep_card.width * 1.35))
                target_h = int(target_w / aspect)
                if target_h > ch - 30:
                    target_h = ch - 30
                    target_w = int(target_h * aspect)
                disp_img = prep_card.resize((target_w, target_h), Image.Resampling.BILINEAR)
            else:
                disp_img = res["stage2_prep"]["annotated_img"]
            tk_img = ImageTk.PhotoImage(disp_img)
            current_data["tk_images"]["stage2"] = tk_img
            canvas.create_image(cw / 2, ch / 2, image=tk_img)

        elif stage == 3:
            lbl_canvas_title.config(text="STAGE 3: OMaR / CLEF HAZARD ROUTING & PROBABILITIES")
            route_target = res["stage3_route"]["target"]
            conf = res["stage3_route"]["confidence"] * 100
            lbl_canvas_meta.config(text=f"DISPATCHED EXPERT: {route_target} ({conf:.1f}% Confidence)")
            
            routing_card = res["stage3_route"].get("routing_card_img")
            if routing_card:
                aspect = routing_card.width / routing_card.height
                target_w = min(cw - 30, int(routing_card.width * 1.35))
                target_h = int(target_w / aspect)
                if target_h > ch - 30:
                    target_h = ch - 30
                    target_w = int(target_h * aspect)
                disp_img = routing_card.resize((target_w, target_h), Image.Resampling.BILINEAR)
            else:
                disp_img = res["stage2_prep"]["annotated_img"]

            tk_img = ImageTk.PhotoImage(disp_img)
            current_data["tk_images"]["stage3"] = tk_img
            canvas.create_image(cw / 2, ch / 2, image=tk_img)

        elif stage == 4:
            lbl_canvas_title.config(text="STAGE 4: GEOSPATIAL FOUNDATION ViT ENCODER")
            inf = res["stage4_inference"]
            lbl_canvas_meta.config(text=f"ENCODER LATENCY: {inf.get('latency_ms', 0):.1f} ms | ATTENTION HEADS: 12")
            img = inf["heatmap_image"]
            max_w, max_h = cw - 40, ch - 40
            scale = min(max_w / img.width, max_h / img.height, 1.8)
            disp_w, disp_h = int(img.width * scale), int(img.height * scale)
            disp_img = img.resize((disp_w, disp_h), Image.Resampling.BILINEAR)
            tk_img = ImageTk.PhotoImage(disp_img)
            current_data["tk_images"]["stage4"] = tk_img
            canvas.create_image(cw / 2, ch / 2, image=tk_img)

        elif stage == 5:
            lbl_canvas_title.config(text="STAGE 5: COMPARATIVE CALAMITY HUD & SEVERITY SCORING")
            cal = res["stage5_calamity"]
            lbl_canvas_meta.config(text=f"STATUS: {cal['urgency_label']} | SEVERITY: {cal['severity_score']*100:.1f}%")
            comp_img = cal["comparison_image"]
            
            aspect = comp_img.width / comp_img.height
            target_w = min(cw - 30, int(comp_img.width * 1.35))
            target_h = int(target_w / aspect)
            if target_h > ch - 30:
                target_h = ch - 30
                target_w = int(target_h * aspect)
                
            disp_img = comp_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
            tk_img = ImageTk.PhotoImage(disp_img)
            current_data["tk_images"]["stage5"] = tk_img
            canvas.create_image(cw / 2, ch / 2, image=tk_img)

        elif stage == 6:
            lbl_canvas_title.config(text="STAGE 6: SITUATIONAL BRIEFING & AI RISK ASSESSMENT")
            vlm = res["stage6_vlm"]
            lbl_canvas_meta.config(text=f"AI NARRATION: {vlm.get('source', 'Ollama Qwen2.5')} ({vlm.get('latency_ms', 0):.1f} ms) | PROTOCOL: IMD / NDRF / SDMA")
            
            # Generate or retrieve text-focused briefing card
            if not current_data.get("briefing_card_img"):
                card = pipeline.vlm_narrator.render_briefing_card(
                    results=res,
                    preprocessed=res["stage2_prep"],
                    ollama_response=vlm.get("ollama_vlm_output")
                )
                current_data["briefing_card_img"] = card
            else:
                card = current_data["briefing_card_img"]

            aspect = card.width / card.height
            target_w = min(cw - 30, int(card.width * 1.35))
            target_h = int(target_w / aspect)
            if target_h > ch - 30:
                target_h = ch - 30
                target_w = int(target_h * aspect)
            disp_img = card.resize((target_w, target_h), Image.Resampling.BILINEAR)
            tk_img = ImageTk.PhotoImage(disp_img)
            current_data["tk_images"]["stage6"] = tk_img
            canvas.create_image(cw / 2, ch / 2, image=tk_img)

    def execute_pipeline_worker(feed_mode: str, preset_id: str, region_name: str, custom_path: str = ""):
        try:
            target_desc = os.path.basename(custom_path) if feed_mode == "custom" else (preset_id.upper() if feed_mode == "preset" else region_name)
            
            # 1. Clear previous photo data and state completely
            current_data["pipeline_result"] = None
            current_data["tk_images"].clear()
            current_data["briefing_card_img"] = None

            # 2. Show active calculation status on canvas
            def show_recalc_splash():
                canvas.delete("all")
                cw = canvas.winfo_width() or 800
                ch = canvas.winfo_height() or 500
                canvas.create_text(
                    cw / 2, ch / 2 - 20,
                    text="RECALCULATING GEOSPATIAL INTELLIGENCE...",
                    fill="#00FFFF",
                    font=("Helvetica", 14, "bold"),
                    justify=tk.CENTER
                )
                canvas.create_text(
                    cw / 2, ch / 2 + 18,
                    text=f"Target: {target_desc} | Mode: {feed_mode.upper()}\nClearing prior state & executing fresh inference pipeline...",
                    fill=TEXT_GRAY,
                    font=("Helvetica", 10),
                    justify=tk.CENTER
                )
                lbl_status_left.config(text=f"RECALCULATING ({feed_mode.upper()})...")
            
            root.after(0, show_recalc_splash)

            # 3. Freshly compute full pipeline from scratch
            res = pipeline.run_full_pipeline(
                feed_mode=feed_mode,
                preset_id=preset_id,
                region_name=region_name,
                custom_image_path=custom_path
            )
            current_data["pipeline_result"] = res

            # 4. Update status and telemetry labels
            cal = res.get("stage5_calamity", {})
            route = res.get("stage3_route", {})
            vlm = res.get("stage6_vlm", {})
            
            target_h = cal.get("target_calamity", "Multi-Hazard Scan")
            expert = route.get("target", "OMaR-Router")
            conf = route.get("confidence", 0.95) * 100.0
            urg_lbl = cal.get("urgency_label", "NORMAL")
            urg_col = cal.get("urgency_color", ACCENT_GREEN)
            
            lbl_status_left.config(text=f"COMPLETED in {res['total_latency_ms']:.1f} ms | AI: {pipeline.vlm_narrator.ollama_model} ({vlm.get('latency_ms', 0):.0f}ms)")
            lbl_status_center.config(text=f"HAZARD: {target_h.upper()} | ROUTED: {expert} ({conf:.1f}%)")
            lbl_status_right.config(text=f"STATUS: {urg_lbl}", fg=urg_col)
            
            root.after(0, render_current_stage)
        except Exception as e:
            lbl_status_left.config(text=f"PIPELINE ERROR: {e}")
            def show_err():
                canvas.delete("all")
                cw = canvas.winfo_width() or 800
                ch = canvas.winfo_height() or 500
                canvas.create_text(cw / 2, ch / 2, text=f"PIPELINE ERROR:\n{e}", fill="#FF4D4D", font=("Helvetica", 11, "bold"), justify=tk.CENTER)
            root.after(0, show_err)

    def on_load_preset():
        idx = combo_preset.current()
        p_id = preset_keys[idx]
        threading.Thread(target=execute_pipeline_worker, args=("preset", p_id, "", ""), daemon=True).start()

    def on_fetch_live():
        reg = combo_region.get()
        threading.Thread(target=execute_pipeline_worker, args=("live", "", reg, ""), daemon=True).start()

    def on_upload_custom():
        file_path = filedialog.askopenfilename(
            title="Select Custom Satellite or Aerial Image",
            filetypes=[
                ("All Supported Images", "*.jpg *.jpeg *.png *.tif *.tiff *.webp *.bmp"),
                ("JPEG Images", "*.jpg *.jpeg"),
                ("PNG Images", "*.png"),
                ("TIFF Satellite GeoTIFF", "*.tif *.tiff"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            set_stage(5)
            threading.Thread(target=execute_pipeline_worker, args=("custom", "", "", file_path), daemon=True).start()

    def on_run_all():
        set_stage(5)
        on_load_preset()

    def toggle_stream():
        if not current_data["is_streaming"]:
            current_data["is_streaming"] = True
            btn_stream.config(text="STREAM: ON", bg=ACCENT_RED, fg=TEXT_WHITE)
            
            def stream_loop():
                while current_data["is_streaming"]:
                    reg = combo_region.get()
                    execute_pipeline_worker("live", "", reg, "")
                    time.sleep(12.0)
            
            t = threading.Thread(target=stream_loop, daemon=True)
            current_data["stream_thread"] = t
            t.start()
        else:
            current_data["is_streaming"] = False
            btn_stream.config(text="STREAM: OFF", bg=BG_CARD, fg=TEXT_WHITE)

    # Attach buttons to control bar
    btn_preset = make_btn(btn_frame, "LOAD PRESET", on_load_preset, bg=BG_CARD, fg=TEXT_WHITE)
    btn_preset.pack(side=tk.LEFT, padx=2)

    btn_live = make_btn(btn_frame, "FETCH LIVE", on_fetch_live, bg=BG_CARD, fg=TEXT_WHITE)
    btn_live.pack(side=tk.LEFT, padx=2)

    btn_custom = make_btn(btn_frame, "CUSTOM IMAGE", on_upload_custom, bg="#1e293b", fg=ACCENT_CYAN)
    btn_custom.pack(side=tk.LEFT, padx=2)

    btn_all = make_btn(btn_frame, "RUN PIPELINE", on_run_all, bg="#2a2a2a", fg=TEXT_WHITE)
    btn_all.pack(side=tk.LEFT, padx=2)

    btn_stream = make_btn(btn_frame, "STREAM: OFF", toggle_stream, bg=BG_CARD, fg=TEXT_WHITE)
    btn_stream.pack(side=tk.LEFT, padx=2)

    # Canvas resize binding
    canvas.bind("<Configure>", lambda event: render_current_stage())

    # Set initial active stage button
    set_stage(1)

    # Auto load initial preset on start
    root.after(300, on_load_preset)

    root.mainloop()


def run_cli_tests(custom_image_path: Optional[str] = None):
    """Runs automated headless verification across presets, live NASA GIBS WMS streams, or a custom image."""
    print("=" * 80)
    print("NABH-DRISHTI MULTI-HAZARD CALAMITY PREDICTION PLATFORM")
    print("AUTOMATED VERIFICATION & OMaR / CLEF HAZARD ROUTING SUITE")
    print("=" * 80)

    runner = CalamityPipelineRunner()

    print(f"[*] Encoder Backbone       : Foundation Vision Transformer (12-layer ViT, 768-D)")
    print(f"[*] Compute Device         : {runner.prithvi_engine.device_name}")
    print(f"[*] Weights Checkpoint     : models/Prithvi_100M.pt (Loaded: {runner.prithvi_engine.weights_loaded})")
    print(f"[*] AI Situational Model   : Ollama ({runner.vlm_narrator.ollama_model})")
    print("-" * 80)

    if custom_image_path:
        print(f"\n[+] Executing Dynamic Pipeline on Custom User Image: {custom_image_path}...")
        res = runner.run_full_pipeline(feed_mode="custom", custom_image_path=custom_image_path)
        cal = res["stage5_calamity"]
        route = res["stage3_route"]
        vlm = res["stage6_vlm"]
        
        print(f"    - Ingested File     : {os.path.basename(custom_image_path)}")
        print(f"    - Target Calamity   : {cal.get('target_calamity', 'Hazard')}")
        print(f"    - Routed Head       : {route['target']} (Confidence: {route['confidence']*100:.1f}%)")
        print(f"    - Urgency Status    : {cal['urgency_label']}")
        print(f"    - Severity Score    : {cal['severity_score']*100:.1f}% (Extent: {cal['affected_area_pct']:.1f}%)")
        print(f"    - Pipeline Latency  : {res['total_latency_ms']:.1f} ms")
        print(f"    - AI Source         : {vlm['source']}")
        print("\n    [OMaR / Clef Calamity Probability Breakdown]")
        for c_name, prob in route.get("calamity_probabilities", {}).items():
            print(f"      • {c_name:<42}: {prob:5.1f}%")
        print("\n" + "=" * 80)
        print("SITUATIONAL DISASTER BRIEFING:\n")
        print(vlm["narrative_text"])
        print("=" * 80)
        return

    # Test all presets
    for p in HISTORICAL_PRESETS:
        p_id = p["id"]
        print(f"\n[+] Executing Pipeline for Preset: {p_id.upper()} ({p['name']})...")
        res = runner.run_full_pipeline(feed_mode="preset", preset_id=p_id)
        cal = res["stage5_calamity"]
        route = res["stage3_route"]
        vlm = res["stage6_vlm"]
        
        print(f"    - Target Calamity   : {cal.get('target_calamity', 'Hazard')}")
        print(f"    - Routed Head       : {route['target']} (Confidence: {route['confidence']*100:.1f}%)")
        print(f"    - Urgency Status    : {cal['urgency_label']}")
        print(f"    - Severity Score    : {cal['severity_score']*100:.1f}% (Extent: {cal['affected_area_pct']:.1f}%)")
        print(f"    - Pipeline Latency  : {res['total_latency_ms']:.1f} ms")
        print("    - Calamity Probabilities:")
        for c_name, prob in route.get("calamity_probabilities", {}).items():
            print(f"        {c_name:<38}: {prob:5.1f}%")

    # Test Live NASA GIBS WMS stream
    print(f"\n[+] Executing Live Real-Time Satellite Feed (NASA GIBS WMS: All India)...")
    live_res = runner.run_full_pipeline(feed_mode="live", region_name="All India Synoptic Composite")
    live_meta = live_res["stage1_raw"]["metadata"]
    print(f"    - Live Stream Status: {live_meta.get('status_note', 'OK')}")
    print(f"    - Live Timestamp    : {live_meta.get('date', 'Live')}")
    print(f"    - Target Calamity   : {live_res['stage5_calamity'].get('target_calamity', 'Scan')}")
    print(f"    - Urgency Status    : {live_res['stage5_calamity']['urgency_label']}")
    print(f"    - Total Latency     : {live_res['total_latency_ms']:.1f} ms")
    print("    - Calamity Probabilities:")
    for c_name, prob in live_res["stage3_route"].get("calamity_probabilities", {}).items():
        print(f"        {c_name:<38}: {prob:5.1f}%")

    print("\n" + "=" * 80)
    print("ALL MULTI-HAZARD CALAMITY PIPELINE TESTS COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nabh-Drishti Multi-Hazard Calamity Prediction Application")
    parser.add_argument("--cli", "--headless", action="store_true", help="Run automated headless CLI verification tests")
    parser.add_argument("-i", "--image", type=str, default=None, help="Path to custom user image file to process dynamically")
    args = parser.parse_args()

    if args.image:
        run_cli_tests(custom_image_path=args.image)
    elif args.cli:
        run_cli_tests()
    else:
        if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
            print("[Nabh-Drishti] Warning: No graphical display detected. Falling back to headless CLI test runner.")
            run_cli_tests()
        else:
            launch_gui()
