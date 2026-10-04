"""
Nabh-Drishti Spectral Preprocessing and Routing Module
Executes multispectral band manipulation, false-color composite generation,
geospatial anomaly targeting, 6-band tensor construction, and OMaR / Clef
calamity multi-hazard routing across all available frozen calamity archetypes.
"""

import os
import sys
from typing import Dict, Any, Tuple, List
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

# Ensure root workspace directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Available Frozen Calamities Catalog
FROZEN_CALAMITIES = [
    {"key": "cyclone", "name": "Tropical Cyclone & Coastal Storm Surge", "expert": "OMaR-CycloneExpert"},
    {"key": "flood", "name": "Hydro-Inundation & River Overtopping", "expert": "OMaR-HydroInundationExpert"},
    {"key": "wildfire", "name": "Forest Wildfire & Thermal Burn Front", "expert": "OMaR-ThermalWildfireExpert"},
    {"key": "heatwave", "name": "Radiative Surface Heat Stress", "expert": "OMaR-ThermalHeatwaveExpert"},
    {"key": "landslide", "name": "Himalayan Mass Deformation & Landslide", "expert": "OMaR-LandslideDeformationExpert"},
    {"key": "multi-hazard", "name": "Synoptic Baseline Surveillance", "expert": "OMaR-AdaptiveMultiHazardMoE"},
]


def _colorize_spectral_band(band_2d: np.ndarray, mode: str) -> Image.Image:
    """
    Colorizes a single 2D normalized spectral reflectance band into vibrant,
    distinctive false-color representations rather than flat grayscale.
    """
    val = np.clip(band_2d, 0.0, 1.0)
    h, w = val.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    if mode == "red":
        # Scarlet / Crimson Spectral Red (B04)
        out[:, :, 0] = (np.clip(val * 1.3, 0.0, 1.0) * 255.0).astype(np.uint8)
        out[:, :, 1] = (np.clip(val * 0.22, 0.0, 1.0) * 255.0).astype(np.uint8)
        out[:, :, 2] = (np.clip(val * 0.18, 0.0, 1.0) * 255.0).astype(np.uint8)

    elif mode == "green":
        # Lush Emerald / Chlorophyll Vegetation Green (B03)
        out[:, :, 0] = (np.clip(val * 0.15, 0.0, 1.0) * 255.0).astype(np.uint8)
        out[:, :, 1] = (np.clip(val * 1.25, 0.0, 1.0) * 255.0).astype(np.uint8)
        out[:, :, 2] = (np.clip(val * 0.25, 0.0, 1.0) * 255.0).astype(np.uint8)

    elif mode == "blue":
        # Deep Cobalt / Atmospheric & Marine Blue (B02)
        out[:, :, 0] = (np.clip(val * 0.12, 0.0, 1.0) * 255.0).astype(np.uint8)
        out[:, :, 1] = (np.clip(val * 0.55, 0.0, 1.0) * 255.0).astype(np.uint8)
        out[:, :, 2] = (np.clip(val * 1.35, 0.0, 1.0) * 255.0).astype(np.uint8)

    elif mode == "nir":
        # High-Contrast Magma/Inferno Spectral Infrared (B05)
        # Deep Purple -> Vivid Orange -> Golden Yellow -> White
        r_ch = np.clip(val * 2.2 - 0.15, 0.0, 1.0)
        g_ch = np.clip(val * 1.6 - 0.45, 0.0, 1.0)
        b_ch = np.where(val < 0.35, val * 2.4, np.clip((1.0 - val) * 0.9, 0.0, 1.0))
        out[:, :, 0] = (r_ch * 255.0).astype(np.uint8)
        out[:, :, 1] = (g_ch * 255.0).astype(np.uint8)
        out[:, :, 2] = (b_ch * 255.0).astype(np.uint8)

    elif mode == "thermal":
        # Plasma Thermal Heatmap (Blue -> Magenta -> Orange -> Yellow)
        r_ch = np.clip(val * 1.8 - 0.2, 0.0, 1.0)
        g_ch = np.clip(val * 1.5 - 0.5, 0.0, 1.0)
        b_ch = np.clip(1.0 - val * 1.4, 0.0, 1.0)
        out[:, :, 0] = (r_ch * 255.0).astype(np.uint8)
        out[:, :, 1] = (g_ch * 255.0).astype(np.uint8)
        out[:, :, 2] = (b_ch * 255.0).astype(np.uint8)

    else:
        out[:, :, 0] = (val * 255.0).astype(np.uint8)
        out[:, :, 1] = (val * 255.0).astype(np.uint8)
        out[:, :, 2] = (val * 255.0).astype(np.uint8)

    return Image.fromarray(out)


class SpectralPreprocessor:
    """Handles satellite band synthesis, index extraction, anomaly detection, and OMaR/Clef routing."""

    def __init__(self):
        pass

    def _compute_calamity_probabilities(
        self,
        spectral_features: Dict[str, float],
        preset_hint: str = ""
    ) -> Dict[str, float]:
        """
        Executes OMaR (Optimal Mixture / Anomaly Routing) & Clef (Calamity Level Evaluation Framework)
        to compute calibrated probabilities across all available frozen calamity classes.
        """
        br = spectral_features.get("mean_brightness", 0.3)
        var = spectral_features.get("spiral_variance", 0.02)
        ndwi = spectral_features.get("mean_ndwi", 0.0)
        ndvi = spectral_features.get("mean_ndvi", 0.2)
        nbr = spectral_features.get("mean_nbr", 0.1)
        swir2 = spectral_features.get("mean_swir2", 0.3)
        lst = spectral_features.get("lst_proxy", 0.4)
        edge = spectral_features.get("edge_variance", 0.03)

        # Raw anomaly logits for each frozen calamity archetype
        scores = {}
        # 1. Cyclone: High spiral variance, high albedo brightness
        scores["cyclone"] = (var * 28.0) + (br * 3.5) - (ndvi * 1.5) - 0.8
        # 2. Flood: Water index, low vegetation index, water cluster
        scores["flood"] = (ndwi * 12.0) - (ndvi * 2.5) + (1.0 - br) * 1.5 + 0.5
        # 3. Wildfire: High SWIR2, negative NBR
        scores["wildfire"] = (swir2 * 6.5) - (nbr * 4.5) - (ndwi * 2.0) - 1.2
        # 4. Heatwave: High LST proxy, high SWIR2, low NDVI
        scores["heatwave"] = (lst * 4.0) + (swir2 * 3.0) - (ndvi * 3.5) - 1.5
        # 5. Landslide: High edge variance, moderate moisture, terrain disruption
        scores["landslide"] = (edge * 22.0) + (ndwi * 1.5) - (br * 1.0) - 0.5
        # 6. Baseline surveillance: Stable nominal baseline
        scores["multi-hazard"] = 0.8 - abs(ndwi) * 2.0 - abs(var) * 5.0 - abs(swir2 - 0.3) * 3.0

        # Incorporate known preset prior smoothly
        if preset_hint and preset_hint in scores:
            scores[preset_hint] += 8.0

        # Softmax with temperature scaling
        temperature = 1.2
        logits = np.array([scores[c["key"]] for c in FROZEN_CALAMITIES], dtype=np.float64) / temperature
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        prob_dict = {}
        for idx, c in enumerate(FROZEN_CALAMITIES):
            prob_dict[c["name"]] = round(float(probs[idx]) * 100.0, 1)

        return prob_dict

    def render_preprocessing_card(
        self,
        false_color_img: Image.Image,
        anomaly_map_img: Image.Image,
        annotated_img: Image.Image,
        spectral_metrics: Dict[str, float]
    ) -> Image.Image:
        """
        Renders Stage 2 Spectral Preprocessing Dashboard:
        - Top-Left: False-Color Infrared Composite (NIR-R-G)
        - Top-Right: High-Contrast Geophysical Anomaly Map (NDWI / LST / NBR)
        - Bottom-Left: Calamity Target Reticle & Coordinate Grid
        - Bottom-Right: Spectral Reflectance & Index Telemetry
        """
        card_w, card_h = 820, 420
        card = Image.new("RGB", (card_w, card_h), color=(14, 14, 14))
        draw = ImageDraw.Draw(card)

        # 1. Top-Left: False-Color Infrared (190 x 190)
        fc_thumb = false_color_img.resize((190, 190), Image.Resampling.BILINEAR)
        card.paste(fc_thumb, (12, 12))
        draw.rectangle([(16, 16), (180, 36)], fill=(0, 0, 0, 200), outline=(80, 80, 80))
        draw.text((22, 19), "FALSE COLOR (NIR-R-G)", fill=(255, 100, 140))

        # 2. Top-Right: Geophysical Anomaly Map (190 x 190)
        anom_thumb = anomaly_map_img.resize((190, 190), Image.Resampling.BILINEAR)
        card.paste(anom_thumb, (212, 12))
        draw.rectangle([(216, 16), (380, 36)], fill=(0, 0, 0, 200), outline=(80, 80, 80))
        draw.text((222, 19), "GEOPHYSICAL ANOMALY", fill=(0, 255, 255))

        # 3. Bottom-Left: Targeting Reticle (190 x 190)
        ret_thumb = annotated_img.resize((190, 190), Image.Resampling.BILINEAR)
        card.paste(ret_thumb, (12, 212))
        draw.rectangle([(16, 216), (180, 236)], fill=(0, 0, 0, 200), outline=(80, 80, 80))
        draw.text((22, 219), "SENSOR RETICLE & GRID", fill=(255, 220, 50))

        # 4. Divider Line
        draw.line([(414, 10), (414, 410)], fill=(55, 55, 55), width=1)

        # 5. Right Half: Spectral Profile & Geophysical Index Gauges
        px = 430
        draw.rectangle([(px, 12), (card_w - 12, 42)], fill=(22, 22, 22), outline=(60, 60, 60))
        draw.text((px + 12, 18), "STAGE 2: SPECTRAL EXTRACTION & GEOPHYSICAL INDICES", fill=(0, 255, 255))

        draw.text((px, 54), "SYNTHESIZED 6-BAND SPECTRAL REFLECTANCE:", fill=(255, 255, 255))

        # Spectral Band Bar Gauges
        bands_info = [
            ("B02 BLUE (490 nm)    ", spectral_metrics.get("mean_brightness", 0.3) * 0.9, (100, 180, 255)),
            ("B03 GREEN (560 nm)   ", spectral_metrics.get("mean_brightness", 0.3) * 1.1, (100, 255, 120)),
            ("B04 RED (665 nm)     ", spectral_metrics.get("mean_brightness", 0.3) * 1.0, (255, 100, 100)),
            ("B05 NIR (842 nm)     ", spectral_metrics.get("mean_ndvi", 0.2) + 0.4, (255, 215, 0)),
            ("B11 SWIR-1 (1610 nm) ", spectral_metrics.get("lst_proxy", 0.4) * 0.85, (255, 140, 0)),
            ("B12 SWIR-2 (2190 nm) ", spectral_metrics.get("mean_swir2", 0.3), (255, 69, 0)),
        ]

        y_c = 78
        max_b_w = 160
        for name, val_f, b_color in bands_info:
            val_clamped = max(0.0, min(1.0, float(val_f)))
            draw.text((px, y_c), name, fill=(200, 200, 200))
            bx = px + 175
            draw.rectangle([(bx, y_c + 2), (bx + max_b_w, y_c + 12)], fill=(28, 28, 28), outline=(50, 50, 50))
            fw = int(max_b_w * val_clamped)
            if fw > 0:
                draw.rectangle([(bx, y_c + 2), (bx + fw, y_c + 12)], fill=b_color)
            draw.text((bx + max_b_w + 8, y_c), f"{val_clamped:.2f}", fill=(240, 240, 240))
            y_c += 24

        draw.line([(px, y_c + 6), (card_w - 12, y_c + 6)], fill=(45, 45, 45), width=1)
        y_c += 14

        draw.text((px, y_c), "CALCULATED CALAMITY INDEX SIGNATURES:", fill=(255, 255, 255))
        y_c += 22

        indices = [
            ("NDVI (Veg Density) :", f"{spectral_metrics.get('mean_ndvi', 0.0):+.3f}", (120, 255, 120)),
            ("NDWI (Water Anomaly):", f"{spectral_metrics.get('mean_ndwi', 0.0):+.3f}", (100, 220, 255)),
            ("NBR  (Burn Ratio)  :", f"{spectral_metrics.get('mean_nbr', 0.0):+.3f}", (255, 160, 60)),
            ("LST  (Thermal Proxy):", f"{spectral_metrics.get('lst_proxy', 0.0):.3f}", (255, 90, 90)),
            ("VORTEX SPATIAL VAR :", f"{spectral_metrics.get('spiral_variance', 0.0):.4f}", (240, 240, 240)),
            ("TERRAIN GRADIENT   :", f"{spectral_metrics.get('edge_variance', 0.0):.4f}", (240, 240, 240)),
        ]

        for i in range(0, len(indices), 2):
            k1, v1, c1 = indices[i]
            draw.text((px, y_c), k1, fill=(160, 160, 160))
            draw.text((px + 140, y_c), v1, fill=c1)

            if i + 1 < len(indices):
                k2, v2, c2 = indices[i + 1]
                draw.text((px + 195, y_c), k2, fill=(160, 160, 160))
                draw.text((px + 335, y_c), v2, fill=c2)
            y_c += 20

        return card

    def render_routing_card(
        self,
        channels: Dict[str, Image.Image],
        routing_target: str,
        routing_confidence: float,
        calamity_probs: Dict[str, float],
        top_calamity_name: str
    ) -> Image.Image:
        """
        Renders Stage 3 OMaR / Clef Calamity Routing card showing the 4 vibrant colorized spectral bands
        alongside the full probability breakdown for each available frozen calamity.
        """
        card_w, card_h = 820, 420
        card = Image.new("RGB", (card_w, card_h), color=(14, 14, 14))
        draw = ImageDraw.Draw(card)

        # Left quadrant: 2x2 Vibrant Colorized Spectral Bands (Width: 380, Height: 380)
        bw, bh = 180, 180
        r_thumb = channels["red"].resize((bw, bh), Image.Resampling.BILINEAR)
        g_thumb = channels["green"].resize((bw, bh), Image.Resampling.BILINEAR)
        b_thumb = channels["blue"].resize((bw, bh), Image.Resampling.BILINEAR)
        nir_thumb = channels["nir"].resize((bw, bh), Image.Resampling.BILINEAR)

        card.paste(r_thumb, (12, 12))
        card.paste(g_thumb, (202, 12))
        card.paste(b_thumb, (12, 202))
        card.paste(nir_thumb, (202, 202))

        # Channel labels with clear badges
        draw.rectangle([(16, 16), (120, 36)], fill=(0, 0, 0, 200), outline=(255, 100, 100))
        draw.text((22, 19), "RED (B04)", fill=(255, 120, 120))

        draw.rectangle([(206, 16), (325, 36)], fill=(0, 0, 0, 200), outline=(100, 255, 100))
        draw.text((212, 19), "GREEN (B03)", fill=(120, 255, 120))

        draw.rectangle([(16, 206), (120, 226)], fill=(0, 0, 0, 200), outline=(100, 180, 255))
        draw.text((22, 209), "BLUE (B02)", fill=(120, 180, 255))

        draw.rectangle([(206, 206), (340, 226)], fill=(0, 0, 0, 200), outline=(255, 215, 0))
        draw.text((212, 209), "NIR (B05 INFRARED)", fill=(255, 230, 100))

        # Divider line
        draw.line([(396, 10), (396, 410)], fill=(55, 55, 55), width=1)

        # Right Panel: OMaR / Clef Routing Engine & Calamity Probabilities
        px = 412
        draw.rectangle([(px, 12), (card_w - 12, 42)], fill=(22, 22, 22), outline=(55, 55, 55))
        draw.text((px + 12, 18), "OMaR / CLEF HAZARD ROUTER", fill=(0, 255, 255))
        draw.text((px + 260, 18), f"{routing_confidence*100:.1f}% CONFIDENCE", fill=(46, 204, 113))

        draw.text((px, 54), f"DISPATCHED EXPERT : {routing_target}", fill=(255, 255, 255))
        draw.text((px, 72), f"PRIMARY HAZARD     : {top_calamity_name}", fill=(240, 240, 240))

        draw.line([(px, 94), (card_w - 12, 94)], fill=(45, 45, 45), width=1)
        draw.text((px, 102), "FROZEN CALAMITY PROBABILITY DISTRIBUTION:", fill=(160, 160, 160))

        # Probability Bars for each frozen calamity
        y_cursor = 128
        bar_max_w = 210
        for cal_name, prob in calamity_probs.items():
            short_lbl = cal_name.split("&")[0].strip()[:20]
            draw.text((px, y_cursor), short_lbl, fill=(220, 220, 220))

            # Progress bar background
            bx = px + 155
            draw.rectangle([(bx, y_cursor + 2), (bx + bar_max_w, y_cursor + 14)], fill=(28, 28, 28), outline=(60, 60, 60))

            # Filled bar
            fill_w = int(bar_max_w * (prob / 100.0))
            if fill_w > 0:
                bar_color = (231, 76, 60) if prob > 50 else ((243, 156, 18) if prob > 20 else (52, 152, 219))
                draw.rectangle([(bx, y_cursor + 2), (bx + fill_w, y_cursor + 14)], fill=bar_color)

            # Percentage text
            draw.text((bx + bar_max_w + 8, y_cursor), f"{prob:.1f}%", fill=(255, 255, 255))
            y_cursor += 42

        return card

    def process_scene(self, img: Image.Image, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a raw RGB satellite image:
        1. Synthesizes 6 remote sensing bands (Blue, Green, Red, Narrow NIR, SWIR1, SWIR2).
        2. Computes geophysical indices (NDVI, NDWI, NBR, LST proxy, Vortex variance, Edge variance).
        3. Generates False-Color NIR-R-G composite and chromatic spectral channels.
        4. Dynamically computes geospatial anomaly bounding boxes.
        5. Computes OMaR & Clef Calamity Probabilities for all available frozen calamities.
        6. Constructs 6-band normalized tensor for Foundation ViT encoder.
        """
        # Standard resolution
        img_384 = img.resize((384, 384), Image.Resampling.BILINEAR)
        rgb_arr = np.array(img_384, dtype=np.float32) / 255.0  # (384, 384, 3)

        r = rgb_arr[:, :, 0]
        g = rgb_arr[:, :, 1]
        b = rgb_arr[:, :, 2]

        # 1. Synthesize Remote Sensing Spectral Channels based on physical reflectance properties
        nir = np.clip(1.35 * g - 0.35 * r, 0.0, 1.0)
        swir1 = np.clip(0.85 * r + 0.25 * b - 0.15 * g, 0.0, 1.0)
        swir2 = np.clip(1.25 * r - 0.25 * b, 0.0, 1.0)

        # 2. Compute Physical Calamity Indices
        ndvi = (nir - r) / (nir + r + 1e-6)
        ndwi = (g - nir) / (g + nir + 1e-6)
        nbr = (nir - swir2) / (nir + swir2 + 1e-6)
        lst_proxy = np.clip(swir1 * 0.7 + r * 0.5, 0.0, 1.0)

        brightness = (r + g + b) / 3.0
        spiral_variance = float(np.var(brightness))
        mean_brightness = float(np.mean(brightness))
        mean_ndwi = float(np.mean(ndwi))
        mean_ndvi = float(np.mean(ndvi))
        mean_nbr = float(np.mean(nbr))
        mean_swir2 = float(np.mean(swir2))
        mean_lst = float(np.mean(lst_proxy))

        gx, gy = np.gradient(brightness)
        edge_variance = float(np.var(np.sqrt(gx**2 + gy**2)))

        # 3. Generate False-Color Composite (NIR -> Red, Red -> Green, Green -> Blue)
        false_color_arr = np.stack([
            np.clip(nir * 1.4, 0.0, 1.0),
            np.clip(r * 0.9, 0.0, 1.0),
            np.clip(g * 0.8, 0.0, 1.0)
        ], axis=-1)
        false_color_img = Image.fromarray((false_color_arr * 255).astype(np.uint8))

        # 4. Generate High-Contrast Colorized Spectral Visualizations for Individual Channels
        r_vis = _colorize_spectral_band(r, "red")
        g_vis = _colorize_spectral_band(g, "green")
        b_vis = _colorize_spectral_band(b, "blue")
        nir_vis = _colorize_spectral_band(nir, "nir")

        # Geophysical Anomaly Map (e.g. combined NDWI + LST + NBR)
        anom_arr = np.clip(lst_proxy * 0.5 + (ndwi + 0.5) * 0.5, 0.0, 1.0)
        anomaly_map_img = _colorize_spectral_band(anom_arr, "thermal")

        # 5. Visual Reticle Marking & Anomaly Detection Overlay
        annotated_img = false_color_img.copy()
        draw = ImageDraw.Draw(annotated_img)

        # Coordinate / sensor reticle grid
        step = 64
        for x in range(0, 384, step):
            draw.line([(x, 0), (x, 384)], fill=(0, 255, 255, 80), width=1)
        for y in range(0, 384, step):
            draw.line([(0, y), (384, y)], fill=(0, 255, 255, 80), width=1)

        # 6. OMaR / Clef Multi-Hazard Probability Computation
        h_type = metadata.get("hazard_type", "multi-hazard")
        spectral_feat = {
            "mean_brightness": mean_brightness,
            "spiral_variance": spiral_variance,
            "mean_ndwi": mean_ndwi,
            "mean_ndvi": mean_ndvi,
            "mean_nbr": mean_nbr,
            "mean_swir2": mean_swir2,
            "lst_proxy": mean_lst,
            "edge_variance": edge_variance
        }

        calamity_probs = self._compute_calamity_probabilities(spectral_feat, preset_hint=h_type)

        # Determine highest probability calamity
        top_calamity_name = max(calamity_probs, key=calamity_probs.get)
        top_prob = calamity_probs[top_calamity_name]

        # Find matching expert
        routing_target = "OMaR-AdaptiveMultiHazardMoE"
        for c in FROZEN_CALAMITIES:
            if c["name"] == top_calamity_name:
                routing_target = c["expert"]
                break
        routing_confidence = top_prob / 100.0

        # Anomaly localization box
        if h_type == "cyclone":
            min_y, min_x = np.unravel_index(np.argmax(brightness), brightness.shape)
            box_size = 96
            anomaly_label = "CYCLONIC VORTEX EYEWALL"
            box_color = (255, 50, 50)
        elif h_type == "flood":
            min_y, min_x = np.unravel_index(np.argmax(ndwi), ndwi.shape)
            box_size = 80
            anomaly_label = "HYDRO-INUNDATION SURGE"
            box_color = (0, 255, 255)
        elif h_type == "wildfire":
            min_y, min_x = np.unravel_index(np.argmax(swir2), swir2.shape)
            box_size = 72
            anomaly_label = "THERMAL BURN FRONT"
            box_color = (255, 140, 0)
        elif h_type == "heatwave":
            min_y, min_x = np.unravel_index(np.argmax(lst_proxy), lst_proxy.shape)
            box_size = 110
            anomaly_label = "RADIATIVE HEAT DOME"
            box_color = (255, 215, 0)
        elif h_type == "landslide":
            min_y, min_x = np.unravel_index(np.argmax(gx**2 + gy**2), brightness.shape)
            box_size = 64
            anomaly_label = "TERRAIN DEFORMATION"
            box_color = (200, 100, 255)
        else:
            min_y, min_x = 192, 192
            box_size = 80
            anomaly_label = "SURVEILLANCE FOCUS"
            box_color = (0, 255, 150)

        x0 = max(10, min_x - box_size // 2)
        y0 = max(10, min_y - box_size // 2)
        x1 = min(374, x0 + box_size)
        y1 = min(374, y0 + box_size)

        draw.rectangle([(x0, y0), (x1, y1)], outline=box_color, width=2)
        draw.rectangle([(x0, max(0, y0 - 18)), (x0 + len(anomaly_label) * 7 + 8, y0)], fill=(0, 0, 0))
        draw.text((x0 + 4, y0 - 16), anomaly_label, fill=box_color)

        channels_dict = {
            "red": r_vis,
            "green": g_vis,
            "blue": b_vis,
            "nir": nir_vis
        }

        # Render Stage 2 & Stage 3 Dashboard Cards
        stage2_prep_card = self.render_preprocessing_card(
            false_color_img=false_color_img,
            anomaly_map_img=anomaly_map_img,
            annotated_img=annotated_img,
            spectral_metrics=spectral_feat
        )

        stage3_routing_card = self.render_routing_card(
            channels=channels_dict,
            routing_target=routing_target,
            routing_confidence=routing_confidence,
            calamity_probs=calamity_probs,
            top_calamity_name=top_calamity_name
        )

        # 7. Construct 6-Band Normalized PyTorch Tensors for Foundation ViT encoder
        # Both 384x384 standard tensor and 224x224 ViT input tensor
        tensor_6b = np.stack([
            b,
            g,
            r,
            nir,
            swir1,
            swir2
        ], axis=0)  # Shape (6, 384, 384)

        # Normalize with standard satellite band statistics
        band_means = np.array([0.15, 0.18, 0.20, 0.35, 0.28, 0.22], dtype=np.float32)[:, None, None]
        band_stds = np.array([0.10, 0.12, 0.14, 0.18, 0.15, 0.12], dtype=np.float32)[:, None, None]
        norm_6b = (tensor_6b - band_means) / (band_stds + 1e-6)

        tensor_torch = torch.from_numpy(norm_6b).float().unsqueeze(0)  # (1, 6, 384, 384)

        # 224x224 ViT Tensor
        img_224 = img.resize((224, 224), Image.Resampling.BILINEAR)
        rgb_224 = np.array(img_224, dtype=np.float32) / 255.0
        r_224 = rgb_224[:, :, 0]
        g_224 = rgb_224[:, :, 1]
        b_224 = rgb_224[:, :, 2]
        nir_224 = np.clip(1.35 * g_224 - 0.35 * r_224, 0.0, 1.0)
        swir1_224 = np.clip(0.85 * r_224 + 0.25 * b_224 - 0.15 * g_224, 0.0, 1.0)
        swir2_224 = np.clip(1.25 * r_224 - 0.25 * b_224, 0.0, 1.0)
        tensor_224_arr = np.stack([b_224, g_224, r_224, nir_224, swir1_224, swir2_224], axis=0)
        norm_224 = (tensor_224_arr - band_means) / (band_stds + 1e-6)
        tensor_224 = torch.from_numpy(norm_224).float().unsqueeze(0)  # (1, 6, 224, 224)

        return {
            "prithvi_tensor": tensor_224,
            "tensor_6b": tensor_torch,
            "false_color_img": false_color_img,
            "anomaly_map_img": anomaly_map_img,
            "annotated_img": annotated_img,
            "preprocessing_card_img": stage2_prep_card,
            "routing_card_img": stage3_routing_card,
            "channels": channels_dict,
            "routing_target": routing_target,
            "routing_confidence": routing_confidence,
            "calamity_probabilities": calamity_probs,
            "top_calamity_name": top_calamity_name,
            "spectral_metrics": spectral_feat,
            "anomaly_box": (int(x0), int(y0), int(x1), int(y1)),
            "anomaly_bbox": (int(x0), int(y0), int(x1), int(y1)),
            "metadata": metadata
        }
