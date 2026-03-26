import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ['YOLO_CONFIG_DIR'] = '/tmp'

print("--- [SYSTEM] STARTING SANCHAY-AI ENGINE ---", file=sys.stderr)

import gradio as gr
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

try:
    model = YOLO("best.pt")
    print("--- [SYSTEM] MODEL LOADED ---", file=sys.stderr)
except Exception as e:
    print(f"--- [ERROR] Model Load Failed: {e} ---", file=sys.stderr)

def sanchay_ai_engine(image, upi_id, manual_total, savings_split):
    if image is None:
        return None, None, None, "⚠️ Please upload a coin photo."

    results = model.predict(source=image, imgsz=640, conf=0.50, iou=0.45)
    annotated_img = results[0].plot()
    detections = results[0].boxes

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_rust = np.array([0, 40, 40])
    upper_rust = np.array([18, 255, 140])
    mask = cv2.inRange(hsv, lower_rust, upper_rust)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    total_rust_pixels = 0
    total_coin_pixels = 0
    rupees_map = {"1": 1, "2": 2, "5": 5, "10": 10, "20": 20}
    ai_total = 0

    if len(detections) > 0:
        for det in detections:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            coin_surface = binary_mask[y1:y2, x1:x2]
            rust_patch = mask[y1:y2, x1:x2]
            total_rust_pixels += np.sum((rust_patch > 0) & (coin_surface > 0))
            total_coin_pixels += np.sum(coin_surface > 0)
            label_idx = int(det.cls[0])
            label_name = model.names[label_idx]
            ai_total += rupees_map.get(label_name, 0)

    rust_percent = (total_rust_pixels / total_coin_pixels * 100) if total_coin_pixels > 0 else 0
    final_amount = manual_total if manual_total > 0 else ai_total

    savings_val = final_amount * (savings_split / 100)
    charity_val = final_amount * 0.05
    upi_val = max(0, final_amount - savings_val - charity_val)

    bonus = 0.50 if (rust_percent < 1.0 and final_amount > 0) else 0.0
    net_deposit = final_amount + bonus

    audit_data = [[
        len(detections), f"₹{ai_total}", f"{rust_percent:.1f}%",
        "FIT ✅" if rust_percent <= 12.0 else "UNFIT ⚠️"
    ]]
    audit_df = pd.DataFrame(audit_data, columns=["Coins", "AI Value", "Oxidation", "Audit Status"])

    savings_data = [[
        f"₹{upi_val:.2f}", f"₹{savings_val:.2f}", f"₹{charity_val:.2f}", f"₹{bonus:.2f}", f"₹{net_deposit:.2f}"
    ]]
    savings_df = pd.DataFrame(savings_data, columns=[
        "UPI Wallet", "Digital Gold (SIP)", "Impact Fund", "Mint Bonus", "Net Credit"
    ])

    status_msg = f"Verified: ₹{final_amount} | Digital Gold Split Active"
    return annotated_img, audit_df, savings_df, status_msg

theme = gr.themes.Soft(primary_hue="emerald", secondary_hue="blue")

with gr.Blocks(theme=theme, title="Sanchay-AI: Smart Coin Bridge") as demo:
    gr.Markdown("# 🇮🇳 Sanchay-AI: Smart Coin Audit & Savings Bridge")
    gr.Markdown("### Bridging Physical Currency to Digital Wealth | A Digital India Innovation")

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Scan Physical Coins", type="numpy")
            upi_input = gr.Textbox(label="Target UPI / CBDC ID", value="resident@rbi")
            with gr.Accordion("⚙️ Savings Configuration", open=True):
                savings_slider = gr.Slider(0, 50, value=20, label="Micro-Investment Split (%)")
                manual_input = gr.Number(label="Manual Override (₹)", value=0)
            btn = gr.Button("🚀 EXECUTE SMART DEPOSIT", variant="primary")

        with gr.Column(scale=2):
            out_label = gr.Textbox(label="Transaction Status")
            out_img = gr.Image(label="Computer Vision Analysis")
            gr.Markdown("#### 🏦 RBI Compliance Ledger")
            audit_table = gr.Dataframe()
            gr.Markdown("#### 📈 Sanchay-AI: Digital Asset Distribution")
            savings_table = gr.Dataframe()

    btn.click(
        fn=sanchay_ai_engine,
        inputs=[input_img, upi_input, manual_input, savings_slider],
        outputs=[out_img, audit_table, savings_table, out_label]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        show_api=False
    )
