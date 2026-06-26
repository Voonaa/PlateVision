import sys
from ui import build_ui, custom_css

if __name__ == "__main__":
    print("[INFO] Memulai PlateVision...")
    demo = build_ui()
    
    # share=True untuk menghasilkan tautan publik gratis (.gradio.live) tanpa akun
    demo.launch(share=True, css=custom_css)