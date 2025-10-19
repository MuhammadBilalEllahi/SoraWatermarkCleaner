from pathlib import Path

from sorawm.core import SoraWM

if __name__ == "__main__":
    fileName = "finding_milk"
    input_video_path = Path(f"outputs/{fileName}_sora_watermark_removed.mp4")
    output_video_path = Path(f"outputs/{fileName}_sora_watermark_removed_refine.mp4")
    sora_wm = SoraWM()
    sora_wm.run(input_video_path, output_video_path)
