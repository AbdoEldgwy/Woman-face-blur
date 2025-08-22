from pipeline import WomanBlurPipeline

pipeline = WomanBlurPipeline()
pipeline.process_video("input.mp4", "output_blurred.mp4", display=True)
