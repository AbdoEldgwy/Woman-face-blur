from pipeline import WomanBlurPipeline

pipeline = WomanBlurPipeline()
pipeline.process_image(r"data\test_images\test_image.jpg", "outputs/output_blurred.jpg", display=True)
