import tensorflow as tf
import numpy as np
from PIL import Image
import io

class TensorBoard(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a buffer
            s = io.BytesIO()
            Image.fromarray(img).save(s, format='png')

            # Create an Image object
            img_sum = tf.io.decode_image(s.getvalue(), channels=3)
            img_sum = tf.expand_dims(img_sum, 0)  # Add batch dimension
            
            with self.writer.as_default():
                tf.summary.image(f'{tag}/{i}', img_sum, step=step)
                self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        with self.writer.as_default():
            tf.summary.histogram(tag, values, step=step, buckets=bins)
            self.writer.flush()
