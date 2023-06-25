import cv2
import streamlit as st
from PIL import Image
import numpy as np
import io

def apply_filter(image, filter_options):
    filtered_image = image.copy()

    for option in filter_options:
        if option == "Grayscale":
            # Apply grayscale filter
            if len(filtered_image.shape) == 3:  # Check if image is color (3 channels)
                filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
        elif option == "Negative":
            # Apply negative filter
            filtered_image = 255 - filtered_image
        elif option == "Gaussian Blur":
            # Apply Gaussian blur filter
            filtered_image = cv2.GaussianBlur(filtered_image, (5, 5), 0)
        elif option == "Edge Detection":
            # Apply edge detection filter
            gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
            filtered_image = cv2.Canny(gray_image, 100, 200)
            # Convert back to 3 channels for consistency
            filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)
        elif option == "Thresholding":
            # Apply thresholding filter
            _, filtered_image = cv2.threshold(filtered_image, 127, 255, cv2.THRESH_BINARY)
        elif option == "Average":
            # Apply average filter
            filtered_image = cv2.blur(filtered_image, (5, 5))
        elif option == "Sobel":
            # Apply Sobel filter
            gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
            gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            filtered_image = cv2.addWeighted(gradient_x, 0.5, gradient_y, 0.5, 0)
            # Normalize the filtered image to the range [0, 255]
            filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            # Convert back to 3 channels for consistency
            filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)
        # Add more filter options here

    return filtered_image

# Set up the Streamlit app
st.title("Image Filter App")

# Display available filter options
filter_options = st.sidebar.multiselect("Select filter options", ["Grayscale", "Negative", "Gaussian Blur", "Edge Detection", "Thresholding", "Average", "Sobel"])

# Load and display the input image
image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if image_file is not None:
    image = Image.open(image_file)
    st.image(image, caption="Original Image", use_column_width=True)

    try:
        # Convert PIL Image to OpenCV format
        image_cv = np.array(image.convert("RGB"))

        # Apply selected filters
        filtered_image_cv = apply_filter(image_cv, filter_options)

        # Convert filtered image to PIL format
        filtered_image = Image.fromarray(filtered_image_cv)

        # Display the filtered image
        st.image(filtered_image, caption="Filtered Image", use_column_width=True)

        # Create a download link for the filtered image
        output_buffer = io.BytesIO()
        filtered_image.save(output_buffer, format='JPEG')
        output_buffer.seek(0)
        st.download_button("Download Output Image", data=output_buffer, file_name='filtered_image.jpg')

    except Exception as e:
        st.error("Error occurred during image processing.")
