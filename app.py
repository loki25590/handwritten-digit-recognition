
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="Digit AI | Handwriting Recognition",
    page_icon="🔢",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_recognition_model.keras")

model = load_model()

# Custom styling
st.markdown("""
<style>
.stApp {
    background: #0e1117;
    color: white;
}
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #58a6ff;
}
.subtitle {
    text-align: center;
    color: #aab4c3;
    font-size: 17px;
}
.result-card {
    background: #1c2333;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #303b50;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔢 Digit AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Handwritten Digit Recognition using Deep Learning</div>',
    unsafe_allow_html=True
)

st.write("")
st.divider()

left, right = st.columns([1.2, 1])

with left:
    st.subheader("✍️ Draw Your Digit")
    st.write("Draw a number between 0 and 9.")

    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=15,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=350,
        height=350,
        drawing_mode="freedraw",
        return_image_data=True,
        key="digit_canvas"
    )

    predict_button = st.button(
        "🚀 Predict Digit",
        use_container_width=True
    )

with right:
    st.subheader("🤖 AI Prediction")

    if predict_button:
        if canvas_result.image_data is None:
            st.warning("Please draw a digit first.")

        else:
            image = canvas_result.image_data.astype(np.uint8)
            gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            coords = cv2.findNonZero(gray)

            if coords is None:
                st.warning("Please draw a digit first.")

            else:
                x, y, w, h = cv2.boundingRect(coords)
                digit = gray[y:y+h, x:x+w]

                size = 20
                scale = size / max(w, h)

                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))

                resized_digit = cv2.resize(
                    digit,
                    (new_w, new_h),
                    interpolation=cv2.INTER_AREA
                )

                processed_image = np.zeros((28, 28), dtype=np.uint8)

                x_offset = (28 - new_w) // 2
                y_offset = (28 - new_h) // 2

                processed_image[
                    y_offset:y_offset+new_h,
                    x_offset:x_offset+new_w
                ] = resized_digit

                processed = processed_image.astype("float32") / 255.0
                processed = processed.reshape(1, 28, 28, 1)

                prediction = model.predict(processed, verbose=0)
                predicted_digit = int(np.argmax(prediction))
                confidence = float(np.max(prediction)) * 100

                st.markdown(
                    f"""
                    <div class="result-card">
                        <p style="color:#aab4c3;">PREDICTED DIGIT</p>
                        <h1 style="font-size:90px;color:#58a6ff;">
                            {predicted_digit}
                        </h1>
                        <p style="font-size:20px;">
                            Confidence: {confidence:.2f}%
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write("")
                st.subheader("Prediction Probabilities")
                st.bar_chart(prediction[0])

                st.image(
                    cv2.resize(
                        processed_image,
                        (280, 280),
                        interpolation=cv2.INTER_NEAREST
                    ),
                    caption="Processed CNN Input",
                    width=220
                )

    else:
        st.info("Draw a digit and click Predict Digit.")

st.divider()
st.caption("Built with Python • TensorFlow • CNN • Streamlit")
