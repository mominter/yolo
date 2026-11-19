# YOLO 코드를 활용해서 streamlit을 활용해서
# 사진을 찍거나(받으면 ), YOLO 추론하는 결과가 나오는 페이지를 만들어보세요.

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import cv2

# ---------------------------------
# 1. 페이지 설정
# ---------------------------------
st.set_page_config(
    page_title="YOLO 객체 탐지",
    page_icon="📷",
    layout="wide"
)

st.title("📷 YOLO 객체 탐지")
st.write("사진을 촬영하거나 업로드하면 YOLO가 객체를 탐지합니다.")

# ---------------------------------
# 2. YOLO 모델 초기화
# ---------------------------------
@st.cache_resource
def load_model():
    model = YOLO("yolo26n.pt")
    return model

model = load_model()

# ---------------------------------
# 3. 이미지 입력
# ---------------------------------
st.subheader("1. 이미지 입력")

input_type = st.radio(
    "이미지 입력 방식을 선택하세요.",
    ["카메라 촬영", "사진 업로드"],
    horizontal=True
)

image = None

if input_type == "카메라 촬영":

    camera_image = st.camera_input(
        "카메라로 사진을 촬영하세요."
    )

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")

else:

    uploaded_file = st.file_uploader(
        "이미지를 업로드하세요.",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")


# ---------------------------------
# 4. YOLO 추론
# ---------------------------------
if image is not None:

    st.divider()
    st.subheader("2. 원본 이미지")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="입력 이미지",
            use_container_width=True
        )

    # PIL 이미지를 numpy 배열로 변환
    image_array = np.array(image)

    # ---------------------------------
    # 5. 추론 실행
    # ---------------------------------
    if st.button("🔍 객체 탐지 시작", type="primary"):

        with st.spinner("YOLO가 객체를 탐지하고 있습니다..."):

            results = model.predict(
                source=image_array,
                conf=0.25
            )

            result = results[0]

            # YOLO 결과 이미지 (BGR)
            result_image = result.plot()

            # BGR -> RGB 변환
            result_image = result_image[..., ::-1].copy()

            # RGB 배열 -> PIL 이미지
            result_pil = Image.fromarray(result_image)

            # 결과 저장
            st.session_state["result_image"] = result_pil
            st.session_state["result"] = result

            if "result_image" in st.session_state:

                result_image = st.session_state["result_image"]
                result = st.session_state["result"]

                with col2:
                    st.image(
                        result_image,
                        caption="YOLO 추론 결과",
                        use_container_width=True
                    )
    # ---------------------------------
    # 6. 추론 결과 출력
    # ---------------------------------
    if "result_image" in st.session_state:

        result_image = st.session_state["result_image"]
        result = st.session_state["result"]

        with col2:
            st.image(
                result_image,
                caption="YOLO 추론 결과",
                use_container_width=True
            )

        st.divider()
        st.subheader("3. 탐지 결과")

        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:

            # 탐지된 객체 정보
            names = [
                result.names[int(cls)]
                for cls in boxes.cls
            ]

            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()

            st.success(f"총 {len(names)}개의 객체를 탐지했습니다.")

            for i, (name, conf, box) in enumerate(
                zip(names, confs, xyxy), start=1
            ):

                x1, y1, x2, y2 = box

                with st.expander(
                    f"객체 {i} : {name} ({conf:.2%})"
                ):
                    st.write(f"**객체 이름:** {name}")
                    st.write(f"**정확도:** {conf:.2%}")
                    st.write(
                        f"**좌표 (x1, y1, x2, y2):** "
                        f"{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}"
                    )

        else:
            st.warning("탐지된 객체가 없습니다.")

        # ---------------------------------
        # 7. 결과 이미지 다운로드
        # ---------------------------------
        result_pil = Image.fromarray(result_image)

        image_buffer = io.BytesIO()
        result_pil.save(image_buffer, format="JPEG")

        st.download_button(
            label="📥 추론 결과 이미지 다운로드",
            data=image_buffer.getvalue(),
            file_name="yolo_result.jpg",
            mime="image/jpeg"
        )