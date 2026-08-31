import cv2
import time
import mediapipe as mp


# Kamera
cap = cv2.VideoCapture(0)


# Yeni MediaPipe API
mpHand = mp.tasks.vision.HandLandmarker

mpDraw = mp.tasks.vision.drawing_utils

mpDrawingStyles = mp.tasks.vision.drawing_styles

mpConnections = mp.tasks.vision.HandLandmarksConnections


# Hand Landmarker parametrləri
options = mp.tasks.vision.HandLandmarkerOptions(

    base_options=mp.tasks.BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),

    running_mode=mp.tasks.vision.RunningMode.VIDEO,

    num_hands=2,

    min_hand_detection_confidence=0.5,

    min_hand_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


# Hand detector yaradılır
hands = mpHand.create_from_options(options)


while True:

    success, img = cap.read()

    if not success:
        break


    # OpenCV BGR -> RGB
    imgRGB = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )


    # NumPy şəkli MediaPipe Image obyektinə çeviririk
    mpImage = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=imgRGB
    )


    # Hər frame üçün timestamp
    timestamp_ms = int(time.monotonic() * 1000)


    # Əli analiz edirik
    results = hands.detect_for_video(
        mpImage,
        timestamp_ms
    )


    print(results.hand_landmarks)


    # Əl tapılıbsa
    if results.hand_landmarks:

        # Hər tapılan əl üçün
        for handLms in results.hand_landmarks:


            # 21 landmark və aralarındakı xətləri çəkirik
            mpDraw.draw_landmarks(

                img,

                handLms,

                mpConnections.HAND_CONNECTIONS,

                mpDrawingStyles.get_default_hand_landmarks_style(),

                mpDrawingStyles.get_default_hand_connections_style()
            )


            # Əldəki 21 nöqtəni ayrı-ayrılıqda oxuyuruq
            for id, lm in enumerate(handLms):

                h, w, c = img.shape


                # Normalized koordinatı pikselə çeviririk
                cx = int(lm.x * w)
                cy = int(lm.y * h)


                # Landmark ID-ni terminalda görmək üçün
                print(id, cx, cy)


                # 20-ci nöqtəni xüsusi göstəririk
                if id == 20:

                    cv2.circle(
                        img,
                        (cx, cy),
                        9,
                        (255, 0, 0),
                        cv2.FILLED
                    )


    cv2.imshow("img", img)


    # q basanda proqram dayansın
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


hands.close()

cap.release()

cv2.destroyAllWindows()