import cv2
import numpy as np
import mediapipe as mp

mp_selfie = mp.solutions.selfie_segmentation

cap = cv2.VideoCapture(0)

# Save MP4
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, 30, (640, 480))

with mp_selfie.SelfieSegmentation(model_selection=1) as segment:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        # segmentation
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = segment.process(rgb)

        mask = res.segmentation_mask
        mask = (mask > 0.5).astype("uint8")  # person = 1, bg = 0

        # background = black
        bg = np.zeros_like(frame)

        # apply mask
        output = frame * mask[:, :, None] + bg * (1 - mask[:, :, None])

        out.write(output)
        cv2.imshow("Background Removed", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
out.release()
cv2.destroyAllWindows()
