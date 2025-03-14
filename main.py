import cv2
import mediapipe as mp
import numpy as np
import keyboard
import screen_brightness_control as sbc
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import math

# Inicializace MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Otevření webkamery
cap = cv2.VideoCapture(0)

# Inicializace hlasitosti (Windows API - pycaw)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Rozsah hlasitosti
vol_min = 0.0  # Minimální hlasitost (0%)
vol_max = 0.15  # Maximální hlasitost (100%)
odchylka_hlasitosti = 30  # Povolená odchylka v stupních

# Rozsah jasu
min_brigthness = 0.5  # Minimální jas (50%)
max_brigthness = 1.0  # Maximální jas (100%)
odchylka_jasu = 60  # Povolená odchylka v stupních

# Stav pro detekci zavření pěsti
fist_closed = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Zpracování snímku
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Získání souřadnic jednotlivých prstů
            thumb_tip = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y])
            index_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y])
            middle_tip = np.array([hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y])
            ring_tip = np.array([hand_landmarks.landmark[16].x, hand_landmarks.landmark[16].y])
            pinky_tip = np.array([hand_landmarks.landmark[20].x, hand_landmarks.landmark[20].y])

            # Získání souřadnic kloubů
            middle_base = np.array([hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y])
            ring_base = np.array([hand_landmarks.landmark[13].x, hand_landmarks.landmark[13].y])
            pinky_base = np.array([hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y])

            # Kontrola, zda jsou malíček, prsteníček a prostředníček zavřené
            fingers_closed = (
                middle_tip[1] > middle_base[1] and
                ring_tip[1] > ring_base[1] and
                pinky_tip[1] > pinky_base[1]
            )

            # Kontrola, zda je pěst zavřená
            if fingers_closed and index_tip[1] > hand_landmarks.landmark[5].y:
                if not fist_closed:
                    keyboard.press_and_release('space')  # Přepnutí play/pause
                    fist_closed = True
            else:
                fist_closed = False

            if fingers_closed:
                # Spočítání vzdálenosti mezi palcem a ukazováčkem
                distance = np.linalg.norm(thumb_tip - index_tip)
                delta_x = index_tip[0] - thumb_tip[0]
                delta_y = index_tip[1] - thumb_tip[1]
                angle = math.degrees(math.atan2(delta_y, delta_x))

                if (-90 - odchylka_hlasitosti) <= angle <= (-90 + odchylka_hlasitosti) or (90 - odchylka_hlasitosti) <= angle <= (90 + odchylka_hlasitosti):
                    min_dist = 0.02
                    max_dist = 0.2
                    normalized_volume = (np.clip((distance - min_dist) / (max_dist - min_dist), 0, 1)) * (vol_max - vol_min)
                    scaled_volume = vol_min + normalized_volume
                    volume.SetMasterVolumeLevelScalar(scaled_volume, None)
                    vol_percentage = int(scaled_volume * 100)
                    cv2.putText(frame, f"Volume: {vol_percentage}%", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.line(frame, tuple((thumb_tip * [frame.shape[1], frame.shape[0]]).astype(int)),
                             tuple((index_tip * [frame.shape[1], frame.shape[0]]).astype(int)),
                             (0, 255, 0), 3)

                elif (-odchylka_jasu <= angle <= odchylka_jasu) or (180 - odchylka_jasu <= angle <= 180 + odchylka_jasu):
                    min_dist = 0.02
                    max_dist = 0.2
                    normalized_brightness = np.clip((distance - min_dist) / (max_dist - min_dist), min_brigthness, max_brigthness)
                    brightness_level = int(normalized_brightness * 100)
                    sbc.set_brightness(brightness_level)
                    cv2.putText(frame, f"Brightness: {brightness_level}%", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
                    cv2.line(frame, tuple((thumb_tip * [frame.shape[1], frame.shape[0]]).astype(int)),
                             tuple((index_tip * [frame.shape[1], frame.shape[0]]).astype(int)),
                             (255, 165, 0), 3)
                else:
                    cv2.putText(frame, "Adjust hand position", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Hand Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
