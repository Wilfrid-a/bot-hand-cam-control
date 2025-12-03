import cv2
import mediapipe as mp
import serial
import serial.tools.list_ports
import time
import numpy as np

# ==========================
# PROCURAR ARDUINO
# ==========================
def encontrar_arduino():
    portas = serial.tools.list_ports.comports()
    for porta in portas:
        if "Arduino" in porta.description or "CH340" in porta.description:
            return porta.device
    return None

def conectar_arduino():
    porta = encontrar_arduino()
    if porta:
        print(f"Arduino encontrado em {porta}. Conectando...")
        try:
            arduino = serial.Serial(porta, 9600, timeout=1)
            time.sleep(2)
            print("Conectado com sucesso!")
            return arduino, porta
        except:
            print("Erro ao conectar no Arduino.")
            return None, None
    else:
        print("Arduino não encontrado.")
        return None, None

arduino, porta_arduino = conectar_arduino()

# ==========================
# MEDIA PIPE
# ==========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# filtro para suavizar
filtro = np.zeros(5)

def smooth(index, novo, alpha=0.3):
    global filtro
    filtro[index] = filtro[index] * (1 - alpha) + novo * alpha
    return int(filtro[index])

# calibração de mão
calibrado = False
offset = np.zeros(5)

def calibrar(raw):
    global calibrado, offset
    offset = np.array(raw)
    calibrado = True
    print("Calibração concluída!")

def y2ang(y):
    return int(max(0, min(180, 180 - y * 180)))

# ==========================
# LOOP PRINCIPAL
# ==========================
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if arduino is None:
            arduino, porta_arduino = conectar_arduino()

        status = "Sem Arduino" if arduino is None else f"Arduino em {porta_arduino}"
        cv2.putText(frame, status, (20, 40), 0, 0.7, (0,255,0), 2)

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]

            raw = [
                y2ang(hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y),
                y2ang(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y),
                y2ang(hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y),
                y2ang(hand.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y),
                y2ang(hand.landmark[mp_hands.HandLandmark.PINKY_TIP].y),
            ]

            if not calibrado:
                calibrar(raw)

            ajustado = np.array(raw) - offset
            ajustado = np.clip(ajustado, 0, 180)

            s = [smooth(i, ajustado[i]) for i in range(5)]

            comando = f"dedo1={s[0]};dedo2={s[1]};dedo3={s[2]};dedo4={s[3]};dedo5={s[4]}\n"

            if arduino:
                try:
                    arduino.write(comando.encode())
                except:
                    print("Arduino desconectado!")
                    arduino = None

            # ==========================
            # LER FEEDBACK DO ARDUINO
            # ==========================
            feedback = ""
            try:
                if arduino and arduino.in_waiting:
                    feedback = arduino.readline().decode().strip()
                    print("ARDUINO:", feedback)

                    if "pos=" in feedback:
                        dados = feedback.split(";")
                        pos_real = dados[0].replace("OK pos=", "").split(",")
                        pos_real = [int(x) for x in pos_real]
                        sensor = int(dados[1].replace("sensor=", ""))
                        erro_servo = int(dados[2].replace("erro_servo=", ""))
                        bateria = int(dados[3].replace("bateria=", ""))

                        cv2.putText(frame, f"Sensor: {sensor}", (20, 80), 0, 0.7, (255,255,0), 2)
                        cv2.putText(frame, f"Erro Servo: {erro_servo}", (20, 120), 0, 0.7, (0,0,255) if erro_servo else (0,255,0), 2)
                        cv2.putText(frame, f"Bateria baixa: {bateria}", (20, 160), 0, 0.7, (0,0,255) if bateria else (0,255,0), 2)

            except:
                pass

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Hand Control", frame)

        key = cv2.waitKey(1)

        if key == 27:  # ESC
            break
        if key == ord('c'):
            calibrado = False

cap.release()
cv2.destroyAllWindows()
