import cv2
import mediapipe as mp
import serial
import serial.tools.list_ports
import time
import numpy as np
from sklearn.neural_network import MLPRegressor

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

def checar_conexao(arduino):
    """Verifica se o Arduino ainda está conectado."""
    if arduino:
        try:
            arduino.write(b'\n')  # envio mínimo para testar
            return True
        except:
            return False
    return False

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
# IA: Aprendizado de movimentos
# ==========================
X_train = []  # entradas (movimentos anteriores)
y_train = []  # saídas (movimentos seguintes)
modelo = MLPRegressor(hidden_layer_sizes=(50,50), max_iter=500)
treinar_ia = False

# ==========================
# LOOP PRINCIPAL
# ==========================
ultimo_movimento = time.time()  # para monitorar o último frame com mão
tempo_espera = 3  # segundos antes da IA assumir

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Reconectar Arduino se necessário
        if arduino is None:
            arduino, porta_arduino = conectar_arduino()

        movimento_camera = [0,0,0,0,0]
        movimento_ia = [0,0,0,0,0]

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
            movimento_camera = s

            # atualizar último tempo com movimento
            ultimo_movimento = time.time()

            # treinar IA
            X_train.append(s)
            if len(X_train) > 1:
                y_train.append(s)
            if len(y_train) > 50 and not treinar_ia:
                modelo.fit(X_train[:-1], y_train)
                treinar_ia = True
                print("IA treinada com movimentos da mão!")

            # enviar para Arduino
            comando = f"dedo1={s[0]};dedo2={s[1]};dedo3={s[2]};dedo4={s[3]};dedo5={s[4]}\n"
            if arduino:
                try:
                    arduino.write(comando.encode())
                except:
                    arduino = None

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        else:
            # Se a mão sumiu, esperar 3s antes da IA assumir
            if treinar_ia and len(X_train) > 0 and (time.time() - ultimo_movimento > tempo_espera):
                s_prev = X_train[-1]
                s_pred = modelo.predict([s_prev])[0]
                s_pred = np.clip(s_pred, 0, 180).astype(int)
                movimento_ia = s_pred
                comando = f"dedo1={s_pred[0]};dedo2={s_pred[1]};dedo3={s_pred[2]};dedo4={s_pred[3]};dedo5={s_pred[4]}\n"
                if arduino:
                    try:
                        arduino.write(comando.encode())
                    except:
                        arduino = None
                X_train.append(s_pred)  # manter histórico para futuras predições

        # ==========================
        # Tabela no canto inferior direito
        # ==========================
        altura, largura = frame.shape[:2]
        tabela_x = largura - 210
        tabela_y = altura - 190
        dedos = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        font = 0
        font_scale = 0.5
        font_thickness = 1
        linha_altura = 25
        barra_max = 80

        overlay = frame.copy()
        rect_x1 = tabela_x - 10
        rect_y1 = tabela_y - 30
        rect_x2 = tabela_x + 180
        rect_y2 = tabela_y + linha_altura*5
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (50,50,50), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Cabeçalho
        cv2.putText(frame, "Dedos", (tabela_x, tabela_y - 10), font, font_scale, (255,255,255), font_thickness)
        cv2.putText(frame, "Cam", (tabela_x + 70, tabela_y - 10), font, font_scale, (255,0,0), font_thickness)
        cv2.putText(frame, "IA", (tabela_x + 130, tabela_y - 10), font, font_scale, (0,0,255), font_thickness)

        for i in range(5):
            y_pos = tabela_y + i*linha_altura
            # Texto do dedo
            cv2.putText(frame, dedos[i], (tabela_x, y_pos), font, font_scale, (255,255,255), font_thickness)

            # Valores
            cam_val = f"{movimento_camera[i]}"
            ia_val = f"{movimento_ia[i]}"
            cv2.putText(frame, cam_val, (tabela_x + 70, y_pos), font, font_scale, (255,0,0), font_thickness)
            cv2.putText(frame, ia_val, (tabela_x + 130, y_pos), font, font_scale, (0,0,255), font_thickness)

            # Barras logo ao lado do número
            barra_cam = int((movimento_camera[i]/180)*barra_max)
            barra_ia = int((movimento_ia[i]/180)*barra_max)
            cv2.rectangle(frame, (tabela_x + 70 + 30, y_pos - 10), (tabela_x + 70 + 30 + barra_cam, y_pos - 2), (255,0,0), -1)
            cv2.rectangle(frame, (tabela_x + 130 + 30, y_pos - 10), (tabela_x + 130 + 30 + barra_ia, y_pos - 2), (0,0,255), -1)

        # ==========================
        # Ícone de conexão Arduino
        # ==========================
        icon_radius = 10
        icon_x = largura - 30
        icon_y = 30
        conectado = checar_conexao(arduino)
        color = (0,255,0) if conectado else (0,0,255)
        cv2.circle(frame, (icon_x, icon_y), icon_radius, color, -1)
        cv2.circle(frame, (icon_x, icon_y), icon_radius, (255,255,255), 1)

        cv2.imshow("Hand IA Control", frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        if key == ord('c'):
            calibrado = False

cap.release()
cv2.destroyAllWindows()
