import cv2
import numpy as np
import mediapipe as mp
import random
import time

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# じゃんけんの手を識別する関数
def classify_hand(hand_landmarks):
    # 各指の先端と第2関節のランドマークを取得
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]

    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]

    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]

    ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]

    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]

    # 指の屈伸状態を判定
    fingers = []

    # 親指
    if thumb_tip.x < thumb_ip.x:
        fingers.append(1)
    else:
        fingers.append(0)

    # 他の指
    for tip, pip in zip(
        [index_finger_tip, middle_finger_tip, ring_finger_tip, pinky_tip],
        [index_finger_pip, middle_finger_pip, ring_finger_pip, pinky_pip]):
        if tip.y < pip.y:
            fingers.append(1)
        else:
            fingers.append(0)

    # 手の形を判定
    if fingers == [0, 0, 0, 0, 0]:
        return 'Rock'
    elif fingers == [0, 1, 1, 0, 0]:
        return 'Scissors'
    elif fingers == [0, 1, 1, 1, 1]:
        return 'Paper'
    else:
        return 'Unknown'

# 顔の向きを識別する関数
def get_face_direction(face_landmarks):
    nose_tip = face_landmarks.landmark[1]

    if nose_tip.y < 0.4:
        return 'Up'
    elif nose_tip.y > 0.6:
        return 'Down'
    elif nose_tip.x < 0.4:
        return 'Left'
    elif nose_tip.x > 0.6:
        return 'Right'
    else:
        return 'Center'

# コンピュータの手をランダムに生成する関数
def computer_hand():
    return random.choice(['Rock', 'Scissors', 'Paper'])

# じゃんけんの勝敗を判定する関数
def judge_janken(player, computer):
    if player == computer:
        return 'draw'
    elif (player == 'Rock' and computer == 'Scissors') or \
         (player == 'Scissors' and computer == 'Paper') or \
         (player == 'Paper' and computer == 'Rock'):
        return 'player'
    elif player in ['Rock', 'Scissors', 'Paper']:
        return 'computer'
    else:
        return 'invalid'

# コンピュータの方向をランダムに生成する関数
def computer_direction():
    return random.choice(['Up', 'Down', 'Left', 'Right'])

def main():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands, \
         mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.7) as face_mesh:

        game_state = 'janken_start'
        result = None
        attacker = None  # 'player' or 'computer'
        player_hand = None
        comp_hand = None
        direction = None
        comp_direction_value = None
        player_direction = None
        hoi_result_next_state = None  # 次のゲーム状態を保持

        # タイマーの初期化
        state_start_time = time.time()
        state_duration = 0  # 各状態での待機時間

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image.flags.writeable = False
            hand_results = hands.process(image)
            face_results = face_mesh.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            h, w, _ = image.shape  # 画像の高さと幅を取得

            current_time = time.time()

            if game_state == 'janken_start':
                cv2.putText(image, 'Get ready for Rock-Paper-Scissors!', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, 'Prepare your hand.', (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 状態が変わったときにタイマーを設定
                if state_duration == 0:
                    state_start_time = current_time
                    state_duration = 2  # 2秒待機

                # 待機時間が過ぎたら次の状態へ
                if current_time - state_start_time > state_duration:
                    game_state = 'janken'
                    state_duration = 0  # タイマーリセット

            elif game_state == 'janken':
                cv2.putText(image, 'Rock, Paper, Scissors!', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(image, 'Show your hand!', (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        player_hand = classify_hand(hand_landmarks)
                        comp_hand = computer_hand()

                        # 状態が変わったときにタイマーを設定
                        if state_duration == 0:
                            state_start_time = current_time
                            state_duration = 1  # 1秒待機

                        # 待機時間が過ぎたら次の処理へ
                        if current_time - state_start_time > state_duration:
                            game_state = 'janken_result'
                            state_duration = 0  # タイマーリセット

            elif game_state == 'janken_result':
                cv2.putText(image, 'Rock, Paper, Scissors!', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(image, f'Your hand: {player_hand}', (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(image, f'Computer\'s hand: {comp_hand}', (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                result = judge_janken(player_hand, comp_hand)
                if result == 'draw':
                    cv2.putText(image, 'Draw! Try again...', (10, 170),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    hoi_result_next_state = 'janken_start'
                elif result == 'player':
                    attacker = 'player'
                    hoi_result_next_state = 'hoi_start'
                    cv2.putText(image, 'You win!', (10, 170),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif result == 'computer':
                    attacker = 'computer'
                    hoi_result_next_state = 'hoi_start'
                    cv2.putText(image, 'Computer wins!', (10, 170),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(image, 'Invalid input. Try again.', (10, 170),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    hoi_result_next_state = 'janken_start'

                # 状態が変わったときにタイマーを設定
                if state_duration == 0:
                    state_start_time = current_time
                    state_duration = 2  # 2秒待機

                # 待機時間が過ぎたら次の状態へ
                if current_time - state_start_time > state_duration:
                    game_state = hoi_result_next_state
                    state_duration = 0  # タイマーリセット

            elif game_state == 'hoi_start':
                cv2.putText(image, 'Get ready for Look This Way!', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 状態が変わったときにタイマーを設定
                if state_duration == 0:
                    state_start_time = current_time
                    state_duration = 2  # 2秒待機

                # 待機時間が過ぎたら次の状態へ
                if current_time - state_start_time > state_duration:
                    game_state = 'hoi'
                    state_duration = 0  # タイマーリセット

            elif game_state == 'hoi':
                cv2.putText(image, 'Look This Way!', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                if attacker == 'player':
                    cv2.putText(image, 'Your turn! Point in a direction.', (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            # 指の方向を推定（人差し指の方向で判定）
                            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                            index_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                            dx = index_finger_tip.x - index_finger_pip.x
                            dy = index_finger_tip.y - index_finger_pip.y
                            if abs(dx) > abs(dy):
                                direction = 'Right' if dx > 0 else 'Left'
                            else:
                                direction = 'Up' if dy < 0 else 'Down'

                            cv2.putText(image, f'Your direction: {direction}', (10, 130),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                            # コンピュータの方向を決定
                            comp_direction_value = computer_direction()

                            # 状態が変わったときにタイマーを設定
                            if state_duration == 0:
                                state_start_time = current_time
                                state_duration = 1  # 1秒待機

                            # 待機時間が過ぎたら次の処理へ
                            if current_time - state_start_time > state_duration:
                                game_state = 'hoi_choice'
                                state_duration = 0  # タイマーリセット

                elif attacker == 'computer':
                    cv2.putText(image, 'Computer\'s turn!', (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(image, 'Computer is choosing...', (10, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # コンピュータの方向を決定
                    if comp_direction_value is None:
                        comp_direction_value = computer_direction()

                    # 状態が変わったときにタイマーを設定
                    if state_duration == 0:
                        state_start_time = current_time
                        state_duration = 1  # 1秒待機

                    # 待機時間が過ぎたら次の処理へ
                    if current_time - state_start_time > state_duration:
                        game_state = 'hoi_choice'
                        state_duration = 0  # タイマーリセット

            elif game_state == 'hoi_choice':
                cv2.putText(image, 'Look This Way!', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                if attacker == 'player':
                    cv2.putText(image, 'Your turn! Point in a direction.', (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    cv2.putText(image, f'Your direction: {direction}', (10, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    # コンピュータの顔を表示
                    center = (int(w / 2), int(h / 2))
                    cv2.circle(image, center, 50, (0, 255, 255), -1)
                    eye_offset = {'Up': (0, -20), 'Down': (0, 20), 'Left': (-20, 0), 'Right': (20, 0)}
                    ex, ey = eye_offset.get(comp_direction_value, (0, 0))
                    cv2.circle(image, (center[0] - 15 + ex, center[1] - 10 + ey), 5, (0, 0, 0), -1)
                    cv2.circle(image, (center[0] + 15 + ex, center[1] - 10 + ey), 5, (0, 0, 0), -1)
                    cv2.putText(image, f'Computer\'s direction: {comp_direction_value}', (10, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # 勝敗判定
                    if direction == comp_direction_value:
                        cv2.putText(image, 'You Win!', (10, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        hoi_result_next_state = 'end'
                    else:
                        cv2.putText(image, 'Try Again!', (10, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        hoi_result_next_state = 'janken_start'

                elif attacker == 'computer':
                    cv2.putText(image, 'Computer\'s turn!', (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(image, f'Computer\'s direction: {comp_direction_value}', (10, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if face_results.multi_face_landmarks:
                        for face_landmarks in face_results.multi_face_landmarks:
                            mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
                            player_direction = get_face_direction(face_landmarks)
                            cv2.putText(image, f'Your head direction: {player_direction}', (10, 160),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    else:
                        player_direction = 'Not detected'
                        cv2.putText(image, 'Face not detected.', (10, 160),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    # 勝敗判定
                    if player_direction == comp_direction_value:
                        cv2.putText(image, 'Computer Wins!', (10, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        hoi_result_next_state = 'end'
                    else:
                        cv2.putText(image, 'Try Again!', (10, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        hoi_result_next_state = 'janken_start'

                # 状態が変わったときにタイマーを設定
                if state_duration == 0:
                    state_start_time = current_time
                    state_duration = 2  # 2秒待機

                # 待機時間が過ぎたら次の状態へ
                if current_time - state_start_time > state_duration:
                    # コンピュータの方向をリセット
                    comp_direction_value = None
                    game_state = hoi_result_next_state
                    state_duration = 0  # タイマーリセット

            elif game_state == 'end':
                cv2.putText(image, 'Game Over!', (10, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.putText(image, 'Press ESC to exit.', (10, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Rock-Paper-Scissors & Look This Way', image)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()