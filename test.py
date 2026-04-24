import json
import uuid
import threading
import time
import websocket

ACCID = None
ROBOT_IP = "10.192.1.2"

should_exit = False
ws_client = None

# 14个关节
joint_values = [-0.05]*14#[+0.0693,-0.1887,-0.1404,+1.1613,+0.0267,+0.3901,+0.2629,-0.1084,+0.2104,+0.1997,-1.1414,+0.0056,-0.1830,-0.1396]

# 当前选中的关节，内部索引 0~13，对应用户输入 1~14
current_joint_index = 0

# 每次增减步长
STEP = 0.05

# movej 时间
MOVE_TIME = 0.2


def generate_guid():
    return str(uuid.uuid4())


def send_request(title, data=None):
    global ACCID, ws_client
    if data is None:
        data = {}

    message = {
        "accid": ACCID,
        "title": title,
        "timestamp": int(time.time() * 1000),
        "guid": generate_guid(),
        "data": data
    }

    message_str = json.dumps(message)
    print(f"\n[Send] {message_str}")

    if ws_client:
        ws_client.send(message_str)
    else:
        print("[Error] ws_client is None")


def send_movej():
    send_request("request_movej", {
        "joint": joint_values,
        "time": MOVE_TIME
    })


def print_status():
    print("\n========== CURRENT STATUS ==========")
    print(f"Selected joint: {current_joint_index + 1}")
    for i, v in enumerate(joint_values, start=1):
        mark = " <==" if i == current_joint_index + 1 else ""
        print(f"J{i:02d}: {v:.4f}{mark}")
    print("====================================\n")


def handle_commands():
    global should_exit, current_joint_index, joint_values

    print("""
控制说明：
  1~14 : 选择关节
  q    : 当前关节 +0.05
  e    : 当前关节 -0.05
  p    : 打印当前关节值
  s    : 手动发送一次 request_movej
  r    : 所有关节清零
  x    : 退出
""")
    print_status()

    while not should_exit:
        cmd = input("请输入指令: ").strip().lower()

        if cmd == "x":
            should_exit = True
            break

        elif cmd.isdigit():
            idx = int(cmd)
            if 1 <= idx <= 14:
                current_joint_index = idx - 1
                print(f"[Info] 已选择关节 J{idx}")
            else:
                print("[Warn] 请输入 1~14")
            print_status()

        elif cmd == "q":
            joint_values[current_joint_index] += STEP
            print(f"[Info] J{current_joint_index + 1} += {STEP}")
            print_status()
            send_movej()

        elif cmd == "e":
            joint_values[current_joint_index] -= STEP
            print(f"[Info] J{current_joint_index + 1} -= {STEP}")
            print_status()
            send_movej()

        elif cmd == "p":
            print_status()

        elif cmd == "s":
            send_movej()

        elif cmd == "r":
            joint_values = [0.0] * 14
            print("[Info] 所有关节已清零")
            print_status()
            send_movej()

        else:
            print("[Warn] 无效指令")


def on_open(ws):
    print("Connected!")
    threading.Thread(target=handle_commands, daemon=True).start()


def on_message(ws, message):
    global ACCID
    try:
        root = json.loads(message)
        title = root.get("title", "")
        recv_accid = root.get("accid", None)

        if recv_accid is not None:
            ACCID = recv_accid

        if title != "notify_robot_info":
            print(f"\n[Recv] {message}")
    except Exception as e:
        print(f"[Error] on_message parse failed: {e}")
        print(f"[Raw] {message}")


def on_error(ws, error):
    print(f"[WebSocket Error] {error}")


def on_close(ws, close_status_code, close_msg):
    print("Connection closed.")


def main():
    global ws_client

    ws_client = websocket.WebSocketApp(
        f"ws://{ROBOT_IP}:5000",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    print("Press Ctrl+C to exit.")
    ws_client.run_forever()


if __name__ == "__main__":
    main()