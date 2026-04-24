# 机器人Tron2的ROS2部署
## 任务介绍
### 预先事情

模型已经训练好了：训练脚本/home/data/Project/lerobot_train/gr00t/train.sh

我在lerobot_py310虚拟conda环境下用lerobot库训练了groot模型，使用方法可参考/home/data/Project/RheoVLA/oracle_realtime_deploy.py

并将训练数据和权重分别存放在了~/TRON2_data/pick_stones和huggingface仓库trantor2nd/tron2-pickup-gr00t

数据采集时订阅了：
/camera/left/color/image_rect_raw/compressed
/camera/right/color/image_rect_raw/compressed
/camera/top/color/image_raw/compressed
/gripper_command
/gripper_state
/joint_command
/joint_states
/left_arm/ee_pose
/left_arm/ee_pose_cmd
/right_arm/ee_pose
/right_arm/ee_pose_cmd

### 目标

现在需要实现Tron2的部署，在本机推理groot模型:
1. 加载我的训练权重
2. 通过ros2获取机械臂的自由度、三个摄像头消息（需要什么可以参考我的lerobot数据集格式）
3. 推理groot得到action chunk
4. 基于websocket包，按照一定频率（默认10Hz），将action chunk逐条发送控制指令给tron2机器人


## 参考控制代码websocket

```python
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
joint_values = [0.0] * 14

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


    """
    可以，下面给你一个基于你这套 WebSocket 通信方式的 键盘控制 joint 脚本。

规则是：

输入 1~14：选择控制第几个关节
输入 q：当前关节 +0.05
输入 e：当前关节 -0.05
输入 p：打印当前 joint
输入 s：发送当前 joint 到机器人
输入 r：全部关节清零
输入 x：退出
    """
```




