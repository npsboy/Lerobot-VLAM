import json
import time
from pathlib import Path
import torch
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
import numpy as np

# We import the model class from train.py
from train import MotionPredictor

LEADER_PORT = "COM5"
FOLLOWER_PORT = "COM6"
HZ = 20.0
PERIOD = 1.0 / HZ
WINDOW_SIZE = 10
MAX_STEP_TICKS = 5  # Max movement per cycle for slow, smooth motion


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper"
]

CALIBRATION_FILE = r"C:\Users\Tusha\.cache\huggingface\lerobot\calibration\robots\so_follower\my_follower_arm.json"


def get_joint_positions(robot) -> list[int]:
    bus = getattr(robot, "bus", robot)
    values = bus.sync_read("Present_Position", normalize=False)
    
    clean_values = []
    for name in JOINT_NAMES:
        raw_val = int(values.get(name, 0))
        # Mask off sign bits/noise beyond 15-bits that cause lerobot encoders to crash.
        # This handles negative numbers reading back as 32878 (which is 0x806E).
        # We strip the top bit if it's set to extract just the position offset.
        if raw_val > 32767:
            # 0x0FFF gives us a safe generic positive value mask if hardware goes negative
            raw_val = raw_val & 0x7FFF
            
        clean_values.append(raw_val)
        
    return clean_values


def set_joint_positions(robot, positions: list[int], calib: dict) -> None:
    clamped_dict = {}
    bus = getattr(robot, "bus", robot)
    
    for i, name in enumerate(JOINT_NAMES):
        pos = positions[i]
        
        # Strict clamping to min/max
        c_min = calib[name]["range_min"]
        c_max = calib[name]["range_max"]
        
        clamped_pos = pos
        if pos > c_max:
            print(f"⚠️ CLAMPED: {name} position {pos} exceeded max {c_max}")
            clamped_pos = c_max
        elif pos < c_min:
            print(f"⚠️ CLAMPED: {name} position {pos} exceeded min {c_min}")
            clamped_pos = c_min
            
        # The API requires mapping joint names exactly as defined in the robot's 
        # internal motor definitions. For lerobot SO101, it uses exact joint names.
        clamped_dict[name] = clamped_pos
    
    bus.sync_write("Goal_Position", clamped_dict, normalize=False)


def build_frame_features(curr_pos: list[float], target_vals: list[float]) -> list[float]:
    sample = []
    # 1. current angle
    for i in range(6):
        sample.append(float(curr_pos[i]))
    # 2. target - current
    for i in range(6):
        sample.append(float(target_vals[i] - curr_pos[i]))
    # 3. target angle
    for i in range(6):
        sample.append(float(target_vals[i]))
    return sample


def main():
    # 1. Load Calibration
    with open(CALIBRATION_FILE, "r") as f:
        calib_data = json.load(f)

    # 2. Load Normalization params
    with open("preprocessed_data.json", "r") as f:
        prep_data = json.load(f)
        
    X_mean = torch.tensor(prep_data["X_mean"], dtype=torch.float32)
    X_std = torch.tensor(prep_data["X_std"], dtype=torch.float32)
    Y_mean = torch.tensor(prep_data["Y_mean"], dtype=torch.float32)
    Y_std = torch.tensor(prep_data["Y_std"], dtype=torch.float32)

    # 3. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MotionPredictor(input_dim=18, sequence_length=WINDOW_SIZE).to(device)
    
    # Checkpoint has pieces
    state_dicts = torch.load("robot_transformer.pt", map_location=device)
    model.input_projection.load_state_dict(state_dicts["embeddings"])
    model.positional_embedding.load_state_dict(state_dicts["positional_embedding"])
    model.transformer.load_state_dict(state_dicts["transformer"])
    model.prediction_head.load_state_dict(state_dicts["prediction_head"])
    model.eval()

    # 4. Init Arms
    config_leader = SO101FollowerConfig(port=LEADER_PORT, id="leader_arm")
    leader = SO101Follower(config_leader)
    
    config_follower = SO101FollowerConfig(port=FOLLOWER_PORT, id="my_follower_arm")
    follower = SO101Follower(config_follower)
    
    print(f"Connecting to Leader on {LEADER_PORT} and Follower on {FOLLOWER_PORT}...")
    try:
        leader.connect(calibrate=False)
    except Exception as e:
        print(f"Failed to connect leader: {e}")
        pass
    try:
        follower.connect(calibrate=False)
    except Exception as e:
        print(f"Failed to connect follower: {e}")
        pass

    leader_bus = getattr(leader, "bus", leader)
    follower_bus = getattr(follower, "bus", follower)
    
    if hasattr(leader_bus, "disable_torque"):
        leader_bus.disable_torque()
        
    # Prevent startup jerk: Before enabling torque, read the current physical position 
    # and set it as the goal position so the arm doesn't snap to an old target.
    curr_startup_pos = get_joint_positions(follower)
    set_joint_positions(follower, curr_startup_pos, calib_data)
    
    if hasattr(follower_bus, "enable_torque"):
        follower_bus.enable_torque()

    try:
        # Phase 1: Record 10 frames
        input("Press Enter to start recording the first 10 frames from the leader arm...")
        leader_history = []
        for _ in range(WINDOW_SIZE):
            pos_dict = leader_bus.sync_read("Present_Position", normalize=False)
            pos_list = [int(pos_dict.get(n, 0)) for n in JOINT_NAMES]
            leader_history.append(pos_list)
            time.sleep(PERIOD)
        print(f"Recorded 10 frames.")

        # Phase 2: Target Selection
        input("Move the leader arm to the target position and press Enter...")
        target_dict = leader_bus.sync_read("Present_Position", normalize=False)
        target_vals = [int(target_dict.get(n, 0)) for n in JOINT_NAMES]
        print(f"Target selected: {target_vals}")

        # Phase 3: Move Follower & Run Model
        print("Starting real-time follower execution... Press Ctrl+C to stop.")
        
        # Follower history will grow as it moves.
        # We keep the commanded positions in the history so the model sees new actions immediately.
        follower_history = []
        
        # Get its current real start position and use it as the commanded baseline.
        curr_follower_pos = get_joint_positions(follower)
        commanded_follower_pos = curr_follower_pos[:]

        while True:
            start_time = time.time()
            
            # Read real follower pos for feedback, but keep the commanded sequence moving forward.
            curr_follower_pos = get_joint_positions(follower)
            
            # Use recorded leader history to pad if follower history is smaller than 10 frames
            if len(follower_history) < WINDOW_SIZE:
                pad_len = WINDOW_SIZE - len(follower_history)
                window_frames = leader_history[-pad_len:] + follower_history
            else:
                window_frames = follower_history[-WINDOW_SIZE:]

            # Build X
            x_window = []
            for frame in window_frames:
                feats = build_frame_features(frame, target_vals)
                x_window.append(feats)
            
            # Normalize X
            X_tensor = torch.tensor([x_window], dtype=torch.float32)
            X_norm = (X_tensor - X_mean) / X_std
            X_norm = X_norm.to(device)

            # Predict delta
            with torch.no_grad():
                pred_norm = model(X_norm)
            
            pred_norm = pred_norm.cpu().squeeze(0)  # shape (6,)
            
            # Denormalize output
            # Y_norm = (Y - Y_mean) / Y_std  =>  Y = Y_norm * Y_std + Y_mean
            predicted_delta = pred_norm * Y_std + Y_mean
            predicted_delta = predicted_delta.numpy()
            
            # Print model output
            print(f"Model Prediction (delta): {[f'{d:.2f}' for d in predicted_delta]}")

            # Calculate next target follower pos from the last commanded state.
            next_follower_pos = []
            for i in range(6):
                desired_pos = commanded_follower_pos[i] + predicted_delta[i]
                c_min = calib_data[JOINT_NAMES[i]]["range_min"]
                c_max = calib_data[JOINT_NAMES[i]]["range_max"]

                target_pos = desired_pos
                if desired_pos > c_max:
                    print(f"⚠️ CLAMPED: {JOINT_NAMES[i]} target {desired_pos:.1f} exceeded max {c_max}")
                    target_pos = c_max
                elif desired_pos < c_min:
                    print(f"⚠️ CLAMPED: {JOINT_NAMES[i]} target {desired_pos:.1f} exceeded min {c_min}")
                    target_pos = c_min

                delta_to_target = target_pos - commanded_follower_pos[i]
                if delta_to_target > MAX_STEP_TICKS:
                    delta_to_target = MAX_STEP_TICKS
                elif delta_to_target < -MAX_STEP_TICKS:
                    delta_to_target = -MAX_STEP_TICKS

                next_pos = int(commanded_follower_pos[i] + delta_to_target)
                next_follower_pos.append(next_pos)
            
            # Print motor commands
            print(f"Motor Commands (ticks): {next_follower_pos}")

            # Advance the sequence with the command that was just sent.
            follower_history.append(next_follower_pos)
            if len(follower_history) > WINDOW_SIZE:
                follower_history = follower_history[-WINDOW_SIZE:]

            commanded_follower_pos = next_follower_pos[:]
            
            # Write out to physical follower (function handles clamping)
            set_joint_positions(follower, next_follower_pos, calib_data)

            # Sleep to match HZ
            elapsed = time.time() - start_time
            sleep_time = PERIOD - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        # Cleanup - Disable torque so both arms can be moved freely by hand
        print("Disabling torque on both arms...")
        if hasattr(leader_bus, "disable_torque"):
            try:
                leader_bus.disable_torque()
            except Exception:
                pass
        if hasattr(follower_bus, "disable_torque"):
            try:
                follower_bus.disable_torque()
            except Exception:
                pass
                
        leader.disconnect()
        follower.disconnect()

if __name__ == "__main__":
    main()