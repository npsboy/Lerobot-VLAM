"""Record multiple sequences of leader arm movements.

Records from leader arm on COM5 and stores recordings as a JSON array in
imitation_learning_recordings.json. Only reads joint positions, doesn't send motor commands.

Usage:
  python record_multiple.py
  
Commands:
  - Press Enter to start/stop each recording
  - Enter 'q' to quit the entire script
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


def read_all_joint_angles(robot_or_bus: Any) -> dict[str, int]:
	"""Read all current joint positions in raw motor ticks.

	Args:
		robot_or_bus: Either a LeRobot robot object that has ``.bus`` or a
			MotorsBus instance directly.

	Returns:
		Mapping ``joint_name -> position_ticks``.
	"""
	bus = getattr(robot_or_bus, "bus", robot_or_bus)
	# Use raw ticks (same scale as calibration values).
	values = bus.sync_read("Present_Position", normalize=False)
	return {joint: int(pos) for joint, pos in values.items()}


def record_single_sequence(leader: Any, sequence_num: int, hz: float = 20.0) -> dict[str, Any] | None:
	"""Record a single sequence until user presses Enter.
	
	Returns:
		Dictionary with 'frames' list, or None if user wants to quit.
	"""
	print(f"\n{'='*60}")
	print(f"Recording {sequence_num}")
	print("Move the leader arm. Press Enter when done, or 'q' to quit.")
	print('='*60)
	
	# Wait for user to be ready
	user_input = input("Press Enter to start recording (or 'q' to quit): ").strip()
	if user_input.lower() == 'q':
		return None
	
	frames: list[dict[str, Any]] = []
	period = 1.0 / float(hz)
	
	print("Recording in progress... (Press Enter to stop)")
	start_t = time.time()
	
	# Non-blocking input reading for Enter key
	import sys
	import threading
	
	stop_recording = False
	
	def wait_for_enter():
		nonlocal stop_recording
		try:
			input()
			stop_recording = True
		except (EOFError, KeyboardInterrupt):
			stop_recording = True
	
	# Start thread to wait for Enter key
	input_thread = threading.Thread(target=wait_for_enter, daemon=True)
	input_thread.start()
	
	try:
		while not stop_recording:
			ts = time.time() - start_t
			pos_dict = read_all_joint_angles(leader)
			
			# Convert dict to ordered array (preserve joint order)
			# Define the expected joint order
			joint_order = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
			pos_array = [pos_dict.get(joint, 0) for joint in joint_order]

			frame = {"t": ts, "positions": pos_array}
			
			frames.append(frame)
			time.sleep(period)
	except KeyboardInterrupt:
		print("\nRecording stopped by user (Ctrl+C).")
		return None
	
	print(f"Recording stopped. Captured {len(frames)} frames.")
	
	# Use the last frame as target
	target = frames[-1] if frames else None
	return {"hz": hz, "frames": frames, "target": target}


def load_recordings(file_path: Path) -> list[dict[str, Any]]:
	"""Load existing recordings or create a new list structure."""
	if file_path.exists():
		try:
			with file_path.open("r", encoding="utf-8") as f:
				data = json.load(f)
				if isinstance(data, list):
					return data
				if isinstance(data, dict):
					items = []
					hz = float(data.get("hz", 20.0))
					for key in sorted(k for k in data.keys() if k.startswith("recording")):
						recording = data[key]
						if isinstance(recording, dict) and "hz" not in recording:
							recording = {"hz": hz, **recording}
						items.append(recording)
					if items:
						return items
					if "frames" in data:
						if "hz" not in data:
							data = {"hz": 20.0, **data}
						return [data]
				return []
		except Exception as e:
			print(f"Warning: Could not load existing recordings: {e}")
			return []
	return []


def save_recordings(data: list[dict[str, Any]], file_path: Path) -> None:
	"""Save recordings to JSON file."""
	file_path.parent.mkdir(parents=True, exist_ok=True)
	with file_path.open("w", encoding="utf-8") as f:
		json.dump(data, f, indent=2)
	print(f"Saved recordings to {file_path}")


def main() -> None:
	leader_port = "COM5"
	hz = 20.0
	out_path = Path("imitation_learning_recordings.json")
	
	# Connect to leader arm
	config = SO101FollowerConfig(port=leader_port, id="leader_arm")
	leader = SO101Follower(config)
	
	print(f"Connecting to leader on {leader_port}...")
	try:
		leader.connect(calibrate=False)
		# Disable torque so the arm can be moved freely
		bus = getattr(leader, "bus", leader)
		if hasattr(bus, "disable_torque"):
			try:
				bus.disable_torque()
				print("Disabled motor torque on leader so it can be moved by hand.")
			except Exception:
				pass
	except Exception as e:
		print(f"Failed to connect to leader: {e}")
		return
	
	# Load existing recordings
	data = load_recordings(out_path)
	
	try:
		while True:
			sequence_num = len(data) + 1
			# Record one sequence
			result = record_single_sequence(leader, sequence_num, hz)
			if result is None:
				print("\nQuitting.")
				break
			
			# Store the recording
			data.append(result)
			save_recordings(data, out_path)
			
	except KeyboardInterrupt:
		print("\nInterrupted by user.")
	finally:
		# Restore torque before disconnecting
		try:
			bus = getattr(leader, "bus", leader)
			if hasattr(bus, "enable_torque"):
				try:
					bus.enable_torque()
					print("Re-enabled motor torque on leader.")
				except Exception:
					pass
		except Exception:
			pass
		leader.disconnect()


if __name__ == "__main__":
	main()
