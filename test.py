import random
from typing import Tuple
import json
import os

def sample_random_state_action(
	pos_bounds: Tuple[Tuple[float,float], Tuple[float,float]] = ((-1.0,1.0),(-1.0,1.0)),
	target_bounds: Tuple[Tuple[float,float], Tuple[float,float]] = ((-1.0,1.0),(-1.0,1.0)),
	max_delta: float = 0.2,
	integer: bool = False,
	n_frames: int = 10,
) -> dict:
	"""
	Return a random current position, target position, and action delta.
	- pos_bounds: ((xmin,xmax),(ymin,ymax))
	- target_bounds: same format for target
	- max_delta: absolute bound for action components
	- integer: if True, sample integers
	"""
	def _rand(a,b):
		if integer:
			return random.randint(int(a), int(b))
		return random.uniform(a,b)

	current = (_rand(*pos_bounds[0]), _rand(*pos_bounds[1]))
	target = (_rand(*target_bounds[0]), _rand(*target_bounds[1]))
	action_delta = (_rand(-max_delta, max_delta), _rand(-max_delta, max_delta))

	next_pos = (current[0] + action_delta[0], current[1] + action_delta[1])
	after_n = (current[0] + n_frames * action_delta[0], current[1] + n_frames * action_delta[1])

	return {
		"current": current,
		"target": target,
		"action_delta": action_delta,
		"next": next_pos,
		"after_n_frames": after_n,
		"n_frames": n_frames,
	}

if __name__ == "__main__":
	# Example: sample a random frame from the recordings JSON and print details
	def sample_random_frame_from_recordings(filepath: str, lookahead: int = 10):
		if not os.path.exists(filepath):
			print("file not found:", filepath)
			return
		with open(filepath, "r") as f:
			data = json.load(f)

		if not data:
			print("no recordings in file")
			return

		rec = random.choice(data)
		frames = rec.get("frames", [])
		if not frames:
			print("no frames in chosen recording")
			return

		i = random.randrange(len(frames))
		cur = frames[i]
		cur_pos = cur.get("positions", [])
		cur_t = cur.get("t")

		# next frame
		next_idx = i + 1
		next_frame = frames[next_idx] if next_idx < len(frames) else None
		next_pos = next_frame.get("positions", []) if next_frame else None

		# after lookahead frames (clamp to last)
		after_idx = min(i + lookahead, len(frames) - 1)
		after_frame = frames[after_idx]
		after_pos = after_frame.get("positions", [])

		target = rec.get("target", {}).get("positions")

		# action delta (next - current) for each joint
		action_delta = None
		if next_pos is not None:
			action_delta = [n - c for n, c in zip(next_pos, cur_pos)]

		print("recording hz:", rec.get("hz"))
		print("picked frame index:", i, "t:", cur_t)
		print("current positions:", cur_pos)
		print("target positions:", target)
		print("action_delta (next - current) [6 joints]:", action_delta)
		print("next positions:", next_pos)
		print(f"positions after {lookahead} frames (index {after_idx}):", after_pos)

	sample_random_frame_from_recordings("imitation_learning_recordings.json", lookahead=10)

