from hand_control.jump_trigger import JumpTrigger

frames = [0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1]   # 1 = fist in that frame
trigger = JumpTrigger()
print("Jump on frames:", [i for i, f in enumerate(frames) if trigger.update(bool(f))])