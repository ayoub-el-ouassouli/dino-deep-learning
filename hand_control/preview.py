import cv2

from hand_control.webcam_controller import WebcamJumpController


def main() -> None:
    controller = WebcamJumpController()
    print("Open and close your hand. Green dots = curled fingers. Press Q in the window to quit.")
    jumps = 0
    try:
        while True:
            if controller.error:
                print("Error:", controller.error)
                break
            if controller.consume_jump():
                jumps += 1
                print(f"JUMP #{jumps}")
            if controller.preview is not None:
                cv2.imshow("Hand control test", controller.preview)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
    finally:
        controller.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()