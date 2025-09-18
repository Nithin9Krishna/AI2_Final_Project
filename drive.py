import carla
import time
import torch
import cv2
from camera_manager import CameraManager
from controller import MLController
from signal_detector.detector import SignalDetector
from record_outputs import OutputRecorder

def main():
    recorder = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.load_world("Town03")

    try:
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("model3")[0]
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicle.set_autopilot(False)

        camera = CameraManager(world, vehicle)
        controller = MLController(model_path="lane_cnn/lane_model.pth", device=device)
        signal_detector = SignalDetector()
        recorder = OutputRecorder()

        print("Autonomous driving started in Town05...")
        time.sleep(2)

        while True:
            seg_img = camera.get_segmentation_mask()
            rgb_img = camera.get_rgb_image()

            if seg_img is None or rgb_img is None:
                continue

            steering = controller.get_steering(seg_img)
            amplified_steering = max(min(steering * 2.5, 1.0), -1.0)

            vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=amplified_steering))

            signal_status = signal_detector.detect_signals(rgb_img)
            debug_frame = recorder.record(rgb_img, amplified_steering, signal_status)

            if debug_frame is not None and debug_frame.shape[1] > 0 and debug_frame.shape[0] > 0:
                cv2.imshow("Autonomous Driving", debug_frame)

            print(f"[DEBUG] Steering: {amplified_steering:.3f}, Signal: {signal_status}")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exit requested by user.")
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        if recorder: recorder.close()
        if 'camera' in locals(): camera.destroy()
        if 'vehicle' in locals(): vehicle.destroy()
        print("Simulation ended.")

if __name__ == "__main__":
    main()
