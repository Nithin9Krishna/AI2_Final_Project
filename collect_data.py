import carla
import numpy as np
import cv2
import os
import csv
import time

SAVE_DIR = "dataset"
IMAGE_DIR = os.path.join(SAVE_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

csv_path = os.path.join(SAVE_DIR, "labels.csv")
csv_file = open(csv_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["filename", "steering"])

def get_lane_mask(image, width, height):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((height, width, 4))[:, :, 2]  # Blue channel
    return array

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    vehicle_bp = blueprint_library.filter("model3")[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True)

    cam_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
    cam_bp.set_attribute("image_size_x", "640")
    cam_bp.set_attribute("image_size_y", "480")
    cam_bp.set_attribute("fov", "110")
    cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    cam = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)

    frame_id = 1
    MAX_FRAMES = 2000

    def process_image(image):
        nonlocal frame_id
        mask = get_lane_mask(image, 640, 480)

        filename = f"{frame_id:04d}.png"
        save_path = os.path.join(IMAGE_DIR, filename)
        cv2.imwrite(save_path, mask)

        control = vehicle.get_control()
        csv_writer.writerow([filename, round(control.steer, 3)])
        print(f"Saved {filename}  steering={control.steer:.3f}")
        frame_id += 1

        if frame_id > MAX_FRAMES:
            print("[DONE] Finished data collection.")
            cam.stop()
            cam.destroy()
            vehicle.destroy()
            csv_file.close()
            exit()

    cam.listen(lambda image: process_image(image))

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print(" Interrupted.")
        cam.stop()
        cam.destroy()
        vehicle.destroy()
        csv_file.close()

if __name__ == "__main__":
    main()
