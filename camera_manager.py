import carla
import numpy as np
import cv2

class CameraManager:
    def __init__(self, world, vehicle, width=640, height=480):
        self.width = width
        self.height = height
        self.image_data = None
        self.seg_data = None

        bp_lib = world.get_blueprint_library()

        self.rgb_bp = bp_lib.find('sensor.camera.rgb')
        self.rgb_bp.set_attribute('image_size_x', str(width))
        self.rgb_bp.set_attribute('image_size_y', str(height))
        self.rgb_bp.set_attribute('fov', '110')

        self.seg_bp = bp_lib.find('sensor.camera.semantic_segmentation')
        self.seg_bp.set_attribute('image_size_x', str(width))
        self.seg_bp.set_attribute('image_size_y', str(height))
        self.seg_bp.set_attribute('fov', '110')

        spawn_point = carla.Transform(carla.Location(x=1.5, z=2.4))

        self.rgb_cam = world.spawn_actor(self.rgb_bp, spawn_point, attach_to=vehicle)
        self.seg_cam = world.spawn_actor(self.seg_bp, spawn_point, attach_to=vehicle)

        self.rgb_cam.listen(lambda image: self._process_rgb(image))
        self.seg_cam.listen(lambda image: self._process_seg(image))

    def _process_rgb(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((self.height, self.width, 4))[:, :, :3]
        self.image_data = array

    def _process_seg(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((self.height, self.width, 4))[:, :, 2]
        self.seg_data = array

    def get_rgb_image(self):
        return self.image_data

    def get_segmentation_mask(self):
        return self.seg_data

    def destroy(self):
        self.rgb_cam.stop()
        self.seg_cam.stop()
        self.rgb_cam.destroy()
        self.seg_cam.destroy()
