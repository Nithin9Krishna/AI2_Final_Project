import cv2
import os
import csv
from datetime import datetime

class OutputRecorder:
    def __init__(self, save_dir="outputs/run_01", width=640, height=480):
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = os.path.join(save_dir, f"drive_{timestamp}.mp4")
        self.csv_path = os.path.join(save_dir, f"log_{timestamp}.csv")

        self.writer = cv2.VideoWriter(
            self.video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height)
        )

        self.csv_file = open(self.csv_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Time", "Steering", "SignalStatus"])  # Header

    def record(self, frame, steering, signal_status):
        overlayed = frame.copy()
        cv2.putText(overlayed, f"Steering: {steering:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(overlayed, f"Signal: {signal_status}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        self.writer.write(overlayed)
        cv2.imshow("Autonomous Driving", overlayed)
        cv2.waitKey(1)

        self.csv_writer.writerow([datetime.now().isoformat(), steering, signal_status])

    def close(self):
        self.writer.release()
        self.csv_file.close()
        cv2.destroyAllWindows()
