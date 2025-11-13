
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time
from FaceDetectionManager import FaceDetectionManager

# if __name__ == "__main__":
#     input_video = "C:\\Users\\ilyab\\DaVinci Projects\\new edit\\антон флекс.avi"
#     output_video = "C:\\Users\\ilyab\\PycharmProjects\\pythonProject5\\output_video.mp4"
#
#     detector = FaceDetector(detection_interval_seconds=0.5)
#     detector.process_video(input_video, output_video)

if __name__ == "__main__":


    input_video = r"D:\DaVinci Projects\new edit\антон флекс.avi"
    output_video = r"C:\Users\ilyab\PycharmProjects\pythonProject5\output_with_tracking.mp4"

    manager = FaceDetectionManager(
        detection_interval=2,  # обнаружение каждые 2 секунды
        confidence_threshold=0.9,  # порог уверенности
        tracker_type='csrt' # тип трекера
    )
    start_time = time.time()
    manager.process_video(input_video, output_video)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Видео обработано за {execution_time:.3f} секунд")