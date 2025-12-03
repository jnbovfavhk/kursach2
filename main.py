
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


    # input_video = r"D:\DaVinci Projects\new edit\антон флекс.avi"

    input_video = r"C:\Users\ilyab\Downloads\паразиты фрагмент.mp4"
    output_video = r"C:\Users\ilyab\PycharmProjects\pythonProject5\parasytes_output_dasiamrpn.mp4"

    manager = FaceDetectionManager(
        detection_interval=2,  # обнаружение каждые ?? секунды
        confidence_threshold=0.8,  # порог уверенности
        tracker_type='dasiamrpn', # тип трекера
        iou_threshold=0.2,
        similarity_threshold=0.8
    )
    start_time = time.time()
    manager.process_video(input_video, output_video)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Видео обработано за {execution_time:.3f} секунд")