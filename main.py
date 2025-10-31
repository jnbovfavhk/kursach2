import cv2

from FaceDetectionManager import FaceDetectionManager

# if __name__ == "__main__":
#     input_video = "C:\\Users\\ilyab\\DaVinci Projects\\new edit\\антон флекс.avi"
#     output_video = "C:\\Users\\ilyab\\PycharmProjects\\pythonProject5\\output_video.mp4"
#
#     detector = FaceDetector(detection_interval_seconds=0.5)
#     detector.process_video(input_video, output_video)

if __name__ == "__main__":
    input_video = "C:\\Users\\ilyab\\DaVinci Projects\\new edit\\антон флекс.avi"
    output_video = "C:\\Users\\ilyab\\PycharmProjects\\pythonProject5\\output_with_tracking.mp4"

    manager = FaceDetectionManager(
        detection_interval=2,  # обнаружение каждые 2 секунды
        confidence_threshold=0.9,  # порог уверенности
        tracker_type='csrt' # тип трекера
    )

    manager.process_video(input_video, output_video)