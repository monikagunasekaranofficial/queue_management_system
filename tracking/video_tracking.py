
import cv2
import torch
import numpy as np
import sys

from pathlib import Path
from ultralytics import YOLO


# --------------------------------------------------
# 1. PROJECT PATHS AND IMPORTS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from queue_logic.zone_manager import QueueZoneManager
from queue_logic.waiting_time import WaitingTimeTracker
from queue_logic.object_mode import classes_for_mode
from analytics.queue_metrics import QueueMetrics
from analytics.little_law import LittleLawStaffing


# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------

MODEL_PATH = "yolo26s.pt"

CONFIDENCE = 0.15          # 0.15 avoids excessive false positives while catching occluded people
IMG_SIZE = 1280            # Increased from 512 so small/crowded people are clearly visible
IOU_THRESHOLD = 0.60       # Keeps overlapping boxes in dense queues
EXIT_GRACE_SECONDS = 1.0



QUEUE_LIMIT = 8
WAITING_LIMIT = 300
WINDOW_SECONDS = 300

DEVICE = 0 if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 3. VIDEO PROCESSING GENERATOR
# --------------------------------------------------

def process_video(
    video_path,
    model_path=MODEL_PATH,
    output_path=None,
    confidence=CONFIDENCE,
    exit_grace_seconds=EXIT_GRACE_SECONDS,
    queue_limit=QUEUE_LIMIT,
    waiting_limit=WAITING_LIMIT,
    window_seconds=WINDOW_SECONDS,
    object_mode="people"
):
    """
    Process a video and yield:
        - Annotated video frame
        - Queue metrics
        - Little's Law metrics
        - Alerts

    The Streamlit dashboard will consume these updates.
    """

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    class_ids = classes_for_mode(object_mode)
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    if exit_grace_seconds < 0:
        raise ValueError("exit_grace_seconds cannot be negative")

    if output_path is None:
        output_path = OUTPUT_DIR / "queue_output(2).mp4"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    writer = None

    try:

        # --------------------------------------------------
        # 4. LOAD MODEL
        # --------------------------------------------------

        print("Loading YOLO26 model...")

        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {model_path}"
            )

        model = YOLO(str(model_path))

        print(f"Using device: {DEVICE}")


        # --------------------------------------------------
        # 5. SELECT QUEUE REGION
        # --------------------------------------------------

        zone_manager = QueueZoneManager()

        print("Select the queue region.")

        queue_region = zone_manager.select_region(
            video_path
        )

        print("Queue region selected.")


        # --------------------------------------------------
        # 6. INITIALIZE TRACKERS AND ANALYTICS
        # --------------------------------------------------

        waiting_tracker = WaitingTimeTracker(
            exit_grace_seconds=exit_grace_seconds
        )

        queue_metrics = QueueMetrics(
            window_seconds=window_seconds
        )

        little_law = LittleLawStaffing(
            window_seconds=window_seconds,
            queue_limit=queue_limit,
            waiting_limit=waiting_limit
        )


        # --------------------------------------------------
        # 7. READ VIDEO INFORMATION
        # --------------------------------------------------

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError("Unable to open video.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        cap.release()

        if fps <= 0:
            fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )

        if not writer.isOpened():
            raise RuntimeError(
                "Unable to create output video."
            )


        # --------------------------------------------------
        # 8. START YOLO + BYTETRACK
        # --------------------------------------------------

       
        results = model.track(
            source=str(video_path),
            tracker=str(ROOT / "custom_bytetrack.yaml"),
            persist=True,
            classes=list(class_ids),
            conf=confidence,
            iou=IOU_THRESHOLD,     # Added: prevent suppressing overlapping people
            imgsz=IMG_SIZE,        # Changed from 512 -> 1280
            max_det=500,           # Added: ensure dense crowds aren't capped
            device=DEVICE,
            stream=True,
            verbose=False
        )

        frame_index = 0

        print("Video processing started.")

        for result in results:

            frame_index += 1

            current_time = frame_index / fps

            # Draw detection boxes and tracking IDs
            frame = result.plot(
                line_width=2,
                labels=True,
                boxes=True,
                conf=False
            )


            # --------------------------------------------------
            # 9. IDENTIFY OBJECTS INSIDE QUEUE REGION
            # --------------------------------------------------

            inside_ids = set()
            track_positions = {}

            boxes = result.boxes

            if (
                boxes is not None
                and boxes.id is not None
                and boxes.cls is not None
                and len(boxes) > 0
            ):

                track_ids = (
                    boxes.id.int().cpu().tolist()
                )

                detected_classes = (
                    boxes.cls.int().cpu().tolist()
                )

                coordinates = (
                    boxes.xyxy.cpu().numpy()
                )

                for track_id, class_id, box in zip(
                    track_ids,
                    detected_classes,
                    coordinates
                ):
                    if class_id not in class_ids:
                        continue

                    x1, y1, x2, y2 = map(int, box)

                    center_x = (x1 + x2) // 2
                    bottom_y = y2
                    if zone_manager.is_box_inside(
                        box,
                        queue_region
                    ):
                        point = (
                            center_x,
                            bottom_y
                        )

                        inside_ids.add(track_id)

                        track_positions[track_id] = point


            # --------------------------------------------------
            # 10. UPDATE WAITING TIME
            # --------------------------------------------------

            current_waiters, completed_now = (
                waiting_tracker.update(
                    inside_ids,
                    current_time
                )
            )


            # --------------------------------------------------
            # 11. UPDATE QUEUE METRICS
            # --------------------------------------------------

            queue_metrics.update(completed_now)

            metrics = queue_metrics.calculate(
                current_waiters,
                current_time
            )


            # --------------------------------------------------
            # 12. UPDATE LITTLE'S LAW
            # --------------------------------------------------

            staffing_metrics = little_law.update(
                current_waiters,
                completed_now,
                current_time,
                entered_now=waiting_tracker.entered_now,
                queue_trend=metrics["queue_trend"],
                queue_trend_per_minute=metrics[
                    "queue_trend_per_minute"
                ]
            )


            # --------------------------------------------------
            # 13. DRAW QUEUE REGION
            # --------------------------------------------------

            polygon = np.array(
                queue_region,
                dtype=np.int32
            )


            cv2.polylines(
                frame,
                [polygon],
                True,
                (0, 255, 255),
                3
            )

            cv2.putText(
                frame,
                f"{object_mode.upper()} QUEUE: {len(current_waiters)}",
                tuple(polygon[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )


            # --------------------------------------------------
            # 14. DRAW INDIVIDUAL WAITING TIMES
            # --------------------------------------------------

            for track_id, waiting_seconds in (
                current_waiters.items()
            ):

                if track_id not in track_positions:
                    continue

                x, y = track_positions[track_id]

                waiting_minutes = waiting_seconds / 60

                label = (
                    f"ID {track_id}: "
                    f"{waiting_minutes:.1f} min"
                )

                cv2.putText(
                    frame,
                    label,
                    (x - 40, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )


            # --------------------------------------------------
            # 15. SAVE ANNOTATED FRAME
            # --------------------------------------------------

            writer.write(frame)


            # --------------------------------------------------
            # 16. SEND DATA TO STREAMLIT
            # --------------------------------------------------

            dashboard_data = {

                "frame": frame,

                "frame_number": frame_index,

                "total_frames": total_frames,

                "video_time": current_time,

                "queue_count": len(current_waiters),
                "object_mode": object_mode,
                "current_waiters": dict(
                    current_waiters
                ),

                "queue_metrics": metrics,

                "staffing_metrics": staffing_metrics,

                "alerts": staffing_metrics["alerts"],

                "completed_now": completed_now,
                "entered_now": list(waiting_tracker.entered_now),

                "output_path": str(output_path)
            }

            # Yield one update per processed frame
            yield dashboard_data


    finally:

        # --------------------------------------------------
        # 17. RELEASE RESOURCES
        # --------------------------------------------------

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()

        print("Video processing finished.")
        print(f"Output saved at: {output_path}")
