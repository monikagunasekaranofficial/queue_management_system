
import cv2
import numpy as np


class QueueZoneManager:

    def __init__(self, max_width=1200, max_height=700):

        self.points = []

        self.selecting = True

        self.max_width = max_width
        self.max_height = max_height

        self.scale_x = 1.0
        self.scale_y = 1.0

        self.original_frame = None
        self.display_frame = None

        self.display_width = 0
        self.display_height = 0

        self.window_name = "Select Queue Region"

    # --------------------------------------------------
    # MOUSE CALLBACK
    # --------------------------------------------------

    def mouse_callback(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:

            # Convert displayed coordinates
            # into original video coordinates

            original_x = int(x * self.scale_x)
            original_y = int(y * self.scale_y)

            # Keep coordinates within frame boundaries

            original_x = max(
                0,
                min(
                    original_x,
                    self.original_frame.shape[1] - 1
                )
            )

            original_y = max(
                0,
                min(
                    original_y,
                    self.original_frame.shape[0] - 1
                )
            )

            self.points.append(
                (original_x, original_y)
            )

            print(
                f"Point selected: ({original_x}, {original_y})"
            )

    # --------------------------------------------------
    # RESIZE VIDEO TO FIT SCREEN
    # --------------------------------------------------

    def resize_frame(self, frame):

        height, width = frame.shape[:2]

        scale = min(
            self.max_width / width,
            self.max_height / height,
            1.0
        )

        self.display_width = int(width * scale)
        self.display_height = int(height * scale)

        self.scale_x = width / self.display_width
        self.scale_y = height / self.display_height

        resized = cv2.resize(
            frame,
            (
                self.display_width,
                self.display_height
            ),
            interpolation=cv2.INTER_AREA
        )

        return resized

    # --------------------------------------------------
    # DRAW SELECTED POINTS
    # --------------------------------------------------

    def draw_selection(self):

        display = self.display_frame.copy()

        display_points = []

        # Convert original points to display coordinates

        for x, y in self.points:

            display_x = int(x / self.scale_x)
            display_y = int(y / self.scale_y)

            display_points.append(
                (display_x, display_y)
            )

            cv2.circle(
                display,
                (display_x, display_y),
                6,
                (0, 255, 0),
                -1
            )

        # Draw polygon lines

        if len(display_points) > 1:

            cv2.polylines(
                display,
                [
                    np.array(
                        display_points,
                        dtype=np.int32
                    )
                ],
                False,
                (0, 255, 255),
                2
            )

        # Close polygon when at least 3 points exist

        if len(display_points) >= 3:

            cv2.polylines(
                display,
                [
                    np.array(
                        display_points,
                        dtype=np.int32
                    )
                ],
                True,
                (0, 255, 0),
                2
            )

        # Instructions

        instructions = [
            "LEFT CLICK: Select points",
            "ENTER: Confirm polygon",
            "R: Reset selection",
            "ESC: Cancel"
        ]

        for index, instruction in enumerate(instructions):

            cv2.putText(
                display,
                instruction,
                (15, 25 + index * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

        cv2.putText(
            display,
            f"Selected Points: {len(self.points)}",
            (15, self.display_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        return display

    # --------------------------------------------------
    # SELECT QUEUE REGION
    # --------------------------------------------------

    def select_region(self, video_path):

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():

            raise ValueError(
                "Unable to open video."
            )

        ret, frame = cap.read()

        cap.release()

        if not ret:

            raise ValueError(
                "Unable to read first video frame."
            )

        self.original_frame = frame.copy()

        # Resize frame to fit screen

        self.display_frame = self.resize_frame(frame)

        self.points = []

        self.selecting = True

        cv2.namedWindow(
            self.window_name,
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            self.window_name,
            self.display_width,
            self.display_height
        )

        cv2.setMouseCallback(
            self.window_name,
            self.mouse_callback
        )

        print("\n========== QUEUE REGION SELECTION ==========")

        print("Left click: Select polygon points")
        print("Enter: Confirm polygon")
        print("R: Reset polygon")
        print("ESC: Cancel selection")

        while self.selecting:

            display = self.draw_selection()

            cv2.imshow(
                self.window_name,
                display
            )

            key = cv2.waitKey(20) & 0xFF

            # Confirm polygon

            if key == 13:

                if len(self.points) >= 3:

                    self.selecting = False

                    print("\nQueue region selected successfully.")

                    break

                else:

                    print(
                        "Please select at least 3 points."
                    )

            # Reset polygon

            elif key == ord("r"):

                self.points.clear()

                print("Selection reset.")

            # Cancel

            elif key == 27:

                cv2.destroyWindow(
                    self.window_name
                )

                raise RuntimeError(
                    "Queue region selection cancelled."
                )

        cv2.destroyWindow(
            self.window_name
        )

        # Return ORIGINAL video coordinates

        return np.array(
            self.points,
            dtype=np.int32
        )

    # --------------------------------------------------
    # CHECK WHETHER POINT IS INSIDE POLYGON
    # --------------------------------------------------

    @staticmethod
    def is_inside(point, polygon):

        result = cv2.pointPolygonTest(
            polygon,
            point,
            False
        )

        return result >= 0

    @staticmethod
    
    def is_box_inside(box, polygon):
        """Check ground-facing points plus lower-body anchor for occluded people."""
        x1, y1, x2, y2 = map(float, box)
        width = x2 - x1
        height = y2 - y1
        
        sample_points = (
            ((x1 + x2) / 2, y2),                 # Bottom center (feet)
            (x1 + width * 0.25, y2),             # Bottom left-ish
            (x1 + width * 0.75, y2),             # Bottom right-ish
            ((x1 + x2) / 2, y1 + height * 0.85), # Lower-body fallback when feet are occluded
        )
        
        inside_votes = sum(
            QueueZoneManager.is_inside(point, polygon)
            for point in sample_points
        )
        return inside_votes >= 1
