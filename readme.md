# AI Queue Management & Staffing Optimization System

A real-time computer vision queue management system built with **YOLO**, **ByteTrack**, **OpenCV**, and **Streamlit**. This application detects and tracks people (or custom object classes) within user-defined polygon zones, calculates individual waiting times, forecasts congestion trends, and provides dynamic staffing recommendations using **Little's Law**.

---

## Features

- **Interactive Polygon Zone Selection:** Draw custom queue boundaries directly on the first frame of your video feed using an OpenCV GUI with automatic resolution scaling.
- **Multi-Object Detection & Tracking:** Powered by Ultralytics YOLO and ByteTrack (`custom_bytetrack.yaml`) for persistent ID assignment across crowded, high-occlusion environments.
- **Occlusion-Resilient Zone Counting:** Evaluates ground-facing anchor points and lower-body fallbacks with debounce filtering to accurately count people even when legs are partially blocked.
- **Individual Waiting Time Tracking:** Measures real-time dwell times per tracked ID with configurable exit grace periods to prevent timer resets from brief detection drops.
- **Queue Analytics & Little's Law Staffing:** Continuously computes arrival rates ($\lambda$), average wait times ($W$), queue length ($L$), queue growth trends, and optimal server/staffing requirements.
- **Real-Time Alerts & Streamlit Dashboard:** Streams annotated video frames, live metrics, and congestion alerts directly to an interactive web dashboard (`dashboard/app.py`) while saving processed `.mp4` outputs.

---

## System Architecture

```text
┌───────────────────┐     ┌──────────────────────────┐     ┌─────────────────────────┐
│  Input Video Feed │ ──▶ │ YOLO26s + ByteTrack      │ ──▶ │ QueueZoneManager        │
│  (videos/*.mp4)   │     │ (High-Res 1280px + IOU)  │     │ (4-Point Ground Filter) │
└───────────────────┘     └──────────────────────────┘     └────────────┬────────────┘
                                                                        │
┌───────────────────┐     ┌──────────────────────────┐     ┌────────────▼────────────┐
│ Streamlit UI      │ ◀── │ QueueMetrics &           │ ◀── │ WaitingTimeTracker      │
│ (dashboard/app.py)│     │ LittleLawStaffing        │     │ (Debounce + Grace Time) │
└───────────────────┘     └──────────────────────────┘     └─────────────────────────┘

## project structure

queue_management/
├── .streamlit/
│   └── config.toml             # Streamlit UI theme and server settings
├── analytics/
│   ├── little_law.py           # Little's Law staffing calculator & congestion alert triggers
│   └── queue_metrics.py        # Arrival/departure rates, wait averages, and trend calculations
├── dashboard/
│   └── app.py                  # Streamlit dashboard entrypoint
├── outputs/                    # Saved annotated output videos (ignored in Git)
├── queue_logic/
│   ├── object_mode.py          # Class ID mapping for people/vehicle/custom tracking modes
│   ├── waiting_time.py         # Per-ID dwell timer with grace periods & debounce logic
│   └── zone_manager.py         # Interactive OpenCV polygon selector & point-in-polygon checks
├── tracking/
│   ├── runs/                   # Ultralytics tracking logs (ignored in Git)
│   └── video_tracking.py       # Main YOLO + ByteTrack video processing & analytics pipeline
├── videos/                     # Input sample videos (ignored in Git)[cite: 3]
├── .gitignore                  # Excludes caches, large videos, and model weights[cite: 3]
├── custom_bytetrack.yaml       # Tuned ByteTrack thresholds for dense crowds[cite: 3]
├── readme.md                   # Project documentation[cite: 3]
└── requirements.txt            # Python dependencies[cite: 3]