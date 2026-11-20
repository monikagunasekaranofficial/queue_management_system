import sys
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking.video_tracking import process_video


st.set_page_config(
    page_title="QueueVision",
    page_icon=":material/groups:",
    layout="wide",
)


def clear_session():
    for key in ("output_path", "history_df", "session_summary"):
        st.session_state.pop(key, None)


st.session_state.setdefault("output_path", None)
st.session_state.setdefault("history_df", None)
st.session_state.setdefault("session_summary", None)


def display_number(value, suffix=""):
    if value is None:
        return "—"
    return f"{value:.1f}{suffix}"


# Sidebar: setup and configuration
with st.sidebar:
    st.markdown("## QueueVision")
    st.caption("Gate and canteen queue monitoring")

    st.divider()
    st.subheader("1. Choose a video")

    selected_object_mode = st.segmented_control(
        "Count objects",
        options=["People", "Vehicles"],
        default="People",
        help="People mode counts people. Vehicles mode counts cars, motorcycles, buses, and trucks.",
    )
    object_mode = selected_object_mode.lower()

    uploaded_video = st.file_uploader(
        "Queue video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Choose a recorded MP4, AVI, MOV, or MKV file.",
    )

    st.subheader("2. Configure the zone")

    zone_name = st.text_input(
        "Zone name",
        value="Gate / Canteen",
        help="Use a name that operators can recognize, such as Gate 1 or Canteen A.",
    )

    queue_limit = st.number_input(
        "High-queue alert",
        min_value=1,
        max_value=100,
        value=8,
        step=1,
        help="Show an alert when this many monitored objects are inside the zone.",
    )

    waiting_limit = st.number_input(
        "Long-wait alert",
        min_value=30,
        max_value=3600,
        value=300,
        step=30,
        help="Show an alert when the average wait exceeds this duration.",
    )

    open_service_points = st.number_input(
        "Open service points",
        min_value=0,
        max_value=50,
        value=1,
        step=1,
        help="Number of counters or gates currently open. This is displayed for context; per-staff capacity is not measured.",
    )

    with st.expander("Advanced detection settings"):
        confidence = st.slider(
            "Detection confidence",
            min_value=0.05,
            max_value=0.90,
            value=0.10,
            step=0.05,
        )
        exit_grace = st.number_input(
            "Exit grace period",
            min_value=0.5,
            max_value=10.0,
            value=1.0,
            step=0.5,
            help="Seconds an ID may be missing before its visit is completed.",
        )
        window_seconds = st.number_input(
            "Analytics window",
            min_value=30,
            max_value=3600,
            value=300,
            step=30,
            help="Time window used for arrival and service-rate estimates.",
        )

    st.subheader("3. Start analysis")

    start_button = st.button(
        ":material/play_arrow: Start monitoring",
        type="primary",
        width="stretch",
        disabled=uploaded_video is None,
    )

    st.button(
        ":material/restart_alt: Clear results",
        width="stretch",
        on_click=clear_session,
    )

    st.divider()
    st.caption("System")
    st.caption("Model: YOLO26s")
    st.caption("Tracker: ByteTrack")
    st.caption(f"Object mode: {selected_object_mode}")


# Main header
st.title("Queue monitoring")
st.caption(
    "Follow queue size, waiting time, arrival pressure, and staffing needs."
)
status_placeholder = st.empty()

if uploaded_video is None:
    status_placeholder.info("Waiting for video")
elif not start_button and st.session_state.get("session_summary") is None:
    status_placeholder.warning("Ready to start")


# Primary live area
video_container = st.container(border=True)
with video_container:
    video_header_left, video_header_right = st.columns(
        [3, 1],
        vertical_alignment="center",
    )
    with video_header_left:
        st.subheader("Live queue view")
    with video_header_right:
        time_placeholder = st.empty()

    video_placeholder = st.empty()
    video_placeholder.info(
        "Upload a video, set the queue zone name, then select Start monitoring."
    )
    progress_placeholder = st.empty()


# KPI summary
st.subheader("Queue overview")
with st.grid("auto", min_column_width=160):
    queue_slot = st.empty()
    current_average_slot = st.empty()
    current_longest_slot = st.empty()
    completed_average_slot = st.empty()
    completed_slot = st.empty()

queue_slot.metric(f"Current {object_mode}", "—", border=True)
current_average_slot.metric("Live average wait", "—", border=True)
current_longest_slot.metric("Longest current wait", "—", border=True)
completed_average_slot.metric("Completed-visit average", "—", border=True)
completed_slot.metric("Completed visits", "—", border=True)


# Detailed analysis
overview_tab, trends_tab, objects_tab, alerts_tab, export_tab = st.tabs(
    [
        ":material/dashboard: Overview",
        ":material/show_chart: Trends",
        ":material/group: Waiting objects",
        ":material/notifications: Alerts",
        ":material/download: Export",
    ]
)

with overview_tab:
    with st.grid("auto", min_column_width=160):
        overview_arrival = st.empty()
        overview_service = st.empty()
        overview_throughput = st.empty()
        overview_estimated_queue = st.empty()
        overview_estimated_wait = st.empty()
        overview_trend = st.empty()

    staffing_placeholder = st.empty()
    staffing_placeholder.caption(
        "Arrival, departure, throughput, and queue estimates appear during analysis."
    )

with trends_tab:
    chart_grid = st.columns(2)
    with chart_grid[0]:
        st.markdown("**Queue count**")
        queue_chart = st.empty()
        queue_chart.info("The queue trend will appear during analysis.")
    with chart_grid[1]:
        st.markdown("**Live average wait**")
        waiting_chart = st.empty()
        waiting_chart.info("The waiting-time trend will appear during analysis.")

with objects_tab:
    st.subheader(f"Current {selected_object_mode.lower()} in the zone")
    waiter_table = st.empty()
    waiter_table.info("Waiting-object details will appear during analysis.")

with alerts_tab:
    alert_area = st.empty()
    alert_area.info("Queue and waiting-time alerts will appear during analysis.")
    staffing_alert_area = st.empty()

with export_tab:
    export_placeholder = st.empty()


if start_button:
    temp_video_path = None
    history = []

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(uploaded_video.name).suffix,
        ) as temp_file:
            temp_file.write(uploaded_video.getbuffer())
            temp_video_path = Path(temp_file.name)

        output_path = ROOT / "outputs" / "dashboard_output.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        st.session_state.pop("output_path", None)
        st.session_state.pop("history_df", None)
        st.session_state.pop("session_summary", None)

        status_placeholder.info("Initializing")
        frame_number = 0
        video_time = 0

        for data in process_video(
            video_path=temp_video_path,
            output_path=output_path,
            confidence=float(confidence),
            exit_grace_seconds=float(exit_grace),
            queue_limit=int(queue_limit),
            waiting_limit=int(waiting_limit),
            window_seconds=int(window_seconds),
            object_mode=object_mode,
        ):
            frame_rgb = cv2.cvtColor(
                data["frame"],
                cv2.COLOR_BGR2RGB,
            )
            video_placeholder.image(
                frame_rgb,
                channels="RGB",
                alt=f"Annotated live queue view for {zone_name}",
                width="stretch",
            )

            video_time = data["video_time"]
            time_placeholder.caption(
                f":material/schedule: "
                f"{int(video_time // 60):02d}:{int(video_time % 60):02d}"
            )

            frame_number = data["frame_number"]
            total_frames = data["total_frames"]
            progress = (
                min(frame_number / total_frames, 1.0)
                if total_frames > 0
                else 0
            )
            progress_placeholder.progress(
                progress,
                text=f"Processing {progress * 100:.1f}% complete",
            )
            status_placeholder.success("Processing")

            metrics = data["queue_metrics"]
            staffing = data["staffing_metrics"]
            waiters = data["current_waiters"]

            queue_slot.metric(
                f"Current {object_mode}",
                data["queue_count"],
                border=True,
                help=f"Confirmed {object_mode} inside the selected queue zone.",
            )
            current_average_slot.metric(
                "Live average wait",
                f"{metrics['current_average_wait']:.1f} sec",
                border=True,
                help="Average waiting time for currently active queue members.",
            )
            current_longest_slot.metric(
                "Longest current wait",
                f"{metrics['longest_current_wait']:.1f} sec",
                border=True,
            )
            completed_average_slot.metric(
                "Completed-visit average",
                (
                    f"{metrics['completed_average_wait']:.1f} sec"
                    if metrics["total_completed"]
                    else "—"
                ),
                border=True,
                help="Lifetime average wait for completed visits.",
            )
            completed_slot.metric(
                "Completed visits",
                metrics["total_completed"],
                border=True,
            )

            overview_arrival.metric(
                "Arrival rate",
                f"{staffing['arrival_rate']:.2f}/min",
                border=True,
            )
            overview_service.metric(
                "Departure rate",
                f"{staffing['departure_rate']:.2f}/min",
                border=True,
            )
            overview_throughput.metric(
                "Throughput",
                f"{metrics['rolling_throughput']:.2f}/min",
                border=True,
                help="Completed queue visits per minute in the selected window.",
            )
            overview_estimated_queue.metric(
                "Estimated queue",
                display_number(staffing["estimated_queue_length"]),
                border=True,
            )
            overview_estimated_wait.metric(
                "Estimated wait",
                display_number(staffing["estimated_current_wait"], " min"),
                border=True,
            )
            overview_trend.metric(
                "Queue trend",
                staffing["queue_trend"].title(),
                delta=f"{staffing['queue_trend_per_minute']:.2f}/min",
                delta_color="inverse",
                border=True,
            )

            staffing_placeholder.info(
                f"**{zone_name}** · Data sufficiency: "
                f"{staffing['data_sufficiency']} · "
                f"Open service points: {int(open_service_points)}.",
                icon=":material/lightbulb:",
            )

            if waiters:
                waiter_data = [
                    {
                        "Object ID": tracking_id,
                        "Waiting (sec)": round(seconds, 1),
                        "Waiting (min)": round(seconds / 60, 2),
                    }
                    for tracking_id, seconds in sorted(waiters.items())
                ]
                waiter_table.dataframe(
                    pd.DataFrame(waiter_data),
                    hide_index=True,
                    column_config={
                        "Object ID": st.column_config.NumberColumn(
                            "Object ID",
                            format="%d",
                        ),
                        "Waiting (sec)": st.column_config.NumberColumn(
                            "Waiting (sec)",
                            format="%.1f",
                        ),
                        "Waiting (min)": st.column_config.NumberColumn(
                            "Waiting (min)",
                            format="%.2f",
                        ),
                    },
                    alt=f"{selected_object_mode} currently waiting in {zone_name}",
                )
            else:
                waiter_table.success(
                    f"No {object_mode} are currently waiting in this zone.",
                    icon=":material/check_circle:",
                )

            alerts = data["alerts"]
            if alerts:
                alert_area.warning(
                    "\n\n".join(
                        f"**{zone_name}:** {alert}"
                        for alert in alerts
                    ),
                    icon=":material/warning:",
                )
            else:
                alert_area.success(
                    "No active queue alerts.",
                    icon=":material/check_circle:",
                )
            if staffing["demand_exceeds_service"]:
                staffing_alert_area.warning(
                    staffing["staffing_recommendation"],
                    title="Staffing advisory",
                    icon=":material/support_agent:",
                )
            else:
                staffing_alert_area.success(
                    "No additional staffing advisory from current flow.",
                    icon=":material/check_circle:",
                )

            active_wait = metrics["current_average_wait"]
            history.append(
                {
                    "Video Time (sec)": round(video_time, 2),
                    "Queue Count": data["queue_count"],
                    "Average Active Wait (sec)": round(active_wait, 2),
                    "Longest Current Wait (sec)": round(
                        metrics["longest_current_wait"],
                        2,
                    ),
                    "Completed Visit Average Wait (sec)": round(
                        metrics["completed_average_wait"],
                        2,
                    ),
                    "Arrival Rate (/min)": round(
                        staffing["arrival_rate"],
                        2,
                    ),
                    "Departure Rate (/min)": round(
                        staffing["departure_rate"],
                        2,
                    ),
                    "Throughput (/min)": round(
                        metrics["rolling_throughput"],
                        2,
                    ),
                    "Estimated Queue Length": round(
                        staffing["estimated_queue_length"] or 0,
                        2,
                    ) if staffing["estimated_queue_length"] is not None else None,
                    "Queue Trend": staffing["queue_trend"],
                    "Queue Trend (/min)": round(
                        staffing["queue_trend_per_minute"],
                        2,
                    ),
                    "Completed Visits": metrics["total_completed"],
                }
            )

            if frame_number % 10 == 0:
                history_df = pd.DataFrame(history)
                chart_data = history_df.set_index("Video Time (sec)")
                queue_chart.line_chart(
                    chart_data[["Queue Count"]],
                    height=300,
                    alt=f"Queue count trend for {zone_name}",
                )
                waiting_chart.line_chart(
                    chart_data[["Average Active Wait (sec)"]],
                    height=300,
                    alt=f"Live average waiting-time trend for {zone_name}",
                )

        if not history:
            raise RuntimeError("The video produced no processable frames.")

        st.session_state["output_path"] = str(output_path)
        st.session_state["history_df"] = pd.DataFrame(history)
        st.session_state["session_summary"] = {
            "zone": zone_name,
            "frames": frame_number,
            "video_time": video_time,
        }

        status_placeholder.success("Completed")
        progress_placeholder.progress(
            1.0,
            text="Video processing completed",
        )

        if history:
            history_df = pd.DataFrame(history)
            chart_data = history_df.set_index("Video Time (sec)")
            queue_chart.line_chart(
                chart_data[["Queue Count"]],
                height=300,
                alt=f"Final queue count trend for {zone_name}",
            )
            waiting_chart.line_chart(
                chart_data[["Average Active Wait (sec)"]],
                height=300,
                alt=f"Final live average waiting-time trend for {zone_name}",
            )

        st.success(
            f"Analysis completed for **{zone_name}**.",
            icon=":material/check_circle:",
        )

    except Exception as error:
        status_placeholder.error("Processing failed")
        st.error(
            f"Could not complete the analysis: {error}",
            icon=":material/error:",
        )

    finally:
        if temp_video_path is not None and temp_video_path.exists():
            temp_video_path.unlink()


# Export area
saved_output = st.session_state.get("output_path")
history_df = st.session_state.get("history_df")

with export_tab:
    export_placeholder.empty()

    if history_df is not None:
        export_csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            ":material/table_chart: Download metrics CSV",
            data=export_csv,
            file_name="queue_metrics.csv",
            mime="text/csv",
            type="secondary",
            width="stretch",
        )
    else:
        st.caption("Metrics will be available for download after processing.")

    if saved_output and Path(saved_output).exists():
        with open(saved_output, "rb") as video_file:
            st.download_button(
                ":material/movie: Download annotated video",
                data=video_file,
                file_name="queue_output.mp4",
                mime="video/mp4",
                type="primary",
                width="stretch",
            )
    else:
        st.caption("The annotated video will be available after processing.")


st.divider()
st.caption("QueueVision · YOLO26 · ByteTrack · Queue Analytics")
