from collections import deque


class LittleLawStaffing:

    def __init__(
        self,
        window_seconds=300,
        queue_limit=8,
        waiting_limit=300
    ):
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if queue_limit <= 0:
            raise ValueError("queue_limit must be positive")
        if waiting_limit <= 0:
            raise ValueError("waiting_limit must be positive")

        self.window_seconds = window_seconds
        self.queue_limit = queue_limit
        self.waiting_limit = waiting_limit
        self.arrivals = deque()
        self.completions = deque()
        self.completed_waits = deque()
        self.previous_active_ids = set()

    def update(
        self,
        current_waiters,
        completed_now,
        current_time,
        entered_now=None,
        queue_trend="stable",
        queue_trend_per_minute=0
    ):
        """Update rolling arrival, departure, wait, and staffing indicators."""
        current_time = max(0, current_time)
        current_ids = set(current_waiters)

        if entered_now is None:
            # Compatibility for callers without explicit visit-start events.
            arrivals_now = [
                (current_time, track_id)
                for track_id in current_ids - self.previous_active_ids
            ]
        else:
            arrivals_now = []
            for event in entered_now:
                if isinstance(event, dict):
                    track_id = event["track_id"]
                    event_time = event.get("entry_time", current_time)
                else:
                    track_id = event
                    event_time = current_time
                arrivals_now.append(
                    (min(current_time, max(0, event_time)), track_id)
                )

        arrivals_now.sort(key=lambda event: event[0])
        for event_time, track_id in arrivals_now:
            self.arrivals.append((event_time, track_id))

        departures_now = []
        for record in completed_now:
            departure_time = min(
                current_time,
                max(
                    0,
                    record.get(
                        "exit_time",
                        record.get("completed_at", current_time),
                    )
                )
            )
            departures_now.append(
                (departure_time, record)
            )

        departures_now.sort(key=lambda event: event[0])
        for departure_time, record in departures_now:
            self.completions.append(
                (departure_time, record["track_id"])
            )
            self.completed_waits.append(
                (departure_time, record["waiting_time"])
            )

        self.previous_active_ids = current_ids

        cutoff = current_time - self.window_seconds
        while self.arrivals and self.arrivals[0][0] < cutoff:
            self.arrivals.popleft()
        while self.completions and self.completions[0][0] < cutoff:
            self.completions.popleft()
        while self.completed_waits and self.completed_waits[0][0] < cutoff:
            self.completed_waits.popleft()

        effective_window_seconds = min(
            self.window_seconds,
            current_time
        )
        effective_window_minutes = effective_window_seconds / 60
        arrival_rate = (
            len(self.arrivals) / effective_window_minutes
            if effective_window_minutes > 0
            else 0
        )
        departure_rate = (
            len(self.completions) / effective_window_minutes
            if effective_window_minutes > 0
            else 0
        )

        completed_wait_times = [
            wait for _, wait in self.completed_waits
        ]
        average_completed_wait_seconds = (
            sum(completed_wait_times) / len(completed_wait_times)
            if completed_wait_times
            else None
        )
        current_wait_times = list(current_waiters.values())
        current_average_wait_seconds = (
            sum(current_wait_times) / len(current_wait_times)
            if current_wait_times
            else 0
        )

        actual_queue_length = len(current_ids)
        enough_arrival_data = (
            len(self.arrivals) >= 3
            and effective_window_seconds >= min(30, self.window_seconds)
        )
        enough_little_law_data = (
            enough_arrival_data
            and len(self.completed_waits) >= 3
            and average_completed_wait_seconds is not None
        )
        estimated_queue_length = (
            arrival_rate * average_completed_wait_seconds / 60
            if enough_little_law_data
            else None
        )
        estimated_current_wait = (
            actual_queue_length / arrival_rate
            if enough_arrival_data and arrival_rate > 0
            else (0 if actual_queue_length == 0 else None)
        )
        demand_exceeds_service = (
            enough_arrival_data
            and arrival_rate > departure_rate
        )

        if (
            len(self.arrivals) >= 10
            and len(self.completions) >= 5
            and effective_window_seconds >= min(60, self.window_seconds)
        ):
            data_sufficiency = "ADEQUATE"
        elif len(self.arrivals) >= 3 and effective_window_seconds >= 30:
            data_sufficiency = "MODERATE"
        else:
            data_sufficiency = "LIMITED"

        alerts = []
        if actual_queue_length >= self.queue_limit:
            alerts.append(
                "HIGH QUEUE: Queue capacity threshold reached."
            )

        rolling_average_wait = (
            average_completed_wait_seconds
            if average_completed_wait_seconds is not None
            else 0
        )
        if (
            current_average_wait_seconds >= self.waiting_limit
            or rolling_average_wait >= self.waiting_limit
        ):
            alerts.append(
                "LONG WAIT: Current or recent completed average wait "
                "exceeds the configured threshold."
            )

        if demand_exceeds_service:
            alerts.append(
                "FLOW ALERT: Arrivals exceed observed departures."
            )

        congestion_threshold = max(2, self.queue_limit / 2)
        if (
            queue_trend == "growing"
            and actual_queue_length >= congestion_threshold
        ):
            alerts.append(
                "CONGESTION ALERT: The queue is growing "
                f"({queue_trend_per_minute:.2f} objects/min)."
            )

        if demand_exceeds_service:
            staffing_recommendation = (
                "Demand is exceeding observed departures. Consider "
                "opening another service point or reallocating staff. "
                "No staff count is estimated because per-staff service "
                "capacity has not been measured."
            )
        else:
            staffing_recommendation = (
                "No additional staffing recommendation from the observed "
                "arrival/departure rates."
            )

        return {
            "arrival_rate": arrival_rate,
            "departure_rate": departure_rate,
            "service_rate": departure_rate,
            "average_wait_seconds": average_completed_wait_seconds or 0,
            "current_average_wait_seconds": current_average_wait_seconds,
            "estimated_queue_length": estimated_queue_length,
            "actual_queue_length": actual_queue_length,
            "estimated_current_wait": estimated_current_wait,
            "alerts": alerts,
            "data_sufficiency": data_sufficiency,
            "effective_window_seconds": effective_window_seconds,
            "queue_trend": queue_trend,
            "queue_trend_per_minute": queue_trend_per_minute,
            "demand_exceeds_service": demand_exceeds_service,
            "staffing_recommendation": staffing_recommendation,
            "arrival_count": len(self.arrivals),
            "departure_count": len(self.completions),
        }
