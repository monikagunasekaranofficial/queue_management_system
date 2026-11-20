from collections import deque
from statistics import median


class QueueMetrics:

    def __init__(self, window_seconds=300, trend_tolerance_per_minute=0.1):
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")

        self.completed_records = []
        self.total_completed = 0
        self.window_seconds = window_seconds
        self.trend_tolerance_per_minute = trend_tolerance_per_minute
        self.queue_history = deque()
        self.rolling_waits = deque()

    def update(self, completed_now):
        """Register newly completed queue visits."""
        for record in sorted(
            completed_now,
            key=lambda item: item.get(
                "exit_time",
                item.get("completed_at", 0),
            ),
        ):
            self.completed_records.append(record)
            self.total_completed += 1
            self.rolling_waits.append(
                (
                    record.get(
                        "exit_time",
                        record.get("completed_at", 0),
                    ),
                    record["waiting_time"],
                )
            )

    def calculate(self, current_waiters, current_time):
        cutoff = current_time - self.window_seconds

        while self.rolling_waits and self.rolling_waits[0][0] < cutoff:
            self.rolling_waits.popleft()

        current_count = len(current_waiters)
        if (
            not self.queue_history
            or current_time - self.queue_history[-1][0] >= 1.0
        ):
            self.queue_history.append((current_time, current_count))

        while (
            len(self.queue_history) > 1
            and self.queue_history[1][0] < cutoff
        ):
            self.queue_history.popleft()

        completed_times = [
            record["waiting_time"]
            for record in self.completed_records
        ]
        rolling_wait_times = [
            wait for _, wait in self.rolling_waits
        ]
        active_times = list(current_waiters.values())

        completed_average_wait = self._average(completed_times)
        rolling_average_wait = self._average(rolling_wait_times)
        current_average_wait = self._average(active_times)
        longest_current_wait = max(active_times, default=0)

        current_time = max(0, current_time)
        elapsed_minutes = current_time / 60
        throughput = (
            self.total_completed / elapsed_minutes
            if elapsed_minutes > 0
            else 0
        )

        effective_window_minutes = (
            min(self.window_seconds, current_time) / 60
        )
        rolling_throughput = (
            len(rolling_wait_times) / effective_window_minutes
            if effective_window_minutes > 0
            else 0
        )

        trend_per_minute = self._queue_trend_rate()
        if trend_per_minute > self.trend_tolerance_per_minute:
            queue_trend = "growing"
        elif trend_per_minute < -self.trend_tolerance_per_minute:
            queue_trend = "shrinking"
        else:
            queue_trend = "stable"

        rolling_peak_queue = max(
            (count for _, count in self.queue_history),
            default=0,
        )
        median_completed_wait = (
            median(completed_times)
            if completed_times
            else 0
        )
        longest_completed_wait = max(completed_times, default=0)

        return {
            "total_completed": self.total_completed,
            "average_wait": completed_average_wait,
            "completed_average_wait": completed_average_wait,
            "longest_wait": longest_completed_wait,
            "longest_completed_wait": longest_completed_wait,
            "median_wait": median_completed_wait,
            "current_average_wait": current_average_wait,
            "longest_current_wait": longest_current_wait,
            "rolling_average_wait": rolling_average_wait,
            "throughput": throughput,
            "rolling_throughput": rolling_throughput,
            "peak_queue": rolling_peak_queue,
            "queue_trend": queue_trend,
            "queue_trend_per_minute": trend_per_minute,
            "window_seconds": self.window_seconds,
        }

    def _queue_trend_rate(self):
        if len(self.queue_history) < 2:
            return 0

        first_time = self.queue_history[0][0]
        points = [
            ((time - first_time) / 60, count)
            for time, count in self.queue_history
        ]
        mean_x = sum(x for x, _ in points) / len(points)
        mean_y = sum(y for _, y in points) / len(points)
        denominator = sum((x - mean_x) ** 2 for x, _ in points)
        if denominator == 0:
            return 0

        return sum(
            (x - mean_x) * (y - mean_y)
            for x, y in points
        ) / denominator

    @staticmethod
    def _average(values):
        return sum(values) / len(values) if values else 0
