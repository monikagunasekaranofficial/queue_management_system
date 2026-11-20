
class WaitingTimeTracker:

    def __init__(self, exit_grace_seconds=1.0):
        # Tracking ID -> waiting information
        self.active_objects = {}

        # Records of objects that have left the queue
        self.completed_objects = []
        self.pending_visits = {}
        self.entered_now = []

        # Allow brief tracking gaps near the queue boundary
        self.exit_grace_seconds = exit_grace_seconds

    def update(self, inside_ids, current_time):
        completed_now = []
        self.entered_now = []
        inside_ids = set(inside_ids)

        # Complete expired visits before handling detections from this frame.
        # If an ID reappears after the grace period, it begins a new visit.
        for track_id, data in list(self.active_objects.items()):
            if current_time - data["last_seen"] > self.exit_grace_seconds:
                record = self._complete_visit(
                    track_id,
                    data,
                    current_time
                )
                completed_now.append(record)
                self.completed_objects.append(record)
                del self.active_objects[track_id]

       # Allow up to 2 missed frames before resetting a pending entry
        for track_id in list(self.pending_visits):
            if track_id not in inside_ids:
                self.pending_visits[track_id]["missed"] = (
                    self.pending_visits[track_id].get("missed", 0) + 1
                )
                if self.pending_visits[track_id]["missed"] > 2:
                    del self.pending_visits[track_id]

        for track_id in inside_ids:
            if track_id in self.active_objects:
                self.active_objects[track_id]["last_seen"] = current_time
                continue

            pending = self.pending_visits.get(track_id)
            if pending is None:
                pending = {
                    "first_seen": current_time,
                    "consecutive_seen": 0,
                }
                self.pending_visits[track_id] = pending

            pending["consecutive_seen"] += 1
            if pending["consecutive_seen"] >= 2:
                self.active_objects[track_id] = {
                    "entry_time": pending["first_seen"],
                    "last_seen": current_time,
                }
                self.entered_now.append(
                    {
                        "track_id": track_id,
                        "entry_time": pending["first_seen"],
                    }
                )
                del self.pending_visits[track_id]

        # Calculate current waiting times
        current_waiters = {}

        for track_id, data in self.active_objects.items():

            current_waiters[track_id] = (
                current_time - data["entry_time"]
            )

        return current_waiters, completed_now

    @staticmethod
    def _complete_visit(track_id, data, completed_at):
        waiting_time = data["last_seen"] - data["entry_time"]
        return {
            "track_id": track_id,
            "entry_time": data["entry_time"],
            "exit_time": data["last_seen"],
            "completed_at": completed_at,
            "waiting_time": max(0, waiting_time),
        }
