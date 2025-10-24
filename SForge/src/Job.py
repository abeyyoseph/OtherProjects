import itertools

# global counter for simple unique job IDs
_job_id_counter = itertools.count()

class Job:
    def __init__(self, estimated_duration, required_material, priority=0, split_potential=False):
        self.job_id = next(_job_id_counter)
        self.estimated_duration = estimated_duration
        self.required_material = required_material
        self.priority = priority
        self.split_potential = split_potential

        # dynamic fields
        self.time_remaining = estimated_duration
        self.progress = 0.0
        self.completed = False

        print(f"[Job {self.job_id}] Created | Duration: {estimated_duration} mins | Material: {required_material} | Priority: {priority} | Split: {split_potential}")

    def update_progress(self, delta):
        """Advance job progress by delta time units."""
        if not self.completed:
            self.time_remaining = max(0, self.time_remaining - delta)
            self.progress = 1.0 - (self.time_remaining / self.estimated_duration)
            print(f"[Job {self.job_id}] Progress updated | Δt: {delta} | Progress: {self.progress*100:.1f}% | Time remaining: {self.time_remaining:.1f} mins")

            if self.time_remaining <= 0:
                self.completed = True
                print(f"[Job {self.job_id}] Completed")

    def is_complete(self):
        """Return True if job has finished."""
        return self.completed

    def split(self, num_parts=2):
        """Split into smaller sub-jobs if allowed."""
        if not self.split_potential:
            print(f"[Job {self.job_id}] Split attempted but not allowed")
            return [self]

        sub_duration = self.estimated_duration / num_parts
        sub_material = self.required_material / num_parts
        print(f"[Job {self.job_id}] Split into {num_parts} sub-jobs | Each duration: {sub_duration:.1f} | Each material: {sub_material:.1f}")

        return [Job(sub_duration, sub_material, self.priority, False) for _ in range(num_parts)]
