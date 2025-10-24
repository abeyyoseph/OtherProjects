from enum import Enum
from Job import Job

class PrinterState(Enum):
    IDLE = "idle"
    PRINTING = "printing"
    REPAIRING = "repairing"
    FAILED = "failed"

class Printer:
    PRINTER_MATERIAL_FULL = 1000

    def __init__(self, printer_id):
        self.id = printer_id
        self.state = PrinterState.IDLE
        self.current_job = None
        self.job_queue = []
        self.time_remaining_on_current_job = 0
        self.material_level = Printer.PRINTER_MATERIAL_FULL

        # Logging flags
        self.completed_job = False
        self.failed_job = False

        print(f"[Printer {self.id}] Initialized | State: {self.state.value} | Material: {self.material_level}")

    def assign_job(self, job: Job):
        if self.state == PrinterState.IDLE and self.has_sufficient_material(job):
            self.current_job = job
            self.time_remaining_on_current_job = job.estimated_duration
            self.material_level -= job.required_material
            self.state = PrinterState.PRINTING
            print(f"[Printer {self.id}] Assigned Job {job.job_id} | Duration: {job.estimated_duration} | Material left: {self.material_level}")
            return True
        else:
            print(f"[Printer {self.id}] Failed to assign Job {job.job_id} | State: {self.state.value} | Material: {self.material_level}")
        return False

    def update(self, time_delta):
        if self.state == PrinterState.PRINTING and self.current_job:
            print(f"[Printer {self.id}] Updating Job {self.current_job.job_id} | Δt: {time_delta}")
            self.current_job.update_progress(time_delta)
            self.time_remaining_on_current_job = self.current_job.time_remaining

            if self.current_job.is_complete():
                self.complete_job()
        elif self.state in (PrinterState.REPAIRING, PrinterState.FAILED):
            print(f"[Printer {self.id}] Currently {self.state.value}, skipping update.")

    def complete_job(self):
        if self.current_job:
            print(f"[Printer {self.id}] Completed Job {self.current_job.job_id}")
            self.current_job.completed = True
            self.completed_job = True
            self.current_job = None
        self.state = PrinterState.IDLE

    def fail(self):
        print(f"[Printer {self.id}] FAILURE | Current Job: {self.current_job.job_id if self.current_job else 'None'}")
        self.state = PrinterState.FAILED
        self.failed_job = True

    def repair(self):
        print(f"[Printer {self.id}] Repair initiated")
        self.state = PrinterState.REPAIRING
        self.failed_job = False

    def refill_material(self, amount=PRINTER_MATERIAL_FULL):
        self.material_level = min(PRINTER_MATERIAL_FULL, self.material_level + amount)
        print(f"[Printer {self.id}] Material refilled | Current: {self.material_level}")

    def has_sufficient_material(self, job):
        return self.material_level >= job.required_material

    def reset(self):
        print(f"[Printer {self.id}] Resetting state")
        self.state = PrinterState.IDLE
        self.current_job = None
        self.job_queue.clear()
        self.time_remaining_on_current_job = 0
        self.material_level = Printer.PRINTER_MATERIAL_FULL
        self.completed_job = False
        self.failed_job = False
