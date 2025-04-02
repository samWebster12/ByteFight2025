# schedules.py
def linear_schedule_fn(progress_remaining, initial_value):
    return progress_remaining * initial_value
