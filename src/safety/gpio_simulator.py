
from datetime import datetime
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[2]

EVENT_DIR = ROOT / "output/demo/events"
GPIO_LOG = EVENT_DIR / "gpio_simulation.json"

EVENT_DIR.mkdir(parents=True, exist_ok=True)


def simulate_machine_lock(violations):
    """
    Software-only machinery lock simulation.

    This function does not access physical GPIO pins
    or control industrial machinery.
    """

    lock_requested = len(violations) > 0

    event = {
        "timestamp": datetime.now().isoformat(),
        "mode": "SIMULATION_ONLY",
        "lock_requested": lock_requested,
        "reason": (
            "Safety violation detected"
            if lock_requested
            else "No safety violation"
        ),
        "violation_count": len(violations)
    }

    existing_events = []

    if GPIO_LOG.exists():
        try:
            existing_events = json.loads(
                GPIO_LOG.read_text()
            )
        except (json.JSONDecodeError, OSError):
            existing_events = []

    existing_events.append(event)

    GPIO_LOG.write_text(
        json.dumps(existing_events, indent=2)
    )

    print("========== GPIO SIMULATION ==========")
    print("Mode:", event["mode"])
    print("Lock requested:", lock_requested)
    print("Reason:", event["reason"])

    return event


if __name__ == "__main__":

    sample_violations = [
        {
            "type": "missing_ppe",
            "equipment": "helmet",
            "severity": "high"
        }
    ]

    simulate_machine_lock(sample_violations)
