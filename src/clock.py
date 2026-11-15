"""IST clock helper (TZ database not reliable in this env)."""
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))
DEADLINE = datetime(2026, 11, 16, 10, 0, tzinfo=IST)  # Mon 16 Nov 10:00 IST


def now_ist() -> datetime:
    return datetime.now(timezone.utc).astimezone(IST)


def ist_str() -> str:
    return now_ist().strftime("%Y-%m-%d %H:%M IST")


def hours_to_deadline() -> float:
    return (DEADLINE - now_ist()).total_seconds() / 3600.0


if __name__ == "__main__":
    print(ist_str(), f"| {hours_to_deadline():.1f}h to deadline")
