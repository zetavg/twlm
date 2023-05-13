from datetime import datetime, timezone


def get_human_timestamp() -> str:
    t = datetime.now(timezone.utc)
    t_str = t.strftime('%Y-%m-%d-%H-%M-%S')
    return t_str
