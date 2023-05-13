def truncate_string_by_lines(s, max_lines=8):
    lines = s.splitlines()

    if len(lines) > max_lines:
        truncated_lines = lines[:max_lines]
        truncated_string = '\n'.join(truncated_lines) + "\n..."
    else:
        truncated_string = s

    return truncated_string
