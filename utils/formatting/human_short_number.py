def human_short_number(num):
    str_num = str(num)
    for unit in ['k', 'm', 'b']:
        num /= 1000.0
        if abs(num) < 1:
            break
        formatted_num = f"{num:g}{unit}"
        if len(formatted_num) <= len(str_num):
            str_num = formatted_num
    return str_num
