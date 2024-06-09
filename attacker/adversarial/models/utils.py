def extract_json(s: str) -> dict:
    """
    Extract json from input string.

    :param s: content include json information.
    :type s: str

    :return: json information
    """
    start_pos = s.find("{")
    end_pos = s.find("}") + 1

    json_str = s[start_pos:end_pos]
    json_str = json_str.replace("\n", "")

    parsed = eval(json_str)
    return parsed
