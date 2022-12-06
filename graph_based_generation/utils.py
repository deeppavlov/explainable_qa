import re


def sanitize(text):
    patterns = [(r"(began his career in )([\d]{3,4})( and)", "began his career in YEAR and"),
                (r"(started his career in )([\d]{3,4})( and)", "started his career in YEAR and")]
    for pattern_fnd, pattern_repl in patterns:
        fnd = re.findall(pattern_fnd, text)
        if fnd:
            pattern_repl = pattern_repl.replace("YEAR", fnd[0][1])
            text = text.replace(pattern_repl, "")
            text = re.sub(r'\s+', ' ', text)
            text = text.strip(".")
            text = f"{text} in {fnd[0][1]}."
    return text
