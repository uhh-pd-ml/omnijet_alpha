"""Tools to help with submitting jobs to the cluster."""
import re


def from_dict(dct):
    """Return a function that looks up keys in dct."""

    def lookup(match):
        key = match.group(1)
        return dct.get(key, f"<{key} not found>")

    return lookup


def convert_values_to_strings(dct):
    """Convert all values in dct to strings."""
    return {k: str(v) for k, v in dct.items()}


def replace_placeholders(file_in, file_out, subs):
    """Replace placeholders of the form @@<placeholder_name>@@ in file_in and write to file_out.

    Parameters
    ----------
    file_in : str
        Input file.
    file_out : str
        Output file.
    subs : dict
        Dictionary mapping placeholders to their replacements, i.e. `{"dummy": "foo"}
        will replace @@dummy@@ with foo.
    """
    with open(file_in) as f:
        text = f.read()
    with open(file_out, "w") as f:
        f.write(re.sub("@@(.*?)@@", from_dict(subs), text))
