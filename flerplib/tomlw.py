"""A small TOML writer for manifests: tables of scalars and lists, and arrays of tables. Reading is tomllib's job."""
import json


def value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (list, tuple)):
        one = "[" + ", ".join(value(x) for x in v) + "]"
        if len(one) <= 100:
            return one
        return "[\n" + "".join(f"    {value(x)},\n" for x in v) + "]"
    if isinstance(v, dict):      # an inline table, e.g. [build] source_options
        return "{ " + ", ".join(f"{json.dumps(k)} = {value(x)}" for k, x in v.items()) + " }"
    raise TypeError(f"no TOML form for {type(v).__name__}")


def table(d):
    width = max((len(k) for k in d), default=0)
    return "".join(f"{k.ljust(width)} = {value(v)}\n" for k, v in d.items())


def dump(doc):
    """doc: list of (name, dict) for a table ('' = top level) or ('[name]', [dict, ...]) for an array of tables."""
    out = []
    for name, body in doc:
        if name.startswith("["):
            for item in body:
                out.append(f"[{name}]\n" + table(item))
        elif name:
            out.append(f"[{name}]\n" + table(body))
        else:
            out.append(table(body))
    return "\n".join(out)
