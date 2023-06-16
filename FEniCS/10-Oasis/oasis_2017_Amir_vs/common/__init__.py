from .io import *
from .utilities import *
import sys
import json
import six


def convert(input):
    if isinstance(input, dict):
        return {convert(key): convert(value) for key, value in input.iter()}
    elif isinstance(input, list):
        return [convert(element) for element in input]
    elif isinstance(input, six.text_type):
        return input.encode('utf-8')
    else:
        return input

# Parse command-line keyword arguments
def parse_command_line():
    commandline_kwargs = {}
    for s in sys.argv[1:]:
        if s.count('=') == 1:
            key, value = s.split('=', 1)
        else:
            raise TypeError((s + " Only kwargs separated with '=' sign " +
                            "allowed. See NSdefault_hooks for a range of " +
                            "parameters. Your problem file should contain " +
                            "problem specific parameters."))
        try:
            value = json.loads(value)

        except ValueError:
            if value in ("True", "False"):  # json understands true/false, but not True/False
                value = eval(value)
            elif "True" in value or "False" in value:
                value = eval(value)
        if isinstance(value, dict):
            value = convert(value)

        commandline_kwargs[key] = value
    return commandline_kwargs

# Note to self. To change a dictionary variable through commandline do, e.g.,
# run NSfracStep velocity_update_solver='{"method":"gradient_matrix"}'
