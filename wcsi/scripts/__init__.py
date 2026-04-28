from ast import literal_eval
from typing import Any
import re

import docopt


def parse_docopt(
    docstring: str,
    argv: list[str] | str | None = None,
    default_help: bool = True,
    version: Any = None,
    options_first: bool = False,
) -> dict:
    """
    Parameters
    ----------
    docstring : str
        Description of your command-line interface.
    argv : list of str or str, optional
        Argument vector to be parsed. sys.argv[1:] is used if not provided.
        If str is passed, the string is split on whitespace.
    default_help : bool (default: True)
        Set to False to disable automatic help on -h or --help options.
    version : any object
        If passed, the object will be printed if --version is in `argv`.
    options_first : bool (default: False)
        Set to True to require options precede positional arguments,
        i.e. to forbid options and positional arguments intermix.

    Returns
    -------
    dict
        A dictionary, where keys are names of command-line elements with special
        characters removed such as e.g. "--verbose" -> "verbose" and "<path>" -> "path"
        and values are the parsed values of those elements parsed as python types where
        possible
    """
    # Load in the arguments
    arguments = docopt.docopt(docstring, argv, default_help, version, options_first)

    # Remove the help argument
    if default_help:
        del arguments["--help"]

    # Parse the remaining arguments
    parsed_arguments = {}
    for arg in arguments.keys():
        # Don't include arguments that have not been specified at the command line to
        # keep the default arguments defined by the function (docopt doesn't allow us
        # to specify default arguments unless it is part of the options list).
        if arguments[arg] is not None:
            # Arguments specified as all upper case (<ARG>)
            if arg.upper() == arg:
                newarg = arg.lower()
            # Arguments specified in angle brackets (<arg>)
            elif re.match(r"<(.*)>", arg):
                newarg = re.match(r"<(.*)>", arg).groups()[0]
            # Arguments specificied with dashes (-arg | --arg)
            elif re.match(r"-+(.*)", arg):
                newarg = re.match(r"-+(.*)", arg).groups()[0]
            # Arguments specified by their presence (True/False arguments)
            elif arguments[arg] is True or arguments[arg] is False:
                newarg = arg
            else:
                raise KeyError("Couldn't parse argument {}".format(arg))

            try:
                parsed_arguments[newarg] = literal_eval(arguments[arg])
            except (ValueError, SyntaxError):
                parsed_arguments[newarg] = arguments[arg]

    # Call the function and return
    return parsed_arguments
