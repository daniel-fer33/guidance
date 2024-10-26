from .._utils import ContentCapture
from .._grammar import grammar


async def execute(string, name=None, hidden=False, _parser_context=None):
    '''
    Execute a dynamically generated string as a guidance program.

    This function allows for the execution of a string as a guidance program segment without preserving the previous context.
    Unlike the `parse` method, `execute` operates independently of the previous execution context, making it suitable for
    dynamically generated guidance programs or program fragments.

    Parameters
    ----------
    string : str
        The program code to parse and execute.
    name : str, optional
        The name of the variable to store the generated content, if applicable. If `None`, the content is not stored in a named variable.
    hidden : bool, optional
        If `True`, the generated content is hidden; otherwise, it is visible. Defaults to `False`.
    _parser_context : dict, optional
        The parser context for execution, containing the `parser` instance and `variable_stack` where
        context-specific variables are managed.

    Notes
    -----
    This method temporarily hides the current `@raw_prefix` in the `variable_stack`, executes the given `string`
    by parsing and visiting its structure, and then captures its content. If `name` is specified, the content
    is stored in the `variable_stack` under that name. Finally, the method restores the original `@raw_prefix`
    with any updates based on `hidden` and `new_raw_prefix` status.

    Raises
    ------
    ParsingError
        Raised if the `string` cannot be parsed by the provided grammar.
    '''

    parser = _parser_context['parser']
    variable_stack = _parser_context['variable_stack']
    prev_raw_prefix = variable_stack["@raw_prefix"]

    # Hide previous prefix
    variable_stack["@raw_prefix"] = ""

    # capture the content of the block
    with ContentCapture(variable_stack, hidden) as new_content:

        # parse and visit the given string
        subtree = grammar.parse_string(string)
        new_content += await parser.visit(subtree, variable_stack)

        # save the content in a variable if needed
        if name is not None:
            variable_stack[name] = str(new_content)

    # Recover state
    new_raw_prefix = variable_stack["@raw_prefix"]
    variable_stack["@raw_prefix"] = prev_raw_prefix
    if not hidden:
        variable_stack["@raw_prefix"] += new_raw_prefix
