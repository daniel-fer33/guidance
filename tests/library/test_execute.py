import guidance


def test_execute():
    """ Test the basic behavior of `execute`.
    """
    llm = guidance.llms.Mock("Marley")
    program = guidance(
        """This is executed: {{execute template name="exec_out"}}. Repeated: {{gen 'repeated' save_prompt='saved_prompt_2'}}""",
        llm=llm)
    out = program(template="My name is {{name}} {{gen 'surname' save_prompt='saved_prompt_1'}}", name="Bob")
    assert str(out) == "This is executed: My name is Bob Marley. Repeated: Marley"

    variables = out.variables()
    assert variables['exec_out'] == 'My name is Bob Marley'
    assert guidance(variables['saved_prompt_1']).text == 'My name is Bob '
    assert guidance(variables['saved_prompt_2']).text == 'This is executed: My name is Bob Marley. Repeated: '


def test_execute_hidden():
    """ Test hidden behavior of `execute`.
    """
    llm = guidance.llms.Mock("Marley")
    program = guidance(
        """This is executed: {{execute template name="exec_out"}}. Repeated: {{gen 'repeated' save_prompt='saved_prompt_2'}}""",
        llm=llm)
    program = guidance(
        """This is executed: {{execute template name="exec_out" hidden=True}}. Repeated: {{gen 'repeated' save_prompt='saved_prompt_2'}}""",
        llm=llm)
    out = program(template="My name is {{name}} {{gen 'surname' save_prompt='saved_prompt_1'}}", name="Bob")
    assert str(out) == "This is executed: . Repeated: Marley"

    variables = out.variables()
    assert variables['exec_out'] == 'My name is Bob Marley'
    assert guidance(variables['saved_prompt_1']).text == 'My name is Bob '
    assert guidance(variables['saved_prompt_2']).text == 'This is executed: . Repeated: '