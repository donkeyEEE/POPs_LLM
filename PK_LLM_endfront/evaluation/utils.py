def wrap_text(text, width):
    """
    Wraps text to a specified width, breaking lines at word boundaries.

    Parameters:
    - text (str): The text to wrap.
    - width (int): The maximum width of a line.

    Returns:
    - str: The wrapped text.
    """
    if not text:
        return ""
    
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        # Check if adding the next word would exceed the width
        if sum(len(w) for w in current_line) + len(word) + len(current_line) - 1 < width:
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]

    # Add the last line
    lines.append(' '.join(current_line))

    return '\n'.join(lines)

def beautify_output(data, max_line_length=100):
    """
    Beautifies and prints the question-answer pairs from the given list of dictionaries,
    wrapping text to a specified width.

    Parameters:
    - data (list): A list of dictionaries, where each dictionary contains a 'question' and an 'answer' key.
    - max_line_length (int): The maximum length of a line.

    Returns:
    - None: This function prints the formatted question-answer pairs directly.
    """
    for idx, qa in enumerate(data, start=1):
        wrapped_question = wrap_text(qa['Question'], max_line_length)
        wrapped_answer = wrap_text(qa['Answer'], max_line_length)

        print(f"Q{idx}: {wrapped_question}\n\nA{idx}: {wrapped_answer}\n{'-'*80}\n\n")