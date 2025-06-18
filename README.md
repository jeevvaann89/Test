def save_script_content(content: str) -> str:
    SCRIPTS_DIR = "Scripts"
    file_name = None
    language = "unknown"

    try:
        os.makedirs(SCRIPTS_DIR, exist_ok=True)
    except OSError as e:
        return f"Error creating directory '{SCRIPTS_DIR}': {e}"

    # Normalize content for easier keyword matching (e.g., remove excess whitespace)
    normalized_content = content.strip()


    if re.search(r'\b(public|private|protected)\s+(class|interface)\b', normalized_content) and \
       re.search(r'\bpublic\s+static\s+void\s+main\s*\(String\[\]\s+args\)\b', normalized_content) or \
       re.search(r'\bimport\s+java\.', normalized_content):
        language = "java"
        file_name = "script.java"

    elif re.search(r'\b(import|def|class)\b', normalized_content) and \
         re.search(r'\b(print|return)\(', normalized_content) or \
         re.search(r'\bfrom\s+\w+\s+import\b', normalized_content):
        language = "python"
        file_name = "script.py"

    elif re.search(r'\b(function|const|let|var)\b', normalized_content) or \
         re.search(r'\bconsole\.log\s*\(', normalized_content) or \
         re.search(r'\b(document|window)\.', normalized_content):
        language = "javascript"
        file_name = "script.js"

    if file_name:
        file_path = os.path.join(SCRIPTS_DIR, file_name)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Content successfully saved to '{file_path}' as {language} code."
        except IOError as e:
            return f"Error saving content to '{file_path}': {e}"
    else:
        return "Could not determine content language. Not saved. Please provide valid Java, Python, or JavaScript code."
