if re.search(r'\b(public|private|protected)\s+(class|interface)\b', normalized_content) and \
       (re.search(r'\bpublic\s+static\s+void\s+main\s*\(String\[\]\s+args\)\b', normalized_content) or \
        re.search(r'\bvoid\b', normalized_content) or \
        re.search(r'\b(int|boolean|double|float|char)\b', normalized_content)):
        language = "java"
        file_name = "script.java"
    elif re.search(r'\bdef\b', normalized_content) or \
         re.search(r'\bfrom\s+\w+\s+import\b', normalized_content) or \
         re.search(r'\bimport\s+\w+\b', normalized_content):
        language = "python"
        file_name = "script.py"
    elif re.search(r'\bfunction\b', normalized_content) or \
         re.search(r'\bconsole\.log\s*\(', normalized_content) or \
         re.search(r'\b(document|window)\.', normalized_content) or \
         re.search(r'\blet|var|const\b', normalized_content):
        language = "javascript"
        file_name = "script.js"
