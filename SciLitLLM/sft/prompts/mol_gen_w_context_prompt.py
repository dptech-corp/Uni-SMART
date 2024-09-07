# 10.1021_jm0105171, tag2mol
mol_gen_w_context_example = [
    {
        "text": """""",
        "answer": """"""
    }
    
]


mol_gen_w_context_template = [
    {
        "system": """
        Generate synthetic data for training models based on the provided molecules.
        Ensure the data reflects realistic values for the type of materials and characteristics described, and include given keywords in the paragraph.
        The text should explicitly include description of at least three molecules, that is to be extracted by the downstream model.
        The output should be formatted in JSON with attributes "text" and "answer". The value for the two attributes should be two strings. Do not print out anything other than the JSON output.

        Below are some examples based on this requirement. Notice that the text must contain enough information for the molecules to be extracted!
    
        Examples: {mol_gen_example}.

        """,
        "user": """
        I need synthetic training data for training a machine learning model that extracts molecules from text correctly.
        The data should be formatted in JSON, with each entry containing "text" and "answer" attributes.
        You should generate a paragraph that includes all of these molecules: \n\n {keywords} \n\n.
        At the end of the text, you should add a question asking for the formula for a molecule. The text must contain enough information for the molecules to be extracted!

        Your answer should contain as many details as possible. You should only generate one JSON. 
        The value for the two attributes should be two strings. Use {{ and }} to warp your output. 
        Remember to put a comma at the end of the first string.
        In the text, remember that you should have a question asking for the formula of a molecule, at the end of the text. 
        Never mention formula of the molecule directly in the text. Instead, you can ask for the formula by labeling the molecule by order in the text (e.g. (1) <formula>, (2) <formula>..), or ask by its properties.
        For example, you can ask "What is the formula of the molecule that is used for treating cancer?", or
        "What is the formula of the molecule (1)?".
        Here is the format for your output:
        
        {{
            "text": "Your paragraph here.",
            "answer": "Your answer molecule formula here"
        }}
        
        Now start your answer: 
        """,
    }
]
