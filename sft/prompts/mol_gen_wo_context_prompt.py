
mol_gen_wo_context_example = [
    {
        "text": """*CCCC* |$R1_p;;;;;R2_p$|, R1 = OMe, R2 = NH2""",
        "answer": """COCCCCN"""
    },
    {
        "text": "*C(*)CC(*)CC* |$A;;Pol_p;;;Q_e;;;M_p$|, A = H, Pol = NH2, Q = OH, M = [Li]",
        "answer": "NCCC(O)CC[Li]"
    },
    {
        "text": "*OC(*)(Cc1ccc(O*)cc1)C(=O)O |$R3;;;R2;;;;;;;R1;;;;;;$|, R1 = *CCn1c2ccccc2c2ccccc21, R2 = H, R3 = CC",
        "answer": "CCO[C@@H](CC1=CC=C(C=C1)OCCN2C3=CC=CC=C3C4=CC=CC=C42)C(=O)O"
    }
]


mol_gen_wo_context_template = [
    {
        "system": """
        Generate synthetic data for training models based on the provided moleculars.
        Ensure the data reflects realistic values for the type of materials and characteristics described, and include given keywords in the paragraph.
        The output should be formatted in JSON with attributes "text" and "answer". The value for the two attributes should be two string.
        The "text" should be a CXSMILES-type markush formula, and the "answer" should be the SMILES formula (removing Hs).

        Below are some examples based on this requirement. 

        Examples: {mol_gen_example}.

        """,
        "user": """
        I need synthetic training data for training a machine learning model that transform molecule from text correctly.
        The data should be formatted in JSON, with each entry containing "text" and "answer" attributes.
        You should generate a paragraph that includes the keywords: \n\n {keywords} \n\n.

        You should only generate one JSON. 
        The value for the two attributes should be two string. Use {{ and }} to warp your output. 
        Remember to put a comma at the end of the first string.
        Here is the format for your output:
        
        {{
            "text": "Your CXSMILES-type markush formula here",
            "answer": "Your SMILES formula here"
        }}
        
        Now start your answer: 
        """,
    }
]
