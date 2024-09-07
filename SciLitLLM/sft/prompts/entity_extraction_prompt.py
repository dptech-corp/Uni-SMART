# The following examples are from these tasks, we take the first example:
# (not used) chemical_entities_recognition
# BM_drug_interactions_recognition
# BM_gene_disease_function
# 

entity_extraction_example = [
    {
        "text": """
Catecholamine-depleting drugs (eg, reserpine) may have an additive effect when given with beta-blocking agents. Patients treated with TENORMIN plus a catecholamine depletor should therefore be closely observed for evidence of hypotension and/or marked bradycardia which may produce vertigo, syncope, or postural hypotension. Calcium channel blockers may also have an additive effect when given with TENORMIN . Beta blockers may exacerbate the rebound hypertension which can follow the withdrawal of clonidine. If the two drugs are coadministered, the beta blocker should be withdrawn several days before the gradual withdrawal of clonidine. If replacing clonidine by beta-blocker therapy, the introduction of beta blockers should be delayed for several days after clonidine administration has stopped. Concomitant use of prostaglandin synthase inhibiting drugs, eg, indomethacin, may decrease the hypotensive effects of beta blockers. Information on concurrent usage of atenolol and aspirin is limited. Data from several studies, ie, TIMI-II, ISIS-2, currently do not suggest any clinical interaction between aspirin and beta blockers in the acute myocardial infarction setting. While taking beta blockers, patients with a history of anaphylactic reaction to a variety of allergens may have a more severe reaction on repeated challenge, either accidental, diagnostic or therapeutic. Such patients may be unresponsive to the usual doses of epinephrine used to treat the allergic reaction.
Identify all the drug-drug interactions:
""",

    "answer": "(reserpine, effect, beta-blocking agent), (beta blockers, effect, clonidine), (beta blocker, advise, clonidine), (beta blockers, advise, clonidine), (indomethacin, effect, beta blockers)"
    },

    {
        "text": """
Autosomal dominant limb-girdle muscular dystrophy associated with conduction defects (LGMD1B): a description of 8 new families with the LMNA gene mutations].\nLa dystrophie musculaire des ceintures autosomique dominante associ\u00e9e \u00e0 des troubles de la conduction cardiaque (LGMD1B). Description de 8 nouvelles familles avec mutations du g\u00e8ne LMNA.\nINTRODUCTION: Limb girdle muscular dystrophy type 1b (LGMD1B), due to LMNA gene mutations, is a relatively rare form of LGMD characterized by proximal muscle involvement associated with heart involvement comprising atrio-ventricular conduction blocks and dilated cardiomyopathy. Its clinical and genetic diagnosis is crucial for cardiac management and genetic counselling. Seven LMNA mutations have been previously reported to be responsible for LGMD1B.\nPATIENTS AND METHODS: We describe the neurological and cardiologic features of 14 patients belonging to 8 families in whom we identified 6 different LMNA mutations, 4 of them having never been reported. Results. Eleven patients had an LGMD1B phenotype with scapulohumeral and pelvic-femoral involvement. Thirteen patients had cardiac disease associating conduction defects (12 patients) or arrhythmias (9 patients). Seven patients needed cardiac device (pacemaker or implantable cardiac defibrillator) and two had heart transplantation.\nCONCLUSION: This study allowed us to specify the clinical characteristics of this entity and to outline the first phenotype/genotype relations resulting from these observations.
Please give me tripples that contain entities and semantic role labeling objects.
""",

    "answer": "(mutations, causeof, due to)"
    },

    {
        "text": """
[Autosomal dominant limb-girdle muscular dystrophy associated with conduction defects (LGMD1B): a description of 8 new families with the LMNA gene mutations].\nLa dystrophie musculaire des ceintures autosomique dominante associ\u00e9e \u00e0 des troubles de la conduction cardiaque (LGMD1B). Description de 8 nouvelles familles avec mutations du g\u00e8ne LMNA.\nINTRODUCTION: Limb girdle muscular dystrophy type 1b (LGMD1B), due to LMNA gene mutations, is a relatively rare form of LGMD characterized by proximal muscle involvement associated with heart involvement comprising atrio-ventricular conduction blocks and dilated cardiomyopathy. Its clinical and genetic diagnosis is crucial for cardiac management and genetic counselling. Seven LMNA mutations have been previously reported to be responsible for LGMD1B.\nPATIENTS AND METHODS: We describe the neurological and cardiologic features of 14 patients belonging to 8 families in whom we identified 6 different LMNA mutations, 4 of them having never been reported. Results. Eleven patients had an LGMD1B phenotype with scapulohumeral and pelvic-femoral involvement. Thirteen patients had cardiac disease associating conduction defects (12 patients) or arrhythmias (9 patients). Seven patients needed cardiac device (pacemaker or implantable cardiac defibrillator) and two had heart transplantation.\nCONCLUSION: This study allowed us to specify the clinical characteristics of this entity and to outline the first phenotype/genotype relations resulting from these observations.


In this Gene-Disease relation extraction task, you need to extract the (gene, function change, disease) triplet from the text. The second element in the triple means the regulation that the gene produces to the disease. Types of regulations are: LOF and GOF, which indicate loss or gain of function; REG, which indicates a general regulatory relationship; COM, which indicates that the functional change between genes and diseases is more complex, and it is difficult to determine whether the functional change is LOF or GOF.
Give your anwer below:
""",

    "answer": "(LMMA, REG, Limb girdle muscular dystrophy type 1b)"
    },
]


entity_extraction_template = [
    {
        "system": """
        Generate synthetic data for training models based on the entity extraction format.
        Ensure the data reflects realistic values for the type of materials and characteristics described, and include given keywords in the paragraph.
        The text should explicitly include a description of a entity extraction question.
        The output should be formatted in JSON with attributes "text" and "answer". The value for the two attributes should be two string. Do not print out anything other than the JSON output.

        Below are some examples based on this requirement.
    
        Examples: {examples}.

        """,
        "user": """
        I need synthetic training data for training a machine learning model.
        The data should be formatted in JSON, with each entry containing "text" and "answer" attributes.
        You should generate a paragraph that includes the keywords: \n\n {keywords} \n\n.
        The "text" part must contain enough information for the entity extraction question to be answered!

        Special notice for the number of entities in the answer: 
        The number of entities in your answer should be greater than 8, but less than 12.

        Your answer should contain as much details as possible. You should only generate one JSON. 
        The value for the two attributes should be two string. Use {{ and }} to warp your output. 
        Remember to put a comma at the end of the first string. Never use a json block to wrap your output.
        Here is the format for your output:
        
        {{
            "text": "Your paragraph and question here, remember that you should have a entity extraction question at the end of the text.",
            "answer": "Your answer here"
        }}
        
        Now start your answer: 
        """,
    }
]


if __name__ == "__main__":
    print(entity_extraction_template[0]["system"])
    print(entity_extraction_template[0]["user"])