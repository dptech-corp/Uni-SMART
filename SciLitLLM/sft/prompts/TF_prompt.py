# The following examples are from these papers:
# 10.3390/ma13214806, Treatment Sequence
# 10.1021/jm0105171, Molecule in Document

TF_example = [
    {
        "text": """
\section*{2. Materials and Methods}

Cast \(\mathrm{NiMnGa}\) samples, of \(\mathrm{Ni}_{50} \mathrm{Mn}_{30} \mathrm{Ga}_{20}\) nominal composition, were prepared by 5 arc melting cycles of the pure elements (electrolytic Ni \(99.97 \%\), electrolytic Mn \(99.5 \%\) and Ga \(99.99 \%\) ) in stoichiometric ratio, in a non-consumable electrode furnace (Leybold LK6/45) (Leybold, Cologne, Germany). As-cast ingot was ground to powder in a planetary ball mill (Fritsch Pulverisette 4) (Fritsch, Idar-Oberstein, Germany) and the powder size selected by means of sieves. Densified pellets were produced by die pressing alloy powders with different average size (lower than \(50 \mu \mathrm{m}\) or between 50 and \(100 \mu \mathrm{m}\) ) at \(0.75 \mathrm{GPa}\) at room temperature and sintered by a thermal treatment at \(925^{\circ} \mathrm{C}\) for 24, 72, and \(168 \mathrm{~h}\) in an Ar atmosphere, followed by slow cooling in the furnace. Sintered pellets had the following dimensions: approximately \(3 \mathrm{~mm}\) in height and \(13 \mathrm{~mm}\) of diameter. Table 1 provides a summary of the prepared sintered samples.

Table 1. Description of sintered samples.

\begin{tabular}{ccc}
\hline Sample & Size of Grain Powders & Sintering Time at \(\mathbf{9 2 5}{ }^{\circ} \mathrm{C}\) \\
\hline A & \(\leq 50 \mu \mathrm{m}\) & \(24 \mathrm{~h}\) \\
B & \(\leq 50 \mu \mathrm{m}\) & \(72 \mathrm{~h}\) \\
C & \(50 \mu \mathrm{m}<\operatorname{size} \leq 100 \mu \mathrm{m}\) & \(72 \mathrm{~h}\) \\
D & \(\leq 50 \mu \mathrm{m}\) & \(168 \mathrm{~h}\) \\
\hline
\end{tabular}


Question: 
"In the upper paper, Is the processing  heat treatment technique before the thermal treatment at 925 C called arc melting?""",

        "answer": "Yes"
    },

    {
        "text": """
Iastatins as well as cytochalasins and disulfiram. Lowaffinity, unspecific binding to tubulin is common, however, and many lipophilic derivatives of combinatorial library syntheses have been reported to interact with this target. \({ }^{3}\)

<smiles>C=CCC(CCC(C)=CC=CCCC=CC1CSC(C2CC2C)=N1)OC</smiles>

\(\operatorname{curacin} A(1)\)

<smiles>CCCC(C/C(C)=C/C=C/CC/C=C\C1CSC([C@@H]2C[C@H]2C)=N1)OC</smiles>

<smiles>C=CCCC1=NC(/C=C\CC/C=C/C=C(\C)CCC(CC=C)OC)CS1</smiles>

<smiles>C=CCC(CC/C(C)=C/C=C/CC/C=C\C1CSC2=NC(C3CC3C)C2C1)OC</smiles>

3. DIBALH, \(\mathrm{CH}_{2} \mathrm{Cl}_{2},-78^{\circ} \mathrm{C} ; 85 \%\)

<smiles>CC(C)(C)OCCC1CCC(CO)C1CO</smiles>

4. EtOH, PPTS; \(80 \%\)

<smiles>COc1cc(/C=N\OCCCCC(C)(C)CCC(O)c2cccs2)cc(OC)c1OC</smiles>

Question: Does \"C/C(=C\\C=C\\CC/C=C\\C1=COC(=N1)C(C)C)/CC[C@H](CC=C)OC\" appear in the document?""",

        "answer": "No"
    }
]
        

TF_template = [
    {
        "system": """
        Generate synthetic data for training models based on the T/F format.
        Ensure the data reflects realistic values for the type of materials and characteristics described, and include given keywords in the paragraph.
        The text should explicitly include a description of a T/F question.
        The output should be formatted in JSON with attributes "text" and "answer". The value for the two attributes should be two string. Do not print out anything other than the JSON output.

        Below are some examples based on this requirement.
    
        Examples: {examples}.

        """,
        "user": """
        I need synthetic training data for training a machine learning model.
        The data should be formatted in JSON, with each entry containing "text" and "answer" attributes.
        You should generate a paragraph that includes the keywords: \n\n {keywords} \n\n.
        The "text" part must contain enough information for the T/F question to be answered!

        Your answer should contain as much details as possible. You should only generate one JSON. 
        The value for the two attributes should be two string. Use {{ and }} to warp your output. 
        Remember to put a comma at the end of the first string. Never use a json block to wrap your output.
        Here is the format for your output:
        
        {{
            "text": "Your paragraph and question here, remember that you should have a T/F question at the end of the text.",
            "answer": "Your answer here, It should be either 'Yes' or 'No'"
        }}
        
        Now start your answer: 
        """,
    }
]


if __name__ == "__main__":
    print(TF_example[0]["system"])
    print(TF_template[0]["user"])