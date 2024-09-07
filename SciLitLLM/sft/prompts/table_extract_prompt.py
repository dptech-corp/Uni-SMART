# The following examples are from these papers:
# 10.1016_j.apsusc.2011.06.120, Composition Extraction
# (Not used) 10.1016_0960-894X__94__00470-Z, Affinity Extraction
# (Not used) 10.1002_adfm.202008332_part8, OLED Property Extraction
# 10.1063_1.2899996.csv, Polymer Property Extraction

table_extract_example = [
    {
        "text": """\section*{1. Introduction}

            MCrAlY ( \(\mathrm{M}=\mathrm{Ni}\) and/or \(\mathrm{Co}\) ) coatings are widely used in gas turbines as a bond coat of thermal barrier coatings(TBCs) to protect the superalloy substrate from oxidation and provide adhesion between the superalloy substrate and ceramic top coat [1,2]. The behavior of alloy bond coat has crucial impact on the overall performance of TBCs [3,4]. As is well known, the growth and adherence of thermally grown oxide (TGO) on the MCrAlY coat during oxidation is a major factor determining the lifetime of TBCs. It is generally accepted that the formation of a continuous and homogeneous TGO protects the coating from further oxidation and avoids the formation of mixed oxide protrusions [5]. The reactive element \(\mathrm{Y}\) is of premier importance to the adherence of the TGO. The doping of Hf could further enhance the adherence of the TGO and then result in the improved oxidation performance of the MCrAlY coating [6].

            MCrAlY coatings can be obtained by electron beam physical vapor deposition (EB-PVD) [7], plasma spray (PS) [8], high velocity oxy-fuel thermal spraying (HVOF) [9] magnetron sputtering (MS) [10], electrophoretic deposition (EPD) [11] and many other

            technologies. Some previous studies about MCrAlY bond coat have been focused on the effects of compositions [12] and preparation technologies [13] on oxidation and corrosion behavior of the coatings, which are associated with the formation and evolution of TGO. However, few studies have been concerned with the effect of substrate characteristics, such as substrate roughness, on performance of MCrAlY coatings. Substrate roughness is a fundamental factor that cannot be ignored for the characteristics of coatings and films. Chang et al. [14] confirmed that the magnetic properties of \(\mathrm{Co} / \mathrm{Pt}\) multilayer thin films were very sensitive to substrate roughness. Lee et al. [15] found that a smooth substrate surface not only provided better adhesion as well as lower friction coefficient of the \(\mathrm{CrN}\) hard coating but also improved its corrosion resistance. Some other studies concerning the effects of substrate roughness on the characteristics of coatings and films have also been carried out.

            In the present paper, the surface roughness, real surface area, adhesion and oxidation behavior of magnetron-sputtered NiCoCrAlY coatings with different superalloy substrate roughness have been studied. Atomic Force Microscopy (AFM) and scratch adhesion test techniques were used to characterize the surface morphology and adhesion strength of the coatings, respectively. The oxidation behaviors of the coatings with different substrate roughness were examined by cyclic oxidation testing. The mechanisms of superal-

            Table 1

            Elemental composition of the superalloy substrate (GH3128).

            \begin{tabular}{llllllllllll}
            \hline Elements & \(\mathrm{Cr}\) & \(\mathrm{W}\) & \(\mathrm{Mo}\) & \(\mathrm{Al}\) & \(\mathrm{Ti}\) & \(\mathrm{Fe}\) & \(\mathrm{C}\) & \(\mathrm{B}\) & \(\mathrm{Ce}\) & \(\mathrm{Zr}\) \\
            \hline Content \((\mathrm{wt} \%)\) & 20.55 & 8.70 & 8.26 & 0.80 & 0.40 & 0.60 & 0.04 & 0.005 & 0.05 & 0.08 & \(\mathrm{Ni}\) \\
            \hline
            \end{tabular}

            loy substrate roughness on adhesion and oxidation behavior of the NiCoCrAlY coatings have been discussed.
        """,
        "answer": """
            AlloyName,Composition,Composition,Composition,Composition,Composition,Composition,Composition,Composition,Composition,Composition
            ,Al,W,Ni,Ti,Cr,B,Mo,Fe,Zr,C
            GH3128,0.80 wt.%,8.70 wt.%,Bal.,0.40 wt.%,20.55 wt.%,0.005 wt.%,8.26 wt.%,0.60 wt.%,0.08 wt.%,0.04 wt.%
        """
    },
    {
        "text": """

more than an order of magnitude less than the typical reported out-of-plane mobility of PCBM. We varied the polymer:fullerene blend ratio, the choice of fullerene, and annealing conditions. We found a maximum efficiency of \(2.3 \%\) for a 1:4 blend ratio having a short-circuit current density \(\left(J_{\mathrm{sc}}\right)\) of \(9.37 \mathrm{~mA} / \mathrm{cm}^{2}\), an open circuit voltage \(\left(V_{\mathrm{oc}}\right)\) of \(0.525 \mathrm{~V}\), and a fill factor (FF) of 0.48 . These initial results show the potential of pBTTT as a light absorbing, hole transporting material for use in high efficiency polymer:PCBM bulk heterojunction solar cells.

The energy levels and molecular structures of pBTTT, \(\mathrm{PC}_{[61]} \mathrm{BM}\), and \(\mathrm{PC}_{[71]} \mathrm{BM}\) are shown in Fig. 1. As can be seen from the energy levels, there is an adequate energy offset for electron transfer from pBTTT to the fullerenes. Absorption and PL spectra of both pure pBTTT and a 1:4 pBTTT: \(\mathrm{PC}_{[71]} \mathrm{BM}\) blend are shown in Fig. 1(b). From the absorption, we see that a film with a \(115 \mathrm{~nm}\) thick active layer and no reflecting electrode absorbs about \(55 \%\) of the light at the maximum. The vibronic peaks in the blend film absorption suggest that PCBM enhances the crystallinity of pBTTT. This unusual result is being explored using synchrotron radiation and will be discussed further in a subsequent publication. The absence of PL from the blend suggests that virtually all of the excitons are dissociated by electron transfer, as expected based on the energy offsets.

An increase in the cell thickness beyond \(115 \mathrm{~nm}\) yielded a lower \(J_{\mathrm{sc}}\) and FF. As mentioned, initial measurements indicate a sublinear dependence of \(J_{\mathrm{sc}}\) on light intensity. Furthermore, decreasing the light intensity, and, thus, the photocur- rent, improved the FF of all devices. For example, decreasing the light intensity from 1 sun to 0.1 sun improved the FF from \(0.42-0.46\) to \(0.50-0.54\) for the \(1: 4\) pBTTT: \(\mathrm{PC}_{[71]} \mathrm{BM}\) devices. A sublinear dependence of \(J_{\mathrm{sc}}\) on light intensity and decreasing FF with increasing light intensity are indicative of SCLP or bimolecular recombination. \({ }^{18}\) SCLP at this thickness is consistent with the results of Mihailetchi et al. \({ }^{7}\) who found \(100 \mathrm{~nm}\) to be the maximum thickness for cells with mobility near \(10^{-4} \mathrm{~cm}^{2} \mathrm{~V}^{-1} \mathrm{~s}^{-1}\) to be free from a space charge limited regime.

Bulk heterojunction solar cells with PCEs of \(2.3 \%\) were fabricated with pBTTT. The high mobility needed to make devices with active layers thicker than \(115 \mathrm{~nm}\) was not achieved due to relatively low mobility in the direction normal to the substrate. When the hole and electron mobilities are matched in the vertical direction, it will be possible to make thick cells that absorb higher fractions of incident radiation and extract carriers in regimes not limited by SCLP. If the absorption and quantum efficiency were \(90 \%\) between 400 and \(650 \mathrm{~nm}\), the short-circuit current density would approach \(14 \mathrm{~mA} / \mathrm{cm}^{2}\). Because of this and the high hole mobility previously reported for pBTTT in the planar direction, \({ }^{10}\) we are optimistic that the pBTTT:PCBM combination is a promising photovoltaic system and that higher efficiencies may be achieved with more optimized processing and a better understanding of what limits the hole mobility, and consequently, efficiency.""",
        
        "answer": """
Nickname,PCE_max(%),PCE_ave(%),Voc (V),Jsc (mA cm^2),FF
PBTTT,2.3,NaN,0.525,9.370,0.370
            """
    },
]




table_extract_template = [
    {
        "system": """
        Generate synthetic data for training models based on the provided table format.
        Ensure the data reflects realistic values for the type of materials and characteristics described, and include given keywords in the paragraph.
        The text should explicitly include a description of a table, that is to be extracted by the downstream model.
        The output should be formatted in JSON with attributes "text" and "answer". The value for the two attributes should be two string. Do not print out anything other than the JSON output.

        Below are some examples based on this requirement. Notice that the text must contain enough information for the table to be extracted!
    
        Examples: {examples}.

        """,
        "user": """
        I need synthetic training data for training a machine learning model that extract tables from text correctly.
        The data should be formatted in JSON, with each entry containing "text" and "answer" attributes.
        You should generate a paragraph that includes the keywords: \n\n {keywords} \n\n.
        The "text" part must contain enough information for the table to be extracted!
        In "text" part, You must you include a table description in latex format.

        Special notice for the table content: \n
        You should generate a table that has complicated numbers and characters, include non-standard characters, and have a variety of values.
        Make sure the value you generated do not follow simple patterns, for example, never include deplicate values or values with constant interval in columns.

        Your answer should contain as much details as possible. You should only generate one JSON. 
        The value for the two attributes should be two string. Use {{ and }} to warp your output. 
        Pay attention to the escape characters in the latex format.
        Remember to put a comma at the end of the first string. Never use a json block to wrap your output.
        Here is the format for your output:
        
        {{
            "text": "Your paragraph here, remember to include a table in latex format",
            "answer": "Your answer table here"
        }}
        
        Now start your answer: 
        """,
    }
]


if __name__ == "__main__":
    print(table_extract_template[0]["system"])
    print(table_extract_template[0]["user"])