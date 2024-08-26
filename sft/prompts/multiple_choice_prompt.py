# The following examples are from these papers:
# 10.3390/ma12182892, Alloy Chart QA
# 10.48550_arXiv.2310.00926, Biology Chart QA
# 10.1002/jps.20046, Drug Chart QA
# (Too long, not used) 10.1063_1.2899996, Polymer Composition QA
# (Too long, not used) 10.1021/acsami.0c21535, Reaction Mechanism QA

multiple_choice_example = [
    {
        "text": """
Figure 9. Mean and maximum pore area of titanium specimens.

\title{
3.2. Microstructure
}

\subsection*{3.2.1. Magnesium}

In lower magnifications of \(200 \times\) (Figure 10), it is possible to verify that not only the general morphology of the solidified melting pools varies between the different batches, but also that all samples present a number of heterogeneities along its microstructure. While the cross-sections associated to batch \(\mathrm{A}_{\mathrm{Mg}}\) ( \(\mathrm{a}\) and d) show many small, single solidified melt pools, batch \(\mathrm{B}_{\mathrm{Mg}}\) (b and e) and \(\mathrm{C}_{\mathrm{Mg}}\) (c and f) present bigger and deeper melt pools, a consequence of lower scan speeds and, therefore, higher energy inputs. Batch \(\mathrm{C}_{\mathrm{Mg}}\), in turn, presents a melt pool superimposition without an overlapping zone, which increases the general energy input and is correlated to the double laser beam exposure. The aforementioned melt pool depths have been measured to be approximately 1, 5-6, and 3-4 times the layer size for batches \(\mathrm{A}_{\mathrm{Mg}}, \mathrm{B}_{\mathrm{Mg}}\), and \(\mathrm{C}_{\mathrm{Mg}}\), respectively. The center of the cross-section of the samples belonging to batch \(\mathrm{A}_{\mathrm{Mg}}\) shows small single melt pools compared to \(\mathrm{B}_{\mathrm{Mg}}\) and \(\mathrm{C}_{\mathrm{Mg}}\). In contrast, the surface of the samples belonging to batch \(\mathrm{A}_{\mathrm{Mg}}\) shows greater melt pools. This is correlated with the decreased thermal conductivity of the surrounding powder bed, resulting in an accumulation of heat, whereas the higher thermal conductivity in the bulk material leads to smaller melt pools.
![](http://deeplinker-cai.oss-cn-zhangjiakou.aliyuncs.com/b9a85942-51c7-4eb1-9cd7-e0c7a843da2e.jpg?OSSAccessKeyId=LTAI5t65Z7BkkgW2MSCZMUML&Expires=1722670714&Signature=xt7lOEqakJgHkzBECm%2BWu9cQS38%3D)


Question: In FIG. 9, what is the closest maximum pore area (with unit um^2) of batches A_Ti?\n\na) 54000\nb) 45000\nc) 5000\nd) 10000"}], "ideal": "a) 54000", "doi": "10.3390/ma12182892
""",
    "answer": "a) 54000"
    },

    {
        "text": """Table 2: Predictive performance of our proposed model using different observation windows quantified with R2. The mean over 5-fold cross validation is reported.

\begin{tabular}{ccc}
\hline Observation window & w/o graph encoder & w graph encoder \\
\hline 7 days & 0.233 & \(\mathbf{0 . 3 0 2}\) \\
14 days & 0.456 & \(\mathbf{0 . 4 7 9}\) \\
21 days & 0.586 & \(\mathbf{0 . 6 0 8}\) \\
28 days & 0.652 & \(\mathbf{0 . 6 5 9}\) \\
\hline
\end{tabular}

of \(7,14,21\), and 28 days, to simulate real-world scenarios where early observations are used to forecast the (future unseen) tumor volume trajectory.

We assessed the predictive performance of our model in two ways. Firstly, we employed R2 to quantify the accuracy of our model in predicting unseen tumor volumes. The results in Table 2 indicate the following: 1) The embedding learned from the heterogeneous graph encoder enhances the predictive performance of our proposed model, and 2) as the observation window size increases, our proposed model captures the unseen tumor dynamic more accurately. Additionally, as it is demonstrated in Panel (B) of Figure 2, the model effectively captures the tumor dynamic trend. However the due to noise inherent in the tumor volume measurements and the clinical significance of mRECIST response category prediction, we also assessed our proposed model's predictive performance as a classifier. The mRECIST categories are derived from the predicted tumor volume time series by applying response criteria. This evaluation measures the model's performance in correctly classifying the treatment responses based on the predicted tumor volume dynamics. Figure 3 summarizes the classification results, revealing an observable trend in which incorporating the heterogeneous graph encoder embedding improves the prediction of response categories across all observation windows.

![](http://deeplinker-cai.oss-cn-zhangjiakou.aliyuncs.com/a8857995-3b5b-46c7-9e8d-6d9f30a83083.jpg?OSSAccessKeyId=LTAI5t65Z7BkkgW2MSCZMUML&Expires=1722673356&Signature=2yhKCQ6Z29V7lJJMFtuzdbkvePs%3D)

Figure 3: Predictive performance of our proposed model as a classifier for mRECIST categories, with and without the heterogeneous graph encoder and considering different lengths of observation windows.

\title{
5 Conclusion
}

In summary, we proposed a novel approach for tumor dynamic prediction that integrates RNA-seq, treatment, disease and longitudinal tumor volume data in an Neural-ODE system in a pre-clinical, PDX setting. We demonstrated that the use of Neural-ODE vastly improved the ability of the model to capture PDX tumor data than a previously proposed TGI model, as well as the benefit of adding the graph encoder to enrich the longitudinal data. As an area for further work, disentangling how the model predictions arise from the multimodal data using explainability techniques and/or attention weights is an important topic to advance our scientific understanding of the complex interplay between gene expression profiles, tumor location and drug targets. This methodology holds significant promise and warrants further evaluations, including in the clinical setting.


Question: In Figure 3, which has a higher accurate score, with the graph encoder or without? \n\na) with graph encoder \nb) w/o graph encoder""",

        "answer": "a) with graph encoder"
    },
]


multiple_choice_template = [
    {
        "system": """
        Generate synthetic data for training models based on the multiple choice format.
        Ensure the data reflects realistic values for the type of materials and characteristics described, and include given keywords in the paragraph.
        The text should explicitly include a description of a multiple choice question.
        The output should be formatted in JSON with attributes "text" and "answer". The value for the two attributes should be two string. Do not print out anything other than the JSON output.

        Below are some examples based on this requirement.
    
        Examples: {examples}.

        """,
        "user": """
        I need synthetic training data for training a machine learning model.
        The data should be formatted in JSON, with each entry containing "text" and "answer" attributes.
        You should generate a paragraph that includes the keywords: \n\n {keywords} \n\n.
        The "text" part must contain enough information for the multiple choice question to be answered!

        Your answer should contain as much details as possible. You should only generate one JSON. 
        The value for the two attributes should be two string. Use {{ and }} to warp your output. 
        Remember to put a comma at the end of the first string. Never use a json block to wrap your output.
        Here is the format for your output:
        
        {{
            "text": "Your paragraph and question here, remember that you should have a multiple choice question at the end of the text.",
            "answer": "Your answer here"
        }}
        
        Now start your answer: 
        """,
    }
]


if __name__ == "__main__":
    print(multiple_choice_template[0]["system"])
    print(multiple_choice_template[0]["user"])