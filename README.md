# Opinion mining with a combination of semantic and syntactic features
A study on opinion mining in news. The goal is to extract opinions in a text corpus without supervision. Many algorithms that tackle this issue are generative models based on lexical features. Our goal is to determine the entities defying an opinion amongst lexical, syntactic and semantic features as well as their compositions. More specifically, we test the hypothesis that an opinion is determined by the composition of  the mentioned facts (SPO triples), the semantic roles of the words and the sentiment lexicon used in it. In this paper we formalise this task and prove that using a composition of the above features provides the best quality when clusterising opinionated texts. To test this hypothesis we gathered and labelled two corpuses of news on political events and proposed unsupervised algorithms for extracting the features.

## Data description
We have collected and labelled two corpora of news:
- 91 news considering enterprise nationalisation in LPR and DPR. The texts average at 200 words. They were extracted from multiple news sources: Russian as well as Ukrainian. We selected texts expressing two opinions: Moscow's opinion and Kiev's opinion, most texts on the topic belong to one of them.
- 220 news considering Donald Trump's decision of quitting the Paris Climate Agreement. The text's sizes once again averaged at around 200 words. The news were equally distributed between two opinions: one of Trump's supporters, those who oppose him (such as Elon Musk).


In [data](./data) there are two files corresponding to the corpora. Each file consists of enumerated lines, where "|text" marks the beginning of the news text itself and opinion follows the label "|mark".

In the corpus on [LPR and DPR](./data/lnr_dnr_labelled.txt) opinions are:
- 0 marks the neutral opinion
- 1 marks pro-Ukrainian position
- 2 marks pro-Russian position

In the corpus on [Trump's decision(./data/trump_labelled.txt) to leave the PCA opinions are:
- 0 marks the neutral opinion
- 1 marks the opinion of Trump's opposers
- 2 marks the opinion of Trump's supporters
