# NLP_Project
 Insertion, Deletion, and Substitution Error Correction in DNA  Using Semantic Communication Techniques Project

 This project investigates the accuracy of semantic communication techniques in correcting insertion, deletion, and substitution (IDS) errors within DNA sequences. Unlike conventional DNA storage systems that encode synthetic text or images, our work centers on real biological data, analyzing how well semantic methods handle real DNA sequencing errors. We evaluated multiple semantic IDS correction approaches, such as DeepJSOC, SemAI-DNA, and Two RRNS, on real genomic datasets. We assess these methods using both characterlevel Levenshtein distance and semantic similarity scoring with DNABERT. Our goal is to provide a comprehensive understanding of how semantic error correction performs in biologically realistic data, ultimately contributing to more reliable DNA error correction systems.

### Added an evaluation metric to see how good ids works and here are the results: 

Evaluation Results:
BLEU Score         : 0.9642
Bit Error Rate     : 0.7108
Levenshtein Dist.  : 881 (Normalized: 0.0881)
BERTScore F1       : 1.0000 (P=1.0000, R=1.0000)
Note: BERTScore is optimized for natural language, not DNA.