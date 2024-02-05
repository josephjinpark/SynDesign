Source code for SynDesign

SynDesign is a web-based tool designed for the creation of prime editing guide RNAs (pegRNAs). It offers a comprehensive solution for pegRNA design, evaluation, and library construction for saturation genome editing (SGE). The tool accommodates various gene identification inputs, including gene symbols and identifiers from popular databases such as NCBI's GI, RefSeq, Ensemble, and HGNC IDs.

The evaluation process relies on DeepPrime, a prime editing efficiency prediction tool based on deep learning. Developed to forecast pegRNA efficiency across various prime editors and cell types, DeepPrime is adept at predicting outcomes for substitutions, insertions, and deletions up to 3 base pairs in size. It also considers primer binding site (PBS) lengths spanning 1 to 17 base pairs, reverse transcription template (RTT) lengths ranging from 1 to 50 base pairs, and edit positions from +1 to +30.

For more details on DeepPrime and its methodology, please refer to  Yu et al., Cell 2023 (https://doi.org/10.1016/j.cell.2023.03.034).


To run SynDesign, please refer to Genet, the base toolkit for all things related genome editing from our lab!
