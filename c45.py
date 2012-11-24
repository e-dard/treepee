
"""
C4.5 is an extension to ID3 by Quinlan, which attempts to address the 
folloing issues:

 - Avoiding overfitting
 - Determining how deeply to grow a decision tree.
 - Reduced error pruning.
 - Rule post-pruning.
 - Handling continuous attirbute values by using constraints.
 - Choosing an appropriate attribute selection measure.
 - Handling training data with missing attribute values.
 - Handling attributes with differing costs.
 - Improving computational efficiency.
 From http://www2.cs.uregina.ca/~dbd/cs831/notes/ml/dtrees/c4.5/tutorial.html
"""