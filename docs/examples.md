@page examples

Examples
========

Directory `examples` contains 2 subfolders: `problem` and `approximation`.
Sources in `problem` show how to generate problem (spatial statistics, minimal
or dense) and how to create STARSH\_problem instance, required for every step
of STARSH. Examples in `approximation` are based on problem generation and have
additional steps on approximation of corresonding matrices.

*Important notice: approximation does not require dense matrix to be stored
anywhere, only required matrix elements are computed when needed.*
