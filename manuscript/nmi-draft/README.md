# Nature Machine Intelligence manuscript

This directory contains the current Diffushadow manuscript, its bibliography,
LaTeX class, corrected Nature bibliography style, the eight figures used in the
manuscript, and the compiled PDF. Earlier figure versions and build files are
kept outside this upload directory.

## Files

- `sn-article.tex`: manuscript source.
- `sn-bibliography.bib`: bibliography database.
- `sn-jnl.cls`: Springer Nature LaTeX class.
- `sn-nature.bst`: corrected Nature bibliography style. Keep this file beside
  the manuscript; it fixes proceedings and book-chapter formatting in the
  supplied template.
- `fig/`: the eight figures referenced by the active manuscript.
- `sn-article.pdf`: compiled manuscript PDF.

## Build

With a LaTeX installation that provides `latexmk`, pdfLaTeX and BibTeX, run from
this directory:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error sn-article.tex
```

The manuscript was compiled with TeX Live 2025 before this directory was
prepared. The packaged sources and PDF are byte-for-byte copies of those files.

## Remaining author-provided material

- The Funding section is intentionally blank for the coauthors to complete.
- The manuscript cites Supplementary Information. That file was not available
  in the source directory and must be supplied before submission.

## Upload

Upload this entire `nmi-draft` directory under the repository's `manuscript/`
directory, retaining the `fig/` subdirectory structure.
