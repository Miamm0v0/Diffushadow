# Diffushadow Supplementary Information

This LaTeX project contains the draft Supplementary Information for Diffushadow. It can be compiled locally or uploaded to Overleaf.

## Files

- `Diffushadow_Supplementary_sections.tex`: the current Supplementary Information manuscript.
- `supplementary_refs.bib`: the BibTeX reference database.
- `Diffushadow_Supplementary_sections.pdf`: the compiled manuscript.
- `figures/`: supplementary figure images.
- `main.tex`: the original Supplementary Information template.
- `.latexmkrc`: default latexmk settings for pdfLaTeX, with output in `build/`.

## Local Compilation

Run these commands from this directory:

```powershell
xelatex Diffushadow_Supplementary_sections.tex
bibtex Diffushadow_Supplementary_sections
xelatex Diffushadow_Supplementary_sections.tex
xelatex Diffushadow_Supplementary_sections.tex
```

The BibTeX step generates the reference list; the following two XeLaTeX runs resolve citation numbers. Alternatively, use `latexmk -xelatex Diffushadow_Supplementary_sections.tex` to manage these steps automatically (output goes to `build/`).

## Overleaf Use

1. Zip the contents of this directory, including `figures/`.
2. Upload the zip file as a new Overleaf project.
3. Set `Diffushadow_Supplementary_sections.tex` as the main file.
4. Select `XeLaTeX` as the compiler. Overleaf runs BibTeX automatically when needed.

## Notes For Adapting

- Replace bracketed placeholders such as `[Full manuscript title]`.
- Keep supplementary figures, tables, equations, videos, audio files, datasets, and software numbered separately and sequentially.
- Keep the supplementary material concise and only include information needed to support the main manuscript.
- Put figure files in `figures/` and reference them with `\includegraphics`.
- For final submission, check the journal portal instructions for the exact required file names and whether references should be pasted directly into the TeX file.

## Official References

- Nature Machine Intelligence submission guidelines: https://www.nature.com/natmachintell/submission-guidelines
- Nature Machine Intelligence initial submission / formatting guidance: https://www.nature.com/natmachintell/submission-guidelines/aip-and-formatting
