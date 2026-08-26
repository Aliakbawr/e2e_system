# Bachelor Project Report

LaTeX sources for the Persian voice-chatbot and ASR-to-LLM error-propagation final-project report.

## Structure

```text
report/
├── main.tex
├── references.bib
├── chapters/
│   ├── chapter1.tex
│   └── ...
└── appendices/
    ├── appendixA.tex
    └── appendixB.tex
```

The original Windows project used the misspelled directory name `apendices/`. It was normalized to `appendices/` during import, and the corresponding `\include` paths in `main.tex` were updated.

## Build

The document uses `xepersian`, so compile it with XeLaTeX rather than pdfLaTeX. The configured fonts are `B Nazanin` and `Times New Roman` and must be installed on the build machine.

Using `latexmk`:

```bash
cd report
latexmk -xelatex main.tex
```

Or run the tools manually:

```bash
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

Generated PDFs, auxiliary files, logs, and SyncTeX data are intentionally excluded from version control.
