# Build the final report

All numbers, tables and figures in main.tex are made by make_assets.py from the REAL data.
Nothing is typed by hand.

## Folder layout (inside your repo)
```
machine-learning-labwork1-pca/
├── data/                 (raw/ and processed/ are filled automatically)
└── report/
    ├── main.tex
    ├── placeholder.tex
    ├── make_assets.py
    └── requirements_report.txt
```

## Steps
1. Open a terminal in the ROOT of the repo (the folder that contains `report/`).
2. Create and activate a virtual environment:
   - Windows:  `python -m venv .venv`  then  `.venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv .venv`  then  `source .venv/bin/activate`
3. Install the libraries: `pip install -r report/requirements_report.txt`
4. Run: `python report/make_assets.py`   (Mac/Linux: `python3 ...`)
5. Check the first two lines of the output:
   `Raisin shape : (900, 8)` and `Abalone shape: (4177, 9)`.
6. Compile `report/main.tex` (pdfLaTeX):
   - Overleaf: New Project -> Upload -> upload the files INSIDE `report/` (main.tex, placeholder.tex,
     results.tex and the folders tables/ and figures/). Compiler: pdfLaTeX. Main document: main.tex.
   - Local: `cd report` then `pdflatex main.tex` (run it twice).
7. Search for `CHECK` in main.tex (4 places) and write the sentence it asks for.
8. Read the final PDF once. Every sentence must match your figures and tables. Count the pages (target: 10).

## If something goes wrong
- `ModuleNotFoundError: No module named ...` -> run step 3 again (and check the virtual environment is active).
- "Could not download ... from UCI" -> download the files by hand and put them in `data/raw/`:
  Raisin  -> `Raisin_Dataset.xlsx`   (inside the zip from the UCI Raisin page)
  Abalone -> `abalone.data`          (inside the zip from the UCI Abalone page)
  Then run step 4 again.
- Shapes are not (900, 8) / (4177, 9) -> stop and check that you use the UCI files.
- The report has 11 pages -> in main.tex make the figures a bit smaller (the number after
  `\safefig{...}{0.8}`, for example 0.8 -> 0.7).
- A figure or table shows "not generated yet" -> step 4 did not finish; read its error message.
- Wrong name on the title page -> edit `\authorname` and `\authorid` at the top of main.tex.
