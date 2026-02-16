![](img/banner.svg)
# Handwriting Synthesis
Implementation of the handwriting synthesis experiments in the paper <a href="https://arxiv.org/abs/1308.0850">Generating Sequences with Recurrent Neural Networks</a> by Alex Graves.  The implementation closely follows the original paper, with a few slight deviations, and the generated samples are of similar quality to those presented in the paper.

Web demo is available <a href="https://seanvasquez.com/handwriting-generation/">here</a>.

## Usage
```python
lines = [
    "Now this is a story all about how",
    "My life got flipped turned upside down",
    "And I'd like to take a minute, just sit right there",
    "I'll tell you how I became the prince of a town called Bel-Air",
]
biases = [.75 for i in lines]
styles = [9 for i in lines]
stroke_colors = ['red', 'green', 'black', 'blue']
stroke_widths = [1, 2, 1, 2]

hand = Hand()
hand.write(
    filename='img/usage_demo.svg',
    lines=lines,
    biases=biases,
    styles=styles,
    stroke_colors=stroke_colors,
    stroke_widths=stroke_widths
)
```
![](img/usage_demo.svg)

Currently, the `Hand` class must be imported from `demo.py`.  If someone would like to package this project to make it more usable, please [contribute](#contribute).

A pretrained model is included, but if you'd like to train your own, read <a href='https://github.com/sjvasquez/handwriting-synthesis/tree/master/data/raw'>these instructions</a>.

## Demonstrations
Below are a few hundred samples from the model, including some samples demonstrating the effect of priming and biasing the model.  Loosely speaking, biasing controls the neatness of the samples and priming controls the style of the samples. The code for these demonstrations can be found in `demo.py`.

### Demo #1:
The following samples were generated with a fixed style and fixed bias.

**Smash Mouth – All Star (<a href="https://www.azlyrics.com/lyrics/smashmouth/allstar.html">lyrics</a>)**
![](img/all_star.svg)

### Demo #2
The following samples were generated with varying style and fixed bias.  Each verse is generated in a different style.

**Vanessa Carlton – A Thousand Miles (<a href="https://www.azlyrics.com/lyrics/vanessacarlton/athousandmiles.html">lyrics</a>)**
![](img/downtown.svg)

### Demo #3
The following samples were generated with a fixed style and varying bias.  Each verse has a lower bias than the previous, with the last verse being unbiased.

**Leonard Cohen – Hallelujah (<a href="https://www.youtube.com/watch?v=dQw4w9WgXcQ">lyrics</a>)**
![](img/give_up.svg)

## Contribute
This project was intended to serve as a reference implementation for a research paper, but since the results are of decent quality, it may be worthwile to make the project more broadly usable.  I plan to continue focusing on the machine learning side of things.  That said, I'd welcome contributors who can:

  - Package this, and otherwise make it look more like a usable software project and less like research code.
  - Add support for more sophisticated drawing, animations, or anything else in this direction.  Currently, the project only creates some simple svg files.

## Experimental Math/LaTeX Layout Engine
A new optional wrapper (`math_layout.py`) adds 2D equation support without changing the core RNN model.

### Architecture
1. **Input parser (Phase 1):** `LatexParser` converts a subset of LaTeX (`^`, `_`, `\frac{...}{...}` and flat symbols) into a tree.
2. **Modular generation (Phase 2):** `ChunkSynthesizer` calls the existing `Hand._sample` model per symbol/chunk, keeping the LSTM+MDN untouched.
3. **Canvas stitcher (Phase 3):** `CanvasStitcher` scales/shifts chunk coordinates for superscripts, fractions, and subscripts, then injects small jitter so layout remains natural.

### Example
```python
from math_layout import MathHandWriter

writer = MathHandWriter(seed=7)
writer.write_svg(r"x^{2}+\\frac{1}{y}", "img/math_demo.svg")
```

The resulting file contains model-generated handwriting chunks stitched into a structured 2D equation.

By default, math stitching now uses zero jitter for more stable, textbook-like placement; pass a non-zero jitter to `MathHandWriter(..., jitter_scale=...)` if you want more hand-drawn variation.


### Getting visible output during testing
If you only want to verify parsing/layout and see terminal output (no TensorFlow/model run), use:
```bash
python math_layout.py 'x^{2}+\\frac{1}{y_0}' --inspect-only
```
This prints the parsed AST and a layout summary to stdout.

To render an SVG and also print progress:
```bash
python math_layout.py 'x^{2}+\\frac{1}{y_0}' --out img/math_demo.svg
```
This prints diagnostics first, then `Rendered SVG: ...` after writing the file.
If rendering dependencies are missing (for example `svgwrite` or `numpy`), the CLI now prints an install hint instead of a full traceback.
It also checks TensorFlow compatibility up front and warns when the runtime/build is incompatible with this TensorFlow 1.x-based project.

> Note: on PowerShell/Windows shells, both `\frac` and `\\frac` forms are accepted by the parser now.
> PowerShell note: use `echo $LASTEXITCODE` (not `echo $?`) to see numeric process exit codes. Example: `python math_layout.py ... --out img/math_demo.svg; echo EXIT:$LASTEXITCODE`.


### Windows quick-start (recommended)
If rendering keeps failing, **do not use base Python 3.8+** for this repo. Use a dedicated Python 3.7 env:

```powershell
conda create -n hwsyn37 python=3.7 -y
conda activate hwsyn37
python -m pip install -r requirements-py37.txt
python math_layout.py 'x^{2}+\frac{1}{y_0}' --doctor --inspect-only
python math_layout.py 'x^{2}+\frac{1}{y_0}' --out img/math_demo.svg
echo EXIT:$LASTEXITCODE
```

`--doctor` prints Python/dependency compatibility before rendering so setup issues are obvious.


> TensorFlow 1.15.x is sensitive to protobuf versions. If you see a `Descriptors cannot not be created directly` error, pin protobuf to `<=3.20.3`.

> If Python 3.7 reports `ModuleNotFoundError: No module named importlib_metadata`, run: `python -m pip install importlib-metadata`.


### PowerShell note for inline Python tests
The Bash-style heredoc (`python - <<'PY'`) does not work in PowerShell.
Use this instead:

```powershell
python math_layout.py 'x^{2}+\\frac{1}{y_0}' --smoke-test --inspect-only
```
