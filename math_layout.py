import random
import argparse
import json


DEFAULT_LAYOUT = {
    'char_width': 18.0,
    'line_height': 64.0,
    'superscript_scale': 0.6,
    'subscript_scale': 0.6,
    'superscript_raise': 22.0,
    'subscript_drop': 12.0,
    'fraction_gap': 12.0,
    'fraction_rule_margin': 10.0,
    'fraction_rule_padding': 5.0,
    'fraction_rule_thickness': 2.0,
    'node_spacing': 6.0,
}


class LatexParser(object):
    """Parse a compact LaTeX subset into an AST suitable for handwriting layout."""

    def __init__(self, expression):
        self.tokens = self._tokenize(expression)
        self.index = 0

    def parse(self):
        node = self._parse_expression(stop_tokens=None)
        if self.index != len(self.tokens):
            raise ValueError('Unexpected token {} at position {}'.format(self.tokens[self.index], self.index))
        return node

    def _tokenize(self, expression):
        tokens = []
        i = 0
        while i < len(expression):
            char = expression[i]
            if char.isspace():
                i += 1
                continue
            if char in ['{', '}', '^', '_']:
                tokens.append(char)
                i += 1
                continue
            if char == '\\':
                j = i + 1
                while j < len(expression) and expression[j].isalpha():
                    j += 1
                command = expression[i + 1:j]
                if not command:
                    raise ValueError('Invalid LaTeX command near position {}'.format(i))
                tokens.append(('CMD', command))
                i = j
                continue
            tokens.append(('TXT', char))
            i += 1
        return tokens

    def _peek(self):
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def _consume(self, expected=None):
        token = self._peek()
        if token is None:
            raise ValueError('Unexpected end of expression.')
        if expected is not None and token != expected:
            raise ValueError('Expected token {} but got {}'.format(expected, token))
        self.index += 1
        return token

    def _parse_expression(self, stop_tokens):
        children = []
        while self._peek() is not None and (stop_tokens is None or self._peek() not in stop_tokens):
            children.append(self._parse_atom_with_scripts())
        if len(children) == 1:
            return children[0]
        return {'type': 'sequence', 'children': children}

    def _parse_atom_with_scripts(self):
        atom = self._parse_atom()

        while self._peek() in ['^', '_']:
            token = self._consume()
            scripted = self._parse_script_operand()
            if token == '^':
                atom = {'type': 'superscript', 'base': atom, 'exponent': scripted}
            else:
                atom = {'type': 'subscript', 'base': atom, 'subscript': scripted}
        return atom

    def _parse_script_operand(self):
        if self._peek() == '{':
            return self._parse_group()
        return self._parse_atom()

    def _parse_group(self):
        self._consume('{')
        node = self._parse_expression(stop_tokens=['}'])
        self._consume('}')
        return node

    def _parse_atom(self):
        token = self._peek()
        if token is None:
            raise ValueError('Unexpected end of expression while parsing atom.')

        if token == '{':
            return self._parse_group()

        if isinstance(token, tuple) and token[0] == 'CMD':
            self._consume()
            cmd = token[1]
            if cmd == 'frac':
                numerator = self._parse_group()
                denominator = self._parse_group()
                return {'type': 'fraction', 'numerator': numerator, 'denominator': denominator}
            if cmd in ['cdot', 'times']:
                return {'type': 'text', 'value': '·' if cmd == 'cdot' else 'x'}
            raise ValueError('Unsupported LaTeX command \\{}'.format(cmd))

        if isinstance(token, tuple) and token[0] == 'TXT':
            self._consume()
            return {'type': 'text', 'value': token[1]}

        raise ValueError('Unexpected token {}'.format(token))


class MathLayoutEngine(object):
    """
    Build layout instructions from an AST.

    Each draw instruction contains:
    - text: token sent to the handwriting model
    - x/y: target origin
    - scale: geometric scaling factor
    - role: semantic tag (base, superscript, numerator, ...)
    """

    def __init__(self, layout_config=None):
        self.config = dict(DEFAULT_LAYOUT)
        if layout_config:
            self.config.update(layout_config)

    def layout(self, ast):
        instructions, bbox = self._layout_node(ast, 0.0, 0.0, 1.0, role='base')
        return {'instructions': instructions, 'bbox': bbox}

    def _layout_node(self, node, x, baseline, scale, role):
        node_type = node['type']
        if node_type == 'text':
            width = self.config['char_width'] * scale
            ascent = self.config['line_height'] * 0.65 * scale
            descent = self.config['line_height'] * 0.35 * scale
            instruction = {
                'text': node['value'],
                'x': x,
                'y': baseline,
                'scale': scale,
                'role': role,
            }
            bbox = {
                'x_min': x,
                'x_max': x + width,
                'y_min': baseline - ascent,
                'y_max': baseline + descent,
                'width': width,
                'height': ascent + descent,
                'baseline': baseline,
            }
            return [instruction], bbox

        if node_type == 'sequence':
            cursor = x
            all_instructions = []
            boxes = []
            for child in node['children']:
                child_instructions, child_box = self._layout_node(child, cursor, baseline, scale, role=role)
                all_instructions.extend(child_instructions)
                boxes.append(child_box)
                cursor = child_box['x_max'] + (self.config['node_spacing'] * scale)

            if not boxes:
                return [], {
                    'x_min': x, 'x_max': x, 'y_min': baseline, 'y_max': baseline,
                    'width': 0.0, 'height': 0.0, 'baseline': baseline,
                }
            bbox = self._merge_boxes(boxes, baseline)
            return all_instructions, bbox

        if node_type == 'superscript':
            base_instructions, base_box = self._layout_node(node['base'], x, baseline, scale, role='base')
            exponent_scale = scale * self.config['superscript_scale']
            exponent_baseline = baseline - (self.config['superscript_raise'] * scale)
            exponent_x = base_box['x_max']
            exponent_instructions, exponent_box = self._layout_node(
                node['exponent'], exponent_x, exponent_baseline, exponent_scale, role='superscript'
            )
            bbox = self._merge_boxes([base_box, exponent_box], baseline)
            return base_instructions + exponent_instructions, bbox

        if node_type == 'subscript':
            base_instructions, base_box = self._layout_node(node['base'], x, baseline, scale, role='base')
            subscript_scale = scale * self.config['subscript_scale']
            subscript_baseline = baseline + (self.config['subscript_drop'] * scale)
            subscript_x = base_box['x_max']
            subscript_instructions, subscript_box = self._layout_node(
                node['subscript'], subscript_x, subscript_baseline, subscript_scale, role='subscript'
            )
            bbox = self._merge_boxes([base_box, subscript_box], baseline)
            return base_instructions + subscript_instructions, bbox

        if node_type == 'fraction':
            num_scale = scale * 0.9
            den_scale = scale * 0.9
            num_baseline = baseline - (self.config['fraction_gap'] + self.config['line_height'] * 0.3) * scale
            den_baseline = baseline + (self.config['fraction_gap'] + self.config['line_height'] * 0.4) * scale

            num_instr, num_box = self._layout_node(node['numerator'], x, num_baseline, num_scale, role='numerator')
            den_instr, den_box = self._layout_node(node['denominator'], x, den_baseline, den_scale, role='denominator')

            frac_width = max(num_box['width'], den_box['width']) + self.config['fraction_rule_margin'] * scale

            num_shift = x + (frac_width - num_box['width']) / 2.0 - num_box['x_min']
            den_shift = x + (frac_width - den_box['width']) / 2.0 - den_box['x_min']
            self._shift_instructions(num_instr, num_shift, 0.0)
            self._shift_instructions(den_instr, den_shift, 0.0)
            num_box = self._shift_box(num_box, num_shift, 0.0)
            den_box = self._shift_box(den_box, den_shift, 0.0)

            rule_instruction = {
                'type': 'rule',
                'x1': x,
                'x2': x + frac_width,
                'y': baseline,
                'role': 'fraction_rule',
                'thickness': self.config['fraction_rule_thickness'] * scale,
            }

            all_instructions = num_instr + den_instr + [rule_instruction]
            bbox = self._merge_boxes([
                num_box,
                den_box,
                {
                    'x_min': x,
                    'x_max': x + frac_width,
                    'y_min': baseline - self.config['fraction_rule_padding'] * scale,
                    'y_max': baseline + self.config['fraction_rule_padding'] * scale,
                    'width': frac_width,
                    'height': self.config['fraction_rule_padding'] * 2.0 * scale,
                    'baseline': baseline,
                }
            ], baseline)
            return all_instructions, bbox

        raise ValueError('Unsupported node type: {}'.format(node_type))

    def _shift_instructions(self, instructions, dx, dy):
        for instruction in instructions:
            if instruction.get('type', 'glyph') == 'rule':
                instruction['x1'] += dx
                instruction['x2'] += dx
                instruction['y'] += dy
            else:
                instruction['x'] += dx
                instruction['y'] += dy

    def _shift_box(self, box, dx, dy):
        shifted = dict(box)
        shifted['x_min'] += dx
        shifted['x_max'] += dx
        shifted['y_min'] += dy
        shifted['y_max'] += dy
        shifted['baseline'] += dy
        return shifted

    def _merge_boxes(self, boxes, baseline):
        return {
            'x_min': min(box['x_min'] for box in boxes),
            'x_max': max(box['x_max'] for box in boxes),
            'y_min': min(box['y_min'] for box in boxes),
            'y_max': max(box['y_max'] for box in boxes),
            'width': max(box['x_max'] for box in boxes) - min(box['x_min'] for box in boxes),
            'height': max(box['y_max'] for box in boxes) - min(box['y_min'] for box in boxes),
            'baseline': baseline,
        }


class ChunkSynthesizer(object):
    """Generate stroke arrays for token-sized text chunks via the untouched Hand model."""

    def __init__(self, hand=None, bias=0.75, style=9):
        if hand is None:
            from demo import Hand
            hand = Hand()
        self.hand = hand
        self.bias = bias
        self.style = style
        self.cache = {}

    def generate_offsets(self, text):
        import numpy as np
        import drawing

        if text in self.cache:
            return np.copy(self.cache[text])

        if text == '·':
            model_text = '.'
        else:
            model_text = text

        sample = self.hand._sample([model_text], biases=[self.bias], styles=[self.style])[0]
        sample[:, :2] *= 1.5
        sample = drawing.offsets_to_coords(sample)
        sample = drawing.denoise(sample)
        sample[:, :2] = drawing.align(sample[:, :2])
        offsets = drawing.coords_to_offsets(sample)
        self.cache[text] = np.copy(offsets)
        return offsets


class CanvasStitcher(object):
    """Compose generated chunks into one canvas with slight jitter for natural handwriting."""

    def __init__(self, jitter_scale=0.015, seed=None):
        self.random = random.Random(seed)
        self.jitter_scale = jitter_scale

    def stitch(self, instructions, synthesizer):
        import numpy as np
        import drawing

        assembled = []
        for instruction in instructions:
            if instruction.get('type') == 'rule':
                assembled.append(self._rule_to_coords(instruction))
                continue

            offsets = synthesizer.generate_offsets(instruction['text'])
            coords = drawing.offsets_to_coords(offsets)

            # normalize local origin
            coords[:, 0] -= coords[:, 0].min()
            coords[:, 1] -= np.median(coords[:, 1])

            scale = instruction['scale'] * self._jitter(1.0)
            coords[:, :2] *= scale
            coords[:, 0] += instruction['x']
            coords[:, 1] += instruction['y'] + self._jitter(0.0, amplitude=2.0)
            assembled.append(coords)

        if not assembled:
            return np.zeros((0, 3))

        merged = [assembled[0]]
        for coords in assembled[1:]:
            prev_end = merged[-1][-1:, :]
            bridge = np.copy(prev_end)
            bridge[0, 2] = 1.0
            merged.append(bridge)
            merged.append(coords)
        return np.vstack(merged)

    def _rule_to_coords(self, instruction):
        import numpy as np
        return np.array([
            [instruction['x1'], instruction['y'], 0.0],
            [instruction['x2'], instruction['y'], 1.0],
        ])

    def _jitter(self, center, amplitude=None):
        amp = amplitude if amplitude is not None else self.jitter_scale
        return center + self.random.uniform(-amp, amp)


class MathHandWriter(object):
    """End-to-end wrapper: parse LaTeX, generate chunks, stitch and render to SVG."""

    def __init__(self, hand=None, layout_config=None, jitter_scale=0.015, seed=None):
        self.parser_cls = LatexParser
        self.layout = MathLayoutEngine(layout_config=layout_config)
        self.synth = ChunkSynthesizer(hand=hand)
        self.stitcher = CanvasStitcher(jitter_scale=jitter_scale, seed=seed)

    def compile(self, expression):
        ast = self.parser_cls(expression).parse()
        layout = self.layout.layout(ast)
        coords = self.stitcher.stitch(layout['instructions'], self.synth)
        return {
            'ast': ast,
            'layout': layout,
            'coords': coords,
        }

    def write_svg(self, expression, filename, stroke_color='black', stroke_width=2):
        import svgwrite

        compiled = self.compile(expression)
        coords = compiled['coords']

        if len(coords) == 0:
            raise ValueError('No coordinates generated for expression {}'.format(expression))

        margin = 20
        x_min, y_min = coords[:, 0].min() - margin, coords[:, 1].min() - margin
        x_max, y_max = coords[:, 0].max() + margin, coords[:, 1].max() + margin
        width = max(200, x_max - x_min)
        height = max(120, y_max - y_min)

        dwg = svgwrite.Drawing(filename=filename)
        dwg.viewbox(minx=x_min, miny=y_min, width=width, height=height)
        dwg.add(dwg.rect(insert=(x_min, y_min), size=(width, height), fill='white'))

        prev_eos = 1.0
        path_commands = []
        for x, y, eos in coords:
            cmd = 'M' if prev_eos == 1.0 else 'L'
            path_commands.append('{}{},{}'.format(cmd, x, y))
            prev_eos = eos

        path = svgwrite.path.Path(' '.join(path_commands))
        path = path.stroke(color=stroke_color, width=stroke_width, linecap='round').fill('none')
        dwg.add(path)
        dwg.save()
        return compiled


def inspect_expression(expression):
    """Print parser/layout diagnostics without loading TensorFlow or running the model."""
    ast = LatexParser(expression).parse()
    layout = MathLayoutEngine().layout(ast)
    print('AST:')
    print(json.dumps(ast, indent=2, sort_keys=True))
    print('')
    print('Layout summary:')
    print('  instructions: {}'.format(len(layout['instructions'])))
    print('  bbox: {}'.format(layout['bbox']))


def _cli():
    parser = argparse.ArgumentParser(description='Inspect or render LaTeX-style handwriting layout.')
    parser.add_argument('expression', help='Expression like x^{2}+\\frac{1}{y}')
    parser.add_argument('--inspect-only', action='store_true', help='Only print AST/layout information.')
    parser.add_argument('--out', default='img/math_demo.svg', help='Output SVG path for rendering mode.')
    args = parser.parse_args()

    inspect_expression(args.expression)
    if not args.inspect_only:
        writer = MathHandWriter(seed=7)
        writer.write_svg(args.expression, args.out)
        print('Rendered SVG: {}'.format(args.out))


if __name__ == '__main__':
    _cli()
