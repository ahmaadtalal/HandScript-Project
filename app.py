from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import cv2
import random
import io
import base64
from PIL import Image, ImageFilter

app = Flask(__name__)

# --- STEP 1: Load the new SLIM model ---
# We no longer need EMNIST_DATA or TARGETS as the images are inside the dict
with open('handwriting_model_slim.pkl', 'rb') as f:
    SLIM_DATA = pickle.load(f)

# --- STEP 2: Simplified Image Fetcher ---
def get_char_image(char):
    """Fetches the pre-stored golden image for the character."""
    if char in SLIM_DATA:
        raw_img = SLIM_DATA[char]
        # EMNIST is naturally rotated/flipped; we fix it here
        return np.fliplr(np.rot90(raw_img, k=3))
    return None

def generate_line(text, slant=0.1, ink_color=(20, 20, 100)):
    chars = []
    base_h, canvas_h = 90, 130
    tall_chars = "bdfhklitABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    descenders = "gjpqy"

    for char in text:
        if char == " ":
            chars.append(np.ones((canvas_h, 35, 3), dtype=np.uint8) * 255)
            continue

        if char in ".,!-":
            mask_np = np.zeros((30, 30), dtype=np.uint8)
            if char == '.':
                cv2.circle(mask_np, (15, 20), random.randint(3, 5), 255, -1)
            elif char == ',':
                cv2.line(mask_np, (15, 15), (12, 25), 255, 3)
            elif char == '-':
                cv2.line(mask_np, (5, 15), (25, 15), 255, 3)
            elif char == '!':
                mask_np = np.zeros((60, 20), dtype=np.uint8)
                cv2.line(mask_np, (10, 5), (10, 40), 255, 3)
                cv2.circle(mask_np, (10, 52), 3, 255, -1)
            mask_pil = Image.fromarray(mask_np)
        else:
            raw_img = get_char_image(char)
            if raw_img is None:
                continue
            mask_pil = Image.fromarray(raw_img.astype(np.uint8))

        mask_np = np.array(mask_pil)
        coords = np.argwhere(mask_np > 15)
        if coords.size == 0:
            continue

        y0, x0, y1, x1 = coords[:, 0].min(), coords[:, 1].min(), coords[:, 0].max(), coords[:, 1].max()
        mask_pil = mask_pil.crop((x0, y0, x1 + 1, y1 + 1))

        if char in ".,-": new_h = int(base_h * 0.2)
        elif char == "!": new_h = int(base_h * 0.8)
        elif char in tall_chars: new_h = base_h
        elif char in descenders: new_h = int(base_h * 0.85)
        else: new_h = int(base_h * 0.65)

        new_w = int(new_h * (mask_pil.width / mask_pil.height))
        mask_pil = mask_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

        mask_pil = mask_pil.rotate(random.uniform(-1.0, 1.0), resample=Image.Resampling.BICUBIC, expand=True)
        w_m, h_m = mask_pil.size
        xshift = abs(slant) * w_m
        mask_pil = mask_pil.transform(
            (w_m + int(xshift), h_m), Image.AFFINE,
            (1, slant, -xshift if slant > 0 else 0, 0, 1, 0),
            Image.Resampling.BILINEAR
        )

        char_canvas = Image.new('RGB', mask_pil.size, (255, 255, 255))
        char_canvas.paste(Image.new('RGB', mask_pil.size, ink_color), (0, 0), mask_pil)

        final_c = Image.new('RGB', (char_canvas.width, canvas_h), (255, 255, 255))
        if char in ".,-": offset = canvas_h - char_canvas.height - 35
        elif char in descenders: offset = canvas_h - char_canvas.height - 10
        else: offset = canvas_h - char_canvas.height - 35

        final_c.paste(char_canvas, (0, offset + random.randint(-1, 1)))
        final_c = final_c.filter(ImageFilter.GaussianBlur(0.2)).filter(ImageFilter.UnsharpMask(1.5, 150, 2))
        chars.append(np.array(final_c))

    if not chars:
        return None

    line_img = chars[0]
    for c in chars[1:]:
        line_img = np.hstack([line_img, np.ones((canvas_h, random.randint(3, 6), 3), dtype=np.uint8) * 255, c])
    return line_img

def render_page(lines_text, slant, ink_color):
    paper_w, paper_h = 2480, 3508
    paper = Image.new('RGB', (paper_w, paper_h), (255, 255, 255))
    margin_left, margin_right = 200, 200
    max_w = paper_w - margin_left - margin_right
    current_y, line_height, max_y = 250, 160, paper_h - 250
    remaining_lines = []
    text_buffer = list(lines_text)

    while text_buffer:
        line_text = text_buffer.pop(0)
        if current_y + line_height > max_y:
            remaining_lines.append(line_text)
            remaining_lines.extend(text_buffer)
            break
        words = line_text.split(' ')
        current_line = ""
        for i, word in enumerate(words):
            test = current_line + (" " if current_line else "") + word
            test_img = generate_line(test, slant, ink_color)
            if test_img is not None and test_img.shape[1] > max_w:
                if current_line:
                    final_img = generate_line(current_line, slant, ink_color)
                    if final_img is not None:
                        paper.paste(Image.fromarray(final_img), (margin_left, current_y))
                        current_y += line_height
                        if current_y + line_height > max_y:
                            remaining_words = " ".join(words[i:])
                            text_buffer.insert(0, remaining_words)
                            current_line = ""
                            break
                current_line = word
            else:
                current_line = test
        if current_line and current_y + line_height <= max_y:
            final_img = generate_line(current_line, slant, ink_color)
            if final_img is not None:
                paper.paste(Image.fromarray(final_img), (margin_left, current_y))
                current_y += line_height
        current_y += 40
    return paper, remaining_lines

def image_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, 'PNG')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    full_text = request.form.get('text', '')
    slant = float(request.form.get('slant', 0.1))
    ink_map = {'navy': (20, 20, 100), 'black': (10, 10, 10), 'blue': (0, 80, 200)}
    ink_color = ink_map.get(request.form.get('ink_color', 'navy'), (20, 20, 100))
    all_lines = full_text.split('\n')
    pages_b64 = []
    remaining = all_lines
    while remaining:
        page_img, remaining = render_page(remaining, slant, ink_color)
        pages_b64.append(image_to_base64(page_img))
        if len(remaining) == len(all_lines): break
    return jsonify({'pages': pages_b64})

if __name__ == '__main__':
    app.run(debug=True)